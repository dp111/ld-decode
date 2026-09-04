"""PROTOTYPE: batched demodulation on the GPU.

demodblock() is ~52% of a decode worker's time and is almost entirely FFTs and
elementwise products over 32768-point blocks - work a GPU does far faster than
a CPU.  The catch is transfer: one block at a time PCIe round-trip is 76% of
the call and the GPU *loses* (0.78x).  Batched, it amortises away.

Measured on real Domesday RF (CommunityNorth ds8), RTX 3060, blocklen 32768:

    CPU demodblock          2.281 ms/block
    GPU batch 4             0.705        3.24x
    GPU batch 8             0.836        2.73x
    GPU batch 16            0.738        3.09x

(a synthetic chain reaches 7x; the real one moves more data per block, so take
~3x.)  With the demod chain at ~52% of worker time that projects to ~1.5x on
the decode as a whole.

Output is BIT-IDENTICAL to demodblock() for all five video products; rfhpf
agrees to 9.7e-12 in float64 and is cast to float32 downstream.  So unlike the
float32 experiment, this is not a quality trade.

What this covers: input rfft, the rfhpf dropout reference (including its
rotdelay slice), the RF filter chain (RFVideo / FcutPAL / MTF**level), the
Hilbert transform, the conjugate-product FM discriminator in its historical
[0, tau) convention, and the batched four-product video irfft.

NOT yet covered - a block needing any of these must use the CPU path:
  * V4300D coherent subtract.  This matters: every decode in the Domesday
    campaign passes --V4300D_coherent_subtract, so the prototype cannot serve
    them as it stands.  Its work is frequency-domain and cheap but iterative
    and data-dependent per block, so it needs a per-block loop inside the
    batch rather than a batched kernel.
  * EFM and analog audio (two more inverse transforms each).
  * rf_echo_cancel.

Also unbuilt: the pipeline currently demodulates one block at a time, so
something has to buffer blocks into groups of 4-16 before dispatch, and with
several worker processes they would contend for one GPU - the natural shape is
one GPU demod stage feeding several CPU field-processing workers.

Requires cupy (plus CUDA toolkit headers); imports cleanly without it.
"""

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy import fft as cfft
    HAVE_GPU = True
except Exception:                                    # pragma: no cover
    cp = None
    cfft = None
    HAVE_GPU = False


class GPUDemod:
    """Holds an RFDecode's filters on the GPU and demodulates blocks in batches.

    The filters are uploaded once and stay resident: only the raw block data
    crosses the bus in, and the demodulated products out.
    """

    def __init__(self, rf):
        if not HAVE_GPU:
            raise RuntimeError("cupy not available")
        self.rf = rf
        self.blocklen = rf.blocklen
        self.blockcut = rf.blockcut
        self.blockcut_end = rf.blockcut_end
        self.system = rf.system
        SF = rf.Filters
        self.d_rfvideo = cp.asarray(SF["RFVideo"])
        self.d_frfhpf = cp.asarray(SF["Frfhpf_half"])
        self.d_fvideo = cp.asarray(SF["FVideo_rfft"])
        self.d_mtf = cp.asarray(SF["MTF"])
        self.d_fcutpal = cp.asarray(SF["FcutPAL"]) if "FcutPAL" in SF else None
        # demodblock slices rfhpf offset by the measured rot delay
        d = getattr(rf, "delays", None)
        self.rotdelay = int(d["video_rot"]) if d and "video_rot" in d else 0
        self._mtf_cache = {}

    def _mtf_pow(self, level):
        """MTF ** level, cached: level changes rarely (only on servo adoption)."""
        key = round(float(level), 6)
        p = self._mtf_cache.get(key)
        if p is None:
            if len(self._mtf_cache) > 8:
                self._mtf_cache.clear()
            p = self.d_mtf ** key
            self._mtf_cache[key] = p
        return p

    def demod_batch(self, blocks, mtf_level=0.0, cut=False, use_fcutpal=False):
        """Demodulate a batch of raw blocks.

        blocks: (B, blocklen) real.  Returns a list of B dicts with "video"
        (a structured array matching demodblock's) and "rfhpf".
        """
        bl = self.blocklen
        x = cp.asarray(np.ascontiguousarray(blocks, dtype=np.float64))
        B = x.shape[0]

        # real input -> half spectrum, then mirror to the full spectrum the
        # RF-domain filters are defined over (as demodblock does)
        half = cfft.rfft(x, axis=-1)
        nr = half.shape[-1]
        full = cp.empty((B, bl), dtype=half.dtype)
        full[:, :nr] = half
        full[:, nr:] = cp.conj(half[:, 1:bl - nr + 1])[:, ::-1]

        # dropout reference: real filter x real input, so the half-spectrum
        # inverse is exact
        rfhpf = cfft.irfft(half * self.d_frfhpf, n=bl, axis=-1)
        rd = self.rotdelay
        rfhpf = rfhpf[:, self.blockcut - rd:bl - self.blockcut_end - rd].astype(cp.float32)

        filt = full * self.d_rfvideo
        if use_fcutpal and self.d_fcutpal is not None:
            filt = filt * self.d_fcutpal
        if mtf_level != 0:
            filt = filt * self._mtf_pow(mtf_level)

        hil = cfft.ifft(filt, axis=-1)
        # conjugate-product FM discriminator (lddecode.filters.unwrap_hilbert):
        # the phase increment arrives already wrapped into (-pi, pi]
        p = hil[:, 1:] * cp.conj(hil[:, :-1])
        d = cp.arctan2(p.imag, p.real)
        # unwrap_hilbert keeps the historical [0, tau) convention - a negative
        # increment is a positive frequency near the top of the range, not a
        # negative one.  Omitting this puts every such sample a full turn out.
        tau = 2.0 * np.pi
        d = cp.where(d < 0.0, d + tau, d)
        demod = cp.empty((B, bl), dtype=cp.float64)
        demod[:, 1:] = d * (self.rf.freq_hz / tau)
        demod[:, 0] = 0.0

        clipped = cp.clip(demod, 1500000, self.rf.freq_hz * 0.75)
        dfft = cfft.rfft(clipped, axis=-1)
        # (B, 1, nr) x (P, nr) -> (B, P, bl): all video products in one call
        vids = cfft.irfft(dfft[:, None, :] * self.d_fvideo, n=bl, axis=-1)

        vids = cp.ascontiguousarray(vids.astype(cp.float32))
        demod32 = demod.astype(cp.float32)
        v_host = cp.asnumpy(vids)
        d_host = cp.asnumpy(demod32)
        r_host = cp.asnumpy(rfhpf)

        names = (["demod", "demod_raw", "demod_05", "demod_burst", "demod_pilot"]
                 if self.system == "PAL" else
                 ["demod", "demod_raw", "demod_05", "demod_burst"])
        out = []
        for i in range(B):
            cols = [v_host[i, 0], d_host[i], v_host[i, 1], v_host[i, 2]]
            if self.system == "PAL":
                cols.append(v_host[i, 3])
            vo = np.rec.array(cols, names=names)
            out.append({
                "video": vo[self.blockcut:-self.blockcut_end] if cut else vo,
                "rfhpf": r_host[i],
            })
        return out
