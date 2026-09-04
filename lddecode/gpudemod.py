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

        # --- V4300D coherent subtract state (mirrors rfdecode's) ------------
        bl, freq = rf.blocklen, rf.freq
        self.v4_on = bool(getattr(rf, "PAL_V4300D_CoherentSubtract", False))
        self.v4_sl = slice(int(bl * (rf.V4300D_WINDOW_MHZ[0] / freq)),
                           int(1 + bl * (rf.V4300D_WINDOW_MHZ[1] / freq)))
        self.v4_carrier = slice(int(bl * (rf.V4300D_CARRIER_MHZ[0] / freq)),
                                int(bl * (rf.V4300D_CARRIER_MHZ[1] / freq)))
        self.fpb = rf.freq_hz / bl
        self.v4_tol = max(2, int(round(rf.V4300D_ANCHOR_TOL_HZ / self.fpb)))
        self.v4_nbtol = max(1, self.v4_tol // 2)
        self.v4_fh = int(round((1e6 / rf.SysParams["line_period"]) / self.fpb))
        self.d_bins = cp.arange(bl, dtype=cp.float64)
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

        # V4300D coherent subtract, per block, entirely on the GPU: the
        # spectrum never goes back across the bus for it
        if self.v4_on:
            full = self._v4300d_batch(full)

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

    # ---- V4300D coherent subtract, per block on the GPU -------------------
    # Ported from rfdecode.v4300d_coherent_subtract.  It does not batch: the
    # line hunt is sequential and its gates are data-dependent, so this runs
    # per block inside the batch.  The arrays are small (a +-2048 bin cut), so
    # even with a kernel launch per operation this stays cheaper than sending
    # the spectrum back to the CPU and returning it.

    def _dirichlet(self, delta, N):
        """Rectangular-window transform of a unit tone at bin offset delta."""
        d = cp.asarray(delta, dtype=cp.float64)
        num = cp.sin(cp.pi * d)
        den = cp.sin(cp.pi * d / N)
        mag = cp.where(cp.abs(den) > 1e-12, num / cp.where(den == 0, 1, den),
                       cp.float64(N))
        return mag * cp.exp(1j * cp.pi * d * (N - 1) / N)

    def _refine_subtract(self, X, k, fit_bins=64, cut_bins=2048):
        N = self.blocklen
        m = cp.arange(k - fit_bins, k + fit_bins + 1)
        Xw = X[m]
        grid = k + cp.linspace(-0.6, 0.6, 9)
        E = self._dirichlet(grid[:, None] - m[None, :], N)
        num = E.conj() @ Xw
        den = cp.sum(E.real ** 2 + E.imag ** 2, axis=1)
        mag = (num.real ** 2 + num.imag ** 2) / den
        g = int(cp.argmax(mag))
        if 0 < g < 8:
            d2 = mag[g - 1] - 2 * mag[g] + mag[g + 1]
            frac = float(cp.clip(0.5 * (mag[g - 1] - mag[g + 1]) / d2, -1.0, 1.0)) \
                if float(d2) != 0.0 else 0.0
        else:
            frac = 0.0
        fbin = float(grid[g]) + frac * float(grid[1] - grid[0])
        mc = cp.arange(k - cut_bins, k + cut_bins + 1)
        Ef = self._dirichlet(fbin - m, N)
        c = cp.dot(Ef.conj(), Xw) / cp.sum(Ef.real ** 2 + Ef.imag ** 2)
        X[mc] -= c * self._dirichlet(fbin - mc, N)
        X[N - mc] = cp.conj(X[mc])
        return fbin * self.fpb

    def _v4300d(self, X0, maxlines=10):
        """Return X0 with the LD-V4300D clock spur removed (or X0 unchanged)."""
        rf = self.rf
        sl, X, lines = self.v4_sl, X0, []
        amp = cp.abs(X0[sl])
        med = float(cp.median(amp))
        if med <= 0:
            return X0
        if float(cp.abs(X0[self.v4_carrier]).max()) <= rf.V4300D_MIN_CARRIER * med:
            return X0                      # no video carrier: leave untouched

        def subtract(k):
            nonlocal X
            if X is X0:
                X = X0.copy()
            lines.append(self._refine_subtract(X, int(k)))

        main_found = False
        for ks in (0,) + tuple(rf.V4300D_SATELLITE_KS):
            if ks != 0 and not main_found:
                break
            k0 = int(round((rf.V4300D_CLOCK_HZ + ks * rf.V4300D_SATELLITE_HZ)
                           / self.fpb))
            t, nb, fh = self.v4_tol, self.v4_nbtol, self.v4_fh
            seg = cp.abs(X0[k0 - t:k0 + t + 1])
            kk = k0 - t + int(cp.argmax(seg))
            peak = float(seg.max())
            comb = max(float(cp.abs(X0[kk - fh - nb:kk - fh + nb + 1]).max()),
                       float(cp.abs(X0[kk + fh - nb:kk + fh + nb + 1]).max()))
            mm, mc = ((rf.V4300D_MAIN_MIN_MED, rf.V4300D_MAIN_MIN_COMB) if ks == 0
                      else (rf.V4300D_SAT_MIN_MED, rf.V4300D_SAT_MIN_COMB))
            if peak > mm * med and peak > mc * comb:
                subtract(kk)
                if ks == 0:
                    main_found = True

        while len(lines) < maxlines:                 # generic lone-tone hunt
            sq = cp.abs(X[sl])
            m2 = float(cp.median(sq))
            if m2 <= 0:
                break
            k = int(cp.argmax(sq))
            ratio = float(sq[k]) / m2
            fpeak = (k + sl.start) * self.fpb
            near = any(abs(fpeak - f) < 30e3 for f in lines)
            if not (ratio > 40 or (near and ratio > 5)):
                break
            subtract(k + sl.start)
        return X

    def _v4300d_batch(self, Xb):
        """V4300D over a whole batch.

        The carrier guard and the anchored line detection are deterministic,
        so they are evaluated for every block in one pass each instead of one
        kernel launch per block per test.  Only the subtraction - which is
        data-dependent, and which most blocks do not need at all - stays per
        block.  Identical decisions to _v4300d(), just taken in bulk.
        """
        rf, B = self.rf, Xb.shape[0]
        sl, t, nb, fh = self.v4_sl, self.v4_tol, self.v4_nbtol, self.v4_fh

        amp = cp.abs(Xb[:, sl])                       # (B, W)
        med = cp.median(amp, axis=1)                  # (B,)
        carrier = cp.abs(Xb[:, self.v4_carrier]).max(axis=1)
        active = (med > 0) & (carrier > rf.V4300D_MIN_CARRIER * med)

        # anchored lines: peak, its bin, and the line-rate comb neighbours,
        # for every block at once
        hits = {}
        main_ok = None
        for ks in (0,) + tuple(rf.V4300D_SATELLITE_KS):
            k0 = int(round((rf.V4300D_CLOCK_HZ + ks * rf.V4300D_SATELLITE_HZ)
                           / self.fpb))
            seg = cp.abs(Xb[:, k0 - t:k0 + t + 1])    # (B, 2t+1)
            j = cp.argmax(seg, axis=1)
            kk = k0 - t + j
            peak = cp.take_along_axis(seg, j[:, None], axis=1)[:, 0]
            # gather the +-fh_bins neighbourhoods around each block's own kk
            off = cp.arange(-nb, nb + 1)
            lo = cp.abs(cp.take_along_axis(
                Xb, (kk[:, None] - fh + off[None, :]), axis=1)).max(axis=1)
            hi = cp.abs(cp.take_along_axis(
                Xb, (kk[:, None] + fh + off[None, :]), axis=1)).max(axis=1)
            comb = cp.maximum(lo, hi)
            mm, mc = ((rf.V4300D_MAIN_MIN_MED, rf.V4300D_MAIN_MIN_COMB) if ks == 0
                      else (rf.V4300D_SAT_MIN_MED, rf.V4300D_SAT_MIN_COMB))
            ok = active & (peak > mm * med) & (peak > mc * comb)
            if ks == 0:
                main_ok = ok
            else:
                ok = ok & main_ok            # satellites only after the main line
            hits[ks] = (cp.asnumpy(ok), cp.asnumpy(kk))

        out, changed = [], False
        act = cp.asnumpy(active)
        for i in range(B):
            X = Xb[i]
            if not act[i]:
                out.append(X)
                continue
            lines = []
            for ks in (0,) + tuple(rf.V4300D_SATELLITE_KS):
                ok, kk = hits[ks]
                if ok[i]:
                    if X is Xb[i]:
                        X = Xb[i].copy(); changed = True
                    lines.append(self._refine_subtract(X, int(kk[i])))
            # generic lone-tone hunt: sequential by nature, per block
            while len(lines) < 10:
                sq = cp.abs(X[sl])
                m2 = float(cp.median(sq))
                if m2 <= 0:
                    break
                k = int(cp.argmax(sq))
                if float(sq[k]) / m2 <= 40 and not (
                        any(abs((k + sl.start) * self.fpb - f) < 30e3 for f in lines)
                        and float(sq[k]) / m2 > 5):
                    break
                if X is Xb[i]:
                    X = Xb[i].copy(); changed = True
                lines.append(self._refine_subtract(X, k + sl.start))
            out.append(X)
        return cp.stack(out) if changed else Xb
