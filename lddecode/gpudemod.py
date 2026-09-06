"""PROTOTYPE: batched demodulation on the GPU.  Opt-in (LDDECODE_GPU=1).

demodblock() is FFTs and elementwise products over 32768-point blocks, and a
field is ~50 blocks - a natural batch.  Batched, the PCIe round trip
amortises away; one block at a time it is 76% of the call and the GPU loses.

This mirrors demodblock() stage for stage, on the same filters, uploaded
once and re-uploaded only when the decoder replaces them (a chroma-servo
adoption rebuilds FVideo_rfft32 through recompute_fvideo(); an AGC
adoption rebuilds every filter through computefilters()).  Holding a stale
copy silently decodes later fields under a filter the CPU has already left
behind - measured as 91% of samples differing from field 6 onward when only
FVideo was tracked - so the resident copies are keyed on the array OBJECTS,
not their ids (CPython reuses an id once the old array is freed).

Parity with demodblock(), in the order the CPU applies them:
  * rfft of the real block, mirrored to the full spectrum;
  * the rfhpf dropout reference, sliced by the measured rot delay;
  * V4300D coherent subtract (per block, on the GPU, see below);
  * RFVideo, then FcutPAL only on blocks whose analog audio carriers are
    detected (pal_audio_carriers_present, the same band-power test);
  * MTF ** level, the level scaled by mtf_mult / mtf_offset / MTF_basemult
    exactly as demodblock does (raw_mtf=True skips that, as there);
  * the conjugate-product discriminator in its historical [0, tau)
    convention;
  * the four (PAL: five) video products in single precision, centred on
    blanking before the cast and given their DC back afterwards, off the
    same FVideo_rfft32 / _centre / _dc the CPU uses.

The video products cannot be bit-identical to the CPU: the float32 transform
is cuFFT's, not pocketfft's, so they agree to float32 rounding (about 1 Hz
at the carrier, well under a 16-bit output LSB) rather than exactly.  The
float64 stages (rfhpf, demod_raw) match to ~1e-11.

NOT covered - a block needing any of these must use the CPU path, and
parallel._worker_gpu() refuses the GPU for such decodes:
  * EFM and analog audio (two more inverse transforms each);
  * rf_echo_cancel;
  * the --fm_pll discriminator.

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

    # Every filter demodblock() reads, by the Filters key the decoder rebuilds
    # it under.  FVideo_rfft32 is rebuilt together with its centre and dc
    # constants (build_video_rfft_stack), so tracking it covers all three.
    _TRACKED = ("RFVideo", "Frfhpf_half", "FVideo_rfft32", "MTF", "FcutPAL")

    def __init__(self, rf):
        if not HAVE_GPU:
            raise RuntimeError("cupy not available")
        self.rf = rf
        self._rs_bases = {}
        self.blocklen = rf.blocklen
        self.blockcut = rf.blockcut
        self.blockcut_end = rf.blockcut_end
        self.system = rf.system
        self._ids = {}
        self._mtf_cache = {}
        self.d_fcutpal = None
        for k in self._TRACKED:
            if k in rf.Filters:
                self._upload(k)

        bl, freq = rf.blocklen, rf.freq
        self.fpb = rf.freq_hz / bl

        # --- PAL analog audio carrier detection (pal_audio_carriers_present):
        # the same bin ranges, evaluated for every block of the batch at once
        self._carrier_slices = []
        if "FcutPAL" in rf.Filters:
            def band(f0, width):
                return (int((f0 - width) / self.fpb),
                        int((f0 + width) / self.fpb) + 1)
            for fc in (rf.SysParams["audio_lfreq"], rf.SysParams["audio_rfreq"]):
                self._carrier_slices.append(
                    (band(fc, 40e3), band(fc - 175e3, 75e3), band(fc + 175e3, 75e3)))

        # --- V4300D coherent subtract state (mirrors rfdecode's) ------------
        self.v4_on = bool(getattr(rf, "PAL_V4300D_CoherentSubtract", False))
        self.v4_sl = slice(int(bl * (rf.V4300D_WINDOW_MHZ[0] / freq)),
                           int(1 + bl * (rf.V4300D_WINDOW_MHZ[1] / freq)))
        self.v4_carrier = slice(int(bl * (rf.V4300D_CARRIER_MHZ[0] / freq)),
                                int(bl * (rf.V4300D_CARRIER_MHZ[1] / freq)))
        self.v4_tol = max(2, int(round(rf.V4300D_ANCHOR_TOL_HZ / self.fpb)))
        self.v4_nbtol = max(1, self.v4_tol // 2)
        self.v4_fh = int(round((1e6 / rf.SysParams["line_period"]) / self.fpb))
        self.d_bins = cp.arange(bl, dtype=cp.float64)

    def _upload(self, k):
        SF = self.rf.Filters
        dev = cp.asarray(SF[k])
        if k == "RFVideo":
            self.d_rfvideo = dev
        elif k == "Frfhpf_half":
            self.d_frfhpf = dev
        elif k == "FVideo_rfft32":
            self.d_fvideo32 = dev                        # (P, nr) complex64
            self.centre = float(SF["FVideo_rfft_centre"])
            self.d_dc = cp.asarray(SF["FVideo_rfft_dc"])  # (P,) float32
        elif k == "MTF":
            self.d_mtf = dev
            self._mtf_cache.clear()
        elif k == "FcutPAL":
            self.d_fcutpal = dev
        # Hold the array OBJECT, not its id: CPython reuses an id once the old
        # object is freed, so an id comparison can silently miss a filter that
        # was replaced.  Keeping the reference makes the identity test sound.
        self._ids[k] = SF[k]

    def _resync(self):
        """Re-upload any filter the decoder has replaced since we last looked."""
        SF = self.rf.Filters
        for k in self._TRACKED:
            if k in SF and self._ids.get(k) is not SF[k]:
                self._upload(k)

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

    def _carriers_present(self, X):
        """pal_audio_carriers_present() over a batch: (B,) bool."""
        ok = cp.ones(X.shape[0], dtype=bool)
        for (clo, chi), (l1, h1), (l2, h2) in self._carrier_slices:
            carrier = cp.mean(cp.abs(X[:, clo:chi]) ** 2, axis=1)
            flank = (cp.mean(cp.abs(X[:, l1:h1]) ** 2, axis=1)
                     + cp.mean(cp.abs(X[:, l2:h2]) ** 2, axis=1)) / 2
            ok &= ~(carrier < 5.0 * flank)
        return ok

    def demod_batch(self, blocks, mtf_level=0.0, cut=False, raw_mtf=False):
        """Demodulate a batch of raw blocks.

        blocks: (B, blocklen) real.  Returns a list of B dicts with "video"
        (a structured array matching demodblock's) and "rfhpf".
        """
        rf = self.rf
        bl = self.blocklen
        if not raw_mtf:
            mtf_level = ((mtf_level * rf.mtf_mult + rf.mtf_offset)
                         * rf.DecoderParams["MTF_basemult"])
        self._resync()
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
        # inverse is exact.  demodblock slices it by the measured rot delay,
        # which the decoder can recalibrate, so read it per call.
        rotdelay = 0
        d = getattr(rf, "delays", None)
        if d is not None and "video_rot" in d:
            rotdelay = int(d["video_rot"])
        rfhpf = cfft.irfft(half * self.d_frfhpf, n=bl, axis=-1)
        rfhpf = rfhpf[:, self.blockcut - rotdelay:
                      bl - self.blockcut_end - rotdelay].astype(cp.float32)

        # V4300D coherent subtract, per block, entirely on the GPU: the
        # spectrum never goes back across the bus for it
        if self.v4_on:
            full = self._v4300d_batch(full)

        filt = full * self.d_rfvideo
        # PAL: notch the analog audio carriers out of the video path, but only
        # on blocks where they are actually present (see demodblock)
        if self.d_fcutpal is not None and self._carrier_slices:
            mask = self._carriers_present(full)
            if bool(mask.any()):
                filt = cp.where(mask[:, None], filt * self.d_fcutpal, filt)
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
        demod[:, 1:] = d * (rf.freq_hz / tau)
        demod[:, 0] = 0.0

        # video products in single precision, as demodblock: clip, centre on
        # blanking in float64, cast, transform, and give each channel its DC
        # gain back on the float32 result
        clipped = (cp.clip(demod, 1500000, rf.freq_hz * 0.75)
                   - self.centre).astype(cp.float32)
        dfft = cfft.rfft(clipped, axis=-1)                       # complex64
        vids = cfft.irfft(dfft[:, None, :] * self.d_fvideo32, n=bl, axis=-1)
        vids = vids + self.d_dc[None, :, None]                   # float32
        v_host = cp.asnumpy(cp.ascontiguousarray(vids))
        d_host = cp.asnumpy(demod.astype(cp.float32))
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

    def _rs_base(self, fit_bins, cut_bins):
        """Cached (arange(-fit..fit), arange(-cut..cut)) for _refine_subtract."""
        key = (fit_bins, cut_bins)
        b = self._rs_bases.get(key)
        if b is None:
            b = (cp.arange(-fit_bins, fit_bins + 1),
                 cp.arange(-cut_bins, cut_bins + 1))
            self._rs_bases[key] = b
        return b

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
        # Bases are allocated once and shifted by k: cp.arange() per call was
        # two kernel launches per line, and this runs ~150 times per batch.
        base = self._rs_base(fit_bins, cut_bins)
        m = base[0] + k
        Xw = X[m]
        # The 9-point grid is deterministic, so host numpy gives bit-identical
        # values without a device round trip.
        grid_h = k + np.linspace(-0.6, 0.6, 9)
        E = self._dirichlet(cp.asarray(grid_h)[:, None] - m[None, :], N)
        num = E.conj() @ Xw
        den = cp.sum(E.real ** 2 + E.imag ** 2, axis=1)
        mag = (num.real ** 2 + num.imag ** 2) / den
        # One 9-element D2H copy replaces five separate host syncs
        # (argmax, d2, clip, grid[g], grid step).  Same float64 arithmetic,
        # just done on the host where it costs nothing.
        mag_h = cp.asnumpy(mag)
        g = int(np.argmax(mag_h))
        if 0 < g < 8:
            d2 = mag_h[g - 1] - 2 * mag_h[g] + mag_h[g + 1]
            frac = float(np.clip(0.5 * (mag_h[g - 1] - mag_h[g + 1]) / d2,
                                  -1.0, 1.0)) if d2 != 0.0 else 0.0
        else:
            frac = 0.0
        fbin = float(grid_h[g]) + frac * float(grid_h[1] - grid_h[0])
        mc = base[1] + k
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
