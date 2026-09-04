"""Multi-capture disc stacking, integrated into ld-decode.

Combine several captures of the *same* disc into one improved output (video
TBC, analog audio, and EFM) so that dropouts are filled, random noise is
reduced by ~sqrt(N), and sub-pixel timing differences between captures are
corrected.  Because it runs inside ld-decode, the per-disc TBCs are consumed
frame-by-frame as they are produced -- there is no need to write twelve ~75 GB
intermediate .tbc files.

Pipeline
--------
1. Open N inputs.  Each is either a pre-decoded ``.tbc`` (with ``.tbc.json``)
   or a raw ``.ldf`` that is decoded live in lockstep (``LDFFrameSource``).
2. Align frames across captures by their CAV VBI picture number (CLV / non-CAV
   discs fall back to sequential frame index).  The recording start may differ
   per capture; aligning on the picture number absorbs that.
3. Quality / master pass: on a sample of shared frames, estimate each capture's
   noise and cluster captures by master (captures of a *different* glass master
   disagree in a shared, structured way; same-master captures differ only by
   independent noise).  Discard unusable captures and captures not belonging to
   the chosen (largest) master group.
4. For every output frame: sub-pixel + level register each capture's fields to a
   reference, then combine dropout-aware and inverse-variance weighted.
5. Audio: sub-sample-align each capture and robust (outlier-rejecting)
   weighted-average the analog audio (primary master only).
6. EFM: (a) average the TBC-locked pre-PLL EFM waveforms across captures (LDF
   source, --tbc_efm --preEFM; lowers the noise floor into one PLL pass) -> .efm,
   and (b) sector-merge the per-capture decoded .efm, OR-filling sectors no single
   disc read -> .data.  (a) and (b) are complementary.
7. Write ``.tbc`` + ``.tbc.json`` + ``.tbc.db`` (sqlite) + ``.pcm`` + ``.efm`` (+
   ``.data``).

Works for PAL and NTSC; the engine is system-agnostic and reads geometry/levels
from the reference capture's videoParameters.
"""

import json
import os
import sqlite3
import sys
from textwrap import dedent

import numpy as np


# --------------------------------------------------------------------------- #
#  VBI helpers
# --------------------------------------------------------------------------- #
def cav_picture(vbidata):
    """CAV picture number from a field's VBI: an 0xFxxxxx code, low 5 nibbles BCD."""
    for x in vbidata or []:
        h = "%06x" % x
        if h[0] == "f" and all(c in "0123456789" for c in h[1:]):
            return int(h[1:])
    return None


def _bcd2(x):
    hi, lo = (x >> 4) & 0xF, x & 0xF
    if hi > 9 or lo > 9:
        return None
    return hi * 10 + lo


def clv_frame(vbidata):
    """Absolute CLV frame number from a frame's VBI codes (both fields'
    vbiData concatenated), or None.

    IEC 60856 10.1.8/10.1.10: the CLV picture number (line 16,
    0x8XEYZZ with X=sec-tens+0xA, Y=sec-units BCD, ZZ=picture BCD) gives
    seconds+frame; the programme time code (lines 17/18, 0xFHDDMM BCD)
    gives hours+minutes. Every CLV programme frame broadcasts both."""
    sec = pic = hr = mn = None
    for v in vbidata:
        if v is None:
            continue
        if (v & 0xF0F000) == 0x80E000:
            x1 = (v >> 16) & 0xF
            su = (v >> 8) & 0xF
            p = _bcd2(v & 0xFF)
            if x1 >= 0xA and su <= 9 and p is not None:
                sec, pic = 10 * (x1 - 0xA) + su, p
        elif (v & 0xF0FF00) == 0xF0DD00:
            h = (v >> 16) & 0xF
            m = _bcd2(v & 0xFF)
            if h <= 9 and m is not None:
                hr, mn = h, m
    if sec is not None and hr is not None:
        return ((hr * 60 + mn) * 60 + sec) * 25 + pic
    return None


def field_vbi(field_json):
    return field_json.get("vbi", {}).get("vbiData", []) or []


# --------------------------------------------------------------------------- #
#  Frame model + sources
# --------------------------------------------------------------------------- #
class Frame:
    """One disc frame from one capture: two fields plus their metadata."""

    __slots__ = ("key", "f0", "f1", "meta0", "meta1", "do0", "do1",
                 "audio", "efm")

    def __init__(self, key, f0, f1, meta0, meta1, do0, do1, audio, efm):
        self.key = key          # picture number (CAV) or sequential index
        self.f0 = f0            # uint16 2D field array (height, width)
        self.f1 = f1
        self.meta0 = meta0      # dict: fieldPhaseID, isFirstField, ...
        self.meta1 = meta1
        self.do0 = do0          # list[(line, startx, endx)] dropouts
        self.do1 = do1
        self.audio = audio      # int16 (n,2) or None
        self.efm = efm          # T-values (tbc path) or waveform (ldf, tbc_efm)


class FrameSource:
    """Abstract: yields Frame objects in disc order and exposes geometry."""

    videoParameters = None      # dict (ld-decode videoParameters)
    name = "?"

    def frames(self):
        raise NotImplementedError

    def efm_path(self):
        """Path to this capture's decoded EFM (.efm T-value stream), or None.
        Used for cross-capture sector-level EFM merge."""
        return None

    def prefm_path(self):
        """Path to this capture's pre-PLL EFM waveform (.prefm), or None."""
        return None

    def close(self):
        pass


class TBCFrameSource(FrameSource):
    """Read a decoded ``.tbc`` (+ ``.tbc.json``).  Also the path used for tests
    and for users who already have decoded captures."""

    def __init__(self, base, cav=True):
        self.base = base
        self.name = os.path.basename(base)
        self.cav = cav
        j = json.load(open(base + ".tbc.json"))
        self.videoParameters = j["videoParameters"]
        self.fields = j["fields"]
        fw = self.videoParameters["fieldWidth"]
        fh = self.videoParameters["fieldHeight"]
        n = len(self.fields)
        # positional bulk reads, NOT np.memmap: on network/9P mounts (WSL
        # drvfs) memmap faults each 4KB page through the filesystem server,
        # which serialises the whole worker pool; one os.pread per field is a
        # single large sequential read. os.pread is offset-explicit, so forked
        # workers can share the fd without seek races.
        self._tbc_fd = os.open(base + ".tbc", os.O_RDONLY)
        self._fshape = (fh, fw)
        self._fbytes = fh * fw * 2
        self._nfields = n
        self.pcm = None
        p = base + ".pcm"
        if os.path.exists(p):
            self.pcm = np.fromfile(p, dtype="<i2").reshape(-1, 2)

    def _read_field(self, idx):
        buf = os.pread(self._tbc_fd, self._fbytes, idx * self._fbytes)
        return np.frombuffer(buf, dtype="<u2").reshape(self._fshape)

    def close(self):
        try:
            os.close(self._tbc_fd)
        except OSError:
            pass

    def _dropouts(self, fj):
        do = fj.get("dropOuts") or {}
        fl = do.get("fieldLine") or []
        sx = do.get("startx") or []
        ex = do.get("endx") or []
        return list(zip(fl, sx, ex))

    def _build_index(self):
        self._idx = {}
        self._spf = 0
        nframes = len(self.fields) // 2
        if self.pcm is not None and nframes:
            self._spf = len(self.pcm) // nframes
        prev = None
        raw = []
        for fi in range(nframes):
            j0, j1 = self.fields[fi * 2], self.fields[fi * 2 + 1]
            vbi = field_vbi(j0) + field_vbi(j1)
            if self.cav:
                key = cav_picture(vbi)
            else:
                # CLV: absolute timecode frame; frames with unreadable codes
                # continue sequentially from the previous good key so captures
                # still align absolutely (falls back to plain index when the
                # source has no CLV codes at all)
                if 0x88FFFF in vbi:        # lead-in: never key (a lead-in
                    continue               # frame must not shadow t=0:00:00)
                key = clv_frame(vbi)
                if key is None:
                    key = fi if prev is None else prev + 1
                prev = key
            if key is not None:
                raw.append((fi, key))
        for fi, key in sieve_keys(raw):
            if key not in self._idx:
                self._idx[key] = fi

    def keys(self):
        if not hasattr(self, "_idx"):
            self._build_index()
        return set(self._idx)

    def get(self, key):
        if not hasattr(self, "_idx"):
            self._build_index()
        fi = self._idx[key]
        j0, j1 = self.fields[fi * 2], self.fields[fi * 2 + 1]
        audio = None
        if self.pcm is not None and self._spf:
            audio = np.asarray(self.pcm[fi * self._spf:(fi + 1) * self._spf])
        return Frame(
            key,
            self._read_field(fi * 2), self._read_field(fi * 2 + 1),
            j0, j1, self._dropouts(j0), self._dropouts(j1),
            audio, None,
        )

    def frames(self):
        for key in sorted(self.keys()):
            yield self.get(key)

    def efm_path(self):
        p = self.base + ".efm"
        return p if os.path.exists(p) and os.path.getsize(p) > 0 else None

    def prefm_path(self):
        """Path to the pre-PLL TBC-locked EFM waveform (.prefm, int16), or None.
        Used for cross-capture EFM-waveform averaging (one PLL pass on the mean)."""
        p = self.base + ".prefm"
        return p if os.path.exists(p) and os.path.getsize(p) > 0 else None


def _do_pairs(fi):
    do = fi.get("dropOuts") or {}
    return list(zip(do.get("fieldLine", []) or [],
                    do.get("startx", []) or [],
                    do.get("endx", []) or []))


def resolve_scratch(explicit, near, need_gb=3):
    """Where to put per-capture scratch (window .tbc/.efm, merge .bin).

    Default to a RAM-backed tmpfs (/dev/shm) so NOTHING touches a physical disk
    -- only the final results are written.  Falls back to the directory of
    `near` if no tmpfs is writable with enough headroom."""
    import shutil as _sh
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    for cand in ("/dev/shm", "/run/shm"):
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            try:
                if _sh.disk_usage(cand).free > need_gb * (1024 ** 3):
                    return cand
            except OSError:
                pass
    return os.path.dirname(os.path.abspath(near))


class LDFFrameSource(FrameSource):
    """Serve frames from an ``.ldf`` capture by decoding the requested window
    with the normal ld-decode CLI (subprocess) to a small temporary ``.tbc``,
    then reading it back via TBCFrameSource (random access).

    Driving ld-decode's readfield() in-process after a programmatic seek is
    unreliable (field-sync state is left inconsistent and stalls), so we run the
    proven CLI path instead.  The window is bounded by --seek/--length so no
    giant .tbc is materialised; a full disc is covered by processing successive
    windows.  EFM is decoded TBC-locked with --tbc_efm --preEFM, so the pre-PLL
    EFM waveform (<base>.prefm) is available for cross-capture EFM-waveform
    stacking (see stack(): average the .prefm waveforms, PLL once).
    """

    def __init__(self, path, system="PAL", seek=None, length=None, cav=True,
                 inputfreq=None, analog_audio=44100, extra_options=None,
                 scratch_dir=None):
        import tempfile, subprocess, sys as _sys
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._tmpdir = tempfile.mkdtemp(
            prefix="lddstack_", dir=resolve_scratch(scratch_dir, path))
        base = os.path.join(self._tmpdir, "cap")
        # --tbc_efm time-base-corrects the EFM onto the disc-rotation time-base
        # (cleaner per-capture PLL).  --preEFM also writes the pre-PLL EFM
        # waveform (<base>.prefm); for CAV-aligned windows (e.g. all captures
        # --seek'd to the same picture) those waveforms align, so the stacker can
        # average them and run ONE PLL pass on the lower-noise average (see
        # stack(): average_efm).  The per-capture .efm is still decoded too, for
        # the complementary sector-level merge.
        cmd = [_sys.executable, os.path.join(repo, "ld-decode"),
               "--NTSC" if system == "NTSC" else "--PAL", "--tbc_efm", "--preEFM"]
        eo = extra_options or {}
        if eo.get("PAL_V4300D_CoherentSubtract"):
            cmd.append("--V4300D_coherent_subtract")
        rfe = eo.get("rf_echo_cancel")
        if isinstance(rfe, (list, tuple)) and rfe:
            cmd += ["--rf_echo", ",".join(f"{d}:{a}" for d, a in rfe)]
        if seek is not None:
            cmd += ["--seek", str(int(seek))]
        if length is not None:
            cmd += ["--length", str(int(length))]
        cmd += [path, base]
        env = dict(os.environ, PYTHONPATH=repo)
        with open(base + ".decode.log", "wb") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        if not os.path.exists(base + ".tbc.json"):
            # decode failed -- clean our scratch dir before bailing so a failed
            # capture doesn't leak RAM (the caller never gets to call close())
            import shutil
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            raise SystemExit(
                f"ld-decode failed on {path} (see {base}.decode.log)")
        self._inner = TBCFrameSource(base, cav=cav)
        self.videoParameters = self._inner.videoParameters

    def keys(self):
        return self._inner.keys()

    def get(self, key):
        return self._inner.get(key)

    def frames(self):
        return self._inner.frames()

    def efm_path(self):
        return self._inner.efm_path()

    def prefm_path(self):
        return self._inner.prefm_path()

    def close(self):
        import shutil
        shutil.rmtree(getattr(self, "_tmpdir", ""), ignore_errors=True)

# --------------------------------------------------------------------------- #
#  Registration + combine engine
# --------------------------------------------------------------------------- #
def _active(vp):
    avs, ave = vp["activeVideoStart"], vp["activeVideoEnd"]
    fh = vp["fieldHeight"]
    return slice(24, fh - 6), slice(avs, ave)


def integer_shift(ref, img):
    a = ref - ref.mean()
    b = img - img.mean()
    R = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    R /= np.abs(R) + 1e-9
    c = np.real(np.fft.ifft2(R))
    p = np.unravel_index(np.argmax(c), c.shape)
    sy = p[0] - (c.shape[0] if p[0] > c.shape[0] // 2 else 0)
    sx = p[1] - (c.shape[1] if p[1] > c.shape[1] // 2 else 0)
    return sy, sx


def _apply_shift_spectrum(F, H, W, dy, dx, _pycache=None, _pxcache=None):
    """irfft2 of a cached rfft2 spectrum with a (dy, dx) phase-ramp shift.
    The ramp is separable, so build it as two 1-D phasor vectors instead of a
    full 2-D exp (identical result to fft2/ifft2 with the full ramp, since a
    real image's shift ramp preserves Hermitian symmetry)."""
    py = None if _pycache is None else _pycache.get(dy)
    if py is None:
        py = np.exp(-2j * np.pi * np.fft.fftfreq(H) * dy)[:, None]
        if _pycache is not None:
            _pycache[dy] = py
    px = None if _pxcache is None else _pxcache.get(dx)
    if px is None:
        px = np.exp(-2j * np.pi * np.fft.rfftfreq(W) * dx)[None, :]
        if _pxcache is not None:
            _pxcache[dx] = px
    return np.fft.irfft2((F * py) * px, s=(H, W))


def fourier_shift(img, dy, dx):
    if dy == 0 and dx == 0:
        return img
    H, W = img.shape
    return _apply_shift_spectrum(np.fft.rfft2(img), H, W, dy, dx)


def _smooth_axis0(a, sigma):
    """Gaussian smooth along axis 0 (the line axis)."""
    if sigma <= 0 or a.shape[0] < 3:
        return a
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, a)


def subpixel_warp_x(img, ref, ra, ca, nblocks=1, max_shift=0.3, line_sigma=4.0):
    """Align ``img`` to ``ref`` with a per-line (nblocks=1) or within-line
    (nblocks>1) sub-pixel HORIZONTAL warp, on top of the global registration.

    The global field shift in ``register_field`` is rigid, so it cannot remove
    the residual per-line / intra-line horizontal misalignment that independent
    wow/flutter leaves between captures (measured ~0.05px RMS between same-master
    Domesday captures).  We estimate that residual by 1D Lucas-Kanade per block,
        dx = sum(g*(ref-img)) / sum(g*g),   g = d ref/dx
    and resample each line at ``x + dx(x)`` so ``warped(x) ~= ref(x)``.

    Because the residual is near the estimator's noise floor, the shift field is
    heavily regularised -- smoothed across lines (``line_sigma``) and clamped to
    +-``max_shift`` -- so the warp tracks genuine slow wow and not pixel noise
    (warping toward noise would just blur the combine).  Only active rows are
    warped; blanking is left untouched.
    """
    R = ref[ra, ca].astype(np.float64)
    I = img[ra, ca].astype(np.float64)
    nlines, w = R.shape
    bw = max(16, w // max(1, nblocks))
    nb = max(1, w // bw)
    centers = ca.start + (np.arange(nb) + 0.5) * bw          # full-field columns
    # vectorised LK per block (over all lines at once)
    dxg = np.zeros((nlines, nb))
    for b in range(nb):
        sl = slice(b * bw, (b + 1) * bw)
        Rb, Ib = R[:, sl], I[:, sl]
        g = np.gradient(Rb, axis=1)
        den = np.sum(g * g, axis=1)
        num = np.sum(g * (Rb - Ib), axis=1)
        dxg[:, b] = np.where(den > 1e-6, num / den, 0.0)
    dxg = np.clip(_smooth_axis0(dxg, line_sigma), -max_shift, max_shift)
    # apply the warp to each active row across the full width
    out = img.astype(np.float64, copy=True)
    full_x = np.arange(img.shape[1], dtype=np.float64)
    for k, li in enumerate(range(ra.start, ra.stop)):
        dxrow = (np.full_like(full_x, dxg[k, 0]) if nb == 1
                 else np.interp(full_x, centers, dxg[k]))
        out[li] = np.interp(full_x + dxrow, full_x, out[li])
    return out


def register_field(field, ref, ra, ca, subpixel=True, line_reg=0, hint=None):
    """Return field resampled to align with ref, plus affine level-matched.
    ra/ca are the active row/col slices used for measuring the shift.
    line_reg>0 adds a per-line (==1) / within-line (>1, = #blocks) sub-pixel
    horizontal warp after the global shift, to remove residual intra-line wow.

    hint: this capture's shift on the previous frame. Inter-capture offsets
    drift slowly, so searching a small window around the hint costs ~9 field
    transforms instead of the ~34 the cold path needs (integer_shift's three,
    plus a 5x5 grid). If the best lands on the window's border the hint has
    gone stale and the full cold search runs, so this cannot mis-register:
    it only skips work when the answer is already bracketed."""
    fa = field.astype(np.float64)
    best = None
    if subpixel and hint is not None:
        best = (float(hint[0]), float(hint[1]))
    if best is None:
        iy, ix = integer_shift(ref[ra, ca], fa[ra, ca])
        best = (float(iy), float(ix))
    if subpixel:
        refblk = ref[ra, ca]
        H, W = fa.shape
        Ffa = np.fft.rfft2(fa)          # forward FFT once for the whole search
        pyc, pxc = {}, {}
        cands = {}

        def ev(dy, dx):
            key = (round(dy, 4), round(dx, 4))
            if key not in cands:
                if dy == 0 and dx == 0:
                    s = fa[ra, ca]
                else:
                    s = _apply_shift_spectrum(Ffa, H, W, dy, dx,
                                              pyc, pxc)[ra, ca]
                cands[key] = np.mean((s - refblk) ** 2)
            return cands[key]

        # coarse 5x5 grid (0.25 step) around the integer estimate, then
        # per-axis parabolic interpolation of the RMS surface for the
        # sub-step optimum (~0.03px, vs 0.05px for the old 11x11 fine grid,
        # at ~5x fewer full-field transforms). A verification eval keeps
        # the result no worse than the best grid point.
        cy, cx = best
        span = 0.25 if hint is not None else 0.5      # warm: 3x3, cold: 5x5
        bestrms = None
        for dy in cy + np.arange(-span, span + 1e-9, 0.25):
            for dx in cx + np.arange(-span, span + 1e-9, 0.25):
                r = ev(dy, dx)
                if bestrms is None or r < bestrms:
                    bestrms, best = r, (dy, dx)
        if hint is not None and (abs(best[0] - cy) >= span - 1e-9
                                 or abs(best[1] - cx) >= span - 1e-9):
            # stale hint: the optimum is outside the warm window - cold search
            iy, ix = integer_shift(ref[ra, ca], fa[ra, ca])
            cy, cx = float(iy), float(ix)
            bestrms = None
            for dy in cy + np.arange(-0.5, 0.5 + 1e-9, 0.25):
                for dx in cx + np.arange(-0.5, 0.5 + 1e-9, 0.25):
                    r = ev(dy, dx)
                    if bestrms is None or r < bestrms:
                        bestrms, best = r, (dy, dx)

        def para(fm, f0, fp, h):
            den = fm - 2.0 * f0 + fp
            if den <= 0:
                return 0.0
            return float(np.clip(0.5 * (fm - fp) / den * h, -h, h))

        by, bx = best
        oy = para(ev(by - 0.25, bx), ev(by, bx), ev(by + 0.25, bx), 0.25)
        ox = para(ev(by, bx - 0.25), ev(by, bx), ev(by, bx + 0.25), 0.25)
        if (oy or ox) and ev(by + oy, bx + ox) <= bestrms:
            best = (by + oy, bx + ox)
    shifted = fourier_shift(fa, best[0], best[1])
    # affine level match (gain+offset) to reference over active region
    x = shifted[ra, ca].ravel()
    y = ref[ra, ca].ravel()
    A = np.vstack([x, np.ones_like(x)]).T
    g, o = np.linalg.lstsq(A, y, rcond=None)[0]
    out = g * shifted + o
    if line_reg:
        out = subpixel_warp_x(out, ref, ra, ca, nblocks=line_reg)
    return out, best


def chroma_align_field(field, ref, ca, fsc_norm=0.25, halfband=0.06):
    """Make a capture's chroma subcarrier phase-coherent with the reference,
    per line, so it survives averaging.

    ld-decode pilot-locks luma but the per-line chroma subcarrier phase wanders
    (~1px) and that wander is INDEPENDENT between captures, so averaging the
    composite partially cancels chroma.  Here we isolate each line's chroma band
    (around fSC = fs/4 for both PAL and NTSC at 4*fSC sampling) and rotate its
    phase to match the reference capture's chroma on the same line.  Luma is left
    untouched (it is already pilot-solid).  The result shares the reference's
    wander -- common-mode -- so it no longer cancels on combine.
    """
    F = field.astype(np.float64)
    seg = F[:, ca]
    R = ref.astype(np.float64)[:, ca]
    n = seg.shape[1]
    fr = np.fft.rfftfreq(n)
    band = (fr > fsc_norm - halfband) & (fr < fsc_norm + halfband)
    Ff = np.fft.rfft(seg, axis=1)
    Rf = np.fft.rfft(R, axis=1)
    cF = np.where(band, Ff, 0)
    cR = np.where(band, Rf, 0)
    # per-line phase offset of this capture's chroma vs the reference's
    dphi = np.angle((cF * np.conj(cR)).sum(axis=1))
    chroma = np.fft.irfft(cF, n, axis=1)
    chroma_aligned = np.fft.irfft(cF * np.exp(-1j * dphi)[:, None], n, axis=1)
    out = F.copy()
    out[:, ca] = seg - chroma + chroma_aligned
    return out


def dropout_mask(shape, dropouts):
    m = np.zeros(shape, dtype=bool)
    for line, sx, ex in dropouts:
        if 0 <= line < shape[0]:
            m[int(line), int(sx):int(ex)] = True
    return m


# reject a capture deviating > K x the per-pixel MAD (0 disables; env override
# exists so the threshold can be tuned/measured without editing the source)
LOCAL_SPREAD_K = float(os.environ.get("LDSTACK_LOCAL_SPREAD_K", "6.0"))
# A capture whose sync region is mistimed over more than this fraction of a
# field's lines is discarded for the frame (the "field the TBC could not
# place" the integrity check was written for).  At or below it the damage is
# treated as local and masked line-by-line instead, so a three-line defect no
# longer costs the 623 good lines around it.
INTEGRITY_DROP_FRAC = float(os.environ.get("LDSTACK_INTEGRITY_DROP_FRAC", "0.20"))
# cross-master fill must sit within K x the captures' noise of a simple vertical
# estimate; 0 disables the check
FILL_SANITY_K = float(os.environ.get("LDSTACK_FILL_SANITY_K", "8.0"))
# picture-number progression: tolerance per frame, and how many consecutive
# consistent frames confirm a genuine resync after a skip/seek
SEQ_TOL = int(os.environ.get("LDSTACK_SEQ_TOL", "2"))
# frames per worker chunk: long contiguous runs keep each worker's registration
# hint warm (a worker that jumps ahead every few frames re-does the cold search)
CHUNK = int(os.environ.get("LDSTACK_CHUNK", "48"))
WARM = os.environ.get("LDSTACK_WARM", "1") not in ("0", "", "no")
SEQ_CONFIRM = int(os.environ.get("LDSTACK_SEQ_CONFIRM", "4"))


def sieve_keys(raw, seq_tol=None, seq_confirm=None, state=None,
               nlo=0, nhi=0, return_state=False):
    """Filter a capture's raw [(frame_index, key)] down to the frames whose
    keys can be trusted, and yield them in order.

    Shared by the indexed .tbc reader and the streaming decode source so both
    agree on which frames a capture really contributes.

    A stream sieves a window at a time, which needs two things beyond the
    whole-list call.  ``nlo``/``nhi`` mark leading and trailing entries that
    are present only to give the median test its neighbours: they are judged
    but never emitted, and the next window emits them for real.  ``state``
    carries the sequence pass's position across windows - both what key it
    expects next and any part-built resync run - so a window boundary is not
    mistaken for a seek.  Pass the returned state back in with
    ``return_state=True``.
    """
    tol = SEQ_TOL if seq_tol is None else seq_tol
    confirm = SEQ_CONFIRM if seq_confirm is None else seq_confirm
    keep = []
    out = []
    expect0, pending0 = state if state else (None, [])
    # An isolated corrupt VBI read (weak inner-radius RF flipping a
    # picture-number digit) can claim a far-away key and, via
    # first-occurrence, SHADOW the real frame (EcoDisc S2: an early frame
    # misreading as picture 2156 displaced the true one on all four
    # captures). Reject keys that deviate wildly from the local trend of
    # their neighbours before assigning slots.
    for idx in range(nlo, len(raw) - nhi):
        fi, key = raw[idx]
        lo, hi = max(0, idx - 2), min(len(raw), idx + 3)
        neigh = [k for j, (_, k) in enumerate(raw[lo:hi], lo) if j != idx]
        if neigh:
            med = sorted(neigh)[len(neigh) // 2]
            if abs(key - med) > 50:
                continue
        keep.append((fi, key))
    # A capture that starts mid-programme and then seeks back to the disc
    # start sweeps the laser across tracks, and every frame it crosses
    # carries a REAL picture number on garbage/black content (CommunitySouth
    # ds1: ...389 390 390 390 340 290 240 190 90 30 1 1 2 3...). Those keys
    # are not misreads, so no value test can spot them - but during genuine
    # playback the picture number advances by one per frame. Accept a key
    # only while it follows that progression, and require several
    # consecutive consistent frames before trusting a new position (a real
    # skip/seek in the middle of a capture).
    expect = expect0
    pending = list(pending0)
    for fi, key in keep:
        if expect is None or abs(key - expect) <= tol:
            expect = key + 1
            pending = []
        else:
            pending.append((fi, key))
            tail = pending[-confirm:]
            if len(tail) >= confirm and all(
                    b[1] - a[1] == 1 for a, b in zip(tail, tail[1:])):
                # a confirmed resync (the capture really is playing from a
                # new position): adopt the confirming run, discard the
                # transient frames that preceded it
                out.extend(tail)
                expect = key + 1
                pending = []
            continue
        out.append((fi, key))
    return (out, (expect, pending)) if return_state else out

# a capture's frame is dropped when its level is this many robust sigmas from
# the consensus of the captures for that frame (catches seek/scan frames)
FRAME_LEVEL_K = float(os.environ.get("LDSTACK_FRAME_LEVEL_K", "8.0"))
DIFF_DOD = os.environ.get("LDSTACK_DIFF_DOD", "0") not in ("0", "", "no")


def combine_fields(fields, masks, weights, sigma, diff_dod=False):
    """Dropout-aware, inverse-variance weighted combine of registered fields.

    fields: list of float 2D arrays (already registered to a common grid)
    masks:  list of bool dropout masks (True = bad pixel, exclude)
    weights: per-capture inverse-variance weights
    sigma:  per-capture noise (for outlier rejection vs the median)
    diff_dod: dropout flags are sometimes over-cautious (idea from
      decode-orc's stacker): where EVERY source flagged the pixel, accept
      the median of the flagged values anyway IF at least 2 of >=3 sources
      still agree within the usual 5-sigma window AND the level is
      plausible video (a real dropout rails low; 8192 is ~half the
      standard 16-bit black level, well below sync tips).
    """
    stack = np.stack(fields, 0)
    good = ~np.stack(masks, 0)
    med = np.median(np.where(good, stack, np.nan), axis=0)
    med = np.where(np.isnan(med), stack.mean(0), med)
    # reject pixels far from the median (catches un-flagged dropouts/glitches)
    thr = (5.0 * np.array(sigma)).reshape(-1, 1, 1)
    good &= np.abs(stack - med[None]) <= thr
    # ...and again against the LOCAL spread. A capture-specific defect that no
    # dropout detector flagged (Community North DS3 put a 26000-level streak
    # through one line where the other five agreed to within ~3000) sits well
    # inside the global 5-sigma window, so it survived into the output. Compare
    # each capture with the per-pixel MAD of the set instead, floored by the
    # captures' own noise so ordinary disagreement is never rejected.
    if len(fields) >= 4 and LOCAL_SPREAD_K > 0:
        dev = np.abs(stack - med[None])
        mad = np.median(dev, axis=0) * 1.4826
        floor = 3.0 * float(np.median(sigma))
        good &= dev <= np.maximum(LOCAL_SPREAD_K * mad, floor)[None]
    w = np.array(weights, dtype=np.float64).reshape(-1, 1, 1) * good
    wsum = w.sum(0)
    out = np.where(wsum > 0, (stack * w).sum(0) / np.maximum(wsum, 1e-9), med)
    if diff_dod and len(fields) >= 3:
        allbad = wsum == 0
        if allbad.any():
            vals = stack[:, allbad]
            dmed = np.median(vals, axis=0)
            nagree = (np.abs(vals - dmed[None])
                      <= 5.0 * float(np.median(sigma))).sum(0)
            rec = (nagree >= 2) & (dmed > 8192.0)
            if rec.any():
                o = out[allbad]; o[rec] = dmed[rec]; out[allbad] = o
                ws = wsum[allbad]; ws[rec] = 1e-6; wsum[allbad] = ws
    return out, wsum


def perframe_weights(regs, masks, global_w, ra, ca, drop=True, drop_mult=3.0,
                     dead_band=1.5):
    """Per-FRAME inverse-variance weights with a registration-confidence guard.

    The per-capture ``global_w`` (from the quality pass) reflects a capture's
    *average* noise, but a capture can be fine overall yet bad on one frame
    (local rot, a mis-registered field).  We measure each registered field's
    residual against the per-pixel median of the set on this frame and:

    * leave the weight UNCHANGED while the residual is within ``dead_band`` x the
      median residual -- so equal-quality captures keep equal weight (weighting
      by 1/residual^2 there would just couple the weights to noise realisations
      and *raise* the combined noise);
    * softly down-weight (x (dead_band*med/resid)^2) a field that is clearly
      noisier than the group but still usable;
    * DROP a field whose residual is a gross outlier (> ``drop_mult`` x median) --
      this catches a field that registered to the wrong sub-pixel offset and
      would otherwise smear the combine without tripping ``combine_fields``'
      per-pixel 5-sigma test (a smooth mis-registration stays within 5-sigma yet
      is systematically wrong).

    The drop only fires with >=3 fields (with 2, median == mean and the two are
    symmetric, so neither can be called the outlier).  Never drops every field.
    """
    gw = np.asarray(global_w, dtype=np.float64)
    if len(regs) < 2:
        return list(gw)
    stack = np.stack([r[ra, ca] for r in regs], 0)
    good = ~np.stack([m[ra, ca] for m in masks], 0)
    med = np.median(np.where(good, stack, np.nan), axis=0)
    med = np.where(np.isnan(med), stack.mean(0), med)
    resid = np.empty(len(regs))
    for i in range(len(regs)):
        d = (stack[i] - med)[good[i]]
        resid[i] = np.sqrt(np.mean(d * d)) if d.size else np.inf
    finite = resid[np.isfinite(resid)]
    medr = np.median(finite) if finite.size else 1.0
    # dead-band: within dead_band*medr -> untouched; worse -> 1/excess^2 penalty
    excess = np.maximum(resid / (dead_band * max(medr, 1e-9)), 1.0)
    eff = gw / excess ** 2
    if drop and len(regs) >= 3 and finite.size:
        bad = resid > drop_mult * medr
        if 0 < bad.sum() < len(regs):
            eff[bad] = 0.0
    if not np.any(eff > 0):
        eff = gw
    return list(eff)


def align_audio(ref, x, maxlag=4000):
    """Sub-sample lag of x vs ref (mono): the integer central-segment xcorr peak
    refined by parabolic interpolation of its two neighbours.  Returns a float;
    a fractional shift removes the +-0.5-sample quantisation that otherwise combs
    out high audio frequencies when capture waveforms are averaged."""
    n = min(len(ref), len(x))
    if n < 256:
        return 0.0
    s, L = n // 4, n // 2
    r = ref[s:s + L] - ref[s:s + L].mean()
    rn = np.linalg.norm(r) + 1e-9
    corr = np.full(2 * maxlag + 1, -2.0)
    for i, k in enumerate(range(-maxlag, maxlag + 1)):
        a0 = s + k
        if a0 < 0 or a0 + L > len(x):     # out of range (avoid negative-index wrap)
            continue
        seg = x[a0:a0 + L]
        seg = seg - seg.mean()
        corr[i] = float((r * seg).sum() / (rn * (np.linalg.norm(seg) + 1e-9)))
    j = int(np.argmax(corr))
    lag = float(j - maxlag)
    if 0 < j < len(corr) - 1:                      # parabolic sub-sample refine
        a, b, c = corr[j - 1], corr[j], corr[j + 1]
        denom = a - 2 * b + c
        if denom != 0 and a > -1.9 and c > -1.9:
            lag += 0.5 * (a - c) / denom
    return lag


def frac_shift_audio(a, lag):
    """Shift audio ``a`` (n,2 float-able) by a fractional sample ``lag`` so it
    aligns to the reference (matches the old integer np.roll(a, -lag) at integer
    lag).  Cubic (Catmull-Rom) interpolation -- near-lossless for band-limited
    audio, unlike linear interpolation which low-passes and would comb out the
    high frequencies the sub-sample alignment is meant to preserve."""
    if lag == 0:
        return np.asarray(a, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    nlen = len(a)
    p = np.arange(nlen) + lag
    ip = np.floor(p).astype(int)
    f = (p - ip)[:, None]                         # (n,1) broadcasts over channels
    def tap(off):
        return a[np.clip(ip + off, 0, nlen - 1)]
    m1, p0, p1, p2 = tap(-1), tap(0), tap(1), tap(2)
    return (p0 + 0.5 * f * (p1 - m1 + f * (2 * m1 - 5 * p0 + 4 * p1 - p2
            + f * (3 * (p0 - p1) + p2 - m1))))


def combine_audio(aligned, weights, reject=6.0):
    """Robust weighted-average of per-capture, already-aligned audio (each n,2).

    Like combine_fields for video: a click/glitch in one capture shows up as a
    per-sample outlier far from the cross-capture median, so reject samples
    beyond ``reject`` x the robust inter-capture spread (MAD) and weighted-mean
    the rest -- the glitch is excluded, not smeared in.  Needs >=3 captures to
    reject (with 2, median == mean and neither can be called the outlier)."""
    stack = np.stack([np.asarray(a, dtype=np.float64) for a in aligned], 0)
    med = np.median(stack, axis=0)
    good = np.ones(stack.shape, dtype=bool)
    if stack.shape[0] >= 3:
        dev = np.abs(stack - med[None])
        scale = 1.4826 * np.median(dev) + 1e-9
        good = dev <= reject * scale
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1, 1) * good
    wsum = w.sum(0)
    return np.where(wsum > 0, (stack * w).sum(0) / np.maximum(wsum, 1e-9), med)


# --------------------------------------------------------------------------- #
#  Quality + master clustering
# --------------------------------------------------------------------------- #
def analyse(frame_dicts, ra, ca, shift_tol=0.3, noise_mult=4.0,
            allow_noise_drop=True):
    """Estimate per-capture noise and cluster captures by master.

    frame_dicts: list of {name: Frame} for a sample of aligned frames.
    Returns dict per source name: {'noise', 'dx', 'master', 'keep', 'reason'}.

    Master discriminator = the horizontal sub-pixel registration offset.
    Different glass masters lay the active video down at a slightly different
    position relative to sync (a fixed, content-independent fractional-pixel
    shift); same-master captures register to ~0 against each other.  This is far
    more robust than residual correlation, which is confounded by shared edge
    residuals on high-detail content (e.g. a test card).  Captures are clustered
    by their median dx against a fixed reference; the largest cluster is the
    primary master, the rest are alt masters (used for cross-master fill).
    """
    names = sorted({n for fd in frame_dicts for n in fd})
    # fixed reference = the capture present in the most sample frames
    refn = max(names, key=lambda n: sum(n in fd for fd in frame_dicts))
    shifts = {n: [] for n in names}
    noise = {n: [] for n in names}
    for fd in frame_dicts:
        if refn not in fd or len(fd) < 2:
            continue
        ref = fd[refn].f0.astype(np.float64)
        reg = {}
        for n, fr in fd.items():
            r, best = register_field(fr.f0.astype(np.float64), ref, ra, ca,
                                     subpixel=True)
            shifts[n].append(best[1])           # dx (sub-pixel)
            reg[n] = r[ra, ca]
        med = np.median(np.stack(list(reg.values()), 0), 0)
        gy, gx = np.gradient(med)
        flat = np.hypot(gy, gx) < np.percentile(np.hypot(gy, gx), 40)
        for n, r in reg.items():
            noise[n].append((r - med)[flat].std())
    info = {n: {} for n in names}
    for n in names:
        info[n]["noise"] = float(np.median(noise[n])) if noise[n] else float("inf")
        info[n]["dx"] = float(np.median(shifts[n])) if shifts[n] else None
    present = [n for n in names if shifts[n]]
    # 1-D clustering of dx: sort, split where the gap exceeds shift_tol
    order = sorted(present, key=lambda n: info[n]["dx"])
    clusters, cur = [], []
    for n in order:
        if cur and (info[n]["dx"] - info[cur[-1]]["dx"]) > shift_tol:
            clusters.append(cur); cur = []
        cur.append(n)
    if cur:
        clusters.append(cur)
    clusters.sort(key=lambda g: (-len(g),
                                 np.mean([info[k]["noise"] for k in g])))
    for ci, g in enumerate(clusters):
        for n in g:
            info[n]["master"] = ci
    primary = clusters[0] if clusters else []
    cutoff = (noise_mult * float(np.median([info[n]["noise"] for n in primary]))
              if primary else float("inf"))
    for n in names:
        if n not in present:
            info[n]["keep"], info[n]["reason"] = False, "no usable frames"
        elif info[n].get("master", -1) != 0:
            info[n]["keep"], info[n]["reason"] = \
                False, "different master (dx=%+.2f)" % info[n]["dx"]
        elif allow_noise_drop and info[n]["noise"] > cutoff:
            info[n]["keep"], info[n]["reason"] = False, "noise outlier"
        else:
            info[n]["keep"], info[n]["reason"] = True, "ok"
    return info


# --------------------------------------------------------------------------- #
#  Output writer (.tbc / .tbc.json / .tbc.db / .pcm / .efm)
# --------------------------------------------------------------------------- #
class StackWriter:
    def __init__(self, outbase, vp, system, git_commit=""):
        self.outbase = outbase
        self.vp = dict(vp)
        self.system = system
        self.git_commit = git_commit
        self.tbc = open(outbase + ".tbc", "wb")
        self.pcm = open(outbase + ".pcm", "wb")
        self.fields_json = []
        self.audio_total = 0
        if os.path.exists(outbase + ".tbc.db"):
            os.unlink(outbase + ".tbc.db")
        self.db = sqlite3.connect(outbase + ".tbc.db")
        self._schema()
        self._field_id = 0

    def _schema(self):
        self.db.executescript(dedent('''\
            PRAGMA user_version = 1;
            CREATE TABLE capture(capture_id INTEGER PRIMARY KEY, system TEXT,
                decoder TEXT, git_branch TEXT, git_commit TEXT,
                video_sample_rate REAL, active_video_start INTEGER,
                active_video_end INTEGER, field_width INTEGER, field_height INTEGER,
                number_of_sequential_fields INTEGER, colour_burst_start INTEGER,
                colour_burst_end INTEGER, is_mapped INTEGER, is_subcarrier_locked INTEGER,
                is_widescreen INTEGER, white_16b_ire INTEGER, black_16b_ire INTEGER,
                blanking_16b_ire INTEGER, capture_notes TEXT);
            CREATE TABLE pcm_audio_parameters(capture_id INTEGER PRIMARY KEY,
                bits INTEGER, is_signed INTEGER, is_little_endian INTEGER, sample_rate REAL);
            CREATE TABLE field_record(capture_id INTEGER, field_id INTEGER,
                audio_samples INTEGER, decode_faults INTEGER, disk_loc REAL,
                efm_t_values INTEGER, field_phase_id INTEGER, file_loc INTEGER,
                is_first_field INTEGER, median_burst_ire REAL, pad INTEGER,
                sync_conf INTEGER, PRIMARY KEY(capture_id, field_id));
            CREATE TABLE vbi(capture_id INTEGER, field_id INTEGER, vbi0 INTEGER,
                vbi1 INTEGER, vbi2 INTEGER, PRIMARY KEY(capture_id, field_id));
            CREATE TABLE drop_outs(capture_id INTEGER, field_id INTEGER,
                field_line INTEGER, startx INTEGER, endx INTEGER);
        '''))

    def write_field(self, arr_u16, meta, vbidata, dropouts, efm_t, audio_n):
        self.tbc.write(np.ascontiguousarray(arr_u16, dtype="<u2").tobytes())
        fid = self._field_id
        self._field_id += 1
        fj = {
            "isFirstField": bool(meta.get("isFirstField", fid % 2 == 0)),
            "syncConf": int(meta.get("syncConf", 100)),
            "seqNo": fid + 1,
            "diskLoc": meta.get("diskLoc", 0),
            "fieldPhaseID": meta.get("fieldPhaseID", 0),
            "medianBurstIRE": meta.get("medianBurstIRE", 0),
            "audioSamples": int(audio_n),
            "efmTValues": int(efm_t),
            "vbi": {"vbiData": list(vbidata)},
        }
        do = {"fieldLine": [], "startx": [], "endx": []}
        for line, sx, ex in dropouts:
            do["fieldLine"].append(int(line)); do["startx"].append(int(sx)); do["endx"].append(int(ex))
        if do["fieldLine"]:
            fj["dropOuts"] = do
        self.fields_json.append(fj)
        self.db.execute(
            "INSERT INTO field_record(capture_id,field_id,audio_samples,efm_t_values,"
            "field_phase_id,is_first_field,sync_conf,median_burst_ire,pad,disk_loc) "
            "VALUES(0,?,?,?,?,?,?,?,0,?)",
            (fid, int(audio_n), int(efm_t), int(fj["fieldPhaseID"]),
             1 if fj["isFirstField"] else 0, fj["syncConf"], fj["medianBurstIRE"],
             fj["diskLoc"]))
        v = list(vbidata) + [0, 0, 0]
        self.db.execute("INSERT INTO vbi VALUES(0,?,?,?,?)", (fid, v[0], v[1], v[2]))
        for line, sx, ex in dropouts:
            self.db.execute("INSERT INTO drop_outs VALUES(0,?,?,?,?)",
                            (fid, int(line), int(sx), int(ex)))

    def write_audio(self, audio):
        if audio is None:
            return
        a = np.ascontiguousarray(np.clip(audio, -32768, 32767), dtype="<i2")
        self.pcm.write(a.tobytes())
        self.audio_total += a.shape[0]

    def close(self, efm_bytes=None):
        self.vp["numberOfSequentialFields"] = len(self.fields_json)
        out = {"videoParameters": self.vp, "fields": self.fields_json}
        if self.git_commit:
            self.vp["gitCommit"] = self.git_commit
        json.dump(out, open(self.outbase + ".tbc.json", "w"))
        vp = self.vp
        self.db.execute(
            "INSERT INTO capture(capture_id,system,decoder,git_commit,video_sample_rate,"
            "active_video_start,active_video_end,field_width,field_height,"
            "number_of_sequential_fields,white_16b_ire,black_16b_ire,"
            "is_mapped,is_subcarrier_locked,is_widescreen,capture_notes) "
            "VALUES(0,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.system, "ld-decode", self.git_commit,
             vp.get("sampleRate"), vp.get("activeVideoStart"), vp.get("activeVideoEnd"),
             vp.get("fieldWidth"), vp.get("fieldHeight"), len(self.fields_json),
             int(vp.get("white16bIre", 0)), int(vp.get("black16bIre", 0)),
             1 if vp.get("isMapped") else 0,
             1 if vp.get("isSubcarrierLocked") else 0,
             1 if vp.get("isWidescreen") else 0,
             "stacked by ld-disc-stack"))
        self.db.execute(
            "INSERT INTO pcm_audio_parameters VALUES(0,16,1,1,?)",
            (vp.get("audioSampleRate", 44100),))
        self.db.commit()
        self.db.close()
        self.tbc.close()
        self.pcm.close()
        if efm_bytes is not None:
            with open(self.outbase + ".efm", "wb") as f:
                f.write(efm_bytes)


# --------------------------------------------------------------------------- #
#  Orchestrator
# --------------------------------------------------------------------------- #
def lockstep(sources):
    """Merge several frame iterators (each yielding Frames in increasing key
    order) into a stream of (key, {name: Frame}) for keys present in ALL
    sources.  Buffer is bounded by the inter-capture alignment skew, so full
    discs stream without holding everything in RAM."""
    its = {s.name: iter(s.frames()) for s in sources}
    buf = {n: {} for n in its}
    head = {n: -1 for n in its}
    done = {n: False for n in its}

    def adv(n):
        try:
            fr = next(its[n])
            buf[n][fr.key] = fr
            head[n] = fr.key
            return True
        except StopIteration:
            done[n] = True
            return False

    for n in its:
        adv(n)
    while True:
        if all(done[n] and not buf[n] for n in its):
            return
        # frontier = the largest of each source's smallest buffered key, so all
        # could plausibly carry it
        mins = [min(buf[n]) for n in its if buf[n]]
        if len(mins) < sum(not done[n] or bool(buf[n]) for n in its):
            for n in its:
                if not done[n] and not buf[n]:
                    adv(n)
            continue
        frontier = max(mins)
        for n in its:
            while not done[n] and head[n] < frontier:
                adv(n)
        present = [n for n in its if frontier in buf[n]]
        if present:
            # emit whatever subset has this key; the stacker decides whether
            # enough primary-master captures are present, and uses any
            # alt-master captures only to fill residual master-level dropouts.
            yield frontier, {n: buf[n][frontier] for n in present}
        for n in its:  # drop frontier and anything older (can never complete)
            for k in [k for k in buf[n] if k <= frontier]:
                del buf[n][k]


def _efm_lag(ref, x, maxlag=240, maxn=4_000_000):
    n = min(len(ref), len(x))
    # the residual lag is small (<maxlag); correlate a central window rather
    # than the whole multi-million-sample stream to bound the FFT cost.
    if n > maxn:
        s = (n - maxn) // 2
        ref, x, n = ref[s:s + maxn], x[s:s + maxn], maxn
    a = ref[:n] - ref[:n].mean()
    b = x[:n] - x[:n].mean()
    L = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(a, L) * np.conj(np.fft.rfft(b, L)), L)
    c = np.concatenate([c[-maxlag:], c[:maxlag + 1]])
    # peak index j -> correlation lag (j - maxlag); the roll d that produced
    # x = roll(ref, d) shows up at lag -d, so return d = maxlag - j to make
    # average_efm's np.roll(x, -lag) undo it.
    return int(maxlag - np.argmax(c))


def merge_efm_sectors(efm_paths, out_data, scratch_dir=None, log=print):
    """Sector-level EFM merge: decode each capture's .efm to its data image with
    ld-process-efm, then fill the output from whichever disc decoded each byte
    (sector) cleanly.  EFM data is master-independent and byte-identical where
    valid, so a byte bad/missing (zero) on one disc is filled from another --
    recovering sectors no single disc has.  Waveform averaging is NOT used (real
    captures are not phase-coherent at the EFM bit rate).  Returns (consensus
    nonzero bytes, best-single nonzero bytes, ndiscs) or None if nothing decoded.
    """
    import subprocess, tempfile
    tmp = tempfile.mkdtemp(prefix="efmmerge_",
                           dir=resolve_scratch(scratch_dir, out_data))
    bins = []
    try:
        for i, p in enumerate(efm_paths):
            b = os.path.join(tmp, f"d{i}.bin")
            r = subprocess.run(["ld-process-efm", "-b", "-q", p, b],
                               capture_output=True)
            if os.path.exists(b) and os.path.getsize(b) > 0:
                bins.append(b)
        if not bins:
            return None
        merged = None; best = 0
        for b in bins:
            d = np.fromfile(b, dtype=np.uint8)
            best = max(best, int(np.count_nonzero(d)))
            if merged is None:
                merged = d.copy()
            else:
                # captures can decode to slightly different .bin lengths; pad the
                # shorter of the two so they align before the per-byte OR-fill.
                n = max(len(merged), len(d))
                if len(merged) < n:
                    merged = np.concatenate([merged, np.zeros(n - len(merged), np.uint8)])
                if len(d) < n:
                    d = np.concatenate([d, np.zeros(n - len(d), np.uint8)])
                m = (merged == 0) & (d != 0)
                merged[m] = d[m]
        merged.tofile(out_data)
        return int(np.count_nonzero(merged)), best, len(bins)
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


def _efm_pll_class():
    """Import EFM_PLL whether stack.py is run as a module (lddecode.stack) or as
    a bare script (python lddecode/stack.py), where the relative import fails."""
    try:
        from .efm_pll import EFM_PLL
    except ImportError:
        from efm_pll import EFM_PLL
    return EFM_PLL


def average_efm(waves, weights):
    """Cross-correlate each TBC-locked EFM waveform to the reference, align, and
    weighted-average.  EFM is continuous (not frame-locked), so a residual phase
    offset remains after TBC and must be removed before averaging."""
    ref = np.asarray(waves[0], np.float64)
    L = len(ref)
    acc = np.zeros(L)
    ws = 0.0
    for w, wt in zip(waves, weights):
        x = np.asarray(w, np.float64)
        if len(x) != L:
            x = np.resize(x, L)
        lag = _efm_lag(ref, x)
        if lag:
            x = np.roll(x, -lag)
        acc += wt * x
        ws += wt
    return (acc / max(ws, 1e-9)) if ws else acc


def _compute_frame(fd, P):
    """Stack one aligned frame dict {name: Frame} -> a writable result dict
    (pure per-frame computation; no shared state, so it can run in a worker).
    Returns None when no primary capture carries this frame."""
    primary, fill_sources = P["primary"], P["fill_sources"]
    ra, ca, fh, fw = P["ra"], P["ca"], P["fh"], P["fw"]
    weights, sig = P["weights"], P["sig"]
    present_primary = [n for n in primary if n in fd]
    present_fill = [n for n in fill_sources if n in fd] if P["cross_fill"] else []
    if not present_primary:
        # picture missing from every primary-master capture: build the frame
        # from the alt-master captures rather than dropping it from the output
        if not present_fill:
            return None
        present_primary, present_fill = present_fill, []
    # Integrity check (ported from ld-disc-stacker's isIntegrityOk): in a
    # correctly timed field the sync region sits well below black on every
    # line. Three consecutive lines that do not is the signature of a skip or
    # sample drop, i.e. a field the TBC could not place - drop that capture for
    # this frame. Catches scrambled frames at normal brightness, which the
    # level test below cannot see.
    integrity_bad = {}          # name -> {"f0": bool per line, "f1": ...}
    if len(present_primary) >= 3 and P.get("black") is not None:
        thr = P["black"] - 2560.0            # 10 IRE below black
        ok = []
        for n in present_primary:
            per_field, worst = {}, 0.0
            for which2 in ("f0", "f1"):
                col = np.asarray(getattr(fd[n], which2))[:, 4].astype(np.float64)
                over = col > thr
                # Keep only runs of three or more: an isolated line over the
                # threshold is noise, a run is mistimed video.  Record WHICH
                # lines rather than a single verdict for the field.
                badline = np.zeros(len(over), dtype=bool)
                run = 0
                for i, v in enumerate(over):
                    run = run + 1 if v else 0
                    if run >= 3:
                        badline[i - run + 1:i + 1] = True
                per_field[which2] = badline
                worst = max(worst, float(badline.mean()))
            integrity_bad[n] = per_field
            if worst <= INTEGRITY_DROP_FRAC:
                ok.append(n)
        if len(ok) >= 2:
            present_primary = ok

    # Reference selection: everything registers to this capture's field, so a
    # garbage frame here poisons the whole combine. Take the capture closest to
    # the consensus level rather than whichever sorts first, and drop captures
    # whose frame is grossly inconsistent with the others (a seek/scan frame
    # carrying a real picture number, but black or scrambled content).
    if len(present_primary) >= 3:
        means = {n: float(np.mean(fd[n].f0)) for n in present_primary}
        vals = sorted(means.values())
        cons = vals[len(vals) // 2]
        spread = max(np.median([abs(v - cons) for v in vals]) * 1.4826, 250.0)
        keep = [n for n in present_primary
                if abs(means[n] - cons) <= FRAME_LEVEL_K * spread]
        if len(keep) >= 2:
            present_primary = keep
        present_primary = sorted(present_primary,
                                 key=lambda n: abs(means[n] - cons))
    ref_frame = fd[present_primary[0]]
    filled = 0
    out_fields = []
    for fidx, which in enumerate(("f0", "f1")):
        ref_field = getattr(ref_frame, which).astype(np.float64)
        do_attr = "do0" if which == "f0" else "do1"

        def combine_set(names, diff_dod=False):
            regs, masks, ws, sigs = [], [], [], []
            for n in names:
                fr = fd[n]
                if fr is ref_frame:
                    # registering/phase-aligning the reference to itself is an
                    # identity (the search provably lands at (0,0), the affine
                    # at gain 1 offset 0, the chroma rotation at 0) -- skip it
                    reg = ref_field
                else:
                    hk = (n, which)
                    reg, shift = register_field(
                        getattr(fr, which).astype(np.float64), ref_field,
                        ra, ca, P["subpixel"], line_reg=P["line_reg"],
                        hint=_SHIFT_HINT.get(hk) if WARM else None)
                    _SHIFT_HINT[hk] = shift
                    if P["chroma_align"]:
                        reg = chroma_align_field(reg, ref_field, ca)
                regs.append(reg)
                m = dropout_mask((fh, fw), getattr(fr, do_attr))
                # Mask this capture's locally mistimed lines instead of
                # discarding its whole frame; the other captures cover them
                # exactly as they cover a dropout.
                ib = integrity_bad.get(n, {}).get(which)
                if ib is not None and ib.any():
                    m[ib, :] = True
                masks.append(m)
                ws.append(weights[n]); sigs.append(sig[n])
            if P["reg_confidence"]:
                ws = perframe_weights(regs, masks, ws, ra, ca,
                                      drop=True, drop_mult=P["reg_drop_mult"])
            return combine_fields(regs, masks, ws, sigs,
                                  diff_dod=diff_dod)

        # dropout-flagged pixels are never used, even when every capture
        # flagged them (user instruction): leave them for the dropout
        # corrector rather than trusting flagged samples
        out, wsum = combine_set(present_primary, diff_dod=DIFF_DOD)
        allbad = (wsum == 0)            # dropout present in EVERY primary disc
        # ---- cross-master fill: patch master-level dropouts from alt master
        if present_fill and allbad.any():
            fout, fws = combine_set(present_fill)
            fillable = allbad & (fws > 0)
            # The fill source is often a SINGLE alt-master capture, so nothing
            # cross-checks it: combine_fields with one field cannot reject
            # anything. A defect on that one capture then lands in the output
            # and is marked good (Community North: ds3's unflagged streak was
            # filled into a line where all five primaries were flagged bad).
            # Require the fill to be plausible against its own vertical
            # neighbourhood; where it is not, leave the pixel flagged so the
            # downstream dropout corrector conceals it instead.
            if fillable.any() and FILL_SANITY_K > 0:
                est = np.empty_like(fout)
                est[1:-1] = 0.5 * (fout[:-2] + fout[2:])
                est[0], est[-1] = fout[1], fout[-2]
                lim = max(FILL_SANITY_K * float(np.median(list(sig.values()))), 1500.0)
                implausible = fillable & (np.abs(fout - est) > lim)
                fillable &= ~implausible
            if fillable.any():
                out = out.copy()
                out[fillable] = fout[fillable]
                filled += int(fillable.sum())
                allbad = allbad & ~fillable
        outu = np.clip(np.round(out), 0, 65535).astype(np.uint16)
        meta = ref_frame.meta0 if which == "f0" else ref_frame.meta1
        resid_do = []                   # still bad after cross-fill
        if allbad.any():
            for ln in np.where(allbad.any(1))[0]:
                xs = np.where(allbad[ln])[0]
                resid_do.append((int(ln), int(xs.min()), int(xs.max()) + 1))
        # EFM: pool ALL discs (the digital data is master-independent)
        efm_src = [n for n in present_primary + present_fill
                   if fd[n].efm is not None]
        efm_avi = None
        if efm_src:
            av = average_efm([fd[n].efm[fidx] for n in efm_src],
                             [weights[n] for n in efm_src])
            efm_avi = np.clip(np.round(av), -32768, 32767).astype("<i2")
        audio_n = (0 if which == "f1" else
                   (ref_frame.audio.shape[0] if ref_frame.audio is not None else 0))
        out_fields.append((outu, meta, field_vbi(meta), resid_do, efm_avi,
                           audio_n))
    # audio: PRIMARY master only (analog audio is master-specific).
    # Sub-sample align each capture to the reference, then robust
    # (outlier-rejecting) weighted-average so a click/glitch in one capture
    # is excluded rather than smeared into the mean.
    audio_out = None
    ref_audio = ref_frame.audio
    if ref_audio is not None:
        refm = ref_audio[:, 0].astype(np.float64)
        aligned, awts = [], []
        for n in present_primary:
            fa = fd[n].audio
            if fa is None or fa.shape != ref_audio.shape:
                continue
            lag = align_audio(refm, fa[:, 0].astype(np.float64))
            aligned.append(frac_shift_audio(fa, lag))
            awts.append(weights[n])
        if aligned:
            audio_out = combine_audio(aligned, awts)
    return {"fields": out_fields, "audio": audio_out, "filled": filled}


_WORKER_STATE = None            # (sources, P) inherited by forked pool workers
_SHIFT_HINT = {}                # (capture, field) -> last registration shift


def _frame_by_key(key):
    sources, P = _WORKER_STATE
    fd = {s.name: s.get(key) for s in sources if key in s.keys()}
    return _compute_frame(fd, P)


def stack(sources, outbase, system="PAL", subpixel=True, chroma_align=True,
          cross_fill=True, masters=None, sample=24, analysis_window=None,
          max_frames=None,
          scratch_dir=None, git_commit="", reg_confidence=True, reg_drop_mult=3.0,
          line_reg=0, jobs=0, log=print):
    aw = analysis_window or max(sample * 2, 40)
    merged = lockstep(sources)

    # ---- buffer an initial window for the quality / master analysis ----
    # (live .ldf sources only know their geometry once decoding has started, so
    # pull frames first, then read videoParameters)
    head_buf = []
    for key, fd in merged:
        head_buf.append((key, fd))
        if len(head_buf) >= aw:
            break
    if not head_buf:
        raise SystemExit("no frames shared across all captures")
    vp = next((s.videoParameters for s in sources
               if s.videoParameters is not None), None)
    if vp is None:
        raise SystemExit("could not determine video parameters from any capture")
    ra, ca = _active(vp)
    fh, fw = vp["fieldHeight"], vp["fieldWidth"]
    # Sample for the quality / master analysis.  The head window is all a
    # streaming .ldf source can offer, but the opening frames of a disc are a
    # poor proxy for the side: on an AIV disc they are often a near-identical
    # still, so inter-capture differences there collapse to almost nothing and
    # the noise cutoff (4x the primary median) becomes hair-trigger.  NationalA
    # dropped two good captures that way - 794 and 1385 against a 37-41
    # baseline over frames 1-48, where all six measure 1015-1286 sampled across
    # the disc.  Indexed .tbc sources are randomly accessible, so spread the
    # sample over the whole side; fall back to the head window otherwise.
    asample = None
    if all(isinstance(s, TBCFrameSource) for s in sources):
        keysets = [s.keys() for s in sources]
        span = sorted(set.intersection(*keysets)) or sorted(set.union(*keysets))
        if span:
            pick = span[::max(1, len(span) // sample)][:sample]
            asample = [{s.name: s.get(k) for s in sources if k in s.keys()}
                       for k in pick]
    indexed = asample is not None
    if not asample:
        step = max(1, len(head_buf) // sample)
        asample = [fd for _, fd in head_buf[::step][:sample]]
    # Which of analyse()'s two decisions the sample can support.  Clustering by
    # sub-pixel dx is content-independent - it measures where the master laid
    # the active video down relative to sync - so the head window is as good as
    # any.  The noise level is not: on an AIV disc the opening frames are often
    # a near-identical still, inter-capture differences there collapse, and the
    # 4x-median cutoff turns hair-trigger.  That is exactly how NationalA threw
    # away two good captures (794 and 1385 against a 37-41 head baseline, where
    # all six measure 1015-1286 sampled across the disc).  So a streaming
    # source, which can only offer its opening, clusters but never drops: a
    # capture that really is noisy is down-weighted by 1/noise^2 and by the
    # per-frame registration confidence, which is the graceful version of the
    # same judgement.
    info = analyse(asample, ra, ca, allow_noise_drop=indexed)
    if not indexed:
        log("[stack] streaming sources: master clustering from the opening "
            "window, noise used for weighting only (no noise-outlier drop)")
    if masters:
        # explicit master grouping (e.g. from the disc PP/NP/AK marks); group 0
        # is the primary, the rest are alt-master fill sources.  Tokens are
        # matched as substrings of the capture name.
        matched = set()
        for mid, grp in enumerate(masters):
            for n in info:
                if any(tok in n for tok in grp):
                    info[n]["master"] = mid
                    info[n]["keep"] = (mid == 0)
                    info[n]["reason"] = "explicit primary" if mid == 0 else "explicit fill"
                    matched.add(n)
        for n in info:
            if n not in matched:
                info[n]["master"] = -1
                info[n]["keep"] = False
                info[n]["reason"] = "not listed in --masters"
    primary, fill_sources = [], []
    for n in sorted(info):
        i = info[n]
        role = ("PRIMARY" if i["keep"]
                else "fill" if i.get("master", -1) >= 1 else "DROP")
        log(f"  {n:24} noise={i['noise']:.4g} master={i.get('master','-')} "
            f"{role} ({i['reason']})")
        if i["keep"]:
            primary.append(n)
        elif i.get("master", -1) >= 1:
            fill_sources.append(n)
    if not primary:
        raise SystemExit("no usable captures in the primary master")
    weights = {n: 1.0 / max(info[n]["noise"], 1e-6) ** 2 for n in info}
    sig = {n: info[n]["noise"] for n in info}
    log(f"[stack] primary master: {len(primary)} captures: {', '.join(primary)}")
    if fill_sources and cross_fill:
        log(f"[stack] cross-master fill from {len(fill_sources)} alt-master "
            f"captures: {', '.join(fill_sources)}")

    writer = StackWriter(outbase, vp, system, git_commit)
    efm_stream = []          # list of averaged int16 EFM waveforms (per field)
    nwritten = 0
    filled_total = 0
    import itertools

    P = dict(primary=primary, fill_sources=fill_sources, cross_fill=cross_fill,
             black=float(vp.get("black16bIre", 0)) or None,
             weights=weights, sig=sig, ra=ra, ca=ca, fh=fh, fw=fw,
             subpixel=subpixel, chroma_align=chroma_align, line_reg=line_reg,
             reg_confidence=reg_confidence, reg_drop_mult=reg_drop_mult)

    def emit(r):
        nonlocal nwritten, filled_total
        if r is None:
            return
        for outu, meta, vbidata, resid_do, efm_avi, audio_n in r["fields"]:
            efm_t = 0
            if efm_avi is not None:
                efm_stream.append(efm_avi); efm_t = len(efm_avi)
            writer.write_field(outu, meta, vbidata, resid_do, efm_t, audio_n)
        if r["audio"] is not None:
            writer.write_audio(r["audio"])
        filled_total += r["filled"]
        nwritten += 1
        if nwritten % 50 == 0:
            log(f"  ... {nwritten} frames")

    all_tbc = all(isinstance(s, TBCFrameSource) for s in sources)
    if jobs and jobs > 1 and all_tbc:
        # frames are independent -> fan the per-frame computation out over a
        # fork pool (sources/P inherited copy-on-write, .tbc via memmap); the
        # parent writes results in key order, so output is byte-identical to
        # the sequential path
        import multiprocessing as mp
        global _WORKER_STATE
        # UNION of pictures, not intersection: a CAV picture readable on any
        # capture is stacked from whichever captures carry it. Intersection
        # silently dropped pictures missing from even one capture, compressing
        # the output timeline (~50 frames / 2s over a Domesday AIV side) and
        # putting burnt-in timecodes out of step.
        keysets = [s.keys() for s in sources]
        shared = sorted(set.union(*keysets))
        n_all = len(set.intersection(*keysets))
        if max_frames:
            shared = shared[:max_frames]
        log(f"[stack] parallel: {jobs} workers over {len(shared)} frames "
            f"({len(shared) - n_all} present on only some captures)")
        _WORKER_STATE = (sources, P)
        try:
            with mp.get_context("fork").Pool(jobs) as pool:
                for r in pool.imap(_frame_by_key, shared, chunksize=CHUNK):
                    emit(r)
        finally:
            _WORKER_STATE = None
    else:
        if jobs and jobs > 1:
            log("[stack] --jobs ignored (needs all-.tbc inputs)")
        for key, fd in itertools.chain(head_buf, merged):
            if max_frames and nwritten >= max_frames:
                break
            emit(_compute_frame(fd, P))

    efm_bytes = None
    efm_merged = False
    src_by_name = {s.name: s for s in sources}
    # EFM stacking -- two complementary methods, run whichever inputs exist:
    #  (1) pre-PLL WAVEFORM averaging: when captures were decoded --tbc_efm
    #      --preEFM their .prefm waveforms are TBC-locked to disc rotation, so
    #      (for CAV-aligned windows) they align and average -> lower noise into
    #      ONE PLL pass -> a cleaner stacked .efm.  Pool primary+fill (the digital
    #      data is master-independent).
    #  (2) SECTOR-level merge of the per-capture decoded .efm -> .data, OR-filling
    #      sectors no single disc read.  (1) lowers the random-noise floor where
    #      every disc is marginal; (2) recovers localized dropouts.
    def _src_paths(getter):
        out = []
        for n in primary + fill_sources:
            s = src_by_name.get(n)
            p = getattr(s, getter)() if s is not None and hasattr(s, getter) else None
            if p:
                out.append((n, p))
        return out

    if efm_stream:
        # legacy in-process per-field path (Frame.efm waveforms), if ever populated
        EFM_PLL = _efm_pll_class()
        tvals = EFM_PLL().process(np.concatenate(efm_stream))
        efm_bytes = np.asarray(tvals).tobytes()
    else:
        prefm = _src_paths("prefm_path")
        if len(prefm) >= 2:
            EFM_PLL = _efm_pll_class()
            streams = [np.fromfile(p, dtype="<i2") for _, p in prefm]
            L = min(len(s) for s in streams)            # common length (aligned window)
            streams = [s[:L].astype(np.float64) for s in streams]
            wts = [weights[n] for n, _ in prefm]
            log(f"[stack] EFM: averaging {len(streams)} TBC-locked pre-PLL "
                f"waveforms ({L} samples), then one PLL pass")
            av = average_efm(streams, wts)
            avi = np.clip(np.round(av), -32768, 32767).astype(np.int16)
            efm_bytes = np.asarray(EFM_PLL().process(avi)).tobytes()

    # sector-level merge (independent of the above; recovers dropouts) -> .data
    efm_paths = [p for _, p in _src_paths("efm_path")]
    if efm_paths:
        log(f"[stack] EFM: sector-merging {len(efm_paths)} captures")
        res = merge_efm_sectors(efm_paths, outbase + ".data",
                                scratch_dir=scratch_dir, log=log)
        if res:
            got, best, nd = res
            efm_merged = True
            log(f"[stack] EFM merge: {got} data bytes recovered from {nd} "
                f"captures (best single: {best}; +{got-best})")
    writer.close(efm_bytes)
    log(f"[stack] wrote {outbase}.tbc (+.tbc.json/.tbc.db), {outbase}.pcm"
        + (f", {outbase}.efm" if efm_bytes else "")
        + (f", {outbase}.data (merged EFM)" if efm_merged else "")
        + f" ({nwritten} frames; {filled_total} px cross-master filled)")
    return info


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def open_source(path, system, seek=None, length=None, extra_options=None,
                cav=True, scratch_dir=None, stream=False, threads=4):
    if path.endswith(".tbc"):
        return TBCFrameSource(path[:-4], cav=cav)
    if path.endswith(".tbc.json"):
        return TBCFrameSource(path[:-9], cav=cav)
    if path.endswith(".ldf") or path.endswith(".lds"):
        if stream:
            from .streamsource import ProcessStreamSource
            return ProcessStreamSource(
                path, system=system, seek=seek, length=length, cav=cav,
                extra_options=extra_options, scratch_dir=scratch_dir,
                threads=threads)
        return LDFFrameSource(path, system=system, seek=seek, length=length,
                              cav=cav, extra_options=extra_options,
                              scratch_dir=scratch_dir)
    # bare base name
    if os.path.exists(path + ".tbc"):
        return TBCFrameSource(path, cav=cav)
    raise SystemExit(f"unrecognised input: {path}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog="ld-disc-stack",
        description="Combine multiple captures of the same disc into one "
                    "improved output (video TBC, audio, EFM).")
    p.add_argument("inputs", nargs="+", help=".ldf captures (decoded live) or "
                   "decoded .tbc files of the same disc")
    p.add_argument("-o", "--output", required=True, help="output base name")
    sysg = p.add_mutually_exclusive_group()
    sysg.add_argument("--PAL", "-p", dest="system", action="store_const",
                      const="PAL", help="PAL source (default)")
    sysg.add_argument("--NTSC", "-n", dest="system", action="store_const",
                      const="NTSC", help="NTSC source")
    p.add_argument("--no-subpixel", action="store_true",
                   help="integer-only frame registration")
    p.add_argument("--no-chroma-align", action="store_true",
                   help="disable per-line chroma subcarrier phase alignment "
                   "(by default chroma is phase-matched across captures so the "
                   "PAL pilot-lock wander does not cancel on combine)")
    p.add_argument("--no-cross-fill", action="store_true",
                   help="disable cross-master dropout fill (by default, after "
                   "stacking the primary master, residual master-level dropouts "
                   "are patched from the registered alt-master stack)")
    p.add_argument("--masters", type=str, default=None,
                   help="explicit master grouping instead of auto-clustering; "
                   "groups separated by ';', captures by ',', first group is the "
                   "primary, e.g. 'ds1,ds4,ds6;ds3,ds5' (tokens match capture names)")
    p.add_argument("--no-reg-confidence", action="store_true",
                   help="disable per-frame inverse-variance weighting + "
                   "registration-confidence drop (by default each field is "
                   "weighted by its residual against the per-frame median, and a "
                   "field that registered to a gross-outlier offset is dropped)")
    p.add_argument("--reg-drop-mult", type=float, default=3.0,
                   help="drop a field whose per-frame registration residual "
                   "exceeds this multiple of the median residual (needs >=3 "
                   "captures; default 3.0)")
    p.add_argument("--line-reg", type=int, default=0, metavar="NBLOCKS",
                   help="after the global field shift, apply a sub-pixel "
                   "horizontal warp per capture to remove residual intra-line "
                   "wow before combining: 1 = one shift per line, N>1 = N blocks "
                   "across each line (within-line warp). 0 (default) = off. "
                   "NOTE: on well-TBC'd same-master captures the residual is only "
                   "~0.05px and the resampling cost slightly outweighs it (combine "
                   "~0.5%% softer); only worth trying on visibly mis-aligned / "
                   "poorly-TBC'd captures.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after writing this many output frames (bounds a "
                   ".tbc stack for testing/preview; .ldf uses --length instead)")
    p.add_argument("--clv", action="store_true",
                   help="align by sequential frame index (CLV / non-CAV)")
    p.add_argument("--sample", type=int, default=24,
                   help="frames sampled for quality/master analysis")
    p.add_argument("--seek", type=int, default=None,
                   help="(.ldf live decode) start at this CAV picture number")
    p.add_argument("--length", type=int, default=None,
                   help="(.ldf live decode) limit to this many frames per input")
    # decode-time options forwarded to the live .ldf decode (must match what a
    # normal ld-decode of these discs would use, since the stack inherits them)
    p.add_argument("--V4300D_coherent_subtract", action="store_true",
                   help="(.ldf) remove the LD-V4300D ~8.5MHz spur by coherent "
                   "subtraction (recommended for Domesday/EFM PAL captures)")
    p.add_argument("--rf_echo", type=str, default=None,
                   help="(.ldf) RF echo cancellation, e.g. 26:0.035,38:0.018")
    p.add_argument("--stream", action="store_true",
                   help="(.ldf) decode every capture concurrently, one process "
                   "each, and stack the frames as they are produced instead of "
                   "decoding each to its own .tbc first.  One file is written "
                   "instead of N+1, and the stacking overlaps the decoding.  "
                   "Note the analysis sample is then the opening window only, "
                   "so captures are clustered by master but never dropped as "
                   "noise outliers (they are weighted instead).")
    p.add_argument("--decode-threads", type=int, default=4,
                   help="(--stream) decoder threads per capture (default 4)")
    p.add_argument("--read-buffer-mb", type=int, default=None,
                   help="(--stream) per-capture .ldf read-ahead buffer in MB.  "
                   "With several captures decoding at once the drive is "
                   "servicing that many interleaved read streams; a large "
                   "buffer turns each into long sequential bursts (e.g. 2048).")
    p.add_argument("--jobs", type=int, default=0,
                   help="parallelise the per-frame stacking over this many "
                   "worker processes (all-.tbc inputs only; output is "
                   "byte-identical to the sequential path). 0 = sequential.")
    p.add_argument("--scratch-dir", type=str, default=None,
                   help="where to put per-capture scratch (window .tbc/.efm, "
                   "merge .bin).  Default: a RAM tmpfs (/dev/shm) if available "
                   "so nothing touches a physical disk; only final results are "
                   "written.  Falls back to the input directory.")
    args = p.parse_args(argv)
    system = args.system or "PAL"
    if args.read_buffer_mb:
        # read by the file reader in each decode process (spawn inherits env)
        os.environ["LDDECODE_READ_BUFFER_MB"] = str(int(args.read_buffer_mb))

    decode_opts = {}
    if system == "PAL" and args.V4300D_coherent_subtract:
        decode_opts["PAL_V4300D_CoherentSubtract"] = True
    if args.rf_echo:
        decode_opts["rf_echo_cancel"] = [
            (float(p.split(":")[0]), float(p.split(":")[1]))
            for p in args.rf_echo.split(",") if ":" in p
        ]

    sources = []
    try:
        # A capture that fails to decode this window (e.g. a badly-degraded disc
        # with an unreadable region) must NOT abort the whole window -- skip it
        # and stack the rest, as long as >=2 usable captures remain.
        for path in args.inputs:
            try:
                sources.append(open_source(
                    path, system, seek=args.seek, length=args.length,
                    extra_options=decode_opts, cav=not args.clv,
                    scratch_dir=args.scratch_dir, stream=args.stream,
                    threads=args.decode_threads))
            except SystemExit as e:
                print("WARNING: skipping capture (decode failed): %s (%s)"
                      % (path, e), file=sys.stderr)
        if len(sources) < 2:
            raise SystemExit("fewer than 2 usable captures for this window")
        masters = ([[t.strip() for t in g.split(",") if t.strip()]
                    for g in args.masters.split(";")] if args.masters else None)
        stack(sources, args.output, system=system,
              subpixel=not args.no_subpixel,
              chroma_align=not args.no_chroma_align,
              cross_fill=not args.no_cross_fill, masters=masters,
              sample=args.sample, scratch_dir=args.scratch_dir,
              max_frames=args.max_frames,
              reg_confidence=not args.no_reg_confidence,
              reg_drop_mult=args.reg_drop_mult, line_reg=args.line_reg,
              jobs=args.jobs, git_commit=_git_commit())
    finally:
        for s in sources:
            s.close()


if __name__ == "__main__":
    main()
