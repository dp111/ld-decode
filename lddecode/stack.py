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

import errno
import json
import os
import sqlite3
import sys
import time
from textwrap import dedent

import numpy as np
import scipy.fft as _sfft


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
        # .tbc.json if it is there, .tbc.db otherwise: the decoder stopped
        # writing the JSON sidecar (bdb9865d) but decodes made either side of
        # that are both still in circulation, so this has to take either.
        j = _tbcmeta().load_decode_metadata(base)
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

    def prefetch(self, keys):
        """Read the fields of `keys` with one large read per contiguous span.

        A stack reads every capture at nearly the same position, so on a
        single hard disk (T:) per-frame reads from 16 workers x 9-12 files
        turn into a seek storm: the 2026-10-12 South stack ran at ~22 MB/s
        with the CPUs 78% idle.  One read per capture per block of frames is
        one seek per ~35 MB instead of one per 0.7 MB."""
        if not hasattr(self, "_idx"):
            self._build_index()
        self._cache = {}
        fis = sorted(self._idx[k] for k in keys if k in self._idx)
        if not fis:
            return
        lo, hi = fis[0] * 2, fis[-1] * 2 + 1
        if hi - lo + 1 > 2 * (2 * len(fis)) + 16:    # scattered: per-field reads
            return
        n = hi - lo + 1
        for attempt in range(8):
            try:
                buf = os.pread(self._tbc_fd, n * self._fbytes, lo * self._fbytes)
                break
            except OSError as e:
                if e.errno != errno.ENOMEM or attempt == 7:
                    raise
                time.sleep(0.5 * 2 ** attempt)
        arr = np.frombuffer(buf, dtype="<u2")
        got = len(arr) // (self._fshape[0] * self._fshape[1])
        arr = arr[: got * self._fshape[0] * self._fshape[1]].reshape(got, *self._fshape)
        self._cache = {lo + i: arr[i] for i in range(got)}

    def drop_prefetch(self):
        self._cache = {}

    def _read_field(self, idx):
        c = getattr(self, "_cache", None)
        if c and idx in c:
            return c[idx]
        # drvfs/9P reads under heavy parallel load occasionally fail with a
        # transient ENOMEM (seen 2026-10-11 at frame ~25k of a 9-capture
        # stack, 22 workers); back off and retry rather than lose the run.
        for attempt in range(8):
            try:
                buf = os.pread(self._tbc_fd, self._fbytes, idx * self._fbytes)
                break
            except OSError as e:
                if e.errno != errno.ENOMEM or attempt == 7:
                    raise
                time.sleep(0.5 * 2 ** attempt)
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
        if not (os.path.exists(base + ".tbc.json")
                or os.path.exists(base + ".tbc.db")):
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
    # the correlation peak sits at img's offset from ref; fourier_shift needs
    # the correction, i.e. minus that (an img = ref shifted by +3 gave +3,
    # and the cold search then centred its grid 6 samples from the answer)
    return -sy, -sx


def integer_shift_luma(ref, img, cut=0.17):
    """Integer (dy, dx) of img vs ref from the luma alone: plain (unwhitened)
    cross-correlation of the two blocks low-passed below cut x fs (0.17 x
    17.7 MHz = 3 MHz).  integer_shift's whitened phase correlation is
    dominated by the colour subcarrier, whose 4-sample period gives false
    peaks a cycle away (Community South ds3: -2.7 instead of +1.3)."""
    def lp(x):
        X = np.fft.rfft(x - x.mean(), axis=1)
        X[:, int(cut * x.shape[1]) + 1:] = 0
        return np.fft.irfft(X, x.shape[1], axis=1)
    a, b = lp(ref), lp(img)
    c = np.real(np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a))))
    p = np.unravel_index(np.argmax(c), c.shape)
    sy = p[0] - (c.shape[0] if p[0] > c.shape[0] // 2 else 0)
    sx = p[1] - (c.shape[1] if p[1] > c.shape[1] // 2 else 0)
    # the correlation peak sits at img's offset from ref; fourier_shift needs
    # the correction, i.e. minus that (an img = ref shifted by +3 gave +3,
    # and the cold search then centred its grid 6 samples from the answer)
    return -sy, -sx


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


def _shift_search(F, H, W, dy, dx, pyc, pxc):
    """float32 counterpart of _apply_shift_spectrum for the shift search."""
    py = pyc.get(dy)
    if py is None:
        py = pyc[dy] = np.exp(-2j * np.pi * np.fft.fftfreq(H) * dy).astype(np.complex64)[:, None]
    px = pxc.get(dx)
    if px is None:
        px = pxc[dx] = np.exp(-2j * np.pi * np.fft.rfftfreq(W) * dx).astype(np.complex64)[None, :]
    return _sfft.irfft2((F * py) * px, s=(H, W))


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


REG_WALK_MAX = 8          # cold registration: re-centred 5x5 grids before giving up
# largest vertical registration shift (field lines) searched; captures are line-locked, so it is ~0
REG_MAX_DY = float(os.environ.get("LDSTACK_REG_MAX_DY", "1.0"))


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
    if best is None and not subpixel:
        iy, ix = integer_shift(ref[ra, ca], fa[ra, ca])
        best = (float(iy), float(ix))
    if subpixel:
        refblk = ref[ra, ca].astype(np.float32)
        H, W = fa.shape
        # The search only ranks candidate shifts against each other, so it
        # runs in float32 on a line zero-padded to a fast FFT length: 1135 is
        # 5 x 227 (prime), the slowest size there is.  12.3 -> 3.6 ms per
        # candidate transform; the active block sits far from the padding.
        # The final resample below stays float64 at the true width.
        WP = _sfft.next_fast_len(W, real=True)
        fpad = np.zeros((H, WP), np.float32)
        fpad[:, :W] = fa
        Ffa = _sfft.rfft2(fpad)         # forward FFT once for the whole search
        pyc, pxc = {}, {}
        cands = {}

        def ev(dy, dx):
            key = (round(dy, 4), round(dx, 4))
            if key not in cands:
                if dy == 0 and dx == 0:
                    s = fpad[ra, ca]
                else:
                    s = _shift_search(Ffa, H, WP, dy, dx, pyc, pxc)[ra, ca]
                cands[key] = float(np.mean((s - refblk) ** 2))
            return cands[key]

        # coarse 5x5 grid (0.25 step) around the integer estimate, then
        # per-axis parabolic interpolation of the RMS surface for the
        # sub-step optimum (~0.03px, vs 0.05px for the old 11x11 fine grid,
        # at ~5x fewer full-field transforms). A verification eval keeps
        # the result no worse than the best grid point.
        def grid(cy, cx, span):
            bestrms, best = None, None
            for dy in cy + np.arange(-span, span + 1e-9, 0.25):
                if abs(dy) > REG_MAX_DY + 1e-9:
                    continue
                for dx in cx + np.arange(-span, span + 1e-9, 0.25):
                    r = ev(dy, dx)
                    if bestrms is None or r < bestrms:
                        bestrms, best = r, (dy, dx)
            # an optimum pinned at the +-REG_MAX_DY limit is not a stale window: only dx can walk on
            edge = ((abs(best[0] - cy) >= span - 1e-9 and abs(best[0]) < REG_MAX_DY - 1e-9)
                    or abs(best[1] - cx) >= span - 1e-9)
            return bestrms, best, edge

        def cold():
            # integer_shift (phase correlation on the composite) is often a
            # sample or two off between masters (Community South: -1 for a
            # master at +0.5, -1/-2 for one at +1.3; the 0.75-sample error
            # survived every later frame through the hints).  The error
            # surface is smooth and single-minimum over +-3 samples, so walk:
            # while the best point is on the 5x5 grid's border, re-centre there.
            cy, cx = (float(v) for v in integer_shift_luma(ref[ra, ca], fa[ra, ca]))
            # captures (and masters) are line-locked: the vertical offset is 0 (+- a fraction).  On content
            # with no vertical structure (colour bars) the correlation is flat in dy and the search used to
            # land anywhere (fq benchmark, PAL bars: alt masters registered at dy -14..+108 lines)
            cy = float(np.clip(cy, -REG_MAX_DY, REG_MAX_DY))
            for _ in range(REG_WALK_MAX):
                bestrms, best, edge = grid(cy, cx, 0.5)
                if not edge:
                    break
                cy, cx = best
            return bestrms, best

        if hint is not None:
            cy, cx = best
            cy = float(np.clip(cy, -REG_MAX_DY, REG_MAX_DY))
            bestrms, best, edge = grid(cy, cx, 0.25)      # warm: 3x3
            if edge:
                # stale hint: the optimum is outside the warm window - cold search
                bestrms, best = cold()
        else:
            bestrms, best = cold()

        def para(fm, f0, fp, h):
            den = fm - 2.0 * f0 + fp
            if den <= 0:
                return 0.0
            return float(np.clip(0.5 * (fm - fp) / den * h, -h, h))

        by, bx = best
        oy = para(ev(by - 0.25, bx), ev(by, bx), ev(by + 0.25, bx), 0.25)
        oy = float(np.clip(by + oy, -REG_MAX_DY, REG_MAX_DY) - by)
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
# residual (every-capture) dropouts are written as runs; runs closer than this many good samples are merged
RESID_DO_GAP = int(os.environ.get("LDSTACK_RESID_DO_GAP", "8"))
# every capture outlier-rejected but most not dropout-flagged: output the median unflagged (0 = flag it)
REJ_KEEP_MEDIAN = int(os.environ.get("LDSTACK_REJ_KEEP_MEDIAN", "1"))
# cross-master fill must sit within K x the captures' noise of a simple vertical
# estimate; 0 disables the check
FILL_SANITY_K = float(os.environ.get("LDSTACK_FILL_SANITY_K", "8.0"))
# Master-defect repair (see master_repair_mask): a defect cut into the
# primary glass master (every primary capture carries it, no dropout detector
# flags it) is replaced from an alt-master capture where that is plausible.
# LDSTACK_REPAIR_K=0 disables.  K: seed threshold in sigma of the
# primary-vs-alt difference; R: plausibility ratio (primary vs alt vertical
# residual, alt vs primary's neighbours, neighbour agreement); MINRUN:
# shortest seed run along the line; SUBBLACK: a seed run must dip below this
# fraction of the black level; KW/GAP: the weaker part of a dashed streak is
# taken within GAP of the seeds on the same line; GROW: margin each side;
# LEVEL_W: box width of the local level match; DUMP: directory for a per-field
# audit trail of every substituted pixel; PRESCREEN(_F): skip registering a
# fill-only alt master when the primary has no candidate at all.
REPAIR_K = float(os.environ.get("LDSTACK_REPAIR_K", "6.0"))
REPAIR_R = float(os.environ.get("LDSTACK_REPAIR_R", "3.0"))
REPAIR_MINRUN = int(os.environ.get("LDSTACK_REPAIR_MINRUN", "6"))
REPAIR_KW = float(os.environ.get("LDSTACK_REPAIR_KW", "2.0"))
REPAIR_GAP = int(os.environ.get("LDSTACK_REPAIR_GAP", "48"))
REPAIR_GROW = int(os.environ.get("LDSTACK_REPAIR_GROW", "4"))
REPAIR_SUBBLACK = float(os.environ.get("LDSTACK_REPAIR_SUBBLACK", "0.5"))
REPAIR_NBR = os.environ.get("LDSTACK_REPAIR_NBR", "1") not in ("0", "", "no")
REPAIR_LEVEL_W = int(os.environ.get("LDSTACK_REPAIR_LEVEL_W", "33"))
REPAIR_DUMP = os.environ.get("LDSTACK_REPAIR_DUMP", "")
REPAIR_PRESCREEN_F = float(os.environ.get("LDSTACK_REPAIR_PRESCREEN_F", "1.5"))
REPAIR_PRESCREEN = os.environ.get("LDSTACK_REPAIR_PRESCREEN", "1") not in ("0", "", "no")
# Short-impulse path of the repair (2026-10-16): 1..IMP_MAXRUN-sample spikes
# cut into the primary master (CommunitySouth picture 3030 bars: ~90-250 per
# frame, in 92-100 % of the primary captures, too short for MINRUN).  Seed:
# the primary deviates from its same-subcarrier-phase horizontal median
# (+-4, +-8 samples) by > IMP_K sigma, the alt does not (< 1/R of it) and the
# alt sits near the primary's own median there.  IMP_K=0 disables.
REPAIR_IMP_K = float(os.environ.get("LDSTACK_REPAIR_IMP_K", "6.0"))
REPAIR_IMP_MAXRUN = int(os.environ.get("LDSTACK_REPAIR_IMP_MAXRUN", "5"))
REPAIR_IMP_GROW = int(os.environ.get("LDSTACK_REPAIR_IMP_GROW", "1"))
# Dip path (2026-10-16): the bars streaks at 3030 are ~5-20 samples long,
# jitter between captures, and in composite show as deep dips at the chroma
# troughs (primary ~0-14000, alt ~21000-32000); the vertical-residual tests
# of the streak path fail on saturated PAL chroma.  Seed: primary below
# DIP_SUB x black, the alt above it by > DIP_D x black and > IMP_K/1.5 sigma.
# Seeds closer than DIP_GAP on a line form one span (<= DIP_MAXLEN), which
# is replaced whole, +-DIP_GROW.  DIP_D=0 disables.
REPAIR_DIP_SUB = float(os.environ.get("LDSTACK_REPAIR_DIP_SUB", "0.9"))
REPAIR_DIP_D = float(os.environ.get("LDSTACK_REPAIR_DIP_D", "0.5"))
REPAIR_DIP_GAP = int(os.environ.get("LDSTACK_REPAIR_DIP_GAP", "6"))
REPAIR_DIP_MAXLEN = int(os.environ.get("LDSTACK_REPAIR_DIP_MAXLEN", "600"))
REPAIR_DIP_GROW = int(os.environ.get("LDSTACK_REPAIR_DIP_GROW", "2"))
# a span also takes, within DIP_GAP, samples where the primary is below the
# alt by > DIP_WEAK sigma (the shallower parts of the same streak)
REPAIR_DIP_WEAK = float(os.environ.get("LDSTACK_REPAIR_DIP_WEAK", "2.0"))
REPAIR_DIP_EDGE = int(os.environ.get("LDSTACK_REPAIR_DIP_EDGE", "24"))
# dip path: the alt sample must be no rougher along its own line than the
# primary (|alt - same-phase median| <= max(ALTK x sigma, the primary's));
# a single-capture alt master's own unflagged dropout (South ds3, pictures
# 16291/16991) otherwise passes "alt brighter" on half its samples in
# saturated colour and is pasted in.  0 disables the test.
REPAIR_DIP_ALTK = float(os.environ.get("LDSTACK_REPAIR_DIP_ALTK", "6.0"))
# every repair path: a substitution must not move the line's local luma
# (8-sample box = two subcarrier cycles, chroma cancels) further from its
# field neighbours' than the primary already is, by more than LUMA_M x sigma.
# A real defect (dropout to sync, dip, spike) pulls the primary's luma away
# and the alt brings it back; an alt master's own unflagged dropout (South
# ds3: picture 18621 line 266 whole stretch near 0, oks True) pushes it away.
# 0 disables it.
REPAIR_LUMA_M = float(os.environ.get("LDSTACK_REPAIR_LUMA_M", "1.0"))
REPAIR_LUMA_W = 8


def _box(x, w=REPAIR_LUMA_W):
    k = np.ones(w) / w
    return np.apply_along_axis(lambda r: np.convolve(r, k, "same"), -1, x)


# per field and alt master: repair only from an alt that lines up with the
# primary.  South picture 29733: ds3 carried a different picture (luma corr
# 0.01) and its repair "removed" every map road; 16374 (ds3, +1.5..4 samples)
# and 50144 (master 1, -1.7) were misregistered, so thin strokes were pasted
# in shifted (ghost lettering).  Box-8 luma (chroma cancels) over the repair
# area: corr at zero shift must be >= ALIGN_C (0.90) and no 1-2 sample / 1 line
# shift may fit better.  Fields with too little picture (std of the luma <
# ALIGN_STD, e.g. black) cannot be judged and are let through.
REPAIR_ALIGN_C = float(os.environ.get("LDSTACK_REPAIR_ALIGN_C", "0.90"))
REPAIR_ALIGN_STD = float(os.environ.get("LDSTACK_REPAIR_ALIGN_STD", "1000"))


REPAIR_ALIGN_BAND = int(os.environ.get("LDSTACK_REPAIR_ALIGN_BAND", "24"))
# a band is only refused when a shifted alt matches clearly better than the
# unshifted one: low correlation alone just means flat picture, different
# master noise, or the primary's own defect in the band (North 12981: run-5
# band gate refused real dropout repairs, corr0 0.30 vs shifted 0.30).
# Refused when shifted - corr0 > max(floor, GAIN * (1 - corr0)): 16272 0.744
# -> 0.890 and 31167 0.799 -> 0.860 still refused.
REPAIR_ALIGN_BAND_GAIN = float(os.environ.get("LDSTACK_REPAIR_ALIGN_BAND_GAIN", "0.1"))
REPAIR_ALIGN_BAND_FLOOR = float(os.environ.get("LDSTACK_REPAIR_ALIGN_BAND_FLOOR", "0.005"))


def alt_aligned(out, alt, ok, ra, ca, minrows=50):
    """(aligned?, corr at zero shift, best corr at a shift, luma std)."""
    if REPAIR_ALIGN_C <= 0:
        return True, 1.0, 0.0, 0.0
    A = _box(out[ra, ca].astype(np.float64))[:, 8:-8]
    B = _box(alt[ra, ca].astype(np.float64))[:, 8:-8]
    good = ok[ra, ca][:, 8:-8]
    rows = good.mean(1) > 0.95
    if rows.sum() < minrows:
        return True, 1.0, 0.0, 0.0
    A = A[rows]; B = B[rows]
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    sd = float(A.std())
    if sd < REPAIR_ALIGN_STD:
        return True, 1.0, 0.0, sd

    def corr(dy, dx):
        a = A[2:-2, 4:-4]
        b = B[2 + dy:B.shape[0] - 2 + dy, 4 + dx:B.shape[1] - 4 + dx]
        return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-9))
    c0 = corr(0, 0)
    cs = max(corr(dy, dx) for dy, dx in ((0, -2), (0, -1), (0, 1), (0, 2), (-1, 0), (1, 0)))
    return (c0 >= REPAIR_ALIGN_C and c0 >= cs), c0, cs, sd


def band_misaligned(out, alt, ok, rm, ca):
    """Bands of REPAIR_ALIGN_BAND lines (holding repairs) where the alt is
    locally misregistered although the field as a whole lines up, e.g. a
    capture whose timebase slips for the first lines of a field (picture
    16272: ds3 4 samples out on lines 22-46, exact below).
    -> list of (row slice, c0, cs, sd)."""
    bad = []
    if REPAIR_ALIGN_BAND <= 0 or REPAIR_ALIGN_C <= 0:
        return bad
    ys = np.where(rm.any(1))[0]
    # bands overlap (step REPAIR_ALIGN_BAND // 3) so a slip that starts part-way
    # down a band is still the majority of some band (South 31167: master 1 is
    # 4 samples out from field line 140 only; 120-144 passed, 136-160 doesn't)
    st = max(1, REPAIR_ALIGN_BAND // 3)
    starts = set()
    for y in ys:
        b = (y // st) * st
        starts.update(range(max(0, b - REPAIR_ALIGN_BAND + st), b + 1, st))
    for b0 in sorted(starts):
        r = slice(max(0, b0 - 4), min(out.shape[0], b0 + REPAIR_ALIGN_BAND + 4))
        al, c0, cs, sd = alt_aligned(out, alt, ok, r, ca, minrows=REPAIR_ALIGN_BAND // 2)
        if not al and cs - c0 > max(REPAIR_ALIGN_BAND_FLOOR, REPAIR_ALIGN_BAND_GAIN * (1 - c0)):
            bad.append((slice(b0, b0 + REPAIR_ALIGN_BAND), c0, cs, sd))
    return bad


def luma_check(out, rm, vals, sig):
    """rm limited to the substitutions that keep (or bring) the local luma
    at least as close to the neighbouring lines as the primary's."""
    if REPAIR_LUMA_M <= 0 or sig <= 0:
        return rm
    trial = out.copy()
    trial[rm] = vals
    ys = np.where(rm.any(1))[0]
    ok = np.zeros_like(rm)
    H = out.shape[0]
    for ln in ys:
        nb = [l for l in (ln - 1, ln + 1) if 0 <= l < H]
        ref = _box(trial[nb]).mean(0)
        lr = np.abs(_box(trial[ln]) - ref)
        lp = np.abs(_box(out[ln]) - ref)
        ok[ln] = lr <= lp + REPAIR_LUMA_M * sig
    return rm & ok


# field lines the repair covers: every line the chroma decoder outputs
# (22..308; 309-310 hold the half-line blanking, which differs between masters), not ra (24..fh-6) - CommunitySouth
# frames 9060/11980/20940 kept primary-master streaks on line 307
REPAIR_ROW0 = int(os.environ.get("LDSTACK_REPAIR_ROW0", "22"))
REPAIR_ROW1 = int(os.environ.get("LDSTACK_REPAIR_ROW1", "309"))


def _vert_est(x):
    """Mean of the lines above and below (same field)."""
    e = np.empty_like(x)
    e[1:-1] = 0.5 * (x[:-2] + x[2:])
    e[0], e[-1] = x[1], x[-2]
    return e


def _phase_med(x):
    """Median of the same-subcarrier-phase samples +-4, +-8 along the line."""
    return np.median(np.stack([np.roll(x, k, -1) for k in (-8, -4, 4, 8)]), 0)


def impulse_mask(p, a, both, d, sig, vp_ok=None):
    """Short primary-only spikes (see REPAIR_IMP_K); p, a, d on the block."""
    ip = p - _phase_med(p)
    ia = a - _phase_med(a)
    thr = REPAIR_IMP_K * sig
    m = (both & (np.abs(ip) > thr) & (np.abs(d) > thr)
         & (np.sign(ip) == np.sign(d)) & (np.abs(ia) * REPAIR_R < np.abs(ip))
         & (np.abs(a - (p - ip)) * REPAIR_R < np.abs(d)))
    if not m.any():
        return m
    keep = np.zeros_like(m)
    for ln in np.where(m.any(1))[0]:
        for x0, x1 in _runs(m[ln]):
            if x1 - x0 <= REPAIR_IMP_MAXRUN:
                keep[ln, x0:x1] = True
    g = keep.copy()
    for k in range(1, REPAIR_IMP_GROW + 1):
        g[:, k:] |= keep[:, :-k]; g[:, :-k] |= keep[:, k:]
    return g & both


def dip_mask(p, a, both, sig, black):
    """Primary-only deep dips (see REPAIR_DIP_D); p, a on the block."""
    seed = (both & (p < REPAIR_DIP_SUB * black)
            & (a - p > max(REPAIR_DIP_D * black, REPAIR_IMP_K / 1.5 * sig)))
    # not at the ends of the active line: the blanking edge differs a
    # little between masters (North: a column of seeds down the left edge)
    seed[:, :REPAIR_DIP_EDGE] = False
    seed[:, -REPAIR_DIP_EDGE:] = False
    m = np.zeros_like(seed)
    w = m.shape[1]
    for ln in np.where(seed.any(1))[0]:
        xs = np.where(seed[ln])[0]
        brk = np.where(np.diff(xs) > REPAIR_DIP_GAP)[0]
        for c0, c1 in zip(np.concatenate(([0], brk + 1)),
                          np.concatenate((brk, [len(xs) - 1]))):
            x0, x1 = xs[c0], xs[c1] + 1
            if x1 - x0 > REPAIR_DIP_MAXLEN:
                continue
            weak = both[ln] & (a[ln] - p[ln] > REPAIR_DIP_WEAK * sig)
            while True:         # grow over weak samples within GAP
                lo = np.where(weak[max(0, x0 - REPAIR_DIP_GAP):x0])[0]
                hi = np.where(weak[x1:x1 + REPAIR_DIP_GAP])[0]
                n0, n1 = x0, x1
                if len(lo): n0 = max(0, x0 - REPAIR_DIP_GAP) + lo[0]
                if len(hi): n1 = x1 + hi[-1] + 1
                if (n0, n1) == (x0, x1) or n1 - n0 > REPAIR_DIP_MAXLEN:
                    break
                x0, x1 = n0, n1
            m[ln, max(0, x0 - REPAIR_DIP_GROW):min(w, x1 + REPAIR_DIP_GROW)] = True
    if REPAIR_DIP_ALTK > 0 and m.any():
        ia = np.abs(a - _phase_med(a))
        ip = np.abs(p - _phase_med(p))
        m &= ia <= np.maximum(REPAIR_DIP_ALTK * sig, ip)
    return m & both


def _runs(row):
    """(start, stop) of the True runs of a 1-D bool array."""
    e = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
    return zip(np.where(e == 1)[0], np.where(e == -1)[0])


def _seed_runs(seed, minrun):
    """Keep only runs of >= minrun samples along each line."""
    keep = np.zeros_like(seed)
    for ln in np.where(seed.any(1))[0]:
        for x0, x1 in _runs(seed[ln]):
            if x1 - x0 >= minrun:
                keep[ln, x0:x1] = True
    return keep


def repair_prescreen(out, valid, ra, ca, thr, black=None):
    """Cheap primary-only test: does any line of the active area deviate from
    its vertical neighbourhood by > thr over >= REPAIR_MINRUN samples (and,
    with the sub-black gate, reach below it inside that run)?  Only then is
    an alt master consulted (registering it costs FFTs).  Every seed run
    master_repair_mask can find lies inside such a run when thr is at most
    K x the primary-vs-alt spread."""
    blk = (ra, ca)
    p = out[blk]
    if black is not None and REPAIR_SUBBLACK > 0:
        low = (p < REPAIR_SUBBLACK * black) & valid[blk]
        if not low.any():
            return False
    rp = np.abs(p - _vert_est(out)[blk]) > thr
    rp &= valid[blk]
    if not rp.any():
        return False
    runs = _seed_runs(rp, REPAIR_MINRUN)
    if black is None or REPAIR_SUBBLACK <= 0:
        return bool(runs.any())
    return bool((runs & low).any())


def repair_values(out, alt, rm, usable):
    """The alt's samples for the pixels in rm, level-matched to the primary.

    register_field removes the masters' global gain/offset, but a residual
    local level difference of a few hundred remains (master nonlinearity,
    ~1 %), which would show as a faint line where one line is swapped.  Add
    the primary-minus-alt difference of the neighbouring lines (+-1, +-2 in
    the field), box-smoothed along the line so chroma and noise cancel."""
    d = np.where(usable & ~rm, out - alt, 0.0)
    c = usable & ~rm
    k = np.ones(REPAIR_LEVEL_W)
    vals = alt[rm].copy()
    ys = np.where(rm.any(1))[0]
    pos = 0
    for ln in ys:
        nb = [l for l in (ln - 2, ln - 1, ln + 1, ln + 2) if 0 <= l < out.shape[0]]
        num = np.convolve(d[nb].sum(0), k, "same")
        den = np.convolve(c[nb].sum(0).astype(np.float64), k, "same")
        corr = np.where(den > 0, num / np.maximum(den, 1), 0.0)
        xs = np.where(rm[ln])[0]
        vals[pos:pos + len(xs)] += corr[xs]
        pos += len(xs)
    return vals


def master_repair_mask(out, valid, alt, alt_ok, ra, ca, black=None):
    """Pixels where the PRIMARY master carries a defect an alt master does not.

    The primary combine cannot see a defect cut into the glass master: every
    primary capture carries it, they agree, and no dropout detector flags it
    (Community North test card, picture 3001 field 2 line 208: a dashed streak
    on all of DS5/8/10/11, absent from DS3 of another master).  Where the
    primary and the registered alt-master estimate disagree grossly, decide
    which side is wrong from its own field: the defective side deviates from
    the mean of its neighbouring lines in the direction of the disagreement,
    the clean side does not.  Genuine fine detail (a one-line-thick horizontal
    stroke, a grating) is on both masters, so they agree there and nothing
    fires; a defect on the alt is the alt's implausibility and is left alone.

    Seeds: |prim - alt| > K sigma, the primary's vertical residual > K sigma
    with the same sign, the alt's < 1/R of it, the alt within 1/R of the
    primary's neighbour mean, the primary's neighbours agreeing with each
    other, in runs >= MINRUN along the line (a master defect is a streak, not
    a speckle) that dip below SUBBLACK x black somewhere.  A line with seeds
    also gives up its weaker (> KW sigma, same sign, same tests) disagreements
    within GAP of the seeds - the gaps of a dashed streak - plus GROW samples
    each side.  sigma = robust (MAD) spread of the primary-vs-alt difference
    over the active area.  Returns (mask, sigma).

    Tuned on the Community trial windows (North/South map, photo, cmap, both
    stacking modes): without the neighbour and sub-black tests, map line work
    and the ends of sloping strokes - real detail the masters render a little
    differently - were taken (up to 2500 px per 60-frame map window)."""
    blk = (ra, ca)
    both = (valid & alt_ok)[blk]
    if both.sum() < 1000:
        return None, 0.0
    p, a = out[blk], alt[blk]
    d = p - a
    dv = d[both]
    sig = 1.4826 * float(np.median(np.abs(dv - np.median(dv))))
    if sig <= 0:
        return None, 0.0
    vp = _vert_est(out)[blk]
    rp = p - vp
    ralt = np.abs(a - _vert_est(alt)[blk])
    same = np.sign(rp) == np.sign(d)
    # the alt must look like the primary WITHOUT the feature: equal to the
    # primary's own neighbouring lines.  Real one-line content that the
    # masters merely render differently (a dashed map line softer on one
    # master) fails this - the alt has the feature too, only weaker.
    flat = np.abs(a - vp) * REPAIR_R < np.abs(d)
    # ...and the primary's own neighbours must agree with each other: a
    # defect sits on otherwise continuous picture, whereas at an edge or the
    # end of a sloping stroke the lines above and below differ, the mean of
    # them is no estimate of the line between, and a sub-sample master
    # difference there is not a defect.
    nb = np.empty_like(out)
    nb[1:-1] = np.abs(out[:-2] - out[2:])
    nb[0], nb[-1] = nb[1], nb[-2]
    if REPAIR_NBR:
        flat &= nb[blk] * REPAIR_R < np.abs(rp)
    seed = (both & same & flat & (np.abs(d) > REPAIR_K * sig)
            & (np.abs(rp) > REPAIR_K * sig) & (ralt * REPAIR_R < np.abs(rp)))
    imp = impulse_mask(p, a, both, d, sig) if REPAIR_IMP_K > 0 else None
    if REPAIR_DIP_D > 0 and black is not None:
        dm = dip_mask(p, a, both, sig, black)
        imp = dm if imp is None else (imp | dm)

    def _only_imp():
        if imp is None or not imp.any():
            return None, sig
        full = np.zeros(out.shape, dtype=bool)
        full[blk] = imp
        return full, sig
    if not seed.any():
        return _only_imp()
    seed = _seed_runs(seed, REPAIR_MINRUN)
    if black is not None and REPAIR_SUBBLACK > 0:
        # a seed run must reach far below black (half way to the sync tip)
        # somewhere: no picture goes there, while a defect - RF lost on the
        # master - rails low.  Ink and edge overshoot stay above it.
        floor = REPAIR_SUBBLACK * black
        for ln in np.where(seed.any(1))[0]:
            for x0, x1 in list(_runs(seed[ln])):
                if p[ln, x0:x1].min() >= floor:
                    seed[ln, x0:x1] = False
    if not seed.any():
        return _only_imp()
    weak = (both & same & flat & (np.abs(d) > REPAIR_KW * sig)
            & (np.abs(rp) > REPAIR_KW * sig) & (ralt * REPAIR_R < np.abs(rp)))
    m = seed.copy()
    w = m.shape[1]
    for ln in np.where(seed.any(1))[0]:
        xs = np.where(seed[ln])[0]
        # seed clusters: seeds closer than GAP belong to one streak, and a
        # streak has one polarity - only weak hits of that sign join it
        brk = np.where(np.diff(xs) > REPAIR_GAP)[0]
        for c0, c1 in zip(np.concatenate(([0], brk + 1)),
                          np.concatenate((brk, [len(xs) - 1]))):
            pol = np.sign(np.median(d[ln, xs[c0:c1 + 1]]))
            lo = max(0, xs[c0] - REPAIR_GAP); hi = min(w, xs[c1] + 1 + REPAIR_GAP)
            m[ln, lo:hi] |= weak[ln, lo:hi] & (np.sign(d[ln, lo:hi]) == pol)
    # GROW samples of margin each side along the line (a streak's soft
    # ends), never onto pixels the alt cannot supply
    g = m.copy()
    for k in range(1, REPAIR_GROW + 1):
        g[:, k:] |= m[:, :-k]; g[:, :-k] |= m[:, k:]
    g &= both
    if imp is not None:
        g |= imp
    full = np.zeros(out.shape, dtype=bool)
    full[blk] = g
    return full, sig


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


def combine_fields(fields, masks, weights, sigma, diff_dod=False, info=None):
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
    if info is not None:
        # every capture rejected by the outlier tests, yet most were not flagged as dropouts: the captures
        # disagree (saturated chroma, ~10-17 sigma between players on Test Card F with 4 captures), they are
        # not all broken.  wsum stays 0 (cross-master fill still gets these pixels: marking them good lost
        # that fill, fq benchmark 0-2 MHz error +14%); the caller only leaves them unflagged if nothing fills
        # them, so the median (out, above) reaches the output instead of an ld-dropout-correct line copy
        unmasked = (~np.stack(masks, 0)).sum(0)
        info["rej_only"] = (wsum == 0) & (2 * unmasked > len(fields))
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


# --------------------------------------------------------------------------- #
#  Cross-master stacking
# --------------------------------------------------------------------------- #
# Much of the noise on a disc is recorded on it: every pressing from one glass
# master carries the same noise, so stacking captures of one master cannot
# remove it.  Measured on the Domesday Community discs (3 masters per side):
# blank-line noise at 5.5-7 MHz is 0.94-0.99 coherent between captures of one
# master and 0.00 between masters, and 40-80% of the noise in every band is
# on-disc.  Each master's disc noise is independent of the others', so
# averaging masters reduces it.  The optimum weights depend on frequency (the
# disc noise steps up ~10x above 5.5 MHz, where a master's capture count no
# longer helps), so masters are blended per horizontal frequency band:
#   1. combine each master's captures as usual -> one estimate per master
#   2. estimate each master estimate's total error variance V_m per band from
#      the differences between master estimates (independent errors, so
#      var(x_m - x_k) = V_m + V_k; three or more masters make it solvable)
#   3. blend with weights 1/V_m per band, on the active picture only (masters
#      place the picture differently relative to sync, so blanking, sync and
#      VBI stay from the primary master).
XM_BAND_EDGES_MHZ = (2.0, 3.5, 5.5)
XM_XOVER_MHZ = 0.3
XM_REJECT_K = 6.0       # a master's pixel this many sigma from the primary's is a defect
XM_HF_BAND_MHZ = (5.5, 7.0)     # disc-noise band used to tell masters apart
XM_HF_CORR_MIN = 0.3            # same master: ~0.6-0.95; different masters: ~0.0
XM_PAD = 64
XM_GROW = 12             # a >K-sigma disagreement is a defect: exclude its tails too (samples)


def xm_band_masks(n, fs_hz, edges_mhz=XM_BAND_EDGES_MHZ, xover_mhz=XM_XOVER_MHZ):
    """Raised-cosine partition of unity over rfft bins of length n: (nb, n//2+1)."""
    f = np.fft.rfftfreq(n, 1.0 / fs_hz) / 1e6
    ups = [0.5 - 0.5 * np.cos(np.pi * np.clip((f - (e - xover_mhz / 2)) / xover_mhz, 0, 1))
           for e in edges_mhz]
    bounds = [np.ones_like(f)] + ups + [np.zeros_like(f)]
    return np.array([bounds[i] - bounds[i + 1] for i in range(len(bounds) - 1)])


def _xm_band_power(x, masks, bad=None):
    """Per-band power of the rows of x (mean-removed), in consistent units.
    Pixels flagged in ``bad`` (dropouts, rejected defects) are replaced by the
    row's median so they neither count nor spread into the band power."""
    if bad is not None and bad.any():
        x = np.where(bad, np.median(x, axis=1, keepdims=True), x)
    X = np.fft.rfft(x - x.mean(1, keepdims=True), axis=1)
    p = (np.abs(X) ** 2).sum(0)
    return (masks * p[None]).sum(1) / x.size


def _grow_x(mask, n):
    """Grow a boolean mask n samples each way along the line."""
    out = mask.copy()
    for s in range(1, n + 1):
        out |= np.roll(mask, s, 1) | np.roll(mask, -s, 1)
    return out


def xm_clusters(frame_dicts, names, ra, ca, fs_hz):
    """Group captures by master from their shared on-disc noise.

    Above 5.5 MHz there is no picture, only noise, and the on-disc part of it
    is identical across captures of one master and unrelated between masters,
    so the correlation of the 5.5-7 MHz band separates masters cleanly where
    the sub-pixel dx clustering can be marginal (Community South ds5/ds11 sit
    at dx +0.06..+0.5, inside the 0.3 tolerance).  Returns a list of groups,
    largest first."""
    width = ca.stop - ca.start
    f = np.fft.rfftfreq(width, 1.0 / fs_hz) / 1e6
    band = (f >= XM_HF_BAND_MHZ[0]) & (f < XM_HF_BAND_MHZ[1])
    num = {}; den = {}
    for fd in frame_dicts:
        present = [n for n in names if n in fd]
        if len(present) < 2:
            continue
        ref = fd[present[0]].f0.astype(np.float64)
        hp = {}
        for n in present:
            r, _ = register_field(fd[n].f0.astype(np.float64), ref, ra, ca, subpixel=True)
            X = np.fft.rfft(r[ra, ca], axis=1)
            X[:, ~band] = 0
            hp[n] = np.fft.irfft(X, width, axis=1)
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                k = (a, b)
                num[k] = num.get(k, 0.0) + float((hp[a] * hp[b]).sum())
                den[k] = den.get(k, 0.0) + float(np.sqrt((hp[a] ** 2).sum() * (hp[b] ** 2).sum()))
    corr = {k: num[k] / den[k] for k in num if den[k] > 0}
    # single-linkage on the correlation graph
    groups = []
    for n in names:
        linked = [g for g in groups
                  if any(corr.get((n, m), corr.get((m, n), 0.0)) >= XM_HF_CORR_MIN for m in g)]
        merged = [n] + [m for g in linked for m in g]
        groups = [g for g in groups if g not in linked] + [merged]
    groups.sort(key=len, reverse=True)
    return groups, corr


def xm_estimate(frame_dicts, groups, ra, ca, fs_hz, weights, log=print):
    """Per-band blend weights for the master estimates (see the section note).

    Noise is measured on the blank vertical-interval lines (field lines 6-13,
    samples 200-1080), where there is no picture: there a difference between
    two captures is noise and nothing else.  Measured on picture content, a
    systematic difference between masters (a slightly different frequency
    response or level - present below ~3.5 MHz on Community South) looks like
    noise and gets blamed on whichever master differs from the others, which
    once handed 90% of the low band to a two-capture master.  Bands where the
    masters' picture differs by much more than their noise explains are
    therefore left to the primary master alone.

    Returns dict with alpha (M, nb) and sig (M,) - the broadband pixel sigma
    of each master estimate's picture difference from the primary's, used to
    reject master-level defects - or None if fewer than two masters overlap.

    xm2 (Oct 2026 study): the error variances are
    measured on the PICTURE area, not on blank lines.  Capture noise on a
    LaserDisc is level-dependent: in the (bright) picture it is 4-8x the
    blank-line power at 0-2 MHz even between two captures of ONE master, so
    the blank-line guard rejected every alt master below 3.5 MHz as
    "systematically different" when nothing but the noise level differed
    (the masters register to a constant sub-sample dx, dy 0, no skew, |H|
    within 2% and phase <1 deg to 4.5 MHz).  The picture-area three-cornered
    hat predicted held-out residual drops within 1-2% on map/photo windows
    and the held-out gains were -17..-27% rms per band for a 2-master blend.
    Dropouts and >K-sigma disagreements (grown by XM_GROW, as in the blend)
    are left out of the estimate so a few gross pixels cannot set a master's
    variance."""
    M = len(groups)
    BL, BX0, BX1 = slice(6, 14), 200, 1080
    bmasks = xm_band_masks(BX1 - BX0, fs_hz)
    amasks = xm_band_masks(ca.stop - ca.start, fs_hz)
    nb = bmasks.shape[0]
    within = {}; pair = {}; apair = {}; pairpix = {}; awithin = {}
    fh, fw = frame_dicts[0][next(iter(frame_dicts[0]))].f0.shape

    def blank(fr):
        x = fr.f0.astype(np.float64)[BL, BX0:BX1]
        return x if np.median(x) - np.median(fr.f0[BL, 10:50]) > 5000 else None

    for fd in frame_dicts:
        ref_name = next((n for n in groups[0] if n in fd), None)
        if ref_name is None:
            continue
        ref = fd[ref_name].f0.astype(np.float64)
        means, ameans, amask = {}, {}, {}
        for m, g in enumerate(groups):
            pres = [n for n in g if n in fd]
            bl = {n: blank(fd[n]) for n in pres}
            pres = [n for n in pres if bl[n] is not None]
            if not pres:
                continue
            w = np.array([weights[n] for n in pres]); w = w / w.sum()
            stackb = np.stack([bl[n] for n in pres], 0)
            means[m] = np.tensordot(w, stackb, 1)
            if len(pres) >= 2:
                for i, n in enumerate(pres):
                    ow = np.delete(w, i)
                    others = np.tensordot(ow / ow.sum(), np.delete(stackb, i, 0), 1)
                    acc = within.setdefault(n, [np.zeros(nb), 0, len(pres)])
                    acc[0] += _xm_band_power(stackb[i] - others, bmasks); acc[1] += 1
            regs = []; dmask = np.zeros((ra.stop - ra.start, ca.stop - ca.start), bool)
            for n in pres:
                dm = dropout_mask((fh, fw), fd[n].do0)
                if n == ref_name:
                    regs.append(ref[ra, ca]); dmask |= dm[ra, ca]; continue
                r, best = register_field(fd[n].f0.astype(np.float64), ref, ra, ca, subpixel=True)
                regs.append(chroma_align_field(r, ref, ca)[ra, ca])
                dmask |= np.roll(dm, (int(round(best[0])), int(round(best[1]))), (0, 1))[ra, ca]
            stacka = np.stack(regs, 0)
            ameans[m] = np.tensordot(w, stacka, 1)
            amask[m] = dmask
            if len(pres) >= 2:          # picture-area player noise per capture (leave-one-out)
                for i, n in enumerate(pres):
                    ow = np.delete(w, i)
                    others = np.tensordot(ow / ow.sum(), np.delete(stacka, i, 0), 1)
                    acc = awithin.setdefault(n, [np.zeros(nb), 0, len(pres)])
                    acc[0] += _xm_band_power(stacka[i] - others, amasks, dmask); acc[1] += 1
        for m in means:
            for k in means:
                if k <= m:
                    continue
                acc = pair.setdefault((m, k), [np.zeros(nb), 0])
                acc[0] += _xm_band_power(means[m] - means[k], bmasks); acc[1] += 1
                if m in ameans and k in ameans:
                    d = ameans[m] - ameans[k]
                    bad = amask[m] | amask[k]
                    # the same K-sigma (grown) rejection the blend applies
                    bad |= _grow_x(np.abs(d) > XM_REJECT_K * np.sqrt(d.var()), XM_GROW)
                    apair.setdefault((m, k), []).append(_xm_band_power(d, amasks, bad))
                    if (~bad).any():
                        pairpix.setdefault((m, k), []).append(float(d[~bad].var()))
    if not any(k[0] == 0 for k in apair):
        return None
    P = {n: s / c * (k - 1) / k for n, (s, c, k) in awithin.items()}
    Q = np.full((M, nb), np.nan)
    for m, g in enumerate(groups):
        ps = [P[n] for n in g if n in P]
        if ps:
            Q[m] = 1.0 / np.sum([1.0 / np.maximum(p, 1e-12) for p in ps], 0)
    Sb = {k: s / c for k, (s, c) in pair.items()}        # blank lines (log only)
    # picture area: median over the sample frames, not the mean - on black
    # frames the Community South alt-master players output an almost
    # noise-free flat level (ds3 vs ds5/ds11 differ by ~0 there), which a
    # pooled mean let hand 92% of the low bands to a 2-capture master
    S = {k: np.median(v, 0) for k, v in apair.items()}
    V = np.full((M, nb), np.nan)
    present = sorted({m for k in S for m in k})
    if len(present) >= 3:
        A = []; y = []
        for (m, k), s in S.items():
            row = np.zeros(M); row[m] = row[k] = 1; A.append(row); y.append(s)
        sol, *_ = np.linalg.lstsq(np.array(A)[:, present], np.array(y), rcond=None)
        for i, m in enumerate(present):
            V[m] = sol[i]
    else:
        # two masters: assume equal on-disc noise; a single-capture master's
        # player noise is taken as the primary's typical capture
        m0, m1 = present
        prim = [P[n] for n in groups[0] if n in P]
        q0 = Q[m0] if np.all(np.isfinite(Q[m0])) else np.zeros(nb)
        q1 = Q[m1] if np.all(np.isfinite(Q[m1])) else (np.median(prim, 0) if prim else q0)
        d = np.maximum((S[(m0, m1)] - q0 - q1) / 2, 0)
        V[m0], V[m1] = q0 + d, q1 + d
    for m in range(M):
        if np.all(np.isfinite(Q[m])):
            V[m] = np.maximum(V[m], Q[m])
        V[m] = np.where(np.isfinite(V[m]), np.maximum(V[m], 1e-12), np.inf)
    alpha = (1.0 / V) / (1.0 / V).sum(0, keepdims=True)
    # picture/blank-line power ratio of each master's difference from the
    # primary: information only (it is 4-8x even within one master, because
    # capture noise is level-dependent - see the docstring)
    ratio = np.zeros((M, nb))
    for m in range(1, M):
        if (0, m) in Sb and (0, m) in S:
            ratio[m] = S[(0, m)] / np.maximum(Sb[(0, m)], 1e-12)
    sysband = np.zeros(nb, bool)
    sig = np.zeros(M)
    for m in range(1, M):
        if (0, m) in pairpix:
            sig[m] = np.sqrt(np.median(pairpix[(0, m)]))
    edges = (0,) + XM_BAND_EDGES_MHZ + ("Nyq",)
    log("[stack] cross-master blend weights per band (MHz), from each master's "
        "error variance measured on the picture (three-cornered hat):")
    log("          " + "".join(f"{str(edges[b]) + '-' + str(edges[b + 1]) + ('*' if sysband[b] else ''):>12}"
                               for b in range(nb)))
    for m, g in enumerate(groups):
        log(f"  master {m} ({len(g)} cap) " + "".join(f"{alpha[m, b]:12.2f}" for b in range(nb))
            + "   V " + " ".join(f"{v:.0f}" for v in V[m])
            + (("   picture/blank " + " ".join(f"{r:.1f}" for r in ratio[m])) if m else "")
            + f"   [{', '.join(g)}]")
    Vb = 1.0 / (1.0 / V).sum(0)
    log("  predicted blend error vs primary alone per band: "
        + " ".join(f"{100 * (np.sqrt(b / a) - 1):+.0f}%" for a, b in zip(V[0], Vb)) + " rms")
    return dict(alpha=alpha, sig=sig)


def xm_blend(ests, alpha, active_rows, active_cols, fs_hz):
    """Blend master estimates [(m, field, ok_mask)] per band on the active area.

    ests[0] must be the primary master.  Outside the active area, and where
    no master is valid, the primary is kept.  Where only some masters are
    valid, those are averaged with their broadband weights."""
    ms = [m for m, _, _ in ests]
    a = alpha[ms]
    a = a / a.sum(0, keepdims=True)
    abb = a.mean(1)                                   # broadband weight per master
    out = ests[0][1].copy()
    blk = (active_rows, active_cols)
    fields = [e[1][blk] for e in ests]
    oks = [e[2][blk] for e in ests]
    wsum = sum(w * ok for w, ok in zip(abb, oks))
    prov = sum(w * ok * f for w, ok, f in zip(abb, oks, fields)) / np.maximum(wsum, 1e-12)
    prov = np.where(wsum > 0, prov, fields[0])
    width = fields[0].shape[1]
    n = width + 2 * XM_PAD
    masks = xm_band_masks(n, fs_hz)
    acc = None
    for i, (f, ok) in enumerate(zip(fields, oks)):
        x = np.where(ok, f, prov)
        x = np.pad(x, ((0, 0), (XM_PAD, XM_PAD)), mode="reflect")
        H = (a[i][:, None] * masks).sum(0)
        X = np.fft.rfft(x, axis=1) * H[None]
        acc = X if acc is None else acc + X
    blended = np.fft.irfft(acc, n, axis=1)[:, XM_PAD:XM_PAD + width]
    allok = np.logical_and.reduce(oks)
    res = np.where(allok, blended, prov)
    out[blk] = res
    anyok = np.zeros(out.shape, dtype=bool)
    anyok[blk] = wsum > 0
    return out, anyok


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
    order) into a stream of (key, {name: Frame}).

    Emits the UNION of the captures' pictures, not the intersection: a picture
    is emitted once every source has either buffered it or moved past it, with
    whatever subset carries it.  Waiting for the point where ALL sources have
    arrived (the largest of their smallest buffered keys) silently discarded
    everything a capture carried below where the latest-starting one began -
    and captures do start at very different pictures, so on a set starting at
    1 and 50 the first 49 pictures were lost outright.  That is the same
    frame loss the indexed path had, and there is no index here to fall back
    on, so it has to be fixed in the merge itself.

    Buffer is bounded by the inter-capture skew: a source that starts early
    has its frames emitted as the frontier reaches them rather than held.
    """
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
        mins = [min(buf[n]) for n in its if buf[n]]
        if len(mins) < sum(not done[n] or bool(buf[n]) for n in its):
            for n in its:
                if not done[n] and not buf[n]:
                    adv(n)
            continue
        # frontier = the smallest key any source is holding.  Every source
        # still behind it is advanced until it has either produced that key or
        # passed it; a source whose head is already beyond it never had it.
        frontier = min(mins)
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


def _tbcmeta():
    """Import tbcmeta whether stack.py is run as a module (lddecode.stack) or
    as a bare script, as _efm_pll_class does for the same reason."""
    try:
        from . import tbcmeta
    except ImportError:
        import tbcmeta
    return tbcmeta


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


def _compute_frame(fd, P, hints=None):
    """Stack one aligned frame dict {name: Frame} -> a writable result dict.

    Pure: `hints` (per (capture, field) registration start points) comes in
    and the shifts actually measured go back out in the result, so nothing is
    carried in module state and a worker's result does not depend on what it
    happened to process before.  Returns None when no primary capture carries
    this frame."""
    primary, fill_sources = P["primary"], P["fill_sources"]
    ra, ca, fh, fw = P["ra"], P["ca"], P["fh"], P["fw"]
    weights, sig = P["weights"], P["sig"]
    out_shifts = {}
    present_primary = [n for n in primary if n in fd]
    present_fill = [n for n in fill_sources if n in fd] if P["cross_fill"] else []
    avail = (list(present_primary), list(present_fill))     # per-frame stats (LDSTACK_FRAME_STATS)
    # cross-master blend only when the primary master carries this frame
    xm = P.get("xm") if present_primary else None
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
    repaired = 0
    extra_reg = 0                   # fields whose alt was registered for repair only
    out_fields = []

    def repair_from(out, valid, alts):
        """Master-defect repair: a defect every primary carries (cut into the
        primary glass master) taken from an alt master that is clean there."""
        n = 0
        rawk = os.environ.get("LDSTACK_REPAIR_RAWDUMP_KEYS", "")
        if REPAIR_DUMP and rawk and str(fd[present_primary[0]].key) in rawk.split(","):
            np.savez_compressed(os.path.join(REPAIR_DUMP, f"raw_{fd[present_primary[0]].key}_{which}.npz"),
                                out=out, valid=valid, alts=np.stack([a for a, _ in alts]),
                                oks=np.stack([o for _, o in alts]), black=P.get("black") or 0,
                                ra=np.array([ra.start, ra.stop]), ca=np.array([ca.start, ca.stop]))
        for ai, (alt, alt_ok) in enumerate(alts):
            rra = slice(REPAIR_ROW0, min(REPAIR_ROW1, out.shape[0] - 1))
            al, c0, cs, sd = alt_aligned(out, alt, valid & alt_ok, rra, ca)
            if REPAIR_DUMP and not al:
                with open(os.path.join(REPAIR_DUMP, "misaligned.txt"), "a") as fh:
                    fh.write(f"{fd[present_primary[0]].key} f{which} a{ai} corr0 {c0:.3f} shifted {cs:.3f} std {sd:.0f}\n")
            if not al:
                continue
            rm, rsig = master_repair_mask(out, valid, alt, alt_ok, rra, ca,
                                          black=P.get("black"))
            if rm is None or not rm.any():
                continue
            for br, bc0, bcs, bsd in band_misaligned(out, alt, valid & alt_ok, rm, ca):
                rm[br] = False
                if REPAIR_DUMP:
                    with open(os.path.join(REPAIR_DUMP, "misaligned.txt"), "a") as fh:
                        fh.write(f"{fd[present_primary[0]].key} f{which} a{ai} lines {br.start}-{br.stop} "
                                 f"corr0 {bc0:.3f} shifted {bcs:.3f} std {bsd:.0f}\n")
            if not rm.any():
                continue
            vals = repair_values(out, alt, rm, valid & alt_ok)
            keep = luma_check(out, rm, vals, rsig)
            if (keep != rm).any():
                vals = vals[keep[rm]]
                rm = keep
            if not rm.any():
                continue
            if REPAIR_DUMP:
                # audit trail: every substituted pixel, old and new
                ys, xs = np.where(rm)
                np.savez_compressed(
                    os.path.join(REPAIR_DUMP, f"{fd[present_primary[0]].key}_{which}_a{ai}.npz"),
                    y=ys, x=xs, prim=out[rm], alt=alt[rm], sig=rsig, new=vals)
            out = out.copy()
            out[rm] = vals
            n += int(rm.sum())
            valid = valid & ~rm     # a second alt master: only the rest
        return out, n

    for fidx, which in enumerate(("f0", "f1")):
        ref_field = getattr(ref_frame, which).astype(np.float64)
        do_attr = "do0" if which == "f0" else "do1"

        def combine_set(names, diff_dod=False, info=None):
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
                        hint=(hints or {}).get(hk) if WARM else None)
                    out_shifts[hk] = shift
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
                                  diff_dod=diff_dod, info=info)

        # dropout-flagged pixels are never used, even when every capture
        # flagged them (user instruction): leave them for the dropout
        # corrector rather than trusting flagged samples
        prim_info = {} if (REJ_KEEP_MEDIAN and len(present_primary) >= 3) else None
        out, wsum = combine_set(present_primary, diff_dod=DIFF_DOD, info=prim_info)
        # ---- cross-master blend: average the masters' estimates per band
        blended = False
        xm_alts = []                # (estimate, ok) per alt master, for repair
        fill_cache = None
        if xm is not None:
            alt_sets = []
            for m, names in enumerate(P["xm_groups"][1:], 1):
                pres = [n for n in names if n in fd]
                if not pres:
                    continue
                fo, fws = combine_set(pres)
                alt_sets.append((m, fo, fws > 0))
                xm_alts.append((fo, fws > 0))
            # repair primary-master defects BEFORE the blend: blended, a defect
            # is diluted by the alt and no longer passes the repair's tests
            # (North test card frames 2996-2999: streak halved, not repaired)
            if REPAIR_K > 0 and xm_alts:
                out, n = repair_from(out, wsum > 0, xm_alts)
                repaired += n
            ests = [(0, out, wsum > 0)]
            for m, fo, ok in alt_sets:
                # a defect on an alt master (often a single capture, so nothing
                # inside its own combine can reject it) must not reach the output,
                # nor its tails (XM_GROW)
                if xm["sig"][m] > 0:
                    dis = (wsum > 0) & ok & (np.abs(fo - out) > XM_REJECT_K * xm["sig"][m])
                    ok = ok & ~_grow_x(dis, XM_GROW)
                ests.append((m, fo, ok))
            if len(ests) > 1:
                out, anyok = xm_blend(ests, xm["alpha"], ra, ca, P["fs_hz"])
                wsum = np.where(anyok, np.maximum(wsum, 1e-6), wsum)
                blended = True
        allbad = (wsum == 0)            # dropout present in EVERY primary disc
        # ---- cross-master fill: patch master-level dropouts from alt master
        if present_fill and allbad.any() and not blended:
            fout, fws = combine_set(present_fill)
            fill_cache = (fout, fws)
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
        # ---- master-defect repair (no cross-master blend: from the fill captures)
        if REPAIR_K > 0 and not xm_alts and present_fill:
            valid = wsum > 0
            go = True
            if fill_cache is None and REPAIR_PRESCREEN:
                # the alt is not registered yet: only pay for it when the
                # primary has a candidate streak at all.  The seed threshold
                # is K x the primary-vs-alt spread, measured at 2.0-13.7x
                # the noise of the best primary capture (median 5.9x, 560
                # trial fields): F = 1.5 keeps a margin below that.
                thr = REPAIR_K * REPAIR_PRESCREEN_F * min(sig[n] for n in present_primary)
                go = repair_prescreen(out, valid, ra, ca, thr, black=P.get("black"))
            if go:
                if fill_cache is None:
                    extra_reg += 1
                    before = set(out_shifts)
                    fill_cache = combine_set(present_fill)
                    # the production path did not register the fill
                    # captures on this frame: keep their hints as they
                    # were so later frames register identically
                    for k in set(out_shifts) - before:
                        del out_shifts[k]
                out, n = repair_from(out, valid, [(fill_cache[0], fill_cache[1] > 0)])
                repaired += n
        outu = np.clip(np.round(out), 0, 65535).astype(np.uint16)
        meta = ref_frame.meta0 if which == "f0" else ref_frame.meta1
        resid_do = []                   # still bad after cross-fill
        if prim_info and "rej_only" in prim_info:
            allbad = allbad & ~prim_info["rej_only"]      # disagreement, not dropout: keep the median
        if allbad.any():
            # one entry per RUN of bad samples (gaps <= RESID_DO_GAP merged), not one min..max span per
            # line: two bad pixels at x=200 and x=1000 used to flag the 800 good samples between them,
            # which ld-dropout-correct then replaced from another line (EcoDisc_S2: 45M samples flagged)
            for ln in np.where(allbad.any(1))[0]:
                xs = np.where(allbad[ln])[0]
                brk = np.where(np.diff(xs) > RESID_DO_GAP + 1)[0]
                for a, b in zip(np.r_[0, brk + 1], np.r_[brk, len(xs) - 1]):
                    resid_do.append((int(ln), int(xs[a]), int(xs[b]) + 1))
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
    fstats = {"avail": avail[0], "avail_alt": avail[1], "used": list(present_primary),
              "used_alt": list(present_fill),
              "resid_do": [sum(b - a for _, a, b in f[3]) for f in out_fields]}
    return {"fields": out_fields, "audio": audio_out, "filled": filled,
            "repaired": repaired, "extra_reg": extra_reg, "shifts": out_shifts,
            "fstats": fstats}


_WORKER_STATE = None            # (sources, P) inherited by forked pool workers

# How many frames back a registration hint comes from.
#
# register_field warm-starts its sub-pixel search from this capture's shift on
# a nearby frame, which is worth ~9 field FFTs.  Taking it from the frame
# immediately before makes the combine order-dependent - what a worker
# processed previously changes where its search starts - so a stack came out
# differently at different --jobs, which is no way to run a preservation
# pipeline.  Taking it from a FIXED number of frames back instead makes the
# hint a function of the frame's position alone, so every path and every
# worker count agree, and it also sets how many frames may be in flight.
#
# Inter-capture offsets drift slowly, so a hint this stale is still inside the
# warm window; when it is not, register_field's border test notices and falls
# back to a cold search, which costs speed and never accuracy.
HINT_LAG = int(os.environ.get("LDSTACK_HINT_LAG", "32"))


def _frame_by_key(args):
    key, hints = args
    sources, P = _WORKER_STATE
    fd = {s.name: s.get(key) for s in sources if key in s.keys()}
    return _compute_frame(fd, P, hints)


# Frames per work item on the indexed (.tbc) path.  A block is read with one
# large read per capture (TBCFrameSource.prefetch) and stacked in order, each
# frame's registration warm-started from the previous frame of its block and
# the first frame of a block starting cold.  Blocks are fixed slices of the
# picture list, so the output does not depend on --jobs.
BLOCK = int(os.environ.get("LDSTACK_BLOCK", "24"))


def _frame_block(keys):
    sources, P = _WORKER_STATE
    for s in sources:
        s.prefetch(keys)
    out, hints = [], None
    try:
        for key in keys:
            fd = {s.name: s.get(key) for s in sources if key in s.keys()}
            r = _compute_frame(fd, P, hints)
            if r:
                hints = r["shifts"]
            out.append(r)
    finally:
        for s in sources:
            s.drop_prefetch()
    return out


def _frame_from_fd(args):
    """Stack one frame the parent has already gathered.

    The indexed path lets each worker fetch its own frame by key, but a
    streaming source holds its frames in the parent's memory and cannot be
    asked twice, so here the frame travels to the worker instead.
    """
    fd, hints = args
    _, P = _WORKER_STATE
    return _compute_frame(fd, P, hints)


HEAD_WINDOW_CAP = 512


def head_window(merged, nsources, aw, cap=HEAD_WINDOW_CAP):
    """Pull the opening frames of a lockstep merge for the quality / master
    analysis.  Returns ([(key, fd), ...], n_full).

    lockstep() emits the UNION of the captures' pictures, and captures start
    at very different pictures (1 and 50, or 222 and 385, in this collection),
    so the first frames out of it are typically carried by the earliest
    starting capture alone.  analyse() registers each capture against a
    reference, which needs frames carried by at least two, and it should see
    every capture, so the window runs until ``aw`` frames carried by ALL
    ``nsources`` have arrived.  ``cap`` bounds the buffer (frames are held in
    memory until the analysis is done) for a set whose captures never all
    overlap; the window then holds whatever it has and the sample is drawn
    from the best-covered frames in it.
    """
    head_buf = []
    nfull = 0
    for key, fd in merged:
        head_buf.append((key, fd))
        if len(fd) >= nsources:
            nfull += 1
        if nfull >= aw or len(head_buf) >= cap:
            break
    return head_buf, nfull


def head_sample(head_buf, sample):
    """Evenly spaced analysis sample from the head window, taken from the
    frames carried by the most captures: a frame carried by one capture alone
    can be neither registered nor scored."""
    if not head_buf:
        return []
    best = max(len(fd) for _, fd in head_buf)
    pool = [fd for _, fd in head_buf if len(fd) == best]
    step = max(1, len(pool) // sample)
    return pool[::step][:sample]


def stack(sources, outbase, system="PAL", subpixel=True, chroma_align=True,
          cross_fill=True, masters=None, sample=24, analysis_window=None,
          max_frames=None,
          scratch_dir=None, git_commit="", reg_confidence=True, reg_drop_mult=3.0,
          line_reg=0, jobs=0, cross_master=False, log=print):
    aw = analysis_window or max(sample * 2, 40)
    merged = lockstep(sources)

    # ---- buffer an initial window for the quality / master analysis ----
    # (live .ldf sources only know their geometry once decoding has started, so
    # pull frames first, then read videoParameters)
    head_buf, nfull = head_window(merged, len(sources), aw)
    if not head_buf:
        raise SystemExit("no frames shared across all captures")
    if nfull < aw:
        log(f"[stack] head window: only {nfull} of the first {len(head_buf)} "
            f"frames are carried by every capture (captures start at "
            f"different pictures); analysing those")
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
        asample = head_sample(head_buf, sample)
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
    fs_hz = float(vp.get("sampleRate") or 4 * 4433618.75)
    if cross_master and not masters:
        # regroup by shared on-disc noise (see xm_clusters) and judge noise
        # outliers against each capture's own master: an alt-master capture's
        # global noise figure is inflated by its master's disc noise
        cand = [n for n in sorted(info) if info[n]["reason"] != "no usable frames"]
        groups, corr = xm_clusters(asample, cand, ra, ca, fs_hz)
        for mid, g in enumerate(groups):
            med = float(np.median([info[n]["noise"] for n in g]))
            for n in g:
                info[n]["master"] = mid
                if len(g) >= 3 and indexed and info[n]["noise"] > 4.0 * med:
                    info[n]["keep"], info[n]["reason"] = False, "noise outlier (within master)"
                else:
                    info[n]["keep"] = mid == 0
                    info[n]["reason"] = ("disc-noise cluster" if mid == 0
                                         else "alt master (disc-noise cluster)")
        log("[stack] masters by shared 5.5-7 MHz disc noise: "
            + " | ".join(",".join(g) for g in groups))
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
    xm, xm_groups = None, None
    if cross_master and fill_sources:
        nmaster = max(info[n]["master"] for n in fill_sources)
        xm_groups = [primary] + [[n for n in fill_sources if info[n]["master"] == m]
                                 for m in range(1, nmaster + 1)]
        xm_groups = [g for g in xm_groups if g]
        xm = xm_estimate(asample, xm_groups, ra, ca, fs_hz, weights, log=log)
        if xm is None:
            log("[stack] cross-master: masters never overlap in the sample - fill only")

    writer = StackWriter(outbase, vp, system, git_commit)
    # LDSTACK_FRAME_STATS=<file>: one line per output frame (does not change the stack):
    # frame, picture, captures available / used (primary master), alt captures used, pixels
    # cross-filled, pixels repaired, samples still flagged dropout, used names, alt names
    fstats_fh = None
    if os.environ.get("LDSTACK_FRAME_STATS"):
        fstats_fh = open(os.environ["LDSTACK_FRAME_STATS"], "w")
        fstats_fh.write("frame\tpicture\tavail\tused\tused_alt\tfilled\trepaired\tresid_do\tused_names\talt_names\n")
    efm_stream = []          # list of averaged int16 EFM waveforms (per field)
    nwritten = 0
    filled_total = 0
    repaired_total = 0
    extra_reg_total = 0
    import itertools

    P = dict(primary=primary, fill_sources=fill_sources, cross_fill=cross_fill,
             black=float(vp.get("black16bIre", 0)) or None,
             weights=weights, sig=sig, ra=ra, ca=ca, fh=fh, fw=fw,
             subpixel=subpixel, chroma_align=chroma_align, line_reg=line_reg,
             reg_confidence=reg_confidence, reg_drop_mult=reg_drop_mult,
             xm=xm, xm_groups=xm_groups, fs_hz=fs_hz)

    def emit(r):
        nonlocal nwritten, filled_total, repaired_total, extra_reg_total
        if r is None:
            return
        for outu, meta, vbidata, resid_do, efm_avi, audio_n in r["fields"]:
            efm_t = 0
            if efm_avi is not None:
                efm_stream.append(efm_avi); efm_t = len(efm_avi)
            writer.write_field(outu, meta, vbidata, resid_do, efm_t, audio_n)
        if r["audio"] is not None:
            writer.write_audio(r["audio"])
        if fstats_fh is not None:
            fs = r.get("fstats") or {}
            vb = [f[2] for f in r["fields"]]
            pic = next((p for p in (cav_picture(v) for v in vb) if p is not None), None)
            if pic is None and vb:
                pic = clv_frame([x for v in vb for x in (v or [])])
            nm = lambda ns: ",".join(os.path.basename(str(n)).replace(".tbc", "") for n in ns)
            fstats_fh.write("%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n" % (
                nwritten, "" if pic is None else pic, len(fs.get("avail", [])),
                len(fs.get("used", [])), len(fs.get("used_alt", [])), r["filled"], r.get("repaired", 0),
                sum(fs.get("resid_do", [])), nm(fs.get("used", [])), nm(fs.get("used_alt", []))))
        filled_total += r["filled"]
        repaired_total += r.get("repaired", 0)
        extra_reg_total += r.get("extra_reg", 0)
        nwritten += 1
        if nwritten % 50 == 0:
            log(f"  ... {nwritten} frames")

    all_tbc = all(isinstance(s, TBCFrameSource) for s in sources)

    def _run(work, submit, depth):
        """Drive the combine, retiring results in order at a fixed lag.

        One shape for all three paths.  `work` yields the per-frame argument,
        `submit(arg, hints)` returns something with .result(), and results are
        retired `depth` frames behind submission - so the hint handed to frame
        i is always the shift measured on frame i-depth, whatever is doing the
        computing.  That is what makes the output independent of --jobs.
        """
        import collections
        inflight = collections.deque()
        hints = {}
        nsub = 0
        for arg in work:
            # bounded on submissions, not on frames written: results trail
            # submission by `depth`, so counting writes would overshoot
            if max_frames and nsub >= max_frames:
                break
            nsub += 1
            inflight.append(submit(arg, hints))
            if len(inflight) >= depth:
                r = inflight.popleft().result()
                hints = r["shifts"] if r else hints
                emit(r)
        while inflight:
            r = inflight.popleft().result()
            hints = r["shifts"] if r else hints
            emit(r)

    class _Now:
        """A completed 'future' for the sequential path."""
        __slots__ = ("_v",)

        def __init__(self, v):
            self._v = v

        def result(self):
            return self._v

    if all_tbc:
        # UNION of pictures, not intersection, and not lockstep's emission
        # either: a picture readable on any capture is stacked from whichever
        # captures carry it.  Intersection silently dropped pictures missing
        # from even one capture, compressing the output timeline (~50 frames /
        # 2s over a Domesday AIV side) and putting burnt-in timecodes out of
        # step; lockstep drops any key sitting behind another capture's
        # position, which on BGB_S1 lost the first 49 pictures outright.  An
        # indexed source can be asked for any key, so ask for all of them.
        keysets = [s.keys() for s in sources]
        shared = sorted(set.union(*keysets))
        if max_frames:
            shared = shared[:max_frames]
        # counted over the frames actually being stacked, so --max-frames does
        # not make it negative
        everywhere = set.intersection(*keysets)
        n_partial = sum(1 for k in shared if k not in everywhere)

        blocks = [shared[i:i + BLOCK] for i in range(0, len(shared), BLOCK)]
        global _WORKER_STATE
        _WORKER_STATE = (sources, P)

        def _retire(results):
            for r in results:
                emit(r)

        try:
            if jobs and jobs > 1:
                # frames are independent -> fan blocks out over a fork pool
                # (sources/P inherited copy-on-write); the parent writes the
                # results in key order
                import collections
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor
                log(f"[stack] parallel: {jobs} workers over {len(shared)} frames "
                    f"in blocks of {BLOCK} ({n_partial} present on only some captures)")
                with ProcessPoolExecutor(
                        max_workers=jobs,
                        mp_context=mp.get_context("fork")) as ex:
                    inflight = collections.deque()
                    for b in blocks:
                        inflight.append(ex.submit(_frame_block, b))
                        if len(inflight) >= 2 * jobs:
                            _retire(inflight.popleft().result())
                    while inflight:
                        _retire(inflight.popleft().result())
            else:
                log(f"[stack] sequential over {len(shared)} frames in blocks of "
                    f"{BLOCK} ({n_partial} present on only some captures)")
                for b in blocks:
                    _retire(_frame_block(b))
        finally:
            _WORKER_STATE = None
    else:
        # Streaming sources cannot be asked for a key twice, so the frames
        # arrive in lockstep order and are stacked as they come.
        def _fds():
            for _key, fd in itertools.chain(head_buf, merged):
                yield fd

        if jobs and jobs > 1:
            # the frames live in this process, so they are shipped to the
            # workers rather than re-read.  Worth the pickling -- measured on
            # three captures the per-frame combine is 1.60s sequential against
            # 0.44s at --jobs 22, and concurrent decodes feed lockstep faster
            # than 0.62 fps, so a sequential combine would make the stack the
            # bottleneck and waste the I/O that streaming saves.
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            log(f"[stack] parallel: {jobs} workers, streaming frames "
                f"({HINT_LAG} in flight)")
            _WORKER_STATE = (None, P)
            try:
                with ProcessPoolExecutor(
                        max_workers=jobs,
                        mp_context=mp.get_context("fork")) as ex:
                    _run(_fds(),
                         lambda fd, h: ex.submit(_frame_from_fd, (fd, h)),
                         HINT_LAG)
            finally:
                _WORKER_STATE = None
        else:
            _run(_fds(), lambda fd, h: _Now(_compute_frame(fd, P, h)),
                 HINT_LAG)

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
    if fstats_fh is not None:
        fstats_fh.close()
    log(f"[stack] wrote {outbase}.tbc (+.tbc.json/.tbc.db), {outbase}.pcm"
        + (f", {outbase}.efm" if efm_bytes else "")
        + (f", {outbase}.data (merged EFM)" if efm_merged else "")
        + f" ({nwritten} frames; {filled_total} px cross-master filled; "
        f"{repaired_total} px primary-master defects repaired from an alt master; "
        f"{extra_reg_total} fields registered an alt master for the repair check)")
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
    p.add_argument("--cross_master", action="store_true",
                   help="blend ALL masters' captures, per frequency band, instead "
                   "of using alt masters only to fill dropouts: on-disc noise is "
                   "per glass master, so averaging masters reduces it.  Masters "
                   "are grouped by their shared 5.5-7 MHz disc noise")
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
                   "worker processes. 0 = sequential.  With .tbc inputs each "
                   "worker fetches its own frames; with --stream they are "
                   "shipped from this process, which is what keeps the "
                   "combine from becoming the bottleneck.  NOTE: registration "
                   "warm-starts from the previous frame, so a worker's hint "
                   "history depends on how frames were shared out -- output "
                   "is reproducible at a given worker count, but differs "
                   "below a quarter pixel between different counts.")
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
              jobs=args.jobs, git_commit=_git_commit(),
              cross_master=args.cross_master)
    finally:
        for s in sources:
            s.close()


if __name__ == "__main__":
    main()
