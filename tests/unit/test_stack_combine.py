"""Combine, registration and output conventions of the multi-capture stacker.

The dropout tests run whole stacks of small synthetic captures written as
.tbc/.tbc.json/.pcm, so the 1-based fieldLine convention is checked end to end:
ld-decode writes line + 1, ld-dropout-correct and newdecode read row
fieldLine - 1, and the stacker must do both the same way.
"""

import json
import os

import numpy as np
import pytest

from lddecode import stack as S

pytestmark = pytest.mark.unit

FH, FW = 64, 256
AVS, AVE = 48, 240
BLACK, WHITE = 16000, 54000
SPF = 1764                              # audio samples per frame


# --------------------------------------------------------------------------- #
#  Synthetic captures
# --------------------------------------------------------------------------- #
def _truth(nframes, seed=0, texture=1500.0):
    """Fields with sync (0-20), porch, and a smooth picture with detail."""
    rng = np.random.default_rng(seed)
    f = np.full((nframes * 2, FH, FW), float(BLACK))
    f[:, :, :20] = 1000.0
    x = np.arange(FW)
    for k in range(nframes * 2):
        pic = 26000 + 9000 * np.sin(2 * np.pi * (x / 37.0 + k * 0.01))[None, :] \
            + 6000 * np.cos(2 * np.pi * np.arange(FH) / 23.0)[:, None]
        pic = pic + rng.normal(0, texture, (FH, FW))
        f[k, 22:FH - 2, AVS:AVE] = pic[22:FH - 2, AVS:AVE]
    return f


def _write_capture(base, fields, pictures, dropouts=None, audio=None, level=(1.0, 0.0)):
    """dropouts: {field index: [(row, x0, x1)]} - written to the TBC at the
    0-based row and recorded 1-based, as ld-decode does."""
    g, o = level
    data = g * fields + o
    recs = {}
    for fi, lst in (dropouts or {}).items():
        for row, x0, x1 in lst:
            data[fi, row, x0:x1] = 200.0          # an RF dropout rails low
            recs.setdefault(fi, []).append((row + 1, x0, x1))
    np.clip(np.round(data), 0, 65535).astype("<u2").tofile(base + ".tbc")
    flds = []
    for fi in range(len(fields)):
        pic = pictures[fi // 2]
        code = 0xF00000 | int("%05d" % pic, 16)
        fj = {"isFirstField": fi % 2 == 0, "seqNo": fi + 1, "fieldPhaseID": fi % 8 + 1,
              "syncConf": 100, "medianBurstIRE": 20, "diskLoc": 0,
              "audioSamples": SPF if fi % 2 == 0 else 0,
              "vbi": {"vbiData": [0, code, code] if fi % 2 == 0 else [0, 0, 0]}}
        if fi in recs:
            fj["dropOuts"] = {"fieldLine": [r[0] for r in recs[fi]],
                              "startx": [r[1] for r in recs[fi]],
                              "endx": [r[2] for r in recs[fi]]}
        flds.append(fj)
    vp = {"system": "PAL", "isSourcePal": True, "fieldWidth": FW, "fieldHeight": FH,
          "activeVideoStart": AVS, "activeVideoEnd": AVE, "sampleRate": 17734475.0,
          "black16bIre": BLACK, "white16bIre": WHITE, "numberOfSequentialFields": len(fields),
          "audioSampleRate": 44100}
    json.dump({"videoParameters": vp, "fields": flds}, open(base + ".tbc.json", "w"))
    if audio is None:
        audio = np.zeros((len(fields) // 2 * SPF, 2))
    np.clip(np.round(audio), -32768, 32767).astype("<i2").tofile(base + ".pcm")


def _read_stack(base):
    j = json.load(open(base + ".tbc.json"))
    t = np.fromfile(base + ".tbc", "<u2").reshape(-1, FH, FW).astype(float)
    pcm = np.fromfile(base + ".pcm", "<i2").reshape(-1, 2)
    return j, t, pcm


def _stack(tmp_path, bases, **kw):
    srcs = [S.TBCFrameSource(b) for b in bases]
    out = str(tmp_path / "stk")
    try:
        S.stack(srcs, out, sample=4, log=lambda *a: None, **kw)
    finally:
        for s in srcs:
            s.close()
    return _read_stack(out)


def _captures(tmp_path, nframes=6, ncap=4, dropouts=None, pictures=None, noise=150.0,
              levels=None, seed=1):
    truth = _truth(nframes)
    pictures = pictures or list(range(100, 100 + nframes))
    rng = np.random.default_rng(seed)
    bases = []
    for c in range(ncap):
        b = str(tmp_path / f"cap{c}")
        f = truth + rng.normal(0, noise, truth.shape)
        _write_capture(b, f, pictures, (dropouts or {}).get(c), level=(levels or {}).get(c, (1, 0)))
        bases.append(b)
    return truth, bases


def _do_rows(j, fi):
    d = j["fields"][fi].get("dropOuts") or {}
    return list(zip(d.get("fieldLine", []), d.get("startx", []), d.get("endx", [])))


# --------------------------------------------------------------------------- #
#  Dropout conventions
# --------------------------------------------------------------------------- #
def test_dropout_mask_is_one_based():
    m = S.dropout_mask((FH, FW), [(1, 10, 20), (FH, 0, 5), (0, 0, 5), (FH + 1, 0, 5)])
    assert m[0, 10:20].all() and m[0].sum() == 10
    assert m[FH - 1, :5].all()
    assert m.sum() == 15                # line 0 and line FH+1 are out of range


def test_single_capture_dropout_is_removed(tmp_path):
    """A dropout on one capture is masked at its real row: the output there
    matches the others, and nothing is left flagged."""
    row, x0, x1 = 40, 100, 160
    truth, bases = _captures(tmp_path, dropouts={1: {4: [(row, x0, x1)]}})
    j, t, _ = _stack(tmp_path, bases)
    err = np.abs(t[4, row, x0:x1] - truth[4, row, x0:x1])
    assert err.max() < 1500, err.max()          # the 200-level rail is gone
    assert _do_rows(j, 4) == []


def test_dropout_on_every_capture_is_flagged_on_its_row(tmp_path):
    """A master-level dropout (every capture) stays flagged, at the right line."""
    row, x0, x1 = 30, 80, 120
    d = {c: {2: [(row, x0, x1)]} for c in range(3)}
    _, bases = _captures(tmp_path, ncap=3, dropouts=d)
    j, t, _ = _stack(tmp_path, bases)
    assert _do_rows(j, 2) == [(row + 1, x0, x1)]
    # its value is concealed from the lines around it (the rail is not written)
    assert t[2, row, x0:x1].min() > 1000
    assert np.abs(t[2, row, x0:x1] - 0.5 * (t[2, row - 1, x0:x1] + t[2, row + 1, x0:x1])).max() < 1


def test_conceal_vertical_uses_nearest_good_lines():
    out = np.arange(10, dtype=np.float64)[:, None] * np.ones((1, 4)) * 100
    bad = np.zeros_like(out, bool); bad[4:6, 1] = True
    o = S.conceal_vertical(out, bad)
    assert o[4, 1] == o[5, 1] == 450.0          # mean of lines 3 and 6
    assert (o[~bad] == out[~bad]).all()


def test_masked_median_ignores_masked_entries():
    st = np.array([[[0.0]], [[10.0]], [[11.0]], [[12.0]]])
    good = np.array([[[False]], [[True]], [[True]], [[True]]])
    assert S.masked_median(st, good)[0, 0] == 11.0
    assert S.masked_median(st, np.zeros_like(good))[0, 0] == 10.5     # none good: all


def test_one_masked_capture_does_not_reject_the_rest():
    """With one capture masked the median was NaN and fell back to the mean,
    dropout included; every unmasked capture then failed the 5-sigma test."""
    rng = np.random.default_rng(0)
    base = rng.normal(20000, 50, (8, 8))
    fields = [base + rng.normal(0, 50, base.shape) for _ in range(4)]
    fields[0][3, 2:6] = 200.0
    masks = [np.zeros(base.shape, bool) for _ in range(4)]
    masks[0][3, 2:6] = True
    out, wsum = S.combine_fields(fields, masks, [1.0] * 4, [50.0] * 4)
    assert np.all(wsum[3, 2:6] == 3.0)
    assert np.abs(out[3, 2:6] - base[3, 2:6]).max() < 200


# --------------------------------------------------------------------------- #
#  Registration / levels
# --------------------------------------------------------------------------- #
def test_level_fit_unbiased_on_a_black_field():
    """A black field is all noise in the picture: least squares squashed the
    gain toward 0 (sync tip at ~black).  The whole-line Deming fit keeps ~1."""
    rng = np.random.default_rng(3)
    f = np.full((FH, FW), float(BLACK)); f[:, :20] = 1000.0
    a = f + rng.normal(0, 400, f.shape)
    b = f + rng.normal(0, 400, f.shape)
    ra, ca = S._active({"activeVideoStart": AVS, "activeVideoEnd": AVE, "fieldHeight": FH})
    out, best, (g, o) = S.register_field(b, a, ra, ca, return_level=True)
    assert abs(g - 1) < 0.01
    assert abs(best[1]) <= S.REG_MAX_DX + 0.5           # nothing to register on: bounded
    x = (b - BLACK)[ra, ca].ravel(); y = (a - BLACK)[ra, ca].ravel()
    assert np.polyfit(x, y, 1)[0] < 0.2                 # what plain least squares gave


def test_registration_is_horizontal_and_bounded():
    t = _truth(1)[0]
    ra, ca = S._active({"activeVideoStart": AVS, "activeVideoEnd": AVE, "fieldHeight": FH})
    img = S.fourier_shift(t, 0, 2.3)
    _, best = S.register_field(img, t, ra, ca)
    assert best[0] == 0.0 and abs(best[1] + 2.3) < 0.05
    flat = np.full((FH, FW), float(BLACK)) + np.random.default_rng(0).normal(0, 300, (FH, FW))
    _, best = S.register_field(flat, flat[::-1].copy(), ra, ca)
    assert abs(best[1]) <= S.REG_MAX_DX + 0.5


def test_chroma_phase_ignores_masked_samples():
    n = 200
    x = np.arange(n)
    ref = np.tile(1000 * np.cos(np.pi / 2 * x), (4, 1)) + 20000
    fld = np.tile(1000 * np.cos(np.pi / 2 * x + 0.3), (4, 1)) + 20000
    fld[:, 50:90] = 16384 + 6000 * np.random.default_rng(2).normal(0, 1, (4, 40))   # dropout
    m = np.zeros(fld.shape, bool); m[:, 50:90] = True
    ca = slice(0, n)
    bad = np.angle(S.chroma_phase(fld, ref, ca))
    good = np.angle(S.chroma_phase(fld, ref, ca, mask=m))
    assert np.allclose(good, 0.3, atol=0.05)
    assert np.abs(bad - 0.3).max() > 4 * np.abs(good - 0.3).max()


def test_cubic_resample_beats_linear():
    x = np.arange(400, dtype=float)
    a = np.sin(2 * np.pi * 0.2 * x)
    p = x[10:-10] + 0.5
    want = np.sin(2 * np.pi * 0.2 * p)
    e_cub = np.abs(S._catmull_rom(a, p) - want).max()
    e_lin = np.abs(np.interp(p, x, a) - want).max()
    assert e_cub < 0.5 * e_lin


# --------------------------------------------------------------------------- #
#  Cross-master weights
# --------------------------------------------------------------------------- #
def test_single_capture_master_variance_is_floored():
    groups = [["a", "b", "c"], ["d"]]
    P = {"a": np.array([4.0, 4.0]), "b": np.array([6.0, 6.0]), "c": np.array([5.0, 5.0])}
    Q = np.array([[2.0, 2.0], [np.nan, np.nan]])
    V = np.array([[10.0, 10.0], [-3.0, 1e-9]])
    Vf = S.xm_floor_variance(V, Q, P, groups)
    assert np.all(Vf[1] >= 5.0)
    alpha = (1 / Vf) / (1 / Vf).sum(0)
    assert np.all(alpha[1] < 0.7)


# --------------------------------------------------------------------------- #
#  Missing pictures, audio
# --------------------------------------------------------------------------- #
def test_missing_picture_is_padded(tmp_path):
    """Picture 103 is in no capture: the output keeps picture-number
    alignment with a flagged repeat and a full frame of audio."""
    pics = [100, 101, 102, 104, 105, 106]
    _, bases = _captures(tmp_path, nframes=6, ncap=3, pictures=pics)
    j, t, pcm = _stack(tmp_path, bases)
    got = [S.cav_picture(f["vbi"]["vbiData"]) for f in j["fields"][::2]]
    assert got == list(range(100, 107))
    assert j["fields"][6].get("pad") and j["fields"][7].get("pad")
    assert np.array_equal(t[6], t[4]) and np.array_equal(t[7], t[5])
    assert len(pcm) == 7 * SPF
    assert sum(f["audioSamples"] for f in j["fields"]) == len(pcm)


def test_picture_vbi_round_trip():
    key = ((1 * 60 + 23) * 60 + 45) * 25 + 17
    v0, v1 = S.picture_vbi(key, cav=False)
    assert S.clv_frame(v0 + v1) == key
    v0, v1 = S.picture_vbi(31234)
    assert S.cav_picture(v0) == 31234


def test_padded_frame_keeps_its_dropout_flags(tmp_path):
    """A repeat of a frame with a residual (every-capture) dropout carries
    the same flags, so the corrector conceals it in the repeat too."""
    pics = [100, 101, 102, 104, 105, 106]
    row, x0, x1 = 30, 80, 120
    d = {c: {4: [(row, x0, x1)]} for c in range(3)}      # picture 102, first field
    _, bases = _captures(tmp_path, nframes=6, ncap=3, pictures=pics, dropouts=d)
    j, _, _ = _stack(tmp_path, bases)
    assert _do_rows(j, 4) == [(row + 1, x0, x1)]
    assert j["fields"][6].get("pad") and _do_rows(j, 6) == [(row + 1, x0, x1)]


def test_no_padding_next_to_a_single_capture_key(tmp_path):
    """A run of keys on one capture far past the others' end (a seek, or a
    mis-sieved run) must not pad the output with ~200 repeats."""
    truth = _truth(8)
    rng = np.random.default_rng(1)
    bases = []
    for c, pics in enumerate(([100, 101, 102, 103, 104, 105, 106, 107],) * 2
                             + ([100, 101, 102, 103, 300, 301, 302, 303],)):
        b = str(tmp_path / f"cap{c}")
        _write_capture(b, truth + rng.normal(0, 150, truth.shape), list(pics))
        bases.append(b)
    j, _, _ = _stack(tmp_path, bases)
    got = [S.cav_picture(f["vbi"]["vbiData"]) for f in j["fields"][::2]]
    assert got == list(range(100, 108)) + [300, 301, 302, 303]
    assert not any(f.get("pad") for f in j["fields"])


def test_audio_lag_confidence():
    rng = np.random.default_rng(5)
    sig = np.cumsum(rng.normal(0, 1, 4000))
    sig -= np.convolve(sig, np.ones(50) / 50, "same")
    lag, pk = S.align_audio(sig, np.roll(sig, 3), maxlag=200, return_peak=True)
    assert abs(lag - 3) < 0.1 and pk > 0.9
    _, pk = S.align_audio(rng.normal(0, 1, 4000), rng.normal(0, 1, 4000), maxlag=200,
                          return_peak=True)
    assert pk < S.AUDIO_LAG_MIN_CORR


def test_write_audio_rounds(tmp_path):
    w = S.StackWriter(str(tmp_path / "w"), {"fieldWidth": 4, "fieldHeight": 2}, "PAL")
    w.write_audio(np.array([[0.6, -0.6], [1.49, -2.5]]))
    w.close()
    assert np.fromfile(str(tmp_path / "w.pcm"), "<i2").tolist() == [1, -1, 1, -2]


class _Streaming(S.FrameSource):
    """A capture served only as a stream (the --stream / .ldf path)."""

    def __init__(self, base):
        self._s = S.TBCFrameSource(base)
        self.name, self.videoParameters = self._s.name, self._s.videoParameters

    def frames(self):
        return self._s.frames()

    def close(self):
        self._s.close()


def test_streaming_path_pads_and_flags_like_the_indexed_one(tmp_path):
    pics = [100, 101, 102, 104, 105, 106]
    row, x0, x1 = 30, 80, 120
    d = {c: {2: [(row, x0, x1)]} for c in range(3)}
    _, bases = _captures(tmp_path, nframes=6, ncap=3, pictures=pics, dropouts=d)
    srcs = [_Streaming(b) for b in bases]
    out = str(tmp_path / "stream")
    try:
        S.stack(srcs, out, sample=4, log=lambda *a: None)
    finally:
        for s in srcs:
            s.close()
    j, t, pcm = _read_stack(out)
    got = [S.cav_picture(f["vbi"]["vbiData"]) for f in j["fields"][::2]]
    assert got == list(range(100, 107))
    assert _do_rows(j, 2) == [(row + 1, x0, x1)]
    assert len(pcm) == 7 * SPF


def _xm_captures(tmp_path, row, x0, x1, alt_rail=False, seed=4):
    """Three primary captures (one master's disc noise) with a dropout
    flagged on all of them, and one alt-master capture; the picture carries
    saturated PAL-like chroma (adjacent field lines in other phases)."""
    truth = _truth(6, texture=200.0)
    x = np.arange(FW)
    for k in range(len(truth)):
        for r in range(22, FH - 2):
            truth[k, r, AVS:AVE] += 4000 * np.cos(np.pi / 2 * x[AVS:AVE] + np.pi / 2 * r)
    rng = np.random.default_rng(seed)
    mA = rng.normal(0, 200, truth.shape); mB = rng.normal(0, 200, truth.shape)
    bases = []
    for c in range(4):
        b = str(tmp_path / f"cap{c}")
        f = truth + (mA if c < 3 else mB) + rng.normal(0, 150, truth.shape)
        if c == 3 and alt_rail:
            f[4, row, x0:x1] = 200.0                # the alt's own, unflagged
        _write_capture(b, f, list(range(100, 106)), {4: [(row, x0, x1)]} if c < 3 else None)
        bases.append(b)
    return truth, bases


@pytest.mark.parametrize("alt_rail", [False, True])
def test_cross_master_fill_of_a_master_level_dropout(tmp_path, alt_rail):
    """Flagged on every primary: filled from a clean alt master (and counted),
    but an implausible alt is refused and the pixels stay flagged."""
    row, x0, x1 = 40, 100, 160
    truth, bases = _xm_captures(tmp_path, row, x0, x1, alt_rail=alt_rail)
    j, t, _ = _stack(tmp_path, bases, cross_master=True,
                     masters=[["cap0", "cap1", "cap2"], ["cap3"]])
    if alt_rail:
        assert _do_rows(j, 4) == [(row + 1, x0, x1)]
    else:
        assert _do_rows(j, 4) == []
        assert np.abs(t[4, row, x0:x1] - truth[4, row, x0:x1]).max() < 2000


def _hat_frames(amp, nframes=6, seed=7):
    """Full-size PAL fields: masters of 4 / 2 / 1 captures, equal disc noise
    300, player noise 400 (the single capture 800).  True error variance per
    master 130k / 170k / 730k -> blend weights 0.51 / 0.39 / 0.09."""
    rng = np.random.default_rng(seed)
    fh, fw = 313, 1135
    caps = {"a1": (0, 400), "a2": (0, 400), "a3": (0, 400), "a4": (0, 400),
            "b1": (1, 400), "b2": (1, 400), "c1": (2, 800)}
    y, x = np.mgrid[0:fh, 0:fw]
    frames = []
    for k in range(nframes):
        t = np.full((fh, fw), 16000.0); t[:, 10:60] = 1000.0
        pic = amp * (np.sin(x / (7.0 + k)) * np.cos(y / 11.0) + 0.5 * np.sin(x / 3.1 + y / 5.0))
        t[24:307, 185:1107] += pic[24:307, 185:1107] + amp
        disc = [rng.normal(0, 300, (fh, fw)) for _ in range(3)]
        fd = {}
        for n, (m, p) in caps.items():
            f = t + disc[m] + rng.normal(0, p, (fh, fw))
            fd[n] = S.Frame(k, f, f, {}, {}, [], [], None, None)
        frames.append(fd)
    w = {n: 1.0 / (300.0 ** 2 + p ** 2) for n, (m, p) in caps.items()}
    return frames, [["a1", "a2", "a3", "a4"], ["b1", "b2"], ["c1"]], w


@pytest.mark.parametrize("amp", [6000.0, 300.0])
def test_three_master_hat_weights(amp):
    """The three-cornered hat recovers the masters' weights on real-picture
    and on near-black (low-SNR) fields.  With the least-squares level match
    the near-black case collapsed the alt masters' variance and gave the
    single noisiest capture ~half the blend (CommunityNorth, Oct 2026)."""
    frames, groups, w = _hat_frames(amp)
    vp = {"activeVideoStart": 185, "activeVideoEnd": 1107, "fieldHeight": 313}
    ra, ca = S._active(vp)
    xm = S.xm_estimate(frames, groups, ra, ca, 17734475.0, w, log=lambda *a: None)
    a = xm["alpha"].mean(1)
    assert abs(a[0] - 0.51) < 0.12 and abs(a[1] - 0.39) < 0.12 and a[2] < 0.2, a


def test_short_pcm_and_odd_field_count(tmp_path):
    """A capture whose .pcm ends early and whose .tbc.json carries an odd
    number of fields still stacks: the short frame's audio is left out of
    the audio combine, the trailing half frame is ignored."""
    truth, bases = _captures(tmp_path, nframes=6, ncap=3)
    p = bases[0] + ".pcm"
    a = np.fromfile(p, "<i2"); a[: len(a) - 2 * SPF + 100].tofile(p)       # last frame short
    j = json.load(open(bases[1] + ".tbc.json")); j["fields"] = j["fields"][:-1]
    json.dump(j, open(bases[1] + ".tbc.json", "w"))
    s = S.TBCFrameSource(bases[0])
    s.keys(); assert s.get(105).audio.shape[0] < SPF and s.get(104).audio.shape[0] == SPF
    assert s._spf == SPF
    s.close()
    j, t, pcm = _stack(tmp_path, bases)
    assert len(j["fields"]) == 12 and len(pcm) == sum(f["audioSamples"] for f in j["fields"])


def test_read_ahead_is_bit_identical(tmp_path, monkeypatch):
    """The read-ahead workers stack exactly what the pool did."""
    truth, bases = _captures(tmp_path, nframes=12, ncap=3, dropouts={1: {4: [(40, 100, 160)]}})
    monkeypatch.setattr(S, "BLOCK", 3)
    outs = {}
    for ra_on in (False, True):
        monkeypatch.setattr(S, "READAHEAD", ra_on)
        srcs = [S.TBCFrameSource(b) for b in bases]
        o = str(tmp_path / f"stk{int(ra_on)}")
        S.stack(srcs, o, sample=4, jobs=2, log=lambda *a: None)
        for s in srcs:
            s.close()
        outs[ra_on] = (open(o + ".tbc", "rb").read(), open(o + ".pcm", "rb").read(),
                       json.load(open(o + ".tbc.json"))["fields"])
    assert outs[False] == outs[True]


def test_content_offsets_per_segment():
    """A capture numbered one low up to a repeated picture, right after it."""
    rng = np.random.default_rng(5)
    H, W = 40, 64
    pics = {k: rng.normal(size=(H, W)).astype(np.float32) * 1000 + 20000 for k in range(1, 4001)}

    def src(name, mapping):
        sv = S.TBCFrameSource.__new__(S.TBCFrameSource)
        sv.name = name
        fis = sorted(mapping)
        sv._idx = {k: i for i, k in enumerate(fis)}
        content = [pics[mapping[k]] for k in fis]
        sv._read_field = lambda idx, c=content: c[idx // 2]
        return sv
    a = src("a", {k: k for k in range(1, 4001)})
    # b's label k shows picture k+1 up to 2399, then picture k (a repeated label at the switch)
    b = src("b", {**{k: k + 1 for k in range(1, 2400)}, **{k: k for k in range(2400, 4001)}})
    old = S.CONTENT_WIN
    S.CONTENT_WIN = 500
    try:
        offs = S.content_offsets([a, b], log=lambda m: None)
    finally:
        S.CONTENT_WIN = old
    segs = offs["b"]
    assert segs[0][2] == 1 and segs[-1][2] == 0
    assert abs(segs[0][1] - 2399) <= 1, segs
