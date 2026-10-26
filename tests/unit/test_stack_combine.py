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
    assert t[2, row, x0:x1].max() < 1000         # and that row is the damaged one


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
    assert abs(out[ra, 5:15].mean() - 1000) < 100      # sync tip kept
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


def test_output_level_is_the_consensus_not_the_reference(tmp_path):
    """Offsets 0, 0, +900: the reference is one of the 0 captures, but the
    output sits at the mean of the three (+300), whichever is the reference."""
    truth, bases = _captures(tmp_path, ncap=3, levels={2: (1.0, 900.0)})
    _, t, _ = _stack(tmp_path, bases)
    d = (t - truth)[:, 24:FH - 6, AVS:AVE].mean()
    assert abs(d - 300) < 60, d


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


def test_picture_vbi_clv_round_trip():
    key = ((1 * 60 + 23) * 60 + 45) * 25 + 17
    prev = [[0x80E000 | (0xA + 4) << 16 | 5 << 8 | 0x16, 0xF1DD23, 0xF1DD23], [0, 0, 0]]
    v0, v1 = S.picture_vbi(prev, key - 1, key)
    assert S.clv_frame(v0 + v1) == key


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
