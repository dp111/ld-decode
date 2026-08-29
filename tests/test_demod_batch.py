"""The batched multi-row irfft over the stacked video-product filters must be
bit-identical to per-product irfft calls (each row is an independent
transform). Guards the FVideo_rfft_batch path in rfdecode.py, including the
recompute_fvideo rebuild."""
import numpy as np
import scipy.fft as npfft

from lddecode.rfdecode import RFDecode
from lddecode.utils import unwrap_hilbert


def _check(system):
    rf = RFDecode(system=system)
    rng = np.random.default_rng(999)
    sig = rng.integers(0, 16384, rf.blocklen).astype(np.float64)
    r = rf.demodblock(data=sig)

    indata_fft = npfft.fft(sig)
    hilbert = npfft.ifft(indata_fft * rf.Filters["RFVideo"])
    demod = unwrap_hilbert(hilbert, rf.freq_hz)
    demod_fft = npfft.rfft(np.clip(demod, 1500000, rf.freq_hz * 0.75))
    nr = demod_fft.shape[0]
    names = ["demod", "demod_05", "demod_burst"]
    keys = ["FVideo", "FVideo05", "FVideoBurst"]
    if system == "PAL":
        names.append("demod_pilot")
        keys.append("FVideoPilot")
    for name, key in zip(names, keys):
        expect = npfft.irfft(demod_fft * rf.Filters[key][:nr], n=rf.blocklen)
        assert np.array_equal(
            np.asarray(r["video"][name]), expect.astype(np.float32)
        ), f"{system} {name} not bit-identical"
    # the batch must follow an MTF recompute
    rf.DecoderParams["inverse_mtf_strength"] = 0.3
    rf.recompute_fvideo()
    r2 = rf.demodblock(data=sig)
    expect = npfft.irfft(
        demod_fft * rf.Filters["FVideo"][:nr], n=rf.blocklen
    ).astype(np.float32)
    assert np.array_equal(np.asarray(r2["video"]["demod"]), expect)


def test_pal_batch_bit_identical():
    _check("PAL")


def test_ntsc_batch_bit_identical():
    _check("NTSC")
