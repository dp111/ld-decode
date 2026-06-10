#!/usr/bin/env python3
"""Experiment: cancellation of FM sideband fold-over distortion (PAL).

Background
----------
PAL LaserDisc video FM is wideband relative to its carrier (6.76-7.9 MHz
carrier, baseband to 5.8 MHz), so second-order lower sidebands of chroma
fold through 0 Hz: at black, fv - 2*4.43 MHz = -1.77 MHz reflects to
+1.77 MHz, passes partially through the RF band-pass low skirt, and beats
against the carrier in the discriminator, producing a spur at
2*(fv - 4.43 MHz) ~ 5.3-6.1 MHz.  Measured with the standard chain and
fully saturated (100 IRE pp) chroma: ~2.2 IRE pp at 0 IRE pedestal,
~1.0 IRE pp at 50 IRE.  Worst case is dark, saturated colour.

The band-pass low edge is the linear-filter control knob, and a scan shows
the default (2.3 MHz, order 2) already sits on the Pareto front:

    low edge        fold@0IRE  fold@25IRE  chroma 4.43  luma 5.5 (dB)
    2.0 MHz o2        2.64       2.02        -1.13       -4.19
    2.3 MHz o2 (def)  2.18       1.62        -1.38       -4.86
    2.5 MHz o3        1.43       0.91        -1.67       -7.10
    2.7 MHz o4        0.78       0.44        -2.43      -10.22
    3.2 MHz o2 (lowb) 1.25       0.89        -2.44       -6.45

i.e. any static filter that suppresses the fold harder also eats the
legitimate lower sidebands of 5-5.8 MHz luma (same RF real estate), so
improving on the default requires a nonlinear approach.

Method (this script)
--------------------
Remodulation cancellation: the fold is deterministic given the video, so
  1. demodulate normally -> x1
  2. reconstruct the RF phase theta with a ZERO-PHASE band-limit (the
     band-pass's own phase at the carrier must not leak into theta, or the
     fold estimate is rotated by 2*phase and cancellation fails)
  3. demodulate cos(theta) (real: regenerates folds) and exp(j*theta)
     (analytic: fold-free) through the identical filter chain; their
     difference D isolates the fold with all in-band shaping common-mode
  4. subtract g*D from x1, where the complex gain g is least-squares
     fitted per block in the 4.5-7 MHz spur band.  g self-calibrates the
     channel phase the model can't know, and goes to ~0 when no
     correlated fold exists (self-disabling).

Results
-------
Synthetic (saturated chroma + noise): spur 9.5 -> 1.4 IRE pp (-16 dB),
|g| = 1.03 at -7 degrees, chroma amplitude preserved to 0.03 dB.

Real capture (kagemusha-leadout-cbar, 75% bars): |g| ~ 0.83, confirming
the fold is present and detectable, but net spur-band change -0.07 dB:
the genuine fold removed is offset by estimation noise added, because on
a healthy disc the spur (~1 IRE) is comparable to the disc noise in that
band.  Cost is ~2.5x the demod FFT work.

Conclusion: not worth enabling by default; potentially worthwhile as an
opt-in for dark/saturated material (e.g. animation) decoded from clean
captures, or combined with multi-capture stacking where the noise floor
drops below the fold.  Kept here so the validated machinery and numbers
aren't lost.

Run:  python3 scripts/fold-cancel-experiment.py [path/to/capture.ldf]
(defaults to the synthetic test; pass an .ldf for the real-capture test)
"""

import sys
import numpy as np
import numpy.fft as npfft
import scipy.signal as sps

sys.path.insert(0, ".")
from lddecode.core import RFDecode
from lddecode.utils import (
    LoadLDF, unwrap_hilbert, build_hilbert, filtfft, gen_bpf_supergauss, genwave,
)

FB = 4.43361875e6


def make_chain(rf):
    fs, bl = rf.freq_hz, rf.blocklen
    DP = rf.DecoderParams
    hp = sps.butter(DP["video_bpf_low_order"], DP["video_bpf_low"] / (fs / 2),
                    btype="highpass")
    lp = sps.butter(DP["video_bpf_high_order"], DP["video_bpf_high"] / (fs / 2),
                    btype="lowpass")
    RFV = filtfft(hp, bl) * filtfft(lp, bl)
    H = build_hilbert(bl)
    # zero-phase reconstruction band; keep the analog audio carriers out
    W = gen_bpf_supergauss(1500000, 14000000, 60, fs / 2, bl)
    if "FcutPAL" in rf.Filters:
        W = W * np.abs(rf.Filters["FcutPAL"])
    m_corr = (npfft.rfftfreq(bl, 1 / fs) > 4.5e6) & (npfft.rfftfreq(bl, 1 / fs) < 7.0e6)
    return RFV, H, W, m_corr


def fold_cancel_block(rf, sig, RFV, H, W, m_corr):
    """Returns (x1 plain demod, xc fold-cancelled demod, g fitted gain)."""
    fs, bl = rf.freq_hz, rf.blocklen
    F = npfft.fft(sig)
    x1 = unwrap_hilbert(npfft.ifft(F * RFV * H), fs)
    theta = np.unwrap(np.angle(npfft.ifft(F * W * H)))
    x2 = unwrap_hilbert(npfft.ifft(npfft.fft(np.cos(theta)) * RFV * H), fs)
    x0 = unwrap_hilbert(npfft.ifft(npfft.fft(np.exp(1j * theta)) * RFV), fs)
    D = x2 - x0
    X1 = npfft.rfft(x1 - x1.mean())
    Df = npfft.rfft(D - D.mean())
    den = np.vdot(Df[m_corr], Df[m_corr]).real
    g = np.vdot(Df[m_corr], X1[m_corr]) / den if den > 0 else 0.0
    Xc = X1.copy()
    Xc[m_corr] -= g * Df[m_corr]
    return x1, x1.mean() + npfft.irfft(Xc, bl), g


def spur_pp(rf, d, fspur, w=0.15e6):
    hz_ire = rf.DecoderParams["hz_ire"]
    cut = d[rf.blockcut:-rf.blockcut_end]
    cut = (cut - cut.mean()) / hz_ire
    sp = np.abs(npfft.rfft(cut * np.hanning(len(cut)))) / (len(cut) / 4)
    fr = npfft.rfftfreq(len(cut), 1 / rf.freq_hz)
    m = (fr > fspur - w) & (fr < fspur + w)
    return 2 * sp[m].max()


def synthetic_test(rf):
    fs, bl = rf.freq_hz, rf.blocklen
    hz_ire = rf.DecoderParams["hz_ire"]
    RFV, H, W, m_corr = make_chain(rf)
    n = np.arange(bl)
    print("synthetic saturated-chroma test (100 IRE pp):")
    for ped in (0, 25, 50):
        fake = np.full(bl, rf.iretohz(ped))
        fake += 50 * hz_ire * np.sin(2 * np.pi * FB / fs * n)
        emp = npfft.ifft(npfft.fft(fake) * rf.Filters["Femp"]).real
        x1, xc, g = fold_cancel_block(rf, genwave(emp, fs / 2), RFV, H, W, m_corr)
        fspur = 2 * (rf.iretohz(ped) - FB)
        print(f"  pedestal {ped:2d}: spur {spur_pp(rf, x1, fspur):5.2f} -> "
              f"{spur_pp(rf, xc, fspur):5.2f} IRE pp   "
              f"|g|={abs(g):.2f} arg={np.degrees(np.angle(g)):+.0f}deg")


def real_test(rf, fname):
    fs, bl = rf.freq_hz, rf.blocklen
    hz_ire = rf.DecoderParams["hz_ire"]
    RFV, H, W, m_corr = make_chain(rf)
    raw = LoadLDF(fname)(None, 3000000, bl * 40).astype(float)
    acc1 = acc2 = None
    gs = []
    for b in range(38):
        sig = raw[b * bl // 2:(b * bl // 2) + bl]
        if len(sig) < bl:
            break
        x1, xc, g = fold_cancel_block(rf, sig - sig.mean(), RFV, H, W, m_corr)
        gs.append(g)
        a = x1[rf.blockcut:-rf.blockcut_end] / hz_ire
        c = xc[rf.blockcut:-rf.blockcut_end] / hz_ire
        w = np.hanning(len(a))
        s1 = np.abs(npfft.rfft((a - a.mean()) * w)) ** 2
        s2 = np.abs(npfft.rfft((c - c.mean()) * w)) ** 2
        acc1 = s1 if acc1 is None else acc1 + s1
        acc2 = s2 if acc2 is None else acc2 + s2
    fr = npfft.rfftfreq(len(a), 1 / fs)
    m = (fr > 4.9e6) & (fr < 6.6e6)
    print(f"{fname}: fold-spur band (4.9-6.6 MHz) "
          f"{10 * np.log10(acc2[m].sum() / acc1[m].sum()):+.2f} dB, "
          f"median |g| = {np.median(np.abs(gs)):.3f}")


if __name__ == "__main__":
    rf = RFDecode(system="PAL", inputfreq=40, decode_analog_audio=0,
                  decode_digital_audio=False, has_analog_audio=True)
    if len(sys.argv) > 1:
        real_test(rf, sys.argv[1])
    else:
        synthetic_test(rf)
