#!/usr/bin/env python3
"""Combine the two analogue channels where the disc says the sound is mono.

IEC 60857 programme status (VBI line 16) declares each region's sound mode;
squelch_pcm.status_audio() decodes it. In a "mono" region both FM carriers
carry the same programme, but each carrier has its own disc noise (per-carrier
FM noise, independent between the two demodulators) and its own clicks. So
in mono regions the two channels are two observations of one signal:

  * R is registered to L (fractional delay, then a 63-tap least-squares FIR
    that matches R's response to L's; CommunityNorth's two sound sections:
    lag +0.04 / -0.42 sample, L/R level 0.985 / 1.16, R 1.3 dB down in the
    mids of the second),
  * a click present in only one channel (|L - R| spike) is replaced from the
    other channel,
  * elsewhere the output is the mean, at the geometric-mean level of L and R,
    written to both channels.

Stereo, bilingual and other regions are passed through unchanged, as is any
region where the measured L/R correlation contradicts the flag.

Input/output s16le stereo 44.1 kHz with the stack's .tbc.json beside the input
(or --json=PATH). Run it AFTER squelch_pcm.py: carrier-less noise inside a
mono region is uncorrelated L/R, which is what the squelch detects, and the
combine would make it identical in both channels.

Usage: mono_pcm.py in.pcm out.pcm [--json=PATH] [--no-click] [--report]
"""

import json
import os
import sys

import numpy as np
import scipy.linalg as sl
import scipy.signal as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from squelch_pcm import status_audio, MIN_RUN   # noqa: E402

RATE = 44100
MONO = 0b0001            # audio-table index "mono"
MIN_CORR = 0.95          # a region must measure this L/R correlation to be combined
XFADE = RATE // 100      # 10 ms crossfade at region edges
FIR_HALF = 32            # fractional-delay windowed-sinc half length
CLICK_K = 8.0            # click: |HP(L-R)| > CLICK_K x robust local level
CLICK_PAD = 32           # samples replaced either side of a click (~0.7 ms)
CLICK_ONESIDED = 2.0     # ...and only when one channel's HF peak is this x the other's
WIENER_TAPS = 63         # R->L response-matching FIR


def mono_runs(json_path, nsamples):
    """(start, end) sample spans whose status code says mono, runs < MIN_RUN fields dropped."""
    with open(json_path) as f:
        fields = json.load(f).get("fields") or []
    pos, cur, st = 0, None, []
    for fj in fields:
        for v in (fj.get("vbi", {}).get("vbiData") or []):
            a = status_audio(v)
            if a is not None:
                cur = a
                break
        st.append((pos, cur == MONO))
        pos += int(fj.get("audioSamples", 0) or 0)
    out, i = [], 0
    while i < len(st):
        j = i
        while j < len(st) and st[j][1] == st[i][1]:
            j += 1
        if st[i][1] and j - i >= MIN_RUN:
            e = st[j][0] if j < len(st) else nsamples
            out.append((st[i][0], min(e, nsamples)))
        i = j
    return [(s, e) for s, e in out if e - s > 4 * RATE]


def frac_delay_fir(d):
    """Kaiser-windowed sinc that delays by d samples (|d| < FIR_HALF)."""
    n = np.arange(-FIR_HALF, FIR_HALF + 1)
    h = np.sinc(n - d) * np.kaiser(len(n), 9.0)
    return h / h.sum()


def register(L, R):
    """Fractional lag of R vs L (R(t) ~ L(t - lag)) and LS gain g (L ~ g * R_shifted),
    from the loud 1 s blocks of the region (cross-spectrum phase slope, 300 Hz-8 kHz)."""
    blk = RATE
    nb = len(L) // blk
    lv = np.array([L[k * blk:(k + 1) * blk].std() for k in range(nb)])
    use = np.where(lv > max(np.percentile(lv, 50), 200))[0][:600]
    if len(use) < 3:
        return 0.0, 1.0
    f = np.fft.rfftfreq(blk, 1 / RATE)
    m = (f > 300) & (f < 8000)
    X = np.zeros(len(f), complex)
    w = np.hanning(blk)
    for k in use:
        a, b = L[k * blk:(k + 1) * blk], R[k * blk:(k + 1) * blk]
        X += np.fft.rfft((a - a.mean()) * w) * np.conj(np.fft.rfft((b - b.mean()) * w))
    k0 = int(np.argmax(np.fft.irfft(X)))
    k0 = k0 if k0 < blk // 2 else k0 - blk
    Y = X * np.exp(-2j * np.pi * f * k0 / RATE)
    ph, wt = np.angle(Y[m]), np.abs(Y[m])
    lag = k0 + np.sum(wt * ph * f[m]) / np.sum(wt * f[m] ** 2) / (2 * np.pi) * RATE
    return float(lag), None


def hp(x):
    return sps.sosfiltfilt(sps.butter(4, 4000, "hp", fs=RATE, output="sos"), x)


def wiener(L, R, ntap=WIENER_TAPS):
    """Short FIR h (centred) with L ~ h * R in the least-squares sense, from the
    loud 1 s blocks: matches R's frequency response (and hum) to L's, so that
    L - h*R is only the two carriers' independent noise and clicks."""
    blk, half = RATE, ntap // 2
    nb = len(L) // blk
    lv = np.array([L[k * blk:(k + 1) * blk].std() for k in range(nb)])
    use = np.where(lv > max(np.percentile(lv, 50), 200))[0][:600]
    if len(use) < 3:
        return None
    # model L[n] = sum_k h[k] R[n-k], k = -half..half (h is applied with mode="same")
    rxx, rxy = np.zeros(ntap), np.zeros(2 * half + 1)
    for k in use:
        a, b = L[k * blk:(k + 1) * blk], R[k * blk:(k + 1) * blk]
        n, q = len(b), ntap
        rxx += [np.dot(b[q:n - q], b[q - m:n - q - m]) for m in range(ntap)]
        rxy += [np.dot(a[q:n - q], b[q - jj:n - q - jj]) for jj in range(-half, half + 1)]
    return sl.solve_toeplitz(rxx, rxy)


def combine_region(L, R, click=True, prev=None):
    """Returns (mono, info) for one mono region (float arrays). prev: a previous
    region's (lag, h) to fall back on when this one is too short or quiet to measure."""
    lag, _ = register(L, R)
    Rs = sps.oaconvolve(R, frac_delay_fir(-lag), mode="same") if abs(lag) > 1e-3 else R.copy()
    h = wiener(L, Rs)
    if h is None:
        if prev is None:
            return None, {"lag": lag, "gain": 0.0, "corr": 0.0, "clicks": 0}
        lag, h = prev
        Rs = sps.oaconvolve(R, frac_delay_fir(-lag), mode="same")
    b = sps.oaconvolve(Rs, h, mode="same")
    g = float(np.dot(L, Rs) / max(np.dot(Rs, Rs), 1e-9))   # broadband R->L gain
    corr = float(np.dot(L - L.mean(), b - b.mean()) /
                 np.sqrt(np.dot(L - L.mean(), L - L.mean()) * np.dot(b - b.mean(), b - b.mean()) + 1e-9))
    info = {"lag": lag, "gain": g, "corr": corr, "clicks": 0, "h": h, "click_at": []}
    if corr < MIN_CORR or not (0.5 < g < 2.0):
        return None, info
    out = 0.5 * (L + b)
    if click:
        d = np.abs(hp(L - b))
        win = RATE // 10
        nb = max(len(d) // win, 1)
        lv = np.maximum(np.median(d[:nb * win].reshape(nb, -1), 1), 2.0)
        lv = np.repeat(lv, win)
        lv = np.r_[lv, np.full(max(len(d) - len(lv), 0), lv[-1])][:len(d)]
        hit = np.where(d > CLICK_K * 1.4826 * lv)[0]
        if len(hit):
            ha, hb = np.abs(hp(L)), np.abs(hp(b))
            br = np.where(np.diff(hit) > 2 * CLICK_PAD)[0]
            for s0, e0 in zip(np.r_[0, br + 1], np.r_[br, len(hit) - 1]):
                s, e = max(hit[s0] - CLICK_PAD, 0), min(hit[e0] + CLICK_PAD + 1, len(out))
                pa, pb = ha[s:e].max(), hb[s:e].max()
                if max(pa, pb) < CLICK_ONESIDED * min(pa, pb):
                    continue                    # both channels alike: programme, not a click
                src = b if pa > pb else L       # the channel without the click
                t = np.ones(e - s)
                r = min(8, (e - s) // 2)
                if r:
                    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r))
                    t[:r], t[-r:] = ramp, ramp[::-1]
                out[s:e] = out[s:e] * (1 - t) + src[s:e] * t
                info["clicks"] += 1
                info["click_at"].append(s)
    # level: geometric mean of L and R (out is at L's level)
    return out / np.sqrt(g), info


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    opts = [x for x in sys.argv[1:] if x.startswith("--")]
    inp, outp = args[0], args[1]
    json_path = next((o.split("=", 1)[1] for o in opts if o.startswith("--json=")),
                     (inp[:-4] if inp.endswith(".pcm") else inp) + ".tbc.json")
    a = np.fromfile(inp, "<i2").reshape(-1, 2)
    b = a.astype(np.float64)
    runs = mono_runs(json_path, len(a)) if os.path.exists(json_path) else []
    done, prev = 0, None
    for s0, e0 in runs:
        m, info = combine_region(b[s0:e0, 0], b[s0:e0, 1], "--no-click" not in opts, prev)
        if m is not None:
            prev = (info["lag"], info["h"])
        t0 = s0 / RATE
        print(f"  {int(t0) // 60}:{int(t0) % 60:02d} - {int(e0 / RATE) // 60}:{int(e0 / RATE) % 60:02d} "
              f"lag {info['lag']:+.3f} gain {info['gain']:.3f} r {info['corr']:.4f} "
              f"{'combined, %d one-channel clicks repaired' % info['clicks'] if m is not None else 'LEFT AS IS'}")
        if m is None:
            continue
        if "--report" in opts and info["click_at"]:
            print("    clicks at " + " ".join("%.3f" % ((s0 + c) / RATE) for c in info["click_at"]))
        # crossfade into / out of the region from the original channels
        t = np.ones(len(m))
        x = min(XFADE, len(m) // 2)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, x))
        t[:x], t[-x:] = ramp, ramp[::-1]
        for c in (0, 1):
            b[s0:e0, c] = b[s0:e0, c] * (1 - t) + m * t
        done += e0 - s0
    np.clip(np.round(b), -32768, 32767).astype("<i2").tofile(outp)
    print(f"{inp}: {len(runs)} mono region(s), {done // RATE}s of {len(a) // RATE}s combined")


if __name__ == "__main__":
    main()
