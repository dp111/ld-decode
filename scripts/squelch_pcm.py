#!/usr/bin/env python3
"""Mute carrier-less spans in a decoded LaserDisc analog-audio .pcm.

When the disc carries no analog audio carriers (runout, stills sections), the
FM demodulator outputs loud broadband noise. Real programme on these discs is
band-limited (nothing above ~14 kHz), while carrier-less noise fills the
spectrum to Nyquist — so a span is muted only when BOTH:
  * absolute HF power (>14 kHz) exceeds HF_THRESH (quietest observed programme
    passages reach ~110 via the exposed noise floor; carrier-off is ~240), and
  * overall level exceeds RMS_THRESH (muting is only needed where the noise is
    loud; quiet windows stay untouched either way).
Hysteresis: 3 consecutive hit-windows to enter mute, 2 misses to exit; edges
get raised-cosine fades. Input is s16le stereo 44.1 kHz; output same format.

Usage: squelch_pcm.py in.pcm out.pcm [--report]
"""
import sys
import numpy as np

RATE = 44100
WIN = RATE            # 1 s analysis windows
HF_CUT = 14000.0
HF_THRESH = 60.0      # RMS-equivalent of >14 kHz band, max over channels
                      # (measured: programme <= ~13 incl. quiet passages and
                      # eruptions; carrier-off runout 145-175)
RMS_THRESH = 700.0
ENTER, EXIT = 3, 2    # hysteresis (windows)
FADE = RATE // 10     # 100 ms


def classify(a):
    n = len(a) // WIN
    hits = np.zeros(n, bool)
    w = np.hanning(WIN)
    wgain = np.sqrt((w ** 2).mean())
    f = np.fft.rfftfreq(WIN, 1 / RATE)
    hf = f > HF_CUT
    for i in range(n):
        seg = a[i * WIN:(i + 1) * WIN].astype(np.float64)
        rms = np.sqrt((seg ** 2).mean())
        if rms <= RMS_THRESH:
            continue
        hf_rms = 0.0
        for c in (0, 1):
            x = seg[:, c] - seg[:, c].mean()
            ps = np.abs(np.fft.rfft(x * w)) ** 2
            hf_rms = max(hf_rms, np.sqrt(2 * ps[hf].sum() / WIN ** 2) / wgain)
        hits[i] = hf_rms > HF_THRESH
    return hits


def spans(hits):
    out, run, start = [], 0, None
    miss = 0
    i = 0
    while i < len(hits):
        if start is None:
            if hits[i:i + ENTER].all() and i + ENTER <= len(hits):
                start = i
                i += ENTER
                miss = 0
                continue
        else:
            if not hits[i]:
                miss += 1
                if miss >= EXIT:
                    out.append((start, i - miss + 1))
                    start = None
            else:
                miss = 0
        i += 1
    if start is not None:
        out.append((start, len(hits)))
    return out


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    a = np.fromfile(inp, "<i2").reshape(-1, 2)
    hits = classify(a)
    sp = spans(hits)
    b = a.astype(np.float64)
    fade = 0.5 * (1 + np.cos(np.linspace(0, np.pi, FADE)))[:, None]
    for s, e in sp:
        s0, e0 = s * WIN, min(e * WIN, len(b))
        b[s0:e0] = 0
        if s0 - FADE >= 0:
            b[s0 - FADE:s0] *= fade
        if e0 + FADE <= len(b):
            b[e0:e0 + FADE] *= fade[::-1]
    np.clip(b, -32768, 32767).astype("<i2").tofile(outp)
    total = sum(e - s for s, e in sp)
    print(f"{inp}: muted {len(sp)} span(s), {total}s of {len(hits)}s")
    for s, e in sp:
        print(f"  {s//60}:{s%60:02d} - {e//60}:{e%60:02d}")


if __name__ == "__main__":
    main()
