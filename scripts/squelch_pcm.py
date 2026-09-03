#!/usr/bin/env python3
"""Mute the spans of a decoded LaserDisc analog-audio .pcm that carry no sound.

Preferred method - the disc says so. IEC 60857 puts a programme status code on
VBI line 16 of every field: 0x8 x1 x2 x3 x4 x5, where x1x2 spell BA (no CX) or
DC (CX present), and the four bits x4.3, x3.0, x4.1, x4.0 give the analogue
audio state. 0b0010 is "no sound carriers" - the stills/data sections of an AIV
disc, where the FM demodulator outputs loud broadband noise. x5 carries even
parity over x4, so a misread status code is rejected rather than believed.
Measured on CommunitySouth: the disc declares no carriers from picture 1906
(1:16.2), and the spectral detector below independently found 1:16.

Fallback - when no .tbc.json is beside the .pcm, or it carries no usable status
codes, spans are detected from the audio itself. Real programme on these discs
is band-limited (nothing above ~14 kHz) while carrier-less noise fills the
spectrum to Nyquist, so a span is muted only when BOTH absolute HF power
(>14 kHz) exceeds HF_THRESH and overall level exceeds RMS_THRESH.

Both paths get raised-cosine fades at the edges. Input is s16le stereo
44.1 kHz; output same format.

Usage: squelch_pcm.py in.pcm out.pcm [--report] [--no-vbi] [--json PATH]
"""

import os
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

# --- VBI programme status code (IEC 60857 10.1.10) ------------------------
NO_CARRIERS = 0b0010     # audio-table index: "no sound carriers"
MIN_RUN = 50             # fields (~1 s): ignore shorter state flips


def status_audio(v):
    """Analogue-audio state (4-bit audio-table index) from a VBI code, or None.

    Accepts only a well-formed programme status code: x1x2 must read BA or DC,
    and x5's bits 3,2,1 must satisfy the even-parity checks over x4. That
    rejects the occasional misread code (0x8ba0a6, 0x8ba126, ...) outright."""
    if (v >> 12) not in (0x8BA, 0x8DC):
        return None
    x3, x4, x5 = (v >> 8) & 0xF, (v >> 4) & 0xF, v & 0xF
    b3, b2, b1, b0 = (x4 >> 3) & 1, (x4 >> 2) & 1, (x4 >> 1) & 1, x4 & 1
    if (((x5 >> 3) & 1), ((x5 >> 2) & 1), ((x5 >> 1) & 1)) != \
       ((b3 ^ b2 ^ b0), (b3 ^ b1 ^ b0), (b2 ^ b1 ^ b0)):
        return None
    return (b3 << 3) | ((x3 & 1) << 2) | (b1 << 1) | b0


def vbi_spans(json_path, nsamples):
    """(sample_start, sample_end) spans declared "no sound carriers", or None
    when the metadata cannot answer."""
    import json
    with open(json_path) as f:
        j = json.load(f)
    fields = j.get("fields") or []
    if not fields:
        return None
    state, pos, seen = [], 0, 0
    cur = None
    for fj in fields:
        a = None
        for v in (fj.get("vbi", {}).get("vbiData") or []):
            a = status_audio(v)
            if a is not None:
                seen += 1
                break
        if a is not None:
            cur = a
        state.append((pos, cur))
        pos += int(fj.get("audioSamples", 0) or 0)
    if seen < len(fields) // 4:
        return None                      # too few status codes to trust
    # run-length filter: drop state runs shorter than MIN_RUN fields
    flags = [s == NO_CARRIERS for _, s in state]
    i = 0
    while i < len(flags):
        j2 = i
        while j2 < len(flags) and flags[j2] == flags[i]:
            j2 += 1
        if j2 - i < MIN_RUN and i > 0:
            for k in range(i, j2):
                flags[k] = flags[i - 1]
            i = j2
        else:
            i = j2
    out, start = [], None
    for idx, f in enumerate(flags):
        if f and start is None:
            start = state[idx][0]
        elif not f and start is not None:
            out.append((start, state[idx][0]))
            start = None
    if start is not None:
        out.append((start, nsamples))
    return [(s, min(e, nsamples)) for s, e in out if e > s]



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


def merge(spans):
    """Coalesce overlapping / touching (start, end) sample intervals."""
    out = []
    for s0, e0 in sorted(spans):
        if out and s0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e0))
        else:
            out.append((s0, e0))
    return out


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    opts = [x for x in sys.argv[1:] if x.startswith("--")]
    inp, outp = args[0], args[1]
    json_path = None
    for o in opts:
        if o.startswith("--json="):
            json_path = o.split("=", 1)[1]
    if json_path is None:
        base = inp[:-4] if inp.endswith(".pcm") else inp
        json_path = base + ".tbc.json"

    a = np.fromfile(inp, "<i2").reshape(-1, 2)

    # The two methods answer different questions and BOTH are needed:
    #   * the VBI status code gives the authored no-carrier regions, including
    #     quiet ones the level test below would skip (it needs rms > RMS_THRESH);
    #   * the spectral detector catches carriers that are actually off inside a
    #     region the disc declares as having sound - inter-segment gaps and
    #     dropouts, which the status word cannot express because it only
    #     changes at region boundaries.
    # Measured on CommunitySouth: the disc declares carriers back at 29:59 but
    # they do not really return until 30:16, and 12 further spans between 30:18
    # and 35:56 are genuinely carrier-less (HF 68-155). VBI alone would have
    # left ~110 s of broadband noise unmuted, so the output is the UNION.
    vsp = []
    if "--no-vbi" not in opts and os.path.exists(json_path):
        try:
            vsp = vbi_spans(json_path, len(a)) or []
        except Exception as e:                      # malformed/partial metadata
            print(f"{inp}: VBI status unusable ({e})", file=sys.stderr)
            vsp = []
    ssp = []
    if "--vbi-only" not in opts:
        ssp = [(s0 * WIN, min(e0 * WIN, len(a))) for s0, e0 in spans(classify(a))]
    sp = merge(vsp + ssp)

    b = a.astype(np.float64)
    fade = 0.5 * (1 + np.cos(np.linspace(0, np.pi, FADE)))[:, None]
    for s0, e0 in sp:
        b[s0:e0] = 0
        if s0 - FADE >= 0:
            b[s0 - FADE:s0] *= fade
        if e0 + FADE <= len(b):
            b[e0:e0 + FADE] *= fade[::-1]
    np.clip(b, -32768, 32767).astype("<i2").tofile(outp)

    total = sum(e0 - s0 for s0, e0 in sp) // RATE
    print(f"{inp}: muted {len(sp)} span(s), {total}s of {len(a) // RATE}s "
          f"[VBI {len(vsp)} + spectral {len(ssp)} -> union]")
    for s0, e0 in sp:
        s1, e1 = s0 // RATE, e0 // RATE
        print(f"  {s1 // 60}:{s1 % 60:02d} - {e1 // 60}:{e1 % 60:02d}")


if __name__ == "__main__":
    main()
