#!/usr/bin/env python3
"""Build a per-frame index for a decoded LaserDisc, for cross-recording
algorithm testing.

For each frame it records:
  - CAV picture number (from the VBI; lets the SAME content be aligned across
    different recordings of the disc),
  - type: video / still / leadin / leadout / unknown,
  - efm  : whether EFM (digital audio) data is present in the field,
  - black: whether the active picture is essentially black,
  - dup  : whether the frame is (near-)identical to the previous one.

The picture number / type / efm columns come from <base>.tbc.json (fast).
black / dup come from <base>.lumaidx.tsv produced by the luma pass over the
.tbc (slow, streams the whole .tbc); they are filled in when that file exists.

Usage: build_disc_index.py <base-path-without-extension>
Writes <base>.index.tsv and prints a summary.
"""
import json
import os
import sys


def pic_of(vbidata):
    """CAV picture number: a 0xFxxxxx VBI code whose low 5 nibbles are BCD."""
    for x in vbidata:
        h = "%06x" % x
        if h[0] == "f" and all(c in "0123456789" for c in h[1:]):
            return int(h[1:])
    return None


def leadcode(vbidata):
    """Lead-in (0x88xxxx) / lead-out (0x80eeee) detection on the status line."""
    for x in vbidata:
        h = "%06x" % x
        if h.startswith("88"):
            return "leadin"
        if h.startswith("80ee"):
            return "leadout"
    return None


def main(base):
    fields = json.load(open(base + ".tbc.json"))["fields"]
    # one record per FRAME (a frame = first field + second field)
    nframes = len(fields) // 2
    recs = []
    for fi in range(nframes):
        f0 = fields[fi * 2]
        f1 = fields[fi * 2 + 1]
        vbi = (f0.get("vbi", {}).get("vbiData", []) or []) + (
            f1.get("vbi", {}).get("vbiData", []) or []
        )
        pic = pic_of(vbi)
        lead = leadcode(vbi)
        efm = bool((f0.get("efmTValues") or 0) or (f1.get("efmTValues") or 0))
        recs.append({"frame": fi, "pic": pic, "lead": lead, "efm": efm})

    # classify video vs still from picture-number progression
    for i, r in enumerate(recs):
        if r["lead"]:
            r["type"] = r["lead"]
            continue
        p, pn = r["pic"], (recs[i + 1]["pic"] if i + 1 < len(recs) else None)
        pp = recs[i - 1]["pic"] if i else None
        if p is None:
            r["type"] = "unknown"
        elif pp is not None and p == pp:
            r["type"] = "still"
        elif (pn is not None and pn == p + 1) or (pp is not None and p == pp + 1):
            r["type"] = "video"
        else:
            r["type"] = "video"  # isolated / segment boundary, treat as video

    # merge luma (black/dup) if the luma pass has produced its file
    luma = {}
    lp = base + ".lumaidx.tsv"
    if os.path.exists(lp):
        with open(lp) as fh:
            next(fh)
            for line in fh:
                c = line.split("\t")
                if len(c) >= 5:
                    fr = int(c[0])
                    mean, p01, p99, diff = map(float, c[1:5])
                    luma[fr] = (mean, p01, p99, diff)

    # black: active p99 still near black level; dup: tiny diff to previous frame
    # thresholds are in raw 16-bit tbc units (sync~0, black~16384, white~54000)
    BLACK_P99 = 20000
    DUP_DIFF = 250
    with open(base + ".index.tsv", "w") as out:
        out.write("frame\tpicture\ttype\tefm\tblack\tdup\tmean\tdiff_prev\n")
        for r in recs:
            mean = diff = ""
            black = dup = ""
            if r["frame"] in luma:
                m, p01, p99, d = luma[r["frame"]]
                mean, diff = "%.0f" % m, "%.1f" % d
                black = "1" if p99 < BLACK_P99 else "0"
                dup = "1" if 0 <= d < DUP_DIFF else "0"
            out.write(
                "%d\t%s\t%s\t%d\t%s\t%s\t%s\t%s\n"
                % (
                    r["frame"],
                    "" if r["pic"] is None else r["pic"],
                    r["type"],
                    int(r["efm"]),
                    black,
                    dup,
                    mean,
                    diff,
                )
            )

    # summary
    from collections import Counter
    types = Counter(r["type"] for r in recs)
    efm_frames = [r["frame"] for r in recs if r["efm"]]
    print("frames: %d" % nframes)
    print("types : %s" % dict(types))
    if efm_frames:
        print("EFM    : frames %d..%d (%d frames)" % (efm_frames[0], efm_frames[-1], len(efm_frames)))
    pics = [r["pic"] for r in recs if r["pic"] is not None]
    if pics:
        print("picture#: %d frames numbered, range %d..%d" % (len(pics), min(pics), max(pics)))
    if luma:
        nblack = sum(1 for r in recs if r["frame"] in luma and luma[r["frame"]][2] < BLACK_P99)
        ndup = sum(1 for r in recs if r["frame"] in luma and 0 <= luma[r["frame"]][3] < DUP_DIFF)
        print("luma   : %d frames analysed, %d black, %d duplicate-of-prev" % (len(luma), nblack, ndup))
    else:
        print("luma   : (lumaidx.tsv not present yet - black/dup columns blank)")
    print("wrote  : %s.index.tsv" % base)


if __name__ == "__main__":
    main(sys.argv[1])
