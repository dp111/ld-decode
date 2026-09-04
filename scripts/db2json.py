#!/usr/bin/env python3
"""Export a .tbc.json from a .tbc.db (sqlite decode metadata).

The sqlite-metadata decoder writes .tbc.db only; upstream tools
(ld-chroma-decoder, ld-disc-stacker, lddecode/stack.py) want .tbc.json.
Usage: db2json.py <base.tbc.db> [...]   ->  writes <base>.tbc.json next to it.
"""
import json, sqlite3, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lddecode.tbcmeta import CAPTURE_QUERY, video_parameters


def convert(db_path):
    out_path = db_path[:-3] + ".json" if db_path.endswith(".tbc.db") \
        else db_path + ".tbc.json"
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cap = db.execute(CAPTURE_QUERY).fetchone()
    if cap is None:
        raise SystemExit(f"{db_path}: empty capture table (decode incomplete?)")
    vp = video_parameters(cap)

    pcm = db.execute("SELECT bits, is_little_endian, is_signed, sample_rate"
                     " FROM pcm_audio_parameters LIMIT 1").fetchone()
    pcmj = None
    if pcm:
        pcmj = {"bits": int(pcm[0]), "isLittleEndian": bool(pcm[1]),
                "isSigned": bool(pcm[2]), "sampleRate": float(pcm[3])}

    vits = {fid: (w, b) for fid, w, b in db.execute(
        "SELECT field_id, w_snr, b_psnr FROM vits_metrics")}
    vbi = {fid: (v0, v1, v2) for fid, v0, v1, v2 in db.execute(
        "SELECT field_id, vbi0, vbi1, vbi2 FROM vbi")}
    drops = {}
    for fid, line, sx, ex in db.execute(
            "SELECT field_id, field_line, startx, endx FROM drop_outs"
            " ORDER BY field_id, field_line, startx"):
        d = drops.setdefault(fid, {"fieldLine": [], "startx": [], "endx": []})
        d["fieldLine"].append(int(line)); d["startx"].append(int(sx))
        d["endx"].append(int(ex))

    fields = []
    for (fid, isff, sconf, dloc, floc, mbire, phase, faults, asamp, efmt,
         pad) in db.execute(
            "SELECT field_id, is_first_field, sync_conf, disk_loc, file_loc,"
            " median_burst_ire, field_phase_id, decode_faults, audio_samples,"
            " efm_t_values, pad FROM field_record ORDER BY field_id"):
        fj = {
            "seqNo": int(fid) + 1,
            "isFirstField": bool(isff),
            "syncConf": int(sconf),
            "diskLoc": float(dloc),
            "fileLoc": int(floc),
            "medianBurstIRE": float(mbire),
            "fieldPhaseID": int(phase),
            "audioSamples": int(asamp),
            "efmTValues": int(efmt),
        }
        if faults is not None:
            fj["decodeFaults"] = int(faults)
        if pad:
            fj["pad"] = True
        if fid in vits:
            w, b = vits[fid]
            vm = {}
            if w is not None:
                vm["wSNR"] = float(w)
            if b is not None:
                vm["bPSNR"] = float(b)
            if vm:
                fj["vitsMetrics"] = vm
        if fid in vbi:
            fj["vbi"] = {"vbiData": [int(x) for x in vbi[fid]]}
        if fid in drops:
            fj["dropOuts"] = drops[fid]
        fields.append(fj)

    nfields = vp["numberOfSequentialFields"]
    if len(fields) != nfields:
        print(f"  warning: {db_path}: capture says {nfields} fields,"
              f" field_record has {len(fields)}", file=sys.stderr)
        vp["numberOfSequentialFields"] = len(fields)

    out = {"videoParameters": vp, "fields": fields}
    if pcmj:
        out["pcmAudioParameters"] = pcmj
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"{db_path} -> {out_path}  ({len(fields)} fields)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
