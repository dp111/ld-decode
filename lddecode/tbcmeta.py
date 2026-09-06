"""The decode metadata, read back from a decode's sqlite sidecar.

The decoder writes .tbc.db; the videoParameters dict that .tbc.json carries -
and that every consumer of a decode (ld-chroma-decoder, the stacker) reads its
geometry from - is a projection of that database's ``capture`` row.  Both
scripts/db2json.py and the stacker's streaming source need it, and they must
not drift apart, so the query and the mapping live here.
"""

CAPTURE_QUERY = (
    "SELECT system, git_branch, git_commit, video_sample_rate,"
    " active_video_start, active_video_end, field_width, field_height,"
    " number_of_sequential_fields, colour_burst_start, colour_burst_end,"
    " white_16b_ire, black_16b_ire, is_mapped, is_subcarrier_locked,"
    " is_widescreen FROM capture LIMIT 1"
)


def video_parameters(row):
    """Build the videoParameters dict from a CAPTURE_QUERY row.

    A stacked output's capture row leaves the colour-burst columns NULL (the
    stacker writes its own row and has no burst window of its own to record),
    so those fall back to 0 rather than raising - reading a stacked .tbc.db
    has to work.
    """
    (system, git_branch, git_commit, srate, avs, ave, fw, fh, nfields,
     cbs, cbe, w16, b16, ismap, issub, iswide) = row
    cbs = 0 if cbs is None else cbs
    cbe = 0 if cbe is None else cbe
    return {
        "system": system,
        "isSourcePal": system != "NTSC",
        "gitBranch": git_branch or "",
        "gitCommit": git_commit or "",
        "numberOfSequentialFields": int(nfields),
        "sampleRate": float(srate),
        "fieldWidth": int(fw),
        "fieldHeight": int(fh),
        "activeVideoStart": int(avs),
        "activeVideoEnd": int(ave),
        "colourBurstStart": int(cbs),
        "colourBurstEnd": int(cbe),
        "white16bIre": int(round(w16)),
        "black16bIre": int(round(b16)),
        "isMapped": bool(ismap),
        "isSubcarrierLocked": bool(issub),
        "isWidescreen": bool(iswide),
    }


def decode_metadata(db_path):
    """The {"videoParameters", "fields", ...} mapping .tbc.json used to carry,
    read from a .tbc.db.  Same shape json.load() of the old sidecar returned,
    so a consumer can take either.
    """
    import sqlite3
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cap = db.execute(CAPTURE_QUERY).fetchone()
        if cap is None:
            raise ValueError(f"{db_path}: empty capture table (decode incomplete?)")
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
    finally:
        db.close()

    if len(fields) != vp["numberOfSequentialFields"]:
        vp["numberOfSequentialFields"] = len(fields)

    out = {"videoParameters": vp, "fields": fields}
    if pcmj:
        out["pcmAudioParameters"] = pcmj
    return out


def load_decode_metadata(base):
    """Metadata for a decode at <base>, from .tbc.json if it exists and from
    .tbc.db otherwise.  The decoder stopped writing .tbc.json (bdb9865d) but
    both are still in circulation, so every consumer has to take either.
    """
    import json
    import os
    jp = base + ".tbc.json"
    if os.path.exists(jp):
        with open(jp) as f:
            return json.load(f)
    return decode_metadata(base + ".tbc.db")
