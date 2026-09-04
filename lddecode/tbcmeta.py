"""The videoParameters block, read back from a decode's sqlite metadata.

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
    """Build the videoParameters dict from a CAPTURE_QUERY row."""
    (system, git_branch, git_commit, srate, avs, ave, fw, fh, nfields,
     cbs, cbe, w16, b16, ismap, issub, iswide) = row
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
