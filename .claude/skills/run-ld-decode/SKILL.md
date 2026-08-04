---
name: run-ld-decode
description: Build, run, and drive ld-decode — the software LaserDisc RF decoder. Use when asked to run ld-decode, decode an .ldf/.lds RF capture, produce a .tbc, benchmark the decoder, or verify a decode change is byte-identical.
---

ld-decode is a Python CLI that demodulates LaserDisc RF captures (`.ldf`/`.lds`/`.flac`) into time-base-corrected video (`.tbc`) plus audio (`.pcm`), EFM (`.efm`), and metadata (`.tbc.json`). Drive it via **`.claude/skills/run-ld-decode/driver.sh`** — it wraps the CLI with `smoke`, `decode`, and a byte-identical `ab` regression check. There is no GUI; it's a batch CLI.

All paths below are relative to the repo root (`ld-decode/`).

## Prerequisites

Core decode needs Python 3 with numpy/scipy/numba/av. **On this container a working interpreter already exists at `/mnt/s/domsday/ldenv/bin/python`** (Python 3.14; numpy 2.4.6, scipy 1.17.1, numba 0.65.1, av) — use it. The system `python3` is **broken** for ld-decode (no `av` module). Verify the interpreter:

```bash
/mnt/s/domsday/ldenv/bin/python -c "import numpy,scipy,numba,av; print('deps OK')"
```

On a clean machine, instead: `python3 -m venv env && env/bin/pip install numpy scipy numba av`. (`matplotlib` is only needed for optional plotting, not decoding.) The EFM post-processor `ld-process-efm` is a separate C++/Qt binary already on PATH here (`/usr/local/bin/ld-process-efm`); built from `tools/ld-process-efm` via cmake if absent.

## Run (agent path) — the driver

The decoder runs from the repo script directly (no install needed). Everything goes through `driver.sh`:

```bash
# 1. Sanity check: synthetic-signal benchmark (NO capture needed) + a 12-frame real decode
.claude/skills/run-ld-decode/driver.sh smoke
# → bench prints Mean ms/block; real decode → "outputs: 17052240 byte .tbc | 24 fields" → SMOKE PASS

# 2. Decode N frames of a specific capture (default 12)
.claude/skills/run-ld-decode/driver.sh decode /path/to/capture.ldf 10
# → "outputs: 14210200 byte .tbc | 20 fields (expected 20)"

# 3. Byte-identical A/B — verify an uncommitted change to the decoder does NOT alter output.
#    Make your edit to lddecode/core.py or utils.py, leave it uncommitted, then:
.claude/skills/run-ld-decode/driver.sh ab /path/to/capture.ldf 150
# → "TBC: BYTE-IDENTICAL  (change preserves output)"   (or "DIFFER" if it changed the decode)
```

Outputs land in `$OUTDIR` (default `/dev/shm`): `<base>.tbc` (video), `.tbc.json` (metadata/field list), `.efm`, `.pcm`, `.run.log`. The `ab` check is the key workflow for decoder refactors/perf work — it decodes the working tree, `git stash`es `core.py`+`utils.py` to get the baseline, decodes again, and `cmp`s the `.tbc`.

Env overrides: `LDDECODE_PY` (interpreter), `LDDECODE_LDF` (default capture), `OUTDIR` (scratch dir). On this machine real `.ldf` captures live under `/mnt/s/domsday/BBC-Domesday-AIV-LaserDisc-DD86-Disc-Set-*/` and the driver defaults to the Set-5 CommunitySouth one.

### Direct invocation (for changes to demod internals)

`tools/bench_rf.py` imports `RFDecode` and times `demodblock` on a synthetic signal — no capture, no full decode. Use it for demod/filter/perf changes:

```bash
/mnt/s/domsday/ldenv/bin/python tools/bench_rf.py --system PAL --blocks 200 --audio --efm
```

## Run (human path)

Invoke the CLI directly (this is what the driver wraps):

```bash
/mnt/s/domsday/ldenv/bin/python ld-decode --PAL -t 4 --seek 1000 --length 12 capture.ldf /dev/shm/out
# → "Took 6.99 seconds to decode 12 frames"; writes /dev/shm/out.{tbc,tbc.json,efm,pcm}
```

`--PAL`/`--NTSC` is required. `--seek N` skips to frame N (skip the lead-in). `--length N` limits frames. `-t` is thread count. Run `ld-decode --help` for the full list.

## Gotchas

- **Use the venv interpreter, not `python3`.** System `python3` lacks `av` → `ModuleNotFoundError: No module named 'av'`. Always `/mnt/s/domsday/ldenv/bin/python`.
- **Short decodes are JIT-bound.** numba compiles on first use (~1.5 s+), so a 10–12 frame smoke runs at ~1.5–2 fps while a 500-frame run hits ~6 fps. Don't read FPS off a short decode — use ≥300 frames for timing.
- **`-t 4` is the measured optimum**, not "more is better": past 4 threads the decode *degrades* (it's bound by the serial per-field sync path, not demod). The CLI default is already 4.
- **`.tbc` is large** — ~1.4 MB/frame for PAL. A full side is tens of GB. Decode to `/dev/shm` (the driver does) and clean up; don't fill `/`.
- **The `ab` check only stashes `core.py` + `utils.py`** (where output-affecting decode logic lives). A change to `main.py` (arg parsing) or the C++ tools won't be A/B'd — it would falsely report identical.
- **CAV lead-in has erratic/repeating VBI frame numbers** (the spiral); judge decode completeness by the "File Frame" counter in the log, not the "CAV Frame #".
- **`--seek` past the very start** avoids the unreadable lead-in; seeking to ~1000 lands in stable program video (used by the driver).

## Troubleshooting

- `ModuleNotFoundError: No module named 'av'` (or `numba`) → you used the wrong interpreter; use `/mnt/s/domsday/ldenv/bin/python`.
- `ab` prints `stash failed (uncommitted change present?)` → the working tree had no changes to `core.py`/`utils.py` to stash; `ab` needs your uncommitted edit in place.
- Decode exits non-zero in the lead-in / "Unable to find any sync pulses" → seek further in (`--seek 1000`) or the capture start is unreadable; the driver's default seek avoids this.
- `ld-process-efm: command not found` → build it from `tools/ld-process-efm` (cmake) or add the prebuilt binary to PATH; only needed for EFM data extraction, not video decode.
