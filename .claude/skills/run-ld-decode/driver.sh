#!/bin/bash
# driver.sh — build/run/drive harness for ld-decode (the PAL/NTSC LaserDisc
# RF decoder CLI).  Verified on this container, ld-decode 7.0.0, Python 3.14.
#
# Subcommands:
#   smoke            bench_rf.py (no input data) + a short real decode; checks outputs
#   decode <ldf> [N] decode N frames (default 12) of an .ldf to $OUTDIR
#   ab <ldf> [N]     byte-identical A/B: decode current tree vs git-stashed baseline,
#                    cmp the .tbc, and report FPS of each.  THE way to verify a change
#                    to the decode internals doesn't alter output.
#
# Env overrides:
#   LDDECODE_PY   python to use      (default: /mnt/s/domsday/ldenv/bin/python, else python3)
#   LDDECODE_LDF  default test .ldf  (default: the Set-5 CommunitySouth CAV PAL capture)
#   OUTDIR        scratch dir        (default: /dev/shm if writable, else /tmp)
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # .../ld-decode
cd "$REPO" || exit 2

PY="${LDDECODE_PY:-/mnt/s/domsday/ldenv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
LDF_DEFAULT="${LDDECODE_LDF:-/mnt/s/domsday/BBC-Domesday-AIV-LaserDisc-DD86-Disc-Set-5/Domesday_DD86-DS5_CommunitySouth_20191111_CAV_PAL_00001-54000.ldf}"
OUTDIR="${OUTDIR:-$([ -w /dev/shm ] && echo /dev/shm || echo /tmp)}"

say() { printf '\n=== %s ===\n' "$*"; }

# Validate a decode's outputs: exit code 0, non-empty .tbc, json field count == 2*N.
check_outputs() {
  local base="$1" frames="$2" rc="$3"
  [ "$rc" = 0 ] || { echo "FAIL: decode exit=$rc"; tail -5 "$base.run.log"; return 1; }
  [ -s "$base.tbc" ] || { echo "FAIL: no/empty .tbc"; return 1; }
  local nf
  nf=$("$PY" -c "import json;print(len(json.load(open('$base.tbc.json'))['fields']))" 2>/dev/null)
  echo "outputs: $(ls -la "$base.tbc" | awk '{print $5}') byte .tbc | $nf fields (expected $((frames*2)))"
  [ "$nf" = "$((frames * 2))" ] || { echo "FAIL: field count $nf != $((frames*2))"; return 1; }
  grep -q "Completed" "$base.run.log" || { echo "FAIL: no 'Completed' in log"; return 1; }
  return 0
}

# Decode helper: $1=ldf $2=frames $3=out-base ; prints FPS, returns decode rc via global RC
RC=0
do_decode() {
  local ldf="$1" frames="$2" base="$3"
  rm -f "$base".*
  local s e
  s=$(date +%s%3N)
  "$PY" ld-decode --PAL -t 4 --seek 1000 --length "$frames" "$ldf" "$base" > "$base.run.log" 2>&1
  RC=$?
  e=$(date +%s%3N)
  awk -v s="$s" -v e="$e" -v f="$frames" 'BEGIN{printf "  %.1fs wall, %.2f fps\n",(e-s)/1000.0,f*1000.0/(e-s)}'
}

cmd_smoke() {
  say "python"; "$PY" --version
  say "bench_rf.py (synthetic signal, no capture needed)"
  "$PY" tools/bench_rf.py --system PAL --blocks 50 2>&1 | grep -E "Initializing|Mean:|Throughput" || { echo "FAIL: bench"; return 1; }
  local ldf="$LDF_DEFAULT"
  if [ ! -f "$ldf" ]; then
    echo; echo "NOTE: default test capture not found ($ldf)"
    echo "      bench passed; pass an .ldf to '$0 decode <file>' for a full decode smoke."
    return 0
  fi
  say "real decode (12 frames from $ldf)"
  do_decode "$ldf" 12 "$OUTDIR/lddec_smoke"
  check_outputs "$OUTDIR/lddec_smoke" 12 "$RC" && { echo; echo "SMOKE PASS"; } || { echo; echo "SMOKE FAIL"; return 1; }
}

cmd_decode() {
  local ldf="${1:-$LDF_DEFAULT}" frames="${2:-12}"
  [ -f "$ldf" ] || { echo "no such .ldf: $ldf"; return 2; }
  say "decode $frames frames: $ldf"
  do_decode "$ldf" "$frames" "$OUTDIR/lddec_out"
  check_outputs "$OUTDIR/lddec_out" "$frames" "$RC"
}

# Byte-identical A/B vs the git-stashed baseline.  Run from a clean-ish tree with
# your uncommitted change in place (or HEAD vs a prior commit by editing baseline).
cmd_ab() {
  local ldf="${1:-$LDF_DEFAULT}" frames="${2:-150}"
  [ -f "$ldf" ] || { echo "no such .ldf: $ldf"; return 2; }
  say "A/B: NEW (working tree)"
  do_decode "$ldf" "$frames" "$OUTDIR/ab_new"; [ "$RC" = 0 ] || { echo "new decode failed"; return 1; }
  say "stashing lddecode/*.py -> baseline"
  git stash push -q lddecode/core.py lddecode/utils.py || { echo "stash failed (uncommitted change present?)"; return 1; }
  say "A/B: OLD (baseline)"
  do_decode "$ldf" "$frames" "$OUTDIR/ab_old"; local oldrc=$RC
  git stash pop -q
  [ "$oldrc" = 0 ] || { echo "old decode failed"; return 1; }
  say "compare"
  if cmp -s "$OUTDIR/ab_new.tbc" "$OUTDIR/ab_old.tbc"; then
    echo "TBC: BYTE-IDENTICAL  (change preserves output)"
  else
    echo "TBC: DIFFER  (change alters output — intended? re-check)"
    cmp "$OUTDIR/ab_new.tbc" "$OUTDIR/ab_old.tbc" | head -1
  fi
}

case "${1:-smoke}" in
  smoke)  cmd_smoke ;;
  decode) shift; cmd_decode "$@" ;;
  ab)     shift; cmd_ab "$@" ;;
  *) echo "usage: $0 {smoke|decode <ldf> [N]|ab <ldf> [N]}"; exit 2 ;;
esac
