# PAL decode audit — session handoff notes (2026-06-10)

Context for continuing the PAL quality audit in `dp111/ld-decode-tools`
(`src/ld-chroma-decoder/`).  Produced during an end-to-end audit of the RF
decoder in this repo; branch `claude/upstream-branch-sync-jnqcx7`.

## What was done in ld-decode (this repo)

All verified against IEC 60856-1986 (spec source:
github.com/simoninns/analogue-video-specifications) and by decoding the
`testdata/` PAL captures (GGV multiburst `ggv-mb-1khz.ldf`, EFM disc
`jason-testpattern.ldf`, analog-audio colour bars
`kagemusha-leadout-cbar.ldf`):

1. `checkMTF`: `np.max(x, 0)` -> `np.maximum` (mtf_level went negative past
   CAV frame 10000 / ~7 min CLV, blurring HF).  Commit 30a38dcb.
2. AGC `whitediff` compared a value against itself (always 0); now compares
   measured white vs 100 IRE.  Same commit.
3. PAL analog audio carriers (683.59/1066.41 kHz, -26 dBc) beat into the
   video band at |fv-fa| ~ 5.7-6.4 MHz; added per-block adaptive notches
   (+-150 kHz, carrier-detection gated since PAL discs have either analog
   audio or EFM).  Real-disc result: +1.1 dB wSNR, +4.0 dB bPSNR on GGV.
4. RF band-pass high edge 14 -> 13 MHz: player/capture chain has nothing
   usable above ~12.5 MHz; +0.1 dB (GGV) to +0.4/+1.7 dB (EFM disc) SNR,
   response and colour-bar chroma unchanged.  Commit 6b9730ba.
5. The post-demod group-delay equaliser (PR #1044) was verified exact
   against IEC 60856 9.1.6 (+10/+35/+85/+135/+200 ns at 2/3/4/4.43/4.8 MHz
   rel 0.5 MHz); pure all-pass; sync-safe.

Known-good reference numbers (GGV, 10 frames, defaults as of 6b9730ba):
wSNR 34.00, bPSNR 35.36.  Decoder-only response (loopback, mtf=0):
-1.38 dB at 4.43 MHz, -2.0 dB at 4.8, -4.9 dB at 5.5 (vs 0.5 MHz),
caused by RF band-pass sideband asymmetry; flat to ~+-0.8 dB below 4 MHz.

## Open items relevant to the chroma decoder audit

- **Chroma band tilt into the decoder**: the TBC luma+chroma arriving at
  ld-chroma-decoder carries the residual response above (-1.4 dB at fsc
  rel LF, falling fast above 5.3 MHz).  Check whether palcolour/Transform
  PAL assumptions (filter symmetry around fsc) compound this.
- **Group delay**: TBC group delay is now flat per IEC 9.1.6 end-to-end.
  If the chroma decoder applies any legacy GD compensation, it would now
  double-correct - verify it doesn't.
- **Things to check in transformpal2d/3d.cpp + palcolour.cpp**: FFT window
  overlap and tile sizes vs chroma bandwidth (+-1.3 MHz spec), threshold
  values' behaviour on noisy input, filter widths/delays (the classic
  audit: too wide/narrow, wrong delays), chroma gain calibration vs burst
  (4.43361875 MHz exact), V-switch/line-pairing, and the luma notch cost.
- **MTF compensation (this repo)** still over-equalises ~+1.5 dB at
  4.8 MHz on GGV at defaults; autoMTF constants are NTSC-calibrated.
  Needs a multi-disc calibration campaign - not yet done.
- **Fold-over distortion**: reflected J2 chroma sidebands produce a
  ~1-2 IRE spur at 2*(fv-4.43MHz) = 5.3-6.1 MHz on dark saturated colour.
  See scripts/fold-cancel-experiment.py for the validated (synthetic
  -16 dB) but not-yet-practical canceller and the band-pass Pareto data.

## Reproducing the test environment

- `pip install -e .` in this repo; `git submodule update --init testdata`.
- Decode: `python3 ld-decode --PAL --start 0 --length 10
  testdata/pal/ggv-mb-1khz.ldf /tmp/out`
- The multiburst frequency-response measurement and SNR comparison
  recipes are in the session history of dp111/ld-decode branch
  `claude/upstream-branch-sync-jnqcx7` commit messages; the multiburst
  packets sit at 0.5/1/2/4/4.8/5.8 MHz on a ~54 IRE pedestal.
