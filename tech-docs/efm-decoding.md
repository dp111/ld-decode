# EFM decoding (digital audio / LV-ROM data)

`ld-decode` extracts the disc's EFM stream (CD-style digital audio, or the
digital data on an LV-ROM / Domesday disc) when `--digital_audio` is enabled,
writing a `.efm` file of T-values that `ld-process-efm` turns into audio or
data sectors.

The EFM path is: RF → linear band-pass filter → **bit-clock PLL** (zero-crossing
detector, `lddecode/efm_pll.py`) → T-values. The PLL is the part that decides
how many sectors survive on a marginal disc.

## Defaults (no flags needed)

The PLL uses a **gear-shift / fast-reacquire** loop **by default**. While it is
locked it behaves identically to the original fixed-gain loop, so clean discs
are byte-for-byte unchanged; it only boosts its phase/frequency gains *while
unlocked* — on cold start, after a drop-out, and through low-SNR regions — which
is exactly where the old loop used to lose framing and drop sectors.

You normally want this on, so just decode as usual:

```
ld-decode --PAL --digital_audio  capture.ldf  out
ld-process-efm -b out.efm out.bin     # data disc  (or omit -b for audio)
```

To restore the original fixed-gain loop for an A/B comparison:

```
LDDECODE_EFM_GEARSHIFT=0 ld-decode --PAL --digital_audio capture.ldf out
```

## Tuning a stubborn disc (advanced)

For a marginal disc you can override individual acquisition parameters via the
environment. Leave them unset to use the tuned defaults.

| Variable | Default | Effect |
|---|---|---|
| `LDDECODE_EFM_PHASEGAIN_ACQ` | 0.05 | phase gain while acquiring (higher = faster, jumpier pull-in) |
| `LDDECODE_EFM_FREQSTEPMUL`   | 20   | frequency-step multiplier while acquiring |
| `LDDECODE_EFM_LOCKERRFRAC`   | 0.125| `|phase error| < frac·period` ⇒ counted as "in lock" |
| `LDDECODE_EFM_LOCKTHRESH`    | 24   | consecutive in-lock edges before declaring lock |

**Ensemble tip:** different settings lock *different* marginal frames. Decoding
a hard disc a few times with varied settings and merging the resulting sectors
(per-byte union, then RS-PC erasure correction at the `ld-process-efm` layer)
recovers more than any single setting alone.

## `LDDECODE_TBC_EFM` (experimental, off)

`LDDECODE_TBC_EFM=1` time-base-corrects the EFM waveform onto the video line
time-base before the PLL. It does **not** improve single-capture decode (same
sector set with or without), so it is off by default; it exists only to align
the pre-PLL EFM of *multiple captures of the same disc* onto a common
disc-position time-base for cross-capture waveform research.
