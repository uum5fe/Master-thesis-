# Local EIS pipeline — what was broken and what changed

Data: `RO2611976-01_Current_45A_Test_01_Karte_1..5.DAT`, 45 A, 5 Dewetron cards,
16 ch each, 10 kHz, 66 wired segments (68 present minus 33 and 59 marked
hardware-bad).

## Result

| | before | after |
|---|---|---|
| segments with a modelled spectrum | **42** | **66** (all measured segments) |
| frequency points per spectrum (median) | 6–18 | 25 |
| R_Ω | 388.9 ± 196.3 mΩ·cm² (CV 50.5 %) | **82.3 ± 23.5 mΩ·cm² (CV 28.5 %)** |
| area actually measured | 168.7 / 304.9 cm² | 289.0 / 304.9 cm² |
| DC closure vs 45 A setpoint | 43.7 A (−2.9 %) | 43.8 A (−2.6 %) |
| plate temperature | fallback constant 58.4 °C | 58.71 … 59.51 °C, measured |
| cell aggregate | 16 frequencies | 27 frequencies |

---

## Fix 1 — the cards were never time-aligned  (`bronze.py`)

`config.py` declared `align_cards = True` and `align_max_lag_s = 2.0`, but no
code read either. The consensus schedule stores each step as raw sample
indices taken from whichever card had the longest dwell, and `process_card`
applied those indices to every card unshifted.

The cards are armed separately. Measured offsets, recovered by
cross-correlating the shared UC reference:

| card | lag vs card 1 | peak corr |
|---|---|---|
| 1 | 0 | 1.000 |
| 2 | −3 samples (−0.0003 s) | 0.997 |
| 3 | **−57 121 samples (−5.7121 s)** | 0.982 |
| 4 | **−25 359 samples (−2.5359 s)** | 0.982 |
| 5 | **−25 493 samples (−2.5493 s)** | 0.983 |

Direct effect on card 3: 7 of 50 steps had reference SNR ≥ 5 dB with the
windows as-is, 15 of 50 after shifting.

Note the old ceiling of 2.0 s was below the largest real offset, so even a
wired-up version of the original setting would have failed silently.

**Changed:**
- new `estimate_card_lags()` — band-passed cross-correlation of the reference
  channel, with the peak height reported and a weak peak refused rather than
  applied
- `consensus_schedule(..., lags=)` converts each card's detected steps onto a
  common time base *before* they are pooled
- `process_card(..., lag=)` shifts each window back onto that card's clock
- `align_max_lag_s` raised to 12.0 s, `align_min_corr = 0.30` added
- the lags and their correlation peaks are written to the bronze manifest

## Fix 2 — polarity decided on the noisiest end of the band  (`bronze.py`)

Segment 1 came out at Re Z ≈ **−460 mΩ·cm²** across its whole low-frequency
range — an inverted but otherwise perfectly good spectrum, which the passivity
gate then destroyed.

The old sign test averaged Re Z over `freqs >= max(1.0, median(freqs))`, the
top half of the band. Up there SNR falls to −40 dB and the per-channel
acquisition skew rotates the phase by 80–170° at 4.5 kHz, so the sign of Re Z
is essentially random.

**Changed:** new `_fix_polarity()` decides on the *lowest decade*, SNR-weighted,
using a weighted median and requiring >60 % of the weight to be negative before
flipping. A delay cannot rotate a 1 Hz point (0.04° at 100 µs) and |Z| is
largest there, so the sign is unambiguous. The weighted median plus the 60 %
requirement preserves the Schneider *et al.* case (ECS Trans. 25(1) 937, 2009)
that motivated the original choice — a genuinely negative low-frequency Re Z
from down-the-channel starvation will not trigger a flip.

## Fix 3 — temperature calibration never applied  (`bronze.py`)

Every card logged `temperature sensors unusable - using 58.4 C`. The FAMOS
channels are `Temp_1 … Temp_4`; lower-casing gives `temp_1`, but
`PlateCalibration` keys are `temp1 … temp4`. The underscore meant the lookup
never matched, in every run.

Second problem: `_segment_temperatures` ran **per card**. Only cards 1 and 3
carry temperature channels at all, so cards 2, 4 and 5 would still have fallen
back even with the key fixed — discarding the whole inlet-to-outlet gradient.

**Changed:**
- `_sensor_key()` strips to the digits and rebuilds the key
- `plate_temperatures()` collects every sensor on every card once and builds
  one field for the plate, passed into `process_card`
- an unmatched sensor or an implausible reading now warns instead of being
  swallowed

Recovered: temp1 = 58.72, temp2 = 58.71, temp3 = 59.28, temp4 = 59.52 °C,
interpolated to 58.71 … 59.51 °C across the segments.

## Fix 4 — passivity gate ran before the de-skew  (`silver.py`)

`process_segment` rejected `Re Z ≤ 0` above 1 Hz *before* `apply_delay`
corrected the phase. On segment 1 that cut 14 surviving points to 6, below the
8-point floor, and the segment was discarded as unmodellable — deleting exactly
the points the next line would have rotated back into the passive half-plane.

**Changed:** the |Z| magnitude bound stays where it is (a delay is all-pass, so
|Z| is invariant and safe to gate early); the sign test moved to immediately
after `apply_delay`. Points dropped there are counted separately and appear as
`dropped_N_non_passive_after_deskew` in the flags column.

## Fix 5 — `_on_grid` ignored a rejected grid fit  (`bronze.py`)

`_on_grid()` read `grid["f0"]` and `grid["ratio"]` without checking
`grid["ok"]`. On this dataset the geometric-grid fit was rejected, yet
`on_grid` was still being set — which swapped the SNR gate for the much looser
`snr_floor_db` on those points. Now returns `False` when the fit was rejected.

## Fix 6 — equal-area mode  (`config.py`, `r2d2_geometry.py`, `utils.py`)

New `equal_areas` flag / `--equal-areas`. Replaces the true per-segment areas
(0.678 … 8.470 cm², a factor of 12.5) with `A_CELL/72 = 4.235 cm²` for every
segment. `utils.segment_areas(cfg)` is now the single resolution point —
precedence `--areas` CSV > `--equal-areas` > true geometry — so silver and gold
can no longer weight the same plate differently.

This is a simplification, not a correction. Local ASR is area-free and does not
move at all (R_Ω is identical, 82.3 ± 23.5 mΩ·cm² either way). What moves is
everything that sums across segments:

| | true areas | equal areas |
|---|---|---|
| measured area | 289.0 cm² | 279.5 cm² |
| DC closure | 43.8 A (−2.6 %) | 45.7 A (+1.6 %) |

The setpoint agreement is coincidentally slightly better under equal areas, but
that is not evidence for it — the true geometry is what the plate has, and the
edge strips really are a twelfth the size of the interior blocks. Use equal
areas when you want every active segment weighted identically in a plate
average, and read the current closure with that in mind.

## Also worth knowing

- `FAMOS_PATTERN` is `Leepa_{leepa}_Current_{cond}_Test_{test}_Karte_*.DAT`,
  which does not match `RO2611976-01_...`. `discover_files` falls back to any
  `*.DAT` so it works, but the log line `5 file(s) matching 'Leepa_*...'` is
  misleading — update the template for this campaign.
- `stage_bronze.py` / `stage_silver.py` / `stage_gold.py` pickle each stage, so
  silver and gold re-run in ~80 s instead of re-reading 780 MB of `.DAT`.

## Remaining, not fixed

- **Segment 67** reads j = 0.4445 A/cm², three times the plate mean of 0.1500.
  Its DC channel level is 0.2825 V against a typical 0.13 V. That looks like a
  wiring or Abgleich-row problem on card 5, not electrochemistry — worth
  checking against the plate before trusting it.
- **Segments 15, 11, 31, 19** sit at R_Ω = 20–33 mΩ·cm² against a plate median
  of 88, all tier C. The arc does not close inside the band, so R_Ω there is an
  extrapolation.
- **The HF arc is open.** |Z| bottoms out near 90 mΩ·cm² at 3 kHz, so R_Ω is
  never measured directly at any segment — it is the model's intercept. The
  fix is a higher sampling rate, not a better inversion.
- **Tiers are still mostly C** (A = 2, B = 19, C = 45), driven by
  `dropped_N_uncertain` at the top of the band. The residual per-card
  differential skew is still 50–70 µs, which is 80–113° at 4.5 kHz.
