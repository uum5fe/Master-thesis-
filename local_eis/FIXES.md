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

---

# Recovering the top decade  (`hf_schedule.py`, September 2026)

Data: `RO2612025-01_Current_45A_Test_01_Karte_4`, fs = 50 kHz, 538 s, UC2 plus
segments 39–53.  The sweep in that file runs downward from **23.9 kHz to
0.48 Hz** at 10 points per decade.  The shipped pipeline recovered eleven
steps of it, topping out at 7.47 Hz.  Nothing was missing from the recording.

| path | steps | band recovered |
|---|---|---|
| shipped — UC2 channel, `f_max_hz` = 4500 | 11 | 0.478 – 7.47 Hz |
| same channel, cap lifted to 0.45·fs | 13 | 0.478 – 7.47 Hz |
| stacked segment ensemble, cap lifted | 21 | 0.478 – 189 Hz |
| + ladder extension, points/decade snapped | 42 | 0.478 – 18 900 Hz |

## Fix 7 — the detector was reading the channel the cell had emptied (`bronze.py`, new `hf_schedule.py`)

`consensus_schedule` ran `detect_schedule` on each card's reference channel,
chosen in `inventory_channels` as the UC* channel with the largest standard
deviation — the cell voltage.  The sweep is **galvanostatic**: the ac current
amplitude is set by the load and is constant across the sweep, so the
amplitude arriving there is

    |u_ref(f)| = |i_ac| · |Z_cell(f)|

and |Z_cell| falls by an order of magnitude from the bottom of the band to the
~45 mΩ·cm² minimum near 8 kHz.  The detector was being asked to find a tone
exactly where the cell had removed it.  No value of `min_snr_db` puts signal
back.

The segment channels behave the other way round: they measure current density,
and current is what the sweep imposes, so their tone amplitude is flat in
frequency.  There are ~14 per card, driven by the same tone at the same
instant, with independent front-end noise.  Above 1 kHz the stacked ensemble
measures **+11.2 dB narrowband over UC2** (+7.4 dB above 5 kHz, +4.3 dB above
10 kHz).

**Changed:** new `hf_schedule.py`, three layers, adding **no new estimator** —
layer 2 hands the work to `eis_local.detect_schedule` unchanged:

1. `polarity_aligned_reference()` standardises every segment channel, checks
   each sign against a provisional sum so a reversed sense pair cannot
   subtract, and adds them.
2. the pipeline's own detector runs on that trace.  **The old trace is pooled
   in, not replaced** — see "what the card set changed" below.
3. the geometric ladder `f_k = f0·r^-k` and its dwell law are fitted on the
   confident steps only, the missing rungs are predicted, and each prediction
   is accepted on a frequency-domain CFAR and a rank-1 (maximum-eigenvalue)
   test across the array — neither of which uses the ladder.

Stack **per card**, not across the plate: pooling all five scored 23/26 against
26/26 for one card, because the cards are not on a common time base until
`estimate_card_lags` has run and the residual offsets make the sum partially
destructive at the top of the band.  `consensus_schedule` still does the
cross-card vote.

## Fix 8 — `Step.valid` thresholded the wrong quantity  (`eis_local.py`)

`Step.valid` gated on `snr_db >= min_snr_db`, and `fit3` defines `snr_db` as
the tone amplitude over the residual rms across the whole Nyquist band.  That
is not what determines a phasor's precision — `N·γ` is (Rife & Boorstyn),
because a longer dwell beats the noise down.

Of 23 rungs located above 100 Hz on card 4, **10 pass the 5 dB gate and all 23
pass `sigma_rel_max` = 0.60**, with σ_rel between 0.3 % and 23 %.  The
11.95 kHz step reports −1.5 dB and has N·γ = 2459 — a 2.0 % phasor, thrown
away by a gate that was never meant to decide this.  `config.py` already
argues exactly this above `sigma_rel_max`; the criterion was simply applied in
silver, after bronze had discarded the step in `detect_schedule`.

**Changed:** `Step.valid` now applies `hf_schedule.crlb_usable`, with a new
`SNR_ABSOLUTE_FLOOR_DB = -20` backstop so that a step with no tone at all
still fails — σ_rel alone would accept pure noise given enough samples.

## Fix 9 — the band ceiling was a config constant, not Nyquist  (`config.py`)

A real card header reads `dx = 1.0e-5 s`, i.e. fs = 100 kHz on 16 channels, so
`cfg.f_hi(fs) = min(f_max_hz, 0.45·fs) = min(4500, 45000) = 4500` Hz.  The
converter had 45 kHz of headroom it was never asked for.

**Changed:** `f_max_hz` 4500 → 30000, and the `--f-max` default with it.
Necessary but **not sufficient**: on its own it took card 4 from 11 recovered
steps to 13, with the same 7.47 Hz top.

## Fix 10 — the five UC channels are five measurements of one voltage  (`bronze.py`)

Averaging segment impedances across cards is wrong — they are different
segments.  Averaging the five UC channels is not, and it pays exactly where it
is needed, because once detection has moved onto the segment ensemble the
reference is the weak phasor in `Z = K·A_ref/A_seg`.

**Changed:** new `pooled_reference_phasors()`, inverse-residual-variance
weighted (`w = N/r_rms²`, the Cramér–Rao weighting), pooling only cards whose
lag was *applied*, each rotated from its own multiplexer slot to slot 0 —
after which `process_card` records `ref_slot = 0` so silver's structural skew
model still reads a consistent geometry.  On the synthetic card set the median
combined SNR went **22.9 → 29.9 dB**.

## What the synthetic card set changed about the design

Run against `make_synth_famos.py`, whose reference amplitude is *flat* in
frequency, the first version of this work was a regression: 13 steps against
the shipped path's 27, and a decade of band lost.  Four things came out of it,
each now carrying a test in `test_hf_schedule.py`:

- **Neither trace dominates.**  Where the reference amplitude is flat, the old
  path locates the top rungs *more precisely* than the stack — 730.3 / 1169.6 /
  1873.2 / 3000.0 Hz exactly, against the stack's 807 / 1182 / 1972 / 3070.  So
  both candidate sets are pooled and the ladder and array tests arbitrate.
  Nothing the shipped path would have found can be lost.
- **A ladder must not be a subdivision of itself.**  Every step of a 10 ppd
  sweep also lies on a 20 ppd ladder, so a fit scored by "most steps on the
  grid" prefers the subdivision, invents the rungs between, and at the top of
  the band those sit a fraction of a DFT bin from a real tone — so both
  acceptance tests see the neighbour's leakage and pass.  Cost when unguarded:
  20 spurious steps.  Guard: the gcd of the observed rung indices.
- **The membership window has to carry the fit's own uncertainty.**  A ladder
  fitted at 4.9013 points/decade against a true 4.8891 — a good fit — runs 4 %
  out at the ends of the band, and a flat 2 % window then discards genuine
  steps.  `Ladder.tol_at` widens it by `|k − k̄|·σ_ln r`.  The ladder is also
  refitted once on its own members, which extends the lever arm from 69 Hz to
  1182 Hz and pins the spacing to 4.8931.
- **Two dwell laws are in common use.**  Fixed cycle count *or* fixed time.
  Assuming the first on a record that used the second mispredicts a
  high-frequency step's start by the length of the sweep.  Both are fitted and
  the better one kept.

## Result on the synthetic card set

| | shipped path | with `hf_use_ensemble` |
|---|---|---|
| schedule | 27 steps | 17 steps |
| true steps found | 17 / 18 | 17 / 18 |
| **spurious steps** | **10** | **0** |
| band | 1.63 – 3709 Hz (3709 is spurious) | 1.63 – **3000 Hz** (the true top) |
| median combined SNR | 22.9 dB | **29.9 dB** |

On a galvanostatic synthetic with short high-frequency dwells — the field
symptom in miniature — 22/34 true with 7 spurious topping out at 1325 Hz
becomes **30/34 true with 1 spurious topping out at 3166 Hz**.

## New configuration

    hf_use_ensemble: bool = True     # detect on the stacked segment ensemble
    hf_ladder_extend: bool = True    # predict-and-verify the missing rungs
    hf_ladder_snap_ppd: bool = True  # snap the fitted spacing to an integer
    hf_ladder_tol: float = 0.02      # base window for ladder membership
    hf_pool_reference: bool = True   # inverse-variance mean of A_uc across cards

All four are A/B switches: set them false and the pipeline takes the old path
exactly.

## Still not fixed

- **Detection gain is not estimation gain.**  The array recovers *which*
  frequency and *when*.  The per-segment impedance still rests on that one
  segment's phasor; what the array buys there is a known f in a known window,
  which drops the Cramér–Rao phase penalty from 6/(N·γ) to 1/(N·γ) and removes
  the runaway-fit mechanism.
- **Chain-response correction above 1 kHz** is still worth doing and is now
  more valuable, because there are finally points up there to correct.  It is
  orthogonal to everything here.
- **Two FAMOS dialects.**  The DASYLab-native export carries one `|CN` block
  per channel with float64 samples and a per-channel byte offset in `|CP`;
  `eis_local.FamosFile` looks for `7,32,<name>` inside a single `|CP` field
  with a hardcoded `<f4` and a 4·n_ch stride.  On the other dialect it raises
  `incomplete FAMOS header`, or — if a file carries both markers — silently
  reads garbage of exactly the right shape.  Worth confirming which dialect
  the cards the thesis processed are in.
- **Aliased steps above fs/2** could in principle be un-folded once the ladder
  is known, since an aliased tone's Nyquist zone is then predictable.  Whether
  anything survives depends entirely on the anti-alias filter in front of the
  Dewetron converters.
