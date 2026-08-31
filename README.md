# Locally-resolved EIS pipeline for a segmented PEM fuel cell

Processes multi-card imc FAMOS recordings of segment shunt voltages and cell
voltage into **per-segment local impedance spectra**, with the timing between
DAQ cards measured rather than assumed and the acquisition chain corrected as a
complex response rather than a scalar.

```
FAMOS .DAT (N cards)
   |
   |  BRONZE  eis/pipeline/bronze.py
   +-> measured inter-card skew + clock drift        [eis/sync]
   +-> time-domain alignment, verified                [eis/sync/resample.py]
   +-> shunt voltage -> local current density j(t)    [eis/calibrate.py]
   |
   |  SILVER  eis/pipeline/silver.py
   +-> Z_k(f) = U_cell(f) / I_k(f), with sigma(Z)     [eis/spectra.py]
   +-> complex instrument response, shared delay      [eis/hf.py]
   +-> Kramers-Kronig + stationarity + plate identity [eis/validate.py]
   +-> a status and a quality score for every segment
   |
   |  GOLD    eis/pipeline/gold.py
   +-> equivalent-circuit parameters with CIs         [eis/model]
   +-> tables, figures                                [eis/viz.py]
   +-> interactive plate view                         [eis/dashboard.py]
```

## The two problems this is built around

### 1. Timing between cards

Phase error from a timing offset is `phi = 2*pi*f*tau`. At the top of a 3.8 kHz
analysis band:

| Source | Magnitude | Phase error @ 3.8 kHz |
| --- | --- | --- |
| One sample at 10 kHz | 100 us | **137 deg** |
| Channel/interleave offset inside a card | 50 us | **68 deg** |
| Free-running clock, 10 ppm over 120 s | up to 1.2 ms | **several wraps** |
| Analog front-end delay | ~120 ns | 0.16 deg |

Merging cards by array index assumes the first row of every file is the same
instant. Nothing guarantees that. To keep the imaginary part accurate to
0.1 deg at 3.8 kHz the residual timing error must stay under **73 ns** — and
that has to be measured, not assumed.

**What makes it solvable.** Every card records its own copy of the *same* cell
voltage. That turns synchronisation into a direct measurement with no
electrochemical assumption: cross-correlate the two copies and read off the
delay.

Two consequences shape the design:

- **Pairing a segment with the cell voltage from its own card makes bulk card
  skew cancel exactly** — it appears in numerator and denominator alike. This
  is the accurate default (`uc_strategy: same_card` / `auto`).
- **Channel-to-channel skew inside a card does not cancel.** It sits directly
  between the segment current and the cell voltage, so it must be measured once
  (feed one signal into every input in parallel) and applied from the delay
  table.

### 2. The instrument is complex, and the calibration says it is real

With the timing budget met to a tenth of a degree, the largest remaining
high-frequency error was **twenty-six degrees**, and no amount of
synchronisation could have found it.

The via shunt is described by two real numbers, `H(T) = c0 + 1e-3*c1*T`. But
`H/A` is **24.8 microohms**, so half a nanohenry of loop inductance is a time
constant of 20-40 microseconds — three orders of magnitude above the 73 ns
timing budget, and 26 degrees at 3.8 kHz. It is invisible to every timing
diagnostic in the pipeline, because it is not a timing error.

| synthetic data, 0.6 nH shunt, everything else perfect | median error | at the top tone |
| --- | --- | --- |
| `H` treated as real | 1.46 % | **48.1 %** |
| `H` treated as complex | 0.18 % | **0.11 %** |

The whole high-frequency chain — and the method used to identify what is left
after it — is written up in
**[docs/HF_ACCURACY_METHOD.md](docs/HF_ACCURACY_METHOD.md)**.

## Quickstart

```bash
pip install -r requirements.txt

# self-contained demonstration on synthetic data with known ground truth
python run_pipeline.py --demo --plots        # or: python -m eis.pipeline.main --demo --plots

# real data
python -m eis.pipeline.main \
    --raw-dir /Volumes/.../Famos --measurement-id 2611976 \
    --shunt-csv cal/curr.csv --temp-csv cal/temp.csv \
    --shunt-inductance-nh 0.6 \
    --out out/2611976 --plots

# timing report only, no spectra
python -m eis.pipeline.main --raw-dir /Volumes/.../Famos --sync-only

# designed multisine: leakage-free synchronous analysis
python -m eis.pipeline.main --config config/example.yaml \
    --base-frequency 1.0 --tones 2,3,5,8,13,21,34,55,89,144,233,377,610
```

In a notebook or on Databricks:

```python
from eis.pipeline import load_config, run_measurement, scalar_map

cfg = load_config("config/example.yaml")
results = run_measurement(cfg)
frame = results["150A"].impedance_frame(cfg.geometry.segment_area_cm2)

# any per-segment scalar, static or frequency-resolved, through one call
rs   = scalar_map(results["150A"], "rs_hf", cfg.geometry.segment_area_cm2)
zmod = scalar_map(results["150A"], "z_mod@f", cfg.geometry.segment_area_cm2,
                  frequency_hz=1000.0)
```

## What the pipeline does at each stage

### Bronze — files to provably aligned, physically scaled signals

**Reading** (`eis/io/famos.py`) recovers the absolute start time (`|NT`), the
sample interval, the channel names and the payload geometry, and
**cross-validates** them: `n_samples * n_channels * 4` must equal the declared
payload, and the channel name count must equal the channel count. A mismatch
raises instead of falling back to defaults, because a wrong channel count
silently de-interleaves every signal.

**Synchronisation** (`eis/sync/`):

| Step | Method |
| --- | --- |
| Coarse delay | GCC-PHAT, whitening restricted to bins that carry signal, search bounded to plausible skew |
| Fine delay | Cross-spectral phase slope, weighted by `gamma^2/(1-gamma^2)`, de-rotated before unwrapping |
| Whole samples | Removed by slicing and iterated, so the fine stage always works on a sub-sample residual |
| Clock drift | Block-wise delay regressed against time; Theil-Sen slope with MAD outlier rejection |
| Correction | Integer shift + FFT phase ramp; polyphase resampler when the delay drifts |
| Verification | Re-measure after correction; closed-loop refinement until the residual is inside budget |
| Consistency | Triplet closure `tau_ab + tau_bc + tau_ca = 0` |

Three details that are easy to get wrong and are handled explicitly:

- **De-rotate before unwrapping.** Fitting `arctan2` output directly breaks as
  soon as the skew exceeds about half a sample.
- **Whiten only where there is signal.** Plain PHAT gives noise-only bins full
  weight; with a multisine it locks onto a random peak *with a
  confident-looking amplitude*.
- **Align in the time domain, before any FFT.** Rotating the final `Z` fixes
  the phase but not the coherence lost inside each window, and cannot express a
  delay that changes during the record.

When a card's cell-voltage channel is dead, the strongest segment channel
stands in as a timing proxy — with the shunt's own phase divided out first,
because a segment channel carries 25 us of shunt time constant that the
estimator would otherwise read as 25 us of card delay. That alone is worth a
factor of five on the proxy-timed card.

**Calibration** (`eis/calibrate.py`): `H(T) = c0 + 1e-3*c1*T` per segment,
evaluated at the **measured** temperature interpolated over the plate, then
`j = V_shunt / H(T)` and `I = j * A`. Evaluating `H` at a fixed nominal
temperature when the plate is 10 K away puts about 4 % onto every resistance.

### Silver — signals to spectra that are measurements

**Spectral estimation** (`eis/spectra.py`), three paths:

- **Coherence-gated Welch** rejects contaminated *windows* before averaging, on
  the excitation level, the response level and phase consistency. A 120 s
  fuel-cell record is not stationary; one load transient poisons the whole
  average and no output-side frequency gate can undo it.
- **Multi-resolution Welch** (the default when the excitation is unknown) gives
  each octave band the shortest window it can afford. One window length cannot
  serve four decades: a window long enough for eight periods of 1 Hz throws
  away almost all the averaging available at 3 kHz. Result: over four times the
  averages in the top decade, where the uncertainty was worst.
- **Synchronous DFT** for designed multisines: an integer number of base
  periods with a rectangular window puts each tone in exactly one bin.

**Estimators.** `H1 = S_iu/S_ii` is biased low by noise on the current — at
`gamma^2 = 0.9` it under-reads `|Z|` by 10 %. `H2` is biased high by the same
factor, so the pair brackets the truth and the width of the bracket is the
noise diagnostic. The default `Hv` is the scale-invariant total-least-squares
solution sitting between them.

**Uncertainty.** Every point carries
`sigma_|Z|/|Z| = sigma_phi = sqrt((1-gamma^2)/(2*gamma^2*n_eff))`, with `n_eff`
corrected for the correlation between overlapping windows, plus the measured
timing uncertainty added in quadrature. The **random** and **total** parts are
kept apart: a systematic rotates a spectrum coherently rather than scattering
it, so it belongs in the verdict on a fit and never in the weights of one.

**High-frequency chain** (`eis/hf.py`) — see
[docs/HF_ACCURACY_METHOD.md](docs/HF_ACCURACY_METHOD.md):

1. the via shunt as a complex element, `H(f) = H_dc*(1 + jw*L/R)`;
2. anti-alias filter mismatch between the segment and cell-voltage channels;
3. the residual delay shared by a card, identified from the *ensemble* by
   **regressing the fitted inductance on the fitted ohmic resistance** — the
   delay is common to the card while `L` is individual, so a non-zero delay
   shows up as a spurious `L`-vs-`Rs` proportionality whose slope is the delay;
4. optionally, in-plane crosstalk between neighbouring segments.

**Validation** (`eis/validate.py`). Lin-KK with automatic model order and
weighted residuals. The residual is returned as a **spectrum**, and its shape
names the fault:

| Residual signature | Diagnosis |
| --- | --- |
| Random, within a percent | Causal, linear, stationary |
| Systematic in Im, growing with f | Uncorrected time delay |
| Divergent at low frequency | Drift / non-stationarity |
| Isolated points off in both parts | Leakage or a contaminated tone |

A point counts as a violation only when it exceeds **both** the absolute
threshold and three times its own uncertainty — so an honestly imprecise
segment is not declared non-causal, and an over-precise one does not pass on a
systematic error. Also: split-half stationarity, and the whole-plate identity
`A_cell/Z_cell = sum_k A_k/Z_k`.

### Gold — parameters, maps, report

Weighted, log-parameterised, multi-start fitting over a ladder of circuits
(`R_L`, `R_L_1RQ`, `R_L_2RQ`, `R_L_1RQ_W`) selected by corrected AIC, with
parameter standard errors from the Jacobian. Parameters whose relative error
exceeds a threshold are flagged and hatched on the plate maps rather than
plotted as if they were known.

`Rs` comes from a weighted `Rs + jwL + B*(jw)^-n` fit over the top decade —
linear in its parameters, so one least-squares solve with an exact covariance —
rather than from the median of the highest real parts, which is biased by the
inductive tail it does not model.

## Every segment stays on the plate

Nothing is rejected. Every segment reaches the output carrying a
machine-written `status`, a list of `flags` and a `quality` score in [0, 1]:

| status | meaning |
| --- | --- |
| `ok` | passed everything |
| `low_coherence` | fewer usable points than the threshold |
| `kk_fail` | Kramers-Kronig residual beyond both the absolute and the statistical limit |
| `nonstationary` | the two halves of the record disagree |
| `no_calibration` | no via-shunt entry; the spectrum is in shunt volts per amp |
| `timing_unverified` | the card's time base could not be verified; the unknown delay is carried as uncertainty |
| `channel_fault` | no spectrum could be formed at all |

Frequency points below the coherence gate are **marked, not deleted**, so all
segments share one frequency grid — which is what makes the frequency-resolved
plate map, the whole-plate admittance sum and the crosstalk deconvolution
possible at all.

## Outputs

`impedance.parquet` — one row per (segment, frequency), every point:

```
segment, card, uc_channel, frequency_hz,
z_real_ohm, z_imag_ohm, z_mod_ohm, z_phase_deg,
z_real_mohm_cm2, z_imag_mohm_cm2,
coherence, sigma_rel, sigma_real_ohm, sigma_imag_ohm, n_eff, nperseg,
used, segment_status, segment_quality, physical_units,
estimator, method,
pipeline_version, param_hash, git_sha, created_utc, channel_delay_table
```

`segments.parquet` — per-segment summary: status, flags, quality,
high-frequency fit (`rs_hf` with its sigma, fitted inductance, crossover
frequency), applied instrument terms and the phase they contribute at `f_max`,
KK verdict / order / mu / residual / shape, stationarity, ECM model, parameters
and their sigmas.

`sync.parquet` — per card: measured skew and its uncertainty, clock rate,
whether drift was corrected, coherence, phase intercept, metadata offset,
**residual skew after alignment**, pass/fail and notes.

`hf_common_mode.parquet` — per card: the identified shared delay and its
uncertainty, the `L`-vs-`Rs` correlation before and after, the inductance
scatter before and after, whether it was clipped to the measured bound, and
whether it was applied. The evidence for the correction, not just its value.

**Figures** (`--plots`): spectra with error bars and ECM overlay,
synchronisation report, KK residuals, and a plate map for every configured
parameter — drawn with **all** segments, the doubtful ones desaturated and the
colour scale computed from the trustworthy ones. Plus small multiples of any
frequency-resolved map across the band, which separates spatially smooth
frequency-flat ohmic losses from the kinetic and transport terms that follow
the flow field.

**`plate_dashboard.html`** — one self-contained file, no server and no external
libraries: the plate heat map on the left, the clicked segment's Nyquist and
Bode with error bars on the right, a parameter selector, and a frequency slider
that redraws `|Z|` or the phase across the plate.

Every row carries `param_hash` + `git_sha`. Two runs with the same pair are the
same computation; changing any parameter changes the hash.

## Validation

`pytest tests/` — 58 tests. Everything is checked against synthetic FAMOS files
built from impedances the test knows exactly, with faults deliberately imposed.

| Property | Result |
| --- | --- |
| Constant skew, 0 to 31 samples | recovered to **< 73 ns** |
| Skew of 137 samples (many phase wraps) | recovered to < 200 ns |
| Clock drift, 6 ppm | recovered to **< 0.2 ppm**; residual after correction < 0.05 ppm |
| Uncorrelated channel | **rejected**, not silently given a number |
| Via-shunt inductance, uncorrected vs corrected | top-tone error **48.1 % -> 0.11 %** |
| Shunt de-embedded from the timing proxy | proxy-timed card **1.99 % -> 0.39 %** |
| Pooled KK delay on that card | **0.39 % -> 0.22 %** |
| Lin-KK model residual on a clean spectrum | **13.2 % -> 0.003 %** |
| Shared delay from L-vs-Rs decorrelation | 400 ns recovered as 445 +/- 445 ns; null case consistent with zero |
| Multi-resolution windows | > 4x the averages in the top decade |
| Crosstalk deconvolution | a 15 % mixing recovered to < 1 % |
| End to end: 4 cards, skew + drift + dead UC + intra-card offsets | **0.12 - 0.15 %** median error on \|Z\| |
| Same run, proxy-timed card | 0.26 % |
| Kramers-Kronig | 20/20 segments pass |
| ECM `Rs` recovery | within 2 % of truth |
| Segments dropped | **0** |

`python run_pipeline.py --demo` reproduces the end-to-end rows.

## Adapting to your data

1. **Check the reader on one real file.** `parse_famos_header(path)` should
   return the right channel count and names; the `scaling_note` field flags a
   `|CR` section that does not look like an identity transform.
2. **Run `--sync-only` first.** It reports skew, drift and closure without
   computing any spectra. That number tells you how large your real problem is.
3. **Measure the shunt loop inductance** and set `hf.shunt_inductance_nh`. It
   is the largest high-frequency error there is and the pipeline records
   `shunt_tau=...` in every output row when it is applied — and records nothing
   when it is not. A bridge measurement of one shunt at 100 kHz is enough.
4. **Declare the excitation.** If you use a designed multisine, put
   `base_frequency_hz` and `excitation_tones_hz` in the config — that switches
   on the leakage-free path.
5. **Measure the channel delay table** (one signal into every input in
   parallel) and fill in `acquisition.channel_delay_table`. Until then the
   pipeline records `assumed-simultaneous` in every output row.
6. **Fix polarity in config**, not by voting on the data.

Three hardware/protocol changes that remove problems rather than correcting
them:

- **A resistive reference channel per card** makes the front-end delay directly
  observable in every measurement and breaks the delay-vs-inductance
  degeneracy that no post-processing can fully resolve.
- **A cell-level reference spectrum** recorded alongside (a potentiostat across
  the whole cell) turns the common-mode delay from an inference into a
  measurement — it is the only anchor that sees the delay shared by *all*
  cards. Pass it as `reference_spectra` and the pipeline uses it first.
- **An odd-random-phase multisine** with a record length that is a whole number
  of base periods enables the synchronous path *and* separates nonlinear
  distortion from noise, for no extra measurement time.

## Layout

```
eis/
  io/famos.py            FAMOS reader with cross-validation
  sync/skew.py           GCC-PHAT + cross-spectral phase slope
  sync/drift.py          block-wise drift, robust regression
  sync/resample.py       fractional delay, affine resampling
  calibrate.py           H(T), temperature field, current density
  spectra.py             gated Welch, multi-resolution Welch, synchronous DFT,
                         H1/H2/Hv, uncertainty
  hf.py                  complex instrument response, shared-delay
                         identification, HFR fit, crosstalk
  validate.py            Lin-KK, stationarity, plate admittance sum
  model/ecm.py           weighted fitting, model selection, CIs
  viz.py                 static figures
  dashboard.py           interactive all-segment plate view
  pipeline/
    config.py            typed configuration, YAML, param_hash
    utils.py             logging, provenance, alignment, tables
    bronze.py            files -> aligned, scaled signals
    silver.py            signals -> spectra with uncertainty and verdicts
    gold.py              spectra -> parameters, maps, report
    main.py              orchestration and the command line
run_pipeline.py          launcher for eis.pipeline.main
tests/                   synthetic generator + 58 tests
config/example.yaml
docs/HF_ACCURACY_METHOD.md          the high-frequency approach
docs/EIS_PIPELINE_DEVELOPMENT_PLAN.md   the three-tier development plan
```

Notebooks contain no algorithms: every numerical step is an importable, tested
module.
