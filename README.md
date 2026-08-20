# Locally-resolved EIS pipeline for a segmented PEM fuel cell

Processes multi-card imc FAMOS recordings of segment shunt voltages and cell
voltage into **per-segment local impedance spectra**, with the timing between
DAQ cards measured rather than assumed.

```
FAMOS .DAT (N cards)
   -> measured inter-card skew + clock drift        [eis/sync]
   -> time-domain alignment, verified                [eis/sync/resample.py]
   -> shunt voltage -> local current density j(t)    [eis/calibrate.py]
   -> Z_k(f) = U_cell(f) / I_k(f), with sigma(Z)     [eis/spectra.py]
   -> Kramers-Kronig + plate-admittance checks       [eis/validate.py]
   -> equivalent-circuit parameters with CIs         [eis/model]
   -> tables + figures                               [eis/viz.py]
```

## The problem this is built around

Phase error from a timing offset is `phi = 2*pi*f*tau`. At the top of a 3.8 kHz
analysis band:

| Source | Magnitude | Phase error @ 3.8 kHz |
| --- | --- | --- |
| One sample at 10 kHz | 100 us | **137 deg** |
| Channel/interleave offset inside a card | 50 us | **68 deg** |
| Free-running clock, 10 ppm over 120 s | up to 1.2 ms | **several wraps** |
| Analog front-end delay | ~120 ns | 0.16 deg |

Merging cards by array index assumes the first row of every file is the same
instant. Nothing guarantees that, and the error it creates is three orders of
magnitude larger than the nanosecond-level analog delays that post-hoc
corrections usually target. To keep the imaginary part accurate to 0.1 deg at
3.8 kHz the residual timing error must stay under **73 ns** — and that has to
be measured, not assumed.

### What makes it solvable

Every card records its own copy of the *same* cell voltage. That turns
synchronisation into a direct measurement with no electrochemical assumption:
cross-correlate the two copies and read off the delay.

Two consequences shape the design:

- **Pairing a segment with the cell voltage from its own card makes bulk card
  skew cancel exactly** — it appears in numerator and denominator alike. This
  is the accurate default (`uc_strategy: same_card` / `auto`).
- **Channel-to-channel skew inside a card does not cancel.** It sits directly
  between the segment current and the cell voltage, so it must be measured
  once (feed one signal into every input in parallel) and applied from the
  delay table.

Synchronisation is still measured on every run, because the cancellation has
to be *verified*, because a card whose cell-voltage channel dies needs another
card's copy, and because the skew and drift are hardware diagnostics in their
own right.

## Quickstart

```bash
pip install -r requirements.txt

# self-contained demonstration on synthetic data with known ground truth
python run_pipeline.py --demo --plots

# real data
python run_pipeline.py \
    --raw-dir /Volumes/.../Famos --measurement-id 2611976 \
    --shunt-csv cal/curr.csv --temp-csv cal/temp.csv \
    --out out/2611976 --plots

# timing report only, no spectra
python run_pipeline.py --raw-dir /Volumes/.../Famos --sync-only

# designed multisine: leakage-free synchronous analysis
python run_pipeline.py --config config/example.yaml \
    --base-frequency 1.0 --tones 2,3,5,8,13,21,34,55,89,144,233,377,610
```

In a notebook or on Databricks:

```python
from eis.config import load_config
from eis.pipeline import run_measurement

cfg = load_config("config/example.yaml")
results = run_measurement(cfg)
frame = results["150A"].impedance_frame(cfg.geometry.segment_area_cm2)
# spark.createDataFrame(frame).write.mode("overwrite") \
#     .option("replaceWhere", "condition = '150A'").saveAsTable("...eis_gold_impedance")
```

## What the pipeline does at each stage

### 1. Reading (`eis/io/famos.py`)

Recovers the absolute start time (`|NT`), the sample interval, the channel
names and the payload geometry, and **cross-validates** them:
`n_samples * n_channels * 4` must equal the declared payload, and the channel
name count must equal the channel count. A mismatch raises instead of falling
back to defaults, because a wrong channel count silently de-interleaves every
signal. The `|CR` trailing fields are parsed and surfaced but never
auto-applied — the dialect is undocumented for these files, and guessing wrong
either corrupts every sample or silently rescales every impedance.

### 2. Synchronisation (`eis/sync/`)

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
  weight; with a multisine — seventeen tones among tens of thousands of empty
  bins — it locks onto a random peak *with a confident-looking amplitude*.
- **Align in the time domain, before any FFT.** Rotating the final `Z` by
  `exp(-j*2*pi*f*tau)` fixes the phase but not the coherence lost inside each
  window, and cannot express a delay that changes during the record.

When a card's cell-voltage channel is dead, the strongest segment channel
stands in as a timing proxy. That recovers whole-sample alignment but not the
nanosecond budget — a segment's own impedance phase is indistinguishable from
a small delay — so the reported uncertainty widens to ~10 us and says so.

### 3. Calibration (`eis/calibrate.py`)

`H(T) = c0 + 1e-3*c1*T` per segment, evaluated at the **measured** temperature
interpolated over the plate, then `j = V_shunt / H(T)` and `I = j * A`.
Evaluating `H` at a fixed nominal temperature when the plate is 10 K away puts
about 4 % onto every resistance — the same order as the segment-to-segment
variation being resolved. Falling back to a constant is allowed but flagged as
a degraded mode.

### 4. Impedance (`eis/spectra.py`)

- **Coherence-gated Welch** rejects contaminated *windows* before averaging,
  on the excitation level, the response level and phase consistency. A 120 s
  fuel-cell record is not stationary; one load transient poisons the whole
  average and no output-side frequency gate can undo it.
- **Synchronous DFT** for designed multisines: an integer number of base
  periods with a rectangular window puts each tone in exactly one bin. No
  scalloping, no picket-fence error, no window correction.
- **Estimators.** `H1 = S_iu/S_ii` is biased low by noise on the current — at
  `gamma^2 = 0.9` it under-reads `|Z|` by 10 %. `H2` is biased high by the same
  factor, so the pair brackets the truth and the width of the bracket is the
  noise diagnostic. The default `Hv` is the scale-invariant total-least-squares
  solution sitting between them. (Naive TLS on the raw spectral matrix is *not*
  scale invariant and silently collapses onto `H1` when current and voltage
  differ by orders of magnitude — see the note in `transfer_estimators`.)
- **Uncertainty.** Every point carries
  `sigma_|Z|/|Z| = sigma_phi = sqrt((1-gamma^2)/(2*gamma^2*n_eff))`, with
  `n_eff` corrected for the correlation between overlapping windows. These
  become the weights for Kramers-Kronig and circuit fitting, and the error bars
  on every plot.

### 5. Validation (`eis/validate.py`)

Lin-KK with automatic model order (Schoenleber mu-criterion) and weighted
residuals. The residual is returned as a **spectrum**, and its shape names the
fault:

| Residual signature | Diagnosis |
| --- | --- |
| Random, within a percent | Causal, linear, stationary |
| Systematic in Im, growing with f | Uncorrected time delay |
| Divergent at low frequency | Drift / non-stationarity |
| Isolated points off in both parts | Leakage or a contaminated tone |

Also: split-half stationarity, and the whole-plate identity
`A_cell/Z_cell = sum_k A_k/Z_k`, which tests the calibration chain, the
polarity map and the delay corrections at every frequency simultaneously.

`kk_optimal_delay` can refine a residual delay from the KK residual, but only
inside a bound taken from the measured skew — over a finite band a delay is
degenerate with an inductance (`L = -Rs*dt` to first order), so an
unconstrained search will happily eat the cell's real inductance.

### 6. Modelling (`eis/model/ecm.py`)

Weighted, log-parameterised, multi-start fitting over a ladder of circuits
(`R_L`, `R_L_1RQ`, `R_L_2RQ`, `R_L_1RQ_W`) selected by corrected AIC, with
parameter standard errors from the Jacobian. Parameters whose relative error
exceeds a threshold are flagged and hatched on the plate maps rather than
plotted as if they were known. A two-arc model forced onto a spectrum with gas
transport returns confident, meaningless numbers; the ladder makes that visible.

## Outputs

`impedance.parquet` — one row per (segment, frequency):

```
segment, card, uc_channel, frequency_hz,
z_real_ohm, z_imag_ohm, z_mod_ohm, z_phase_deg,
z_real_mohm_cm2, z_imag_mohm_cm2,
coherence, sigma_rel, sigma_real_ohm, sigma_imag_ohm, n_eff,
estimator, method,
pipeline_version, param_hash, git_sha, created_utc, channel_delay_table
```

`segments.parquet` — per-segment summary: status and rejection reason, `Rs`,
KK verdict / order / mu / residual / shape, ECM model, parameters and their
sigmas, `chi2_reduced`, temperature and shunt factor used.

`sync.parquet` — per card: measured skew and its uncertainty, clock rate,
whether drift was corrected, coherence, phase intercept, metadata offset,
**residual skew after alignment**, pass/fail and notes.

Figures (`--plots`): spectra with error bars and ECM overlay, synchronisation
report, KK residuals, plate maps of `Rs`, `Rp` and coherence.

Every row carries `param_hash` + `git_sha`. Two runs with the same pair are the
same computation; changing any parameter changes the hash.

## Validation

`pytest tests/` — 33 tests. Everything is checked against synthetic FAMOS files
built from impedances the test knows exactly, with timing faults deliberately
imposed.

| Property | Result |
| --- | --- |
| Constant skew, 0 to 31 samples | recovered to **< 73 ns** |
| Skew of 137 samples (many phase wraps) | recovered to < 200 ns |
| Clock drift, 6 ppm | recovered to **< 0.2 ppm**; residual after correction < 0.05 ppm |
| Uncorrelated channel | **rejected**, not silently given a number |
| End to end: 4 cards, skew + drift + dead UC + intra-card offsets | **0.12 – 0.15 %** median error on \|Z\| |
| Same run, proxy-timed card | 0.41 %, inside its stated uncertainty |
| Kramers-Kronig | 20/20 segments pass at < 1 % |
| ECM `Rs` recovery | within 2 % of truth |
| Predicted `sigma` vs observed scatter | agree within a factor of 3 |

`python run_pipeline.py --demo` reproduces the end-to-end row.

## Adapting to your data

1. **Check the reader on one real file.** `parse_famos_header(path)` should
   return the right channel count and names; the `scaling_note` field flags a
   `|CR` section that does not look like an identity transform.
2. **Run `--sync-only` first.** It reports skew, drift and closure without
   computing any spectra. That number tells you how large your real problem is.
3. **Declare the excitation.** If you use a designed multisine, put
   `base_frequency_hz` and `excitation_tones_hz` in the config — that switches
   on the leakage-free path and is the single largest accuracy gain available.
4. **Measure the channel delay table** (one signal into every input in
   parallel) and fill in `acquisition.channel_delay_table`. Until then the
   pipeline records `assumed-simultaneous` in every output row.
5. **Fix polarity in config**, not by voting on the data.

Two hardware/protocol changes that remove problems rather than correcting them:

- **A resistive reference channel per card** makes the front-end delay directly
  observable in every measurement and breaks the delay-vs-inductance
  degeneracy that no post-processing can resolve.
- **An odd-random-phase multisine** with a record length that is a whole number
  of base periods enables the synchronous path *and* separates nonlinear
  distortion from noise, for no extra measurement time.

## Layout

```
eis/
  config.py        typed configuration, YAML, param_hash
  io/famos.py      FAMOS reader with cross-validation
  sync/skew.py     GCC-PHAT + cross-spectral phase slope
  sync/drift.py    block-wise drift, robust regression
  sync/resample.py fractional delay, affine resampling
  calibrate.py     H(T), temperature field, current density
  spectra.py       gated Welch, synchronous DFT, H1/H2/Hv, uncertainty
  validate.py      Lin-KK, stationarity, plate admittance sum
  model/ecm.py     weighted fitting, model selection, CIs
  pipeline.py      orchestration
  viz.py           figures
run_pipeline.py    CLI
tests/             synthetic generator + 33 tests
config/example.yaml

databricks/local_eis/   the pipeline that runs in the Databricks workspace:
                        bronze/silver/gold on FAMOS, the CSV evaluation path,
                        both plate maps, and the runner notebook

docs/EIS_PIPELINE_DEVELOPMENT_PLAN.md    three-tier development plan
docs/GEN2_PLATE_AND_CSV_PIPELINE.md      the gen2 plate and the CSV path
```

Notebooks contain no algorithms: every numerical step is an importable, tested
module.

## Two plates, two source formats

The Databricks pipeline covers both hardware revisions of the R2-D2 measuring
plate and both measurement file formats. Neither is a default you can ignore:

| | |
| --- | --- |
| **Plate** | `gen1` = green / Kashyyyk, `gen2` = blue / Naboo. Same 45×20 pad grid and 72 segments; segments 37…72 are cut differently, so their areas — and those of the interior segments that gave pad rows to them — differ. Selecting the wrong one does not fail, it draws the right numbers on the wrong squares. |
| **Source** | `famos` = five free-running cards, which need the measured inter-card synchronisation. `csv` = one logger with one clock, where that stage is not merely unnecessary but undefined; it is a separate pipeline sharing only the geometry and the Abgleich. |

See [`docs/GEN2_PLATE_AND_CSV_PIPELINE.md`](docs/GEN2_PLATE_AND_CSV_PIPELINE.md)
for the gen2 reconstruction and its evidence, what the Abgleich files say, the
−11°-at-4.5 kHz measuring-chain roll-off that has never been corrected, and
what the CSV path does instead of synchronisation.
