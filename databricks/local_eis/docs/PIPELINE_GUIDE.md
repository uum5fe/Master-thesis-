# Local EIS pipeline — the complete guide

**For:** whoever takes over this software. It assumes you know what impedance spectroscopy is. It does not assume you have seen the code.
**Covers:** the final version on branch `claude/inspiring-bohr-krz72l` (October 2026), folder `databricks/local_eis/`.

The slide deck that goes with this guide is linked from the README.

---

## Contents

1. [What the software does](#1-what-the-software-does)
2. [Running it on Databricks in five minutes](#2-running-it-on-databricks-in-five-minutes)
3. [Where the input data comes from](#3-where-the-input-data-comes-from)
4. [The fixed evaluation settings, the complete band and its corrections](#4-the-fixed-evaluation-settings-the-complete-band-and-its-corrections)
5. [Folder and module map](#5-folder-and-module-map)
6. [End-to-end data flow](#6-end-to-end-data-flow)
7. [Bronze: raw waveforms → raw impedance](#7-bronze-raw-waveforms--raw-impedance)
8. [Silver: correct, gate, model, grade](#8-silver-correct-gate-model-grade)
9. [Gold: the plate as a field](#9-gold-the-plate-as-a-field)
10. [Checks: is the plate physically coherent?](#10-checks-is-the-plate-physically-coherent)
11. [Formula sheet](#11-formula-sheet)
12. [Output files, one by one](#12-output-files-one-by-one)
13. [The runner notebook, cell by cell](#13-the-runner-notebook-cell-by-cell)
14. [The result cache](#14-the-result-cache)
15. [Plate geometry and the 36-tile statistics](#15-plate-geometry-and-the-36-tile-statistics)
16. [Configuration reference](#16-configuration-reference)
17. [Command line, tests and development](#17-command-line-tests-and-development)
18. [Known limitations and open items](#18-known-limitations-and-open-items)
19. [Troubleshooting](#19-troubleshooting)
20. [Glossary](#20-glossary)

---

## 1. What the software does

A Bosch R2-D2 segmented plate sits on one side of a PEM fuel cell. Its 900 pads (45 × 20) are wired into **72 segments**. Each segment carries its own current through a shunt/via, so its local current density can be measured.

A Gamry potentiostat runs a **galvanostatic EIS sweep** on the whole cell. While it does, five imc **FAMOS** cards record raw waveforms with 16 channels each, typically at 25 kHz:
- one cell-voltage channel per card (`UC1`/`UC2`);
- the segment current channels (named `1`…`72`);
- four plate temperature sensors (`Temp_1`…`Temp_4`).

The software turns those waveforms into a **local impedance spectrum for every segment**. From the spectra it derives, per segment:
- the ohmic resistance R_ohmic;
- Re Z at 1 kHz;
- the charge-transfer resistance R_ct;
- the mass-transport resistance R_mt;
- the polarisation resistance R_pol;
- the DC current density j;
- the temperature.

It draws them as **plate heat maps**, compares the sum of all segments with the Gamry whole-cell spectrum, and runs a set of **plausibility checks**.

The core relation is simple:

> **Z_seg(f) = K_seg(T) · A_U(f) / A_seg(f)**
>
> A_U is the cell-voltage phasor, A_seg the segment's current-channel phasor, both at the excitation frequency f. K_seg(T) is the segment's calibration constant (V per A/cm²) at its temperature. The result is an area-specific impedance in Ω·cm².

Most of the code exists to make that ratio **trustworthy**:
- **the right time windows:** which stretch of a 2-minute record holds which frequency;
- **the right frequencies:** taken from the Gamry log;
- **the right timing between channels:** five cards armed at different moments, a multiplexed ADC, and a per-channel chain delay;
- **honest bookkeeping:** every point that is dropped is dropped for a recorded reason;
- **an end-to-end cross-check:** the parallel sum of all segments against the Gamry whole-cell measurement.

The architecture is the "medallion" pattern:

```
FAMOS .DAT (Volumes)  ─┐
 or datago Delta table ─┤
                       ▼
  BRONZE   find steps, align cards, fit phasors, Z = K·A_U/A_seg     → bronze/
                       ▼
  SILVER   de-skew, channel lag, gates, DRT model, R_ohmic, KK, tiers → silver/
                       ▼
  GOLD     R_ct/R_mt split, spatial field, plate maps, 36-tile stats  → gold/
                       ▼
  CHECKS   Gamry whole-cell comparison, chain response, plausibility  → run folder
                       ▼
  RUNNER   cache on the Volume, interactive viewers, ECM, pol. curves
```

Each layer only adds; nothing upstream is overwritten. Every layer writes its own folder, so a later layer can be re-run from saved files (`--reevaluate`).

---

## 2. Running it on Databricks in five minutes

1. **Import the zip into an EMPTY workspace folder.** Do not import on top of an older copy. A mixed folder (new `silver.py`, old `gold.py`) is the one way to get silently wrong maps, and the notebook's build check stops the run if it sees one (§13, cell 2).
2. Open **`Local EIS Pipeline Runner`** and attach a cluster (any ML runtime; numpy, scipy, matplotlib, pandas and plotly are already there).
3. Run the first three cells. The widget bar appears:

   | Widget | What to choose |
   |---|---|
   | **Order ID (Leepa)** | the order number, e.g. `2612030` (the list is read from datago and the Volumes) |
   | **Measurement file format** | `famos` (normal) or `csv` (the bench tool's exported spectra) |
   | **CSV folder** | only for `csv`: which campaign folder |
   | **Conditions (multi-select)** | e.g. `45A`, `450A`, or `ALL` |
   | **Exclude segments** | leave empty unless a segment is known broken (it is then not read at all) |
   | **Rebuild from neighbours** | segments whose own data is discarded and rebuilt from adjacent segments |
   | **Fill unmeasured segments** | `no` (default) / `yes` |
   | **Run mode** | `cache` (use stored results when the settings match, otherwise run), `rerun` (always run), `cache-only` (never run) |
   | **Write results to cache** | `yes` (store on the Volume) / `no` (trial run) |

   **There are no band, SNR, profile or stage dropdowns any more.** Those settings are fixed (§4), and the notebook removes the old widgets from the bar.
4. **Run all.** The "Run plan" table shows, per condition, whether the result is loaded from the cache or computed (5–10 min per condition). The run cell prints the pipeline log, including the plausibility verdicts.
5. Scroll to the result cells: Nyquist/Bode, Gamry vs aggregate, plate heat maps, ECM fits, summary table and polarisation curves. Each is **one plot with drop-downs** (condition, parameter, view).

After a code change, always use **Run mode = `rerun`**. The cache key includes every setting but not the code version.

---

## 3. Where the input data comes from

All Volumes paths are under `/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev` (`_EV_ROOT` in the notebook).

| Input | Source | Notes |
|---|---|---|
| **FAMOS waveforms** | **1st: datago** (Delta tables `ps_xplatform_dev.rvadvtec_ops.datago_advtec_values_delta`, `…_metadata`, `…_generalproperties`). **2nd: Volumes** `…/Famos/Leepa_<id>_Current_<cond>_Test_01_Karte_<n>.DAT` | See "datago first" below |
| **Gamry whole-cell sweeps** `.DTA` | `…/RO<id>_Gamry`, `…/<id>_Gamry`, or the shared `…/Gamry` | The sweep's build token (e.g. `V26_092`) must match the order's. A folder holding several builds and no known one is refused rather than guessed. |
| **Current calibration** `curr.csv` | `/Workspace/Users/uum5fe@bosch.com/curr.csv` | 72 rows `c0;c1`. K(T) = c0 + 1e-3·c1·T |
| **Temperature calibration** `temp.csv` | `/Workspace/Users/uum5fe@bosch.com/temp.csv` | 4 rows. T = (u − c0)/c1 |
| **Chain gain file** | `/Workspace/Users/uum5fe@bosch.com/chain_gain_gen1.csv` | Built once from the Abgleich `bode/` sweeps by the "Chain response" cell. It is the current amplifiers' own response (−2.5° at 1 kHz). |
| **Abgleich folder** | found automatically, or set `ABGLEICH_DIR` | `coefficients/curr.csv`, `coefficients/temp.csv`, `bode/*.DTA`, `Step*_<T>Grad.csv` |
| **Bench log** (MF4) | `…/Gamry/*RO<id>*.mf4` | Used for the DC current closure and the test-bench viewer |
| **CSV spectra** (alternative source) | `…/csv_files/<campaign>/<condition folder>` | The bench tool's own evaluation, read by `csv_pipeline.py` |

**The `uum5fe` paths are personal.** A new owner must copy `curr.csv`, `temp.csv` and `chain_gain_gen1.csv` into their own workspace and change the three paths in the notebook (search for `uum5fe`; cells 2, 3, 4 and 8).

### Datago first, Volumes as fallback (cell 5)

The "Datago Source" cell decides automatically where each condition is read from.

`discover_famos_file_ids(LEEPA)` does the lookup:
- It queries the metadata tables for `orderId = 'RO<leepa>'` and `measurement_type IS NULL` (FAMOS files carry no type).
- It groups the files by start minute and keeps groups of **exactly five** (one per card).
- It maps the groups, in time order, to conditions. The mapping uses the Volumes file names if they cover every group; otherwise it assumes the protocol order 45 → 60 → 150 → 450 A.

What happens next:
- **If the map is not empty**, `DATA_SOURCE = 'datago'`. `bronze.FamosFile` and `bronze.discover_files` are replaced by shims that read the waveforms with Spark SQL (`DatagoFamosFile`, same interface as the binary reader).
- **If the map is empty, or the query fails** (no table access, for example), `DATA_SOURCE = 'volumes'` and the `.DAT` files are read directly.

The run cell prints the source per condition. Volumes is about 5 s per card; datago is about 30–60 s per card.

Two datago caveats (see §18):
- the condition mapping is a heuristic;
- the ADC slot order of the channels is approximated, because Delta rows carry no header order.

---

## 4. The fixed evaluation settings, the complete band and its corrections

| Setting | Value | Where |
|---|---|---|
| Evaluated band | **complete**: 0.15 Hz – 0.49·fs (12.25 kHz at 25 kHz, i.e. up to the 11.95 kHz step) | runner `BAND = 'complete'` → `FULL_BAND = True`, `EVAL_F_MAX = 0` |
| Alternative | `BAND = 'trusted'`: 0.15 Hz – 1.2 kHz, where all five cards agree with no correction | runner `BAND` |
| Step-detection ceiling | 0.45·fs | `f_max_hz = 30000`, `f_hi_frac_fs = 0.45` |
| Phasors | maximum-likelihood card frequency + linear least-squares sine fit at it (`phasor_method = "card_ml"`) | bronze |
| Point uncertainty | Cramér–Rao bound of every phasor (`uncertainty_model = "crlb"`) | silver |
| R_ct, R_mt, R_pol | **Bayesian DRT** (GP prior, closed-form posterior, CRLB-weighted), split at `tau_split_kinetic_s` | `param_method = "drt"` |
| Circuit fit | R_ohmic + jωL + (R_ct‖CPE) + (R_mt‖CPE), run **beside** the DRT: `R_ct_ecm`, `R_mt_ecm`, `R_pol_ecm` | `ecm_compare = True` |
| R_ohmic | the high-frequency real-axis crossing (complete band); extrapolated to Im Z = 0 when a band ends before it | `r_ohmic_method = "auto"` |
| Plate maps | only 2D interpolated maps, one per parameter | `heatmap_style = "interpolated"` |

### 4.1 What the raw-signal audit found (2612030, 450 A, all five cards)

The FAMOS files were audited channel by channel, independently of this pipeline:

- **Each card runs on its own clock**: cards 2–5 are 10–36 ppm slow against card 1, up to 5.2 ms over the sweep. Z = K·U/u is formed on ONE card, so this cancels in every impedance; it matters only when cards are compared with each other.
- **Up to ~1 kHz all five cards agree**: the shared cell voltage UC2 within ±3 % and ±2°.
- **Cards 3–5 pick the excitation up inductively** (~7 nH, visible even on their temperature channels). On UC2 this is a 7–18 % dip at 2.4–3.8 kHz and a ×3.5–4.3 rise by 9.5 kHz. That was the "filter response" of cards 3–5, and the "~11 µs lead" of cards 1–2 seen against them.
- **The segment channels differ from each other above ~1.5 kHz**: ±13° at 1.9 kHz, ±26° at 3 kHz, downstream of the plate amplifiers (which match to ±0.3°).
- **UC2 gain**: +1.2 % on card 3, +0.9 % on card 5 against card 1; card 4's UC2 has a −2.3 mV input offset.
- **Faults**: segment 33 reads 18 % of its neighbours' current; segment 58 hit the ADC full scale once.
- **Interference**: 150 Hz mains on UC2 of cards 4–5 (corrupts the 150 Hz step there); card switching tones at 1000 / 1012.5 Hz; the cell's own 1.75 Hz current see-saw between outlet and inlet.
- **Aliasing**: the 15–30 kHz steps arrive aliased (no effective anti-aliasing); they are outside the band.
- **Current formula**: j = u/K (Bosch transfer design; confirmed by the Kashyyyk DC steps). The pipeline has always used it: Z = K·U/u is the same thing.

### 4.2 The corrections, in the order they act

| # | Correction | What it removes | Module |
|---|---|---|---|
| 1 | ADC slot delay (slot − slot_UC2)/(n_ch·fs) | the multiplexer's sampling skew (2.5 µs/slot; the audit confirms 2.41 ± 0.75 µs) | silver skew |
| 2 | Chain gain file | the plate amplifier response per segment (Abgleich bode sweeps) | `gain_file` |
| 3 | Card voltage gain, **from the AC steps 1–300 Hz** (phasor SNR ≥ 30 dB) | UC2 gain per card (+1.2 % / +0.9 %); a DC ratio is reported but never applied (it cannot tell a gain from an input offset) | `card_gain.py` |
| 4 | **UC2 magnitude against the cleanest card** above 300 Hz, step by step | the inductive pickup on cards 3–5's UC2 (magnitude); needs no timing between cards | `card_gain.hf_response`, `uc_hf_equalise`, `uc_reference_card = "Karte_1"` |
| 5 | Per-channel lag | the per-channel delays (−57…+76 µs), as delays below the coherent limit | `channel_lag.py` |
| 6 | Per-card phase above 1 kHz against the Gamry sweep | the card-common phase above the coherent limit (pickup phase, card response) | `silver.gamry_card_phase` |
| 7 | Points dropped: ADC full scale; coherent background at the step frequency before the sweep > 5 % | clipped windows; mains / switching tones / the cell's oscillation at a step | bronze, gate `background` |
| 8 | Segments left out: DC current < 0.5× or > 2× its neighbours | channel faults (segment 33) — rebuilt from neighbours when `FILL_GAPS` is on | `silver.dc_share_faults` |

### 4.3 What the corrections cannot do

- **Correction 6 anchors each card's phase to the Gamry above 1 kHz.** Above that frequency the phase of the local spectra therefore follows the Gamry sweep; it is not an independent measurement there. The Gamry's own top decade is ~0.75 nH of wiring on a 0.15 mΩ cell.
- **The per-channel spread above ~1.5 kHz** (±13–26°) is removed only as far as it is a delay (correction 5). What each channel does beyond that is measurable only with an **injected calibration sweep**: one sine, 1 Hz – 12 kHz, into UC2 and every segment input of each card at once, with the same DASYLab setup. Its result is a per-channel response file in the `gain_file` format (segment, f, re, im); the pipeline applies it as correction 2 with no code change, and correction 6 can then be switched off (`gamry_card_phase = False`).
- **Hardware for the next measurements**: twist or re-route the UC2 and segment leads on cards 3–5 (pickup); give the five cards a common clock or a shared pilot tone; end the sweep at 11.95 kHz; ≥ 8 periods per step below 10 Hz; scale the excitation with the current.

With `BAND = 'trusted'` none of corrections 4 and 6 is needed: up to 1.2 kHz all five cards agree as recorded.

### 4.4 The multi-condition review (45 / 60 / 150 / 450 A, all corrections on)

What the four runs of 2612030 still showed, and what was done:

- **Segment 36 was 4.1 % low at every frequency.** Its Abgleich sweep has a glitch at its first point (1 Hz), and every sweep used to be normalised to that point. Sweeps are now normalised to their **median over 5–500 Hz** (`gamry_dta.GamrySweep.z_flat`). Re-run the "Chain response" cell once to rewrite `chain_gain_gen1.csv`; the file's content is in the cache key, so the next runs are re-evaluated.
- **A ~19° phase spread at 11.95 kHz and card |Z| steps up to ±13 % remained above 1 kHz.** Per segment the phase against the plate is a straight line in f over 2–12.5 kHz (0.9° rms after one line, 0.4° after a quadratic); per card the magnitude step grows with f (card 1 +15 %, card 3 −13 % at 7–13 kHz). Both are **identical at all four currents** (r = 1.00), so they are the acquisition, not the cell, and a regional ohmic difference cannot make them (it is flat in f). `checks/hf_response.py` measures them on all conditions, validates leave-one-current-out and writes a frozen per-segment response; silver divides by it after the per-card phase (`hf_response_file`, runner `HF_RESPONSE = 'auto'`, cell "High-frequency response across conditions").

  | held-out current | phase sd at 11.95 kHz | largest card step ≥ 4 kHz |
  |---|---|---|
  | 45 A | 14.3° → 0.9° | 13.8 % → 2.2 % |
  | 60 A | 14.3° → 1.3° | 12.7 % → 2.9 % |
  | 150 A | 14.0° → 0.8° | 13.5 % → 1.9 % |
  | 450 A | 14.3° → 2.2° | 13.3 % → 4.6 % |

  Confirmed end to end: 450 A re-evaluated with the file built from 45/60/150 A only gives 2.2° and 4.6 %.
- **The per-segment scale factors (sd 5.6 %, 28 segments repeatable) are not applied.** If they were K errors, |Z| and j_dc would be anti-correlated with slope −1 at every current; they are at 450 A (r = −0.88) but hardly at 45–60 A (r = −0.4) — the high-current anticorrelation is the cell redistributing current. They stay reported (`segment_scale.py`); a cable swap between two channels tells a channel factor (it moves) from a segment one (it stays).
- **Card j_dc**: card 4 goes from +5.4 % (45 A) to −4.4 % (450 A) — load-dependent, the cell; card 3 is +3…5 % at every current — a candidate K offset for that card, to check with the same swap test.

**Card diagnostics against the Gamry** (`checks/card_diagnostics.py`, runner cell of the same name): per operating point, the median Nyquist, Bode |Z| and Bode phase of every card with the Gamry in black, and the card-to-Gamry transfer H = Z_card/Z_Gamry. `cross_current.png` puts H of every card at every current side by side. Read it like this: a deviation that is the same at every current comes from the acquisition; one that every card shares comes from the chain or the comparison; one that changes with current and differs between cards comes from the cell. On 2612030 (before the hf_response file):

| finding | 45 A | 60 A | 150 A | 450 A | verdict |
|---|---|---|---|---|---|
| plate − Gamry, Re, 100–3000 Hz [mΩ·cm²] | +9.8 | +8.1 | +11.0 | +12.1 | frequency-dependent, see below; **not** a series R → `uc_series_mohm_cm2` stays 0 |
| cards 3–5 \|H\| at 10 kHz | ≈ −15 % | ≈ −15 % | ≈ −15 % | ≈ −15 % | card hardware → `hf_response` |
| card phases < 1 kHz | ±3° | ±3° | ±3° | ±3° | no synchronisation error |
| outlet cards 1/4/5, 3–10 Hz | small | small | larger | +20…+60 %, −10…−20° | the cell (outlet mass transport) |

**Series resistance tested and rejected (final 150 / 450 A runs, `HF_RESPONSE = 'auto'`, `UC_SERIES_MOHM_CM2` = 0 and 10).** The plate-minus-Gamry offset is the same at every current but not the same at every frequency: about +25 below 10 Hz, +7…+12 from 20 Hz to 1 kHz, and −1…+2 at the Gamry's HFR crossing (3.4–3.8 kHz). Compared like for like, uc_series = 0 already gives the Gamry's HFR:

| HFR [mΩ·cm²] | 150 A | 450 A |
|---|---|---|
| Gamry whole cell (crossing) | 45.2 (3.8 kHz) | 44.4 (3.4 kHz) |
| FAMOS, area-weighted harmonic mean of the segment HFRs, uc = 0 | 44.7 | 46.7 |
| FAMOS aggregate Re at the Gamry's crossing, uc = 0 / 10 | 45.0 / 32.9 | 46.6 / 33.6 |
| bench tool CSV (separate test, 65 °C), harmonic mean | 45.8 | 46.9 |

The plate *median* (52–54) is higher than the Gamry because the whole cell is a parallel (harmonic) mean, not because of a series resistance. With 10 subtracted, 14–17 segments lose their real-axis crossing (points pushed to Re ≤ 0 are removed by the passivity gate), the segment phase spread at 4.7–12 kHz rises from < 2° to ~7°, and R_pol drops by 13–24 mΩ·cm². The remaining arc-band and low-frequency excess (whole-cell R_pol 196 vs 168 at 150 A, 122 vs 95 at 450 A) is open: candidates are the current chain's AC gain against its DC calibration (the gain file is normalised to 1 over 5–500 Hz) or a contact with capacitance in the voltage path. A four-wire impedance measurement between the UC2 tap and the Gamry sense point decides it.


---

## 5. Folder and module map

```
databricks/local_eis/
├── Local EIS Pipeline Runner.py   the notebook you run (23 cells)
├── Local EIS Extra Analyses.py    optional diagnostics; %runs the runner first
├── eis_paths.py                   puts the sub-folders on sys.path
├── run.py                         command line: python run.py <module> [args]
├── requirements.txt               numpy, scipy, matplotlib (+ pandas, plotly)
├── core/
│   ├── config.py                  EVERY tunable number (one frozen dataclass)
│   ├── utils.py                   sine fits, CRLB, outlier tests, IO, logging
│   ├── r2d2_geometry.py           both plate maps: pads → segments, areas, centroids
│   ├── plate_stats.py             the 36-tile mean/median/sd
│   └── plate_conditions.py        T / p / RH fields across the plate
├── pipeline/
│   ├── main.py                    run_pipeline(), reevaluate(), CLI, self-test
│   ├── bronze.py                  layer 1: files → raw Z per segment
│   ├── eis_local.py               FAMOS readers, calibration, step detection, sine fits
│   ├── tone_estimation.py         one exact frequency per card and step; phasors
│   ├── hf_schedule.py             high-frequency rung recovery (optional), CFAR, rank-1
│   ├── ladder_snap.py             snap steps to the geometric ladder; repair windows
│   ├── gamry_sync.py              put the FAMOS record on the Gamry clock
│   ├── silver.py                  layer 2: correct, gate, DRT, R_ohmic, KK, tiers
│   ├── channel_lag.py             per-segment chain lag against the plate median
│   ├── eis_measurement_model.py   Boukamp/Agarwal KK model, aggregate
│   ├── eis_validation.py          lin-KK and Z-HIT tests
│   ├── neighbours.py              adjacency, neighbour outliers, spectrum rebuild
│   ├── gold.py                    layer 3: process split, spatial field, maps
│   └── csv_pipeline.py            the separate CSV-source path
├── checks/
│   ├── plausibility.py            the plate-level verdicts → plausibility.csv
│   ├── gamry_compare.py           Gamry whole-cell sweep vs local aggregate
│   ├── frequency_response.py      chain response ex-situ (bench) and in-situ
│   ├── card_gain.py               card voltage gain, impedance factors
│   ├── dc_closure.py              plate current vs bench current (multi-run)
│   ├── segment_scale.py           per-segment scale factors (multi-run)
│   ├── via_resistance.py          via/shunt resistance vs temperature (Abgleich)
│   └── validate_hf.py             synthetic self-validation
├── analysis/
│   ├── ecm_drt.py                 DRT-informed equivalent-circuit fit
│   └── polcurve.py                polarisation curves (FAMOS and Gamry)
├── plotting/
│   ├── nyquist.py                 Nyquist/Bode figure coloured along the air path
│   ├── plate_figure.py            static plate maps (segments / interpolated)
│   ├── plate_plotly.py            interactive plate map
│   ├── plate_maps.py              value maps used by the runner
│   ├── plate_style.py             colours, ports, orientation
│   ├── viewers.py                 the one-plot drop-down viewers
│   ├── figure_panels.py           split multi-panel figures
│   └── bench_plots.py             MF4 test-bench channels
├── readers/
│   ├── gamry_dta.py               Gamry .DTA reader; chain-gain file writer
│   ├── abgleich.py                calibration files from the Abgleich steps
│   ├── csv_source.py              every CSV dialect (bench tool, logger, …)
│   └── _vendor/zstd.py            shim needed by asammdf
├── tests/                         ~430 pytest tests + a synthetic FAMOS generator
└── docs/                          this guide, analyses, figures, change log
```

`eis_paths.py` adds `core/ pipeline/ checks/ analysis/ plotting/ readers/` to `sys.path`. That is why every module imports the others by plain name (`import silver`, `from config import DEFAULT`). The notebook, `run.py`, `pipeline/main.py` and the tests all load it.

---

## 6. End-to-end data flow

`pipeline/main.py: run_pipeline(cfg, stop_after="gold")` is the one function the notebook calls per condition:

```
geom.use_plate(cfg.plate)                       # gen1 = green Kashyyyk plate
cfg.save(out/"config_used.json")
br = bronze.run(cfg)  ; bronze.save(br, cfg)    # → out/bronze/
sr = silver.run(br, cfg) ; silver.save(sr, cfg) # → out/silver/
gr = gold.run(sr, cfg) ; gold.save(gr, sr, cfg) # → out/gold/
whole_cell_and_plausibility(cfg, manifest, sr)  # → gamry_comparison*, frequency_response/, plausibility.csv
write run_manifest.json
```

The notebook wraps it:
1. builds `cfg = DEFAULT.replace(...)` with the paths and the fixed settings;
2. checks the cache;
3. runs or loads;
4. copies the run folder to the Volume cache;
5. collects `PIPELINE_RESULTS[cond] = {'manifest', 'cfg', 'out_dir'}`. Every display cell reads it.

---

## 7. Bronze: raw waveforms → raw impedance

`pipeline/bronze.py`, helped by `eis_local.py`, `tone_estimation.py`, `ladder_snap.py`, `gamry_sync.py` and (optionally) `hf_schedule.py`.

**Bronze applies no physics corrections.** It writes measured values, or values that follow algebraically from the calibration. The only things it changes are:
- the chain gain;
- the card voltage gain;
- an obvious polarity flip.

### 7.1 Steps, in the order `bronze.run()` performs them

1. **Find the card files.** `discover_files(cfg)` tries the patterns in `config.FAMOS_PATTERNS`, e.g. `Leepa_{leepa}_Current_{cond}_Test_{test}_Karte_*.DAT`. It refuses a set of files that spans more than one current.
2. **Inventory the channels.** `inventory_channels` opens each card with `FamosFile`, which handles both FAMOS dialects (`FamosV1`, `FamosV2`). It classifies each channel:
   - digits → segment;
   - `UC*` → voltage reference (fixed: `ref_channel = "UC2"`);
   - `Temp*` → temperature.

   It also records each channel's **ADC slot** (its position in the header). Silver's skew model needs it.
3. **Calibration.** `PlateCalibration.load(curr_cal, temp_cal)`. A segment missing from `curr.csv` is imputed from the plate median and flagged `K_imputed`.
4. **Align the cards** (`estimate_card_lags`). The five cards are armed separately, by up to several seconds (one recording had card 3 5.7 s late).
   - Each card's UC2 channel is band-passed to 0.5–300 Hz and cross-correlated (FFT) with the anchor card's.
   - The anchor is the card with the highest median correlation.
   - A lag is **applied** only if |r| ≥ `align_min_corr` (0.5) and the peak's **prominence** (peak above the median background, in robust sd) ≥ `align_min_prominence` (15). A weaker lag is accepted if another card corroborates it.
   - Cards with a different sample rate are resampled for the correlation, and their windows converted with `to_own(idx) = round(idx·ratio) + lag`.
   - Clock drift (ppm) and pairwise closure are reported, not applied.
5. **Plate temperatures** (`plate_temperatures`). The four `Temp_*` channels are converted with `temp.csv` and interpolated linearly along x to every segment's centroid. Without sensors the fallback is 58.4 °C.
6. **Build one consensus schedule** (`consensus_schedule`). This is "which stretch of the record holds which frequency".
   1. Per card, `eis_local.detect_schedule` scans a log grid of trial frequencies on the UC reference:
      - demodulate the signal;
      - keep the envelope peaks that stand 4× above the median;
      - grow a dwell window, then fit with `fit4` (frequency free);
      - polish the window;
      - de-duplicate and fill gaps.
   2. Every card's steps are moved onto the common time base and clustered by frequency (`grid_tol` = 1 %).
   3. A **geometric grid** (fixed points per decade) is fitted to the confident clusters. A step is accepted if ≥ 2 cards saw it **or** it lies on the grid.
   4. **Ladder snap** (`ladder_snap.snap_steps`): duplicates merge and frequencies snap to the exact rungs. **Window repair** (`repair_windows`): windows that break the time order of a monotonic sweep get interpolated positions, and the step is refused if more than 25 % would need it.
   5. **Gamry clock** (`gamry_sync`, mode `guide` in the notebook), described in 7.2.
   6. Overlapping neighbour windows are cut at the midpoint (`separate_windows`).
7. **Exact frequency per card and step** (`tone_estimation.card_frequency`). A multichannel maximum-likelihood search (variable projection over all 16 channels of the card) gives one frequency per card and step, within ±1 FFT bin of the schedule.
8. **Phasors and impedance** (`process_card`). For every segment and step:
   - read the window `[round(start·ratio)+lag, round(stop·ratio)+lag]`;
   - fit the voltage and segment phasors at that frequency (`phasors_at`: least squares on cos, sin, offset and ramp);
   - compute **`Z = K(T) · A_ref / A_seg`**;
   - record THD (2nd and 3rd harmonic), a χ² stationarity test (drift), SNR of the reference, of the segment and combined, and the samples per step;
   - divide by the chain gain (`gain_at`) and by the card's voltage gain;
   - flip the polarity if Re Z is negative over the lowest decade.
9. **Duplicates.** If two cards report the same segment, the one with the higher median SNR wins.

### 7.2 The Gamry clock (`gamry_sync.py`)

The FAMOS cards record the cell's response to the **Gamry's own sweep**. The `.DTA` file of the same condition therefore knows each step's exact frequency and the second it ended.

- **`align`.** Every confidently detected FAMOS window votes for an offset (window end − Gamry time). The median is accepted if ≥ 5 voters agree within 1.5 s. Each Gamry point is then judged `ok`, `misplaced` or `missing`.
- **`apply_frequencies`.** Steps judged `ok` take the Gamry frequency exactly.
- **`relocate`.** Misplaced, missing, interpolated and **empty** windows (median channel SNR < −3 dB) are searched again inside their Gamry slot. A candidate is accepted only if two independent array tests agree (`hf_schedule.cfar_channel_count` and the rank-1 test). A prediction alone is never accepted.
- **`confine_windows`.** For low frequencies (interval ≥ 3 s) the window is placed strictly between two consecutive Gamry step ends, with a guard and half a period of settling. If fewer than 3 cycles would remain, the settle time is dropped first, then the guard is halved. This fixed windows that overran into the neighbouring step (e.g. 0.19 Hz at 450 A).
- **`corroborate_card_lags`.** A card lag that alignment refused is applied if the Gamry clock independently implies the same lag.

### 7.3 Bronze files (`<run>/bronze/`)

| File | Content |
|---|---|
| `raw_spectra.csv` | segment, card, freq_hz, z_re/z_im (Ω·cm²), snr_ref/seg/comb_db, thd, drift, n_samples, on_grid |
| `segment_meta.csv` | per segment: card, area, centroid, channel_slot, ref_slot, fs, K, K_imputed, T, u_dc, j_dc |
| `schedule.csv` | one row per step: freq, start/stop sample, amp, SNR, THD, drift, `window_source` (detected / gamry / gamry_step / interpolated), coverage |
| `card_alignment.csv` | per card: lag (samples, s), corr, prominence, applied, ratio, rescued, clock ppm |
| `card_reference.csv` | per card UC gain against the median card |
| `gamry_sync.csv` | per Gamry point: offset residual, verdict (ok / misplaced / missing), relocation |
| `frequency_coverage.csv`, `channels.csv`, `bronze_manifest.json` | coverage, channel inventory, summary |

---

## 8. Silver: correct, gate, model, grade

`pipeline/silver.py` with `channel_lag.py`, `eis_measurement_model.py`, `eis_validation.py` and `neighbours.py`. **Nothing is deleted silently**: every dropped point goes into a ledger with the reason.

### 8.1 Timing corrections

1. **ADC skew (structural model).** One converter scans the 16 channels in turn, so channel p is sampled p/(16·fs) after channel 0: 2.5 µs per slot at 25 kHz.
   - Z is de-skewed with `Z·exp(+j2πf·dt)`, where `dt = k·(slot_seg − slot_ref)`.
   - k is fitted per card (it must lie within 0.5–1.6× nominal, otherwise the nominal value is used).
   - The common delay `dt0` is held at 0 (`fit_common_delay = False`): it cannot be told apart from a series inductance.
2. **Channel lag** (`channel_lag.py`, mode `correct`).
   - Each segment's shunt, trace and amplifier chain adds its own delay. On 2612030 the lags span −57…+76 µs, i.e. up to ~30° at 1 kHz.
   - Per segment, the phase of `Z_seg / Z_plate-median` between 50 Hz and the coherent limit (0.16·fs, 4 kHz) is fitted with a pure delay τ (`−ωτ`). The fit iterates 3 times as the plate median is rebuilt, and the lags are re-centred on the median segment.
   - τ is applied only if the rms residual is ≤ 10° and |τ| ≤ 250 µs. Otherwise the segment is flagged (`not_first_order`, `too_large`) and demoted to tier C.
   - **A delay that every channel shares is invisible to this stage** (it is measured against the plate median).
3. **Card phase against the Gamry.** Full band only, so off in the shipped setup (§4).

### 8.2 Point gates (`gate_points`), in order; the first gate to reject a point owns it

| # | Gate (ledger name) | Rule (config field, default) |
|---|---|---|
| 1 | `not_finite` | f > 0 and Z finite |
| 2 | `outside_band` | `f_min_hz` (0.15) ≤ f ≤ `f_eval_max()` (2400) |
| 3 | `snr` | backstop only: SNR ≥ −40 dB (`silver_snr_floor_db`, `silver_snr_gate_db`) |
| 4 | `off_tone` | off (`off_tone_drop_db = None`) |
| 5 | `thd` | THD ≤ 0.10 (`max_thd`) |
| 6 | `drift` | drift ≤ 0.25 (`max_drift`) |
| 7 | `magnitude` | 0.5 < \|Z\| < 800 mΩ·cm² |
| 8 | `uncertainty` | CRLB σ_rel = √(1/(N·γ)) ≤ 0.60 (`sigma_rel_max`). **This is the gate that really decides.** |
| 9 | `cycles` | cycles in the window ≥ 3 (`min_cycles_per_dwell`) |
| 10 | `zmag_outlier` | log10\|Z\| within 4.5 robust sd of a 7-point running median |
| 11 | `passivity` | after the timing corrections: Re Z > 0 for f ≥ 1 Hz (negative Re Z below 1 Hz is kept and flagged; it can be real) |

A segment with fewer than 8 surviving points is dropped, and its verdict is recorded.

### 8.3 R_ct, R_mt, R_pol: the Bayesian DRT, and the circuit fit beside it

**The primary method is the Bayesian DRT** (`param_method = "drt"`, `bayesian_drt`): a Gaussian-process prior on the distribution of relaxation times gives a closed-form posterior, every point weighted by its Cramér–Rao uncertainty. R_pol is the area of the DRT, R_ct the part faster than `tau_split_kinetic_s`, R_mt the slower part; R_ohmic is the high-frequency intercept below, with the DRT's own R_∞ reported as `R_ohmic_drt`.

**The standard equivalent circuit is fitted beside it** (`ecm_compare = True`, `pipeline/standard_eis.py`) and written as `R_ct_ecm`, `R_mt_ecm`, `R_pol_ecm`, `ecm_rms_pct`, so both evaluations of every segment come out of one run. `param_method = "ecm"` makes the circuit the primary method instead. The circuit, the way ZView or Gamry Echem Analyst fit it:

```
Z(f) = R_ohmic + jωL + R_ct / (1 + (jωτ_ct)^n_ct) + R_mt / (1 + (jωτ_mt)^n_mt)
R_pol = R_ct + R_mt
```

1. **R_ohmic from the high-frequency end** (`hf_intercept`):
   - where the spectrum crosses the real axis inside the band, the interpolated crossing;
   - otherwise Re Z extrapolated to Im Z = 0 along a straight line through the top 6 points (Re Z against Im Z).

   It is then **held fixed**. With a band ending before the crossing, a free R_ohmic trades off against an arc starting above the band: on 10 % of the 450 A segments it fell to ~30 mΩ·cm² instead of ~58.
2. **Fitted part of the spectrum.** Only the capacitive part is fitted (`capacitive_band`). At high current the spectrum turns inductive below ~1 Hz (the low-frequency loop); the circuit has no element for it, so those lowest points are left out of the fit. They still count for `R_lf`.
3. **The fit.** Complex non-linear least squares (`scipy.optimize.least_squares`):
   - **modulus weighting**: real and imaginary residuals divided by |Z|;
   - **bounds**: R ≥ 0, L ≥ 0, 0.5 ≤ n ≤ 1, τ_mt ≥ 3·τ_ct (the arcs cannot swap), and both arc peaks inside the measured band;
   - **18 starting points** read off the spectrum (−Im peak, split of the polarisation); the best is kept.
4. **Reported per segment** (`silver/segments_summary.csv`):
   - `R_ohmic` and how it was read (`R_ohmic_how`: crossing / extrapolated);
   - `R_ct`, `R_mt`, R_pol;
   - the arc peak frequencies `f_ct_hz` and `f_mt_hz`;
   - the fit's rms relative residual `ecm_rms_pct` (good ≤ 3 %, acceptable ≤ 8 %);
   - the **graphical** polarisation resistance `R_pol_lf` = Re Z(f_min) − R_ohmic;
   - `R_ohmic_topband` for comparison.

On 2612030 at 450 A (67 segments):
- the charge-transfer arc peaks at 110–170 Hz and the mass-transport arc at 3–10 Hz;
- R_pol from the fit (median 112) agrees with the graphical value (107);
- the fit residual is 3.6 % at the air inlet and 8 % at the air outlet, where the measured low-frequency points scatter (§18).

The notebook's ECM cells (16–20) use the same function, so the fit you see there is the fit behind the maps.

`param_method = "drt"` switches back to the earlier approach (Bayesian DRT, top-band R_ohmic, DRT split at τ = 10⁻⁴ / 10⁻² s). It is kept for comparison only.

### 8.5 Validation and tiers

- **lin-KK** (`eis_validation.lin_kk`, Boukamp with Schönleber's μ criterion): the maximum residual relative to |Z|.
- **Z-HIT** (`z_hit`): reconstructs ln|Z| from the phase. It sees pure delays, which KK cannot.
- **Tier**:
  - **A**: KK ≤ 2 % and CV(R_ohmic) ≤ 10 %;
  - **B**: KK ≤ 6 % and CV ≤ 25 %;
  - **C**: otherwise.

  An imputed calibration caps the tier at B. An unreliable channel lag forces C.

### 8.6 Plate products

- **DC closure.** I = Σ j_dc·A over the measured segments, extrapolated to 304.92 cm².
- **Rebuilt segments** (`neighbours.fill_spectrum`): the area-weighted mean of the adjacent segments' model spectra (pads sharing an edge, up to 2 rings out). Used only for the segments you list under "Rebuild from neighbours" (or every gap, with "Fill unmeasured" = yes). Labelled `reconstructed`, never given a tier.
- **Cell aggregate** (`cell_aggregate_full`): the segments in parallel, `Y = Σ A_s/Z_s`. ASR over the covered area = A_used/Y; full-plate ASR = A_cell/Y.

### 8.7 Silver files (`<run>/silver/`)

| File | Content |
|---|---|
| `spectra_clean.csv` | **the corrected spectra**: segment, freq_hz, z_re/z_im (mΩ·cm²), model and its sd, sigma_rel |
| `segments_summary.csv` | per segment: tier, R_ohmic (+sd, xint), R_pol, L, τ_peak, dt applied, chain τ and status, T, j_dc, KK %, Z-HIT %, SNR, THD, points used/dropped, flags |
| `channel_lag.csv` | per segment: τ (µs), applied, residual, status |
| `point_rejections.csv` | **the ledger**: one row per segment and point, with kept and reason |
| `segment_reach.csv` | per segment: highest kept frequency, and which gate stopped it |
| `skew_model.csv` | per card: basis, dt0, slot µs, applied |
| `cell_aggregate.csv` | the parallel sum, covered-area and full-plate ASR, area coverage |
| `card_factors.csv` | per card \|Z\| and j_dc against the plate |
| `spectra_reconstructed.csv` | rebuilt segments, with their donors |
| `silver_manifest.json` | tier counts, DC closure, skew per card |

---

## 9. Gold: the plate as a field

`pipeline/gold.py`.

1. **R_ct and R_mt** (`split_processes`) are the two arcs of the standard circuit fit (§8.3). Only with `param_method = "drt"` is the DRT split at 10⁻⁴ / 10⁻² s used instead.
2. **Other parameters** (`collect_parameters`):
   - `ReZ_1kHz`: Re Z interpolated at 1 kHz between the two neighbouring points, never extrapolated;
   - `Z_mag_100Hz` and `phase_100Hz` from the model;
   - `j_dc`, `T_degC`, `chain_tau_us`, `sigma_rel`.
3. **Spatial field** (`fit_field`). Gaussian-process regression (Matérn 5/2, length 70 mm along the flow and 35 mm across) on the segment centroids.
   - **Measured segments keep their own values.** The field only fills segments with no data (class `inferred`, tier `D`), and only if no more than 60 % are missing.
   - Excluded segments stay empty. Rebuilt segments take the donors' area-weighted mean.
4. **Fault labels**, from ratios to the plate median: drying, flooding, starvation, contact_loss (thresholds in `config.FAULT_RULES`).
5. **Statistics over the 36 tiles** (`plate_stats.tile_stats`, §15).
6. **Maps: only 2D interpolated plate maps**, `plate_<param>_interp.png`, one per parameter: R_ohmic, Re Z 1 kHz, R_ct, R_mt, R_pol, j, |Z| and phase at 100 Hz, T. The per-segment value maps are no longer written. Also `nyquist.png/.html`.

### Gold files (`<run>/gold/`)

| File | Content |
|---|---|
| `plate_summary.csv` | **one row per segment, all 72**: class, tier, position, area, fault, flags, and every parameter with its sd. Resistances in mΩ·cm², sigma_rel in %. Exception: `R_hf_extra` is in Ω·cm². |
| `plate_statistics.csv` | per parameter: n tiles, mean, median, sd, CV, min, max, missing tiles |
| `plate_*_interp.png` | the 2D interpolated plate maps, one per parameter |
| `nyquist.png`, `nyquist.html` | all segments; the HTML also holds the \|Z\| and phase panels |
| `gold_manifest.json` | statistics, class counts, spatial trends, DC closure |

---

## 10. Checks: is the plate physically coherent?

`pipeline/main.py: whole_cell_and_plausibility` runs three things after gold. None of them can stop the run.

### 10.1 Gamry whole-cell comparison (`checks/gamry_compare.py`)

- It finds the condition's Gamry sweep (`find_cell_sweeps`, filtered by order and build) and converts it to ASR: Z·304.92 cm².
- It compares it with silver's `cell_aggregate.csv` **on the shared band only**: magnitude difference, phase difference, rms, and the high-frequency resistance of each.
- Outputs:
  - `gamry_comparison.csv` (one row per condition, including the bench current at the time of the sweep);
  - `gamry_comparison_curves.csv`;
  - `gamry_reference_full.csv`;
  - `gamry_comparison.png` (the whole Gamry sweep in black, the aggregate in red).
- The numbers compare the aggregate with the Gamry's own point at the same step. **The plots draw the Gamry as recorded**: all points, exact frequencies (hover shows 5 significant digits), no band limit, no inductance removed (`nyquist.gamry_frame`).

### 10.2 Chain frequency response (`checks/frequency_response.py`)

- **Ex-situ:** the bench (Abgleich) response G_d of every current amplifier, from the chain gain file or the bode folder.
- **In-situ:** each segment against the plate median, before and after the timing correction.
- Writes `frequency_response/` and two checks.

### 10.3 Plausibility (`checks/plausibility.py` → `plausibility.csv`)

| Check | Tests | Pass / warn / fail |
|---|---|---|
| coverage | measured **area** fraction | ≥ 60 % / ≥ 25 % / below |
| measured block | whole plate, block 1–36, or block 37–72? | whole or 1–36 pass; otherwise warn |
| current closure | plate current from j (mean over the 36 tiles × 304.92 cm²) against the set or bench current | ≤ 10 % / ≤ 30 % / above |
| passivity | Re Z > 0 everywhere after correction | pass / – / any Re Z ≤ 0 |
| card alignment | every card's lag applied | pass / – / any card not aligned |
| channel lag | share of segments with an unreliable lag | none / some, or a lag ≥ 30 µs / ≥ 20 % |
| card voltage gain | UC gain of each card against the median card | ≤ tolerance / not removed or not flat / > 10 % |
| card impedance factor | a card that looks like a pure scale error | pass / warn |
| R_ohmic smoothness | neighbour contrast against 200 random shuffles | ≤ 0.6 / ≤ 0.85 / above |
| R_ohmic along flow | correlation of R_ohmic with position along the flow | \|r\| ≥ 0.15 pass, else warn (flat) |
| R_s matched-band closure | top-band Re Z, aggregate vs Gamry on the same frequencies | ≤ tol / ≤ 2·tol / above (tol grows as coverage falls) |
| aggregate vs whole cell | median \|Z_agg\|/\|Z_Gamry\| − 1 over the shared band | same rule |
| chain response (ex-situ) | amplifier response flat within 1 % at 5–500 Hz, segments within 2° | pass / warn |
| segment frequency response (in-situ) | phase spread over segments at the top of the band | ≤ 5° / ≤ 10° / above |

### 10.4 Multi-run checks in the notebook

- **Card gain and DC closure** (cell 9): plate current against bench current over all conditions, fitted as I_plate = a·I_bench + b. a − 1 is a current-scale error that also sits in every Z.
- **Segment scale** (`segment_scale.py`): per-segment scale factors that persist across conditions. Reported, not applied.
- **Via resistance** (`via_resistance.py`, cell 11): from the Abgleich steps, the via/shunt slope against temperature per card.

---

## 11. Formula sheet

Units: impedances are area-specific (Ω·cm² inside the code, mΩ·cm² in the output files), currents are in A, areas in cm², temperatures in °C.

| Quantity | Formula | Code |
|---|---|---|
| Segment calibration | K(T) = c0 + 10⁻³·c1·T (V per A/cm²), from `curr.csv` | `eis_local.PlateCalibration.K` |
| Sensor temperature | T = (u − c0)/c1, from `temp.csv`; segment T linear in x between the 4 sensors | `PlateCalibration.temperature`, `r2d2_geometry.segment_temperatures` |
| Phasor | least squares y = a·cos ωt + b·sin ωt + c + d·t → A = a − jb | `tone_estimation.phasors_at` |
| **Local impedance** | **Z = K(T) · A_U / A_seg**, then ÷ chain gain(f), ÷ card voltage gain | `bronze.process_card` |
| ADC de-skew | Z ← Z·exp(+j2πf·k·(slot_seg − slot_ref)), k ≈ 1/(16·fs) | `silver.process_segment` |
| Channel lag | fit arg(Z/Z_plate-median) = −2πfτ; Z ← Z·exp(+j2πfτ) | `channel_lag.estimate`, `.correction` |
| Point uncertainty | σ_rel = √(1/(N·γ)), γ = linear SNR, N = samples per step (CRLB) | `utils.sigma_rel_from_snr` |
| DC current density | **j = u_dc / K(T)** (A/cm²) | `SilverSpectrum.j_dc` |
| DRT model | Z = R_inf + jωL + Σ γ_m/(1 + jωτ_m), γ ≥ 0 | `silver.bayesian_drt` |
| R_ohmic | high-frequency intercept: Im Z = 0 crossing, or Re Z = a + b·Im Z through the top 6 points read at Im Z = 0 | `standard_eis.hf_intercept` |
| Standard circuit | Z = R_ohmic + jωL + R_ct/(1+(jωτ_ct)^n_ct) + R_mt/(1+(jωτ_mt)^n_mt), CNLS, weights 1/\|Z\| | `standard_eis.fit_spectrum` |
| R_pol | R_ct + R_mt (graphical check: Re Z(f_min) − R_ohmic) | `standard_eis.fit_spectrum` |
| Arc peak frequency | f = 1/(2π·τ) | `standard_eis.fit_spectrum` |
| R_ohmic_topband (comparison) | Σ w_i·Re Z_i / Σ w_i over the top 0.35 decade, w = 1/(σ_rel·\|Z\|)² | `silver.r_ohmic_topband` |
| R_ohmic_xint | Re Z = a + b·Im Z fitted to the top 6 points, read at Im Z = 0 | `silver.r_ohmic_axis_fit` |
| R_ohmic sd | √(sd_top² + hf_arc_open²), hf_arc_open = 2·\|min(Im Z_model(f_top), 0)\| | `silver.process_segment` |
| Re Z at 1 kHz | linear in ln f between the two kept points around 1 kHz | `gold.re_at` |
| Cell aggregate | Y(f) = Σ A_s/Z_s(f); Z_cell,ASR = A_used/Y (covered) or A_cell/Y (full) | `silver.cell_aggregate_full` |
| Gamry ASR | Z_Gamry[Ω] · 304.92 cm² | `gamry_compare` |
| Plate current | I = mean(j over the 36 tiles) · 304.92 cm²; DC closure: I = Σ j·A · A_cell/A_meas | `plausibility.current_closure`, `silver.dc_closure` |
| Tile statistics | mean, median, sd (ddof = 1), CV = sd/mean over the measured segments 1–36 | `plate_stats.tile_stats` |
| lin-KK residual | max over f of \|Z − Z_KK\|/\|Z\| | `eis_validation.lin_kk` |
| Tier | A: KK ≤ 2 % and CV(R_ohmic) ≤ 10 %; B: KK ≤ 6 % and CV ≤ 25 %; C: otherwise | `silver.process_segment` |
| Cathode λ | λ = flow / (I/(4F)/x_O₂ · 22.414 · 60) | `plate_conditions.cathode_stoichiometry` |

---

## 12. Output files, one by one

One run folder per condition:

```
<run>/                              e.g. EIS_Results/2612030/45A/mode_default/snr_0.0_<id>/
├── config_used.json                every setting of this run
├── run_manifest.json               stage summaries, elapsed time
├── plausibility.csv                the verdicts (§10.3)
├── gamry_comparison.csv / _curves.csv / gamry_reference_full.csv / .png
├── frequency_response/             ex-situ + in-situ chain response
├── bronze/                         §7.3
├── silver/                         §8.7
└── gold/                           §9
```

**What to open first when judging a run:**
1. `plausibility.csv`
2. `gold/plate_statistics.csv`
3. `silver/segments_summary.csv` (tiers and flags)
4. `silver/segment_reach.csv` (why a spectrum stops)
5. `bronze/gamry_sync.csv` (were the windows where the Gamry says)

---

## 13. The runner notebook, cell by cell

| # | Title | What it does |
|---|---|---|
| 1 | (restart) | `restartPython()` for a clean module reload. Skipped when `Local EIS Extra Analyses` runs the notebook. |
| 2 | Setup | Finds the module folder and imports and reloads every module. **Build check**: each module must come from this folder and contain this version's functions, otherwise "These modules are not from this build" (fix: delete the folder, import the zip into an empty one). Also the plate self-check and the display helpers. |
| 3 | Widgets | Discovers the orders (datago + Volumes), the conditions, the CSV folders and the Gamry folder and build. Creates the widgets, removes the retired ones, and sets the **FIXED SETTINGS** (§4). |
| 4 | Chain response | Builds `chain_gain_gen1.csv` from the Abgleich `bode/` sweeps (once). Prints the amplifier response and verifies the calibration. |
| 5 | Datago source | Datago first, Volumes as fallback (§3). |
| 6 | (skipped, SQL) | A diagnostic query of the datago FAMOS files. |
| 7 | Run dashboard | The cache inventory and the run plan per condition (load / run / skip), with age, segment count and band. |
| 8 | Build config + run | `cfg = DEFAULT.replace(...)`, then `run_pipeline` per condition, and the cache write. Defines `PIPELINE_RESULTS`. **The plausibility verdicts are printed here.** |
| 9 | Card gain and DC closure | UC gain per card, card impedance factors, plate current vs bench current over all conditions, per-segment scale. |
| 10 | Frequency response | Ex-situ amplifiers and in-situ per condition. |
| 11 | Via resistance | Via slope against temperature per card (needs the Abgleich folder). |
| 12 | Nyquist / Bode | Every segment coloured along the air path (inlet blue → outlet red), and the condition's **Gamry in black** in all three views (Nyquist, \|Z\|, phase): every point of the `.dta` at the frequency the file gives (`gamry_spectra`). Rebuilt segments are dotted. |
| 13 | Gamry vs aggregate | **Two curves only:** the aggregate of the segments (red, `cell_aggregate.csv` as written) and the Gamry (black, every point of the file, 0.19 Hz – 30 kHz). Nothing scaled, cut to a band, interpolated or removed (`nyquist.comparison_figure`). |
| 14 | Plate heat maps | Condition × parameter × view (interpolated / per segment). Fixed colour scales. Also writes the static PNGs. |
| 15 | MF4 test-stand | Every bench channel of the order, with the FAMOS plate temperatures as reference lines. |
| 16 | ECM fitting | Rs + L + (R_ct‖CPE) [+ (R_mt‖CPE)] per segment and for the aggregate; arc count chosen by AICc. Defines `ECM_ALL`. |
| 17–20 | ECM views | Fit per segment, fit of the aggregate, parameter tables, ECM plate maps. |
| 21 | Summary table | Plate statistics from `run_manifest.json` and `plate_statistics.csv`. |
| 22–23 | Polarisation curves | FAMOS (whole plate and three flow bands) and Gamry (with iR-free). |

`Local EIS Extra Analyses.py` covers:
- the plate numbering map;
- the Gamry-vs-local decomposition (scale, series R and L);
- Lin-KK;
- the Tikhonov DRT;
- the DRT-informed ECM (method B) and its comparison with the runner's ECM (method A).

---

## 14. The result cache

- **Location:** `/Volumes/…/EIS_Results/<leepa>/<cond>/mode_default/snr_0.0_<id>/`. Viewers are saved to `EIS_Results/<leepa>/viewers/`.
- **The `<id>`** is a 10-character hash of every setting that changes a result (`_CACHE_IDENTITY_KEYS`): band, gates, window placement, alignment, estimators, excluded or rebuilt segments, Gamry build, and so on. Change a setting and you get a new entry, never a stale one.
- **What the hash does not see is the code version.** After updating the code, run once with `rerun`.
- **Run modes:**
  - `cache`: load if present, else run;
  - `rerun`: always run, and overwrite if "write to cache" is yes;
  - `cache-only`: never run.
- An entry counts only if `silver/spectra_clean.csv` exists. A folder holding only a manifest is a failed run.
- Fresh runs are written to `/tmp/eis_<uid>/<leepa>/<cond>` and copied to the Volume (top-level files plus `bronze/ silver/ gold/`). `frequency_response/` is recomputed for cached runs.

---

## 15. Plate geometry and the 36-tile statistics

`core/r2d2_geometry.py` is the single source of truth:
- 45 × 20 pads of 5.60 × 6.05 mm, giving a 252 × 121 mm plate and 304.92 cm²;
- 72 segments;
- `gen1` = green Kashyyyk plate (verified against the plant's own pad matrix);
- `gen2` = blue Naboo plate (a reconstruction; there is no FAMOS recording of it).

On gen1 the segment areas range from 1.36 to 8.47 cm². Segments 1–36 cover the interior (68.6 % of the area); 37–72 are the smaller edge segments at both ends.

**The plate statistics use 36 equal tiles.** The 45 × 20 pads split exactly into 9 × 4 tiles of 5 × 5 pads (8.47 cm²). Segment k = 1…36 sits at the centre of tile k:

```
tile column i = (k − 1) // 4    → pad columns 5i+1 … 5i+5
tile row    j = (k − 1) %  4    → pad rows    5j+1 … 5j+5
```

- Segment k **stands for its whole tile**, so the values of edge segments 37–72 are not used.
- Every tile has the same weight, so mean and median are area-weighted, and the denser segmentation at the plate ends cannot pull them.
- A tile without a measured value (unmeasured, excluded or rebuilt) is left out, and the count says so ("35 of 36 tiles").
- This is used in every map subtitle, in `plate_statistics.csv`, in the run manifest, and in the plausibility current closure.

The **red label boxes** of the coordinate drawing are where each segment's value is drawn. Segments 1–36 sit at pad columns 3, 8, …, 43 and rows 3, 8, 13, 18; segments 37–72 sit at columns 1, 5, 10, 36, 41, 45 and rows 1, 5, 9, 12, 16, 20.

The ports, viewed from the drawing side:
- **left end:** AIR OUT, COOLANT OUT, H₂ IN;
- **right end:** H₂ OUT, COOLANT IN, AIR IN (counter-flow).

`config.PLATE_VIEW_MIRRORED = True` mirrors the whole drawing.

---

## 16. Configuration reference

Every number lives in `core/config.py`, one frozen dataclass `Config`. The default instance is `DEFAULT`; derive variants with `DEFAULT.replace(field=value)`. The fields that matter most:

| Group | Field = default | Meaning |
|---|---|---|
| Band | `f_min_hz=0.15`, `eval_f_max_hz=2400`, `f_max_hz=30000`, `f_hi_frac_fs=0.45` | evaluated band; detection ceiling |
| Band (developer) | `full_band=False`, `full_band_frac_fs=0.49`, `coherent_f_max_frac_fs=0.16`, `r_ohmic_method="auto"` | full-band diagnostic |
| Detection | `ppd=12`, `grid_tol=0.01`, `min_ref_channels=2` (runner), `min_snr_db` (runner: 0) | step grid and consensus |
| Alignment | `ref_channel="UC2"`, `align_min_corr=0.5`, `align_min_prominence=15`, `align_max_lag_s=30` | card alignment |
| Gamry clock | `gamry_sync` (runner: `guide`), `window_confine=True`, `window_min_snr_db=-3`, `window_guard_s=0.5`, `window_settle_periods=0.5` | window placement |
| Phasors | `phasor_method="card_ml"`, `freq_refine=False`, `fit_at_exact_frequency=False` | estimator |
| Silver gates | `sigma_rel_max=0.60`, `min_cycles_per_dwell=3`, `max_thd=0.10`, `max_drift=0.25`, `zmag_outlier_mad=4.5`, `passivity_gate_min_hz=1.0`, `min_points_per_spectrum=8` | §8.2 |
| Timing | `skew_model="structural"`, `fit_common_delay=False`, `channel_lag="correct"`, `channel_lag_max_resid_deg=10`, `channel_lag_max_us=250` | §8.1 |
| DRT | `drt_enable=True`, `drt_nonneg=True`, `drt_n_tau_per_decade=10`, `drt_tau_pad_slow=1.0` | §8.3 |
| Grading | `kk_tol=0.02`, `mu_crit=0.85` | tiers |
| Gold | `tau_split_ohmic_s=1e-4`, `tau_split_kinetic_s=1e-2`, `spatial_len_x_mm=70`, `spatial_len_y_mm=35`, `max_inferred_fraction=0.60` | §9 |
| Segments | `exclude_segments`, `substitute_segments`, `fill_missing_from_neighbours=False`, `fill_max_hops=2` | widgets |
| Calibration | `curr_cal`, `temp_cal`, `gain_file`, `uc_series_mohm_cm2=0.0`, `card_gain="report"` | inputs |
| Maps | `heatmap_style="interpolated"`, `HEATMAP_LIMITS` (R_ohmic 40–80, ReZ_1kHz 45–85, R_ct 25–225, R_mt 0–300, R_pol 50–450 mΩ·cm²), `PLATE_VIEW_MIRRORED=False` | look |

`preset("permissive")` and `preset("strict")` exist for exploratory work. The notebook does not offer them any more.

---

## 17. Command line, tests and development

```bash
cd databricks/local_eis
python -m pytest tests                                  # ~430 tests, ~2 min
python run.py main --self-test                          # geometry, estimators, readers
python run.py main --reevaluate RUN_DIR --out OUT       # silver/gold again from saved bronze
python pipeline/main.py --dat <Famos dir> --curr-cal curr.csv --temp-cal temp.csv \
       --gamry <Gamry dir> --leepa 2612030 --condition 45A --out out/45A
python run.py frequency_response RUN_DIR --bode <Abgleich>/bode
python run.py dc_closure RUN_45A RUN_60A ... -o OUT
```

- `tests/make_synth_famos.py` writes synthetic FAMOS cards with a known impedance. Most end-to-end tests use it.
- **To ship a new version**, from the repo root:
  ```bash
  git archive --format=zip --prefix=local_eis/ HEAD:databricks/local_eis > Local_Eis.zip
  ```
  Then import it into an **empty** workspace folder.
- **Changing a module:** keep `_BUILD_MARKERS` in the notebook's setup cell in step with any function that a new version depends on.
- **Adding a setting that changes results:** add its name to `_CACHE_IDENTITY_KEYS` in the notebook, or the cache will serve old results under the new setting.

---

## 18. Known limitations and open items

1. **The top of the band.** With `BAND = 'complete'` the spectra cross the real axis inside the band and R_ohmic is that crossing. Above ~1 kHz the card phase is anchored to the Gamry (§4.3), so the crossing is not independent of the Gamry until the injected calibration sweep exists.
2. **Low-frequency scatter at the 450 A air outlet is in the measurement.**
   - **Not caused by the processing:** silver changes the < 20 Hz values by 0.0 %, and the same scatter is in the old v3 results.
   - **Larger at the outlet:** the outlet segments (x < 60 mm) scatter 6.5 % rms against 1.7 % at the inlet, with lower SNR (10.7 vs 18 dB) and higher THD.
   - **Correlated between segments:** the outlet segments scatter together (r = 0.87), so the cell itself changes during the long low-frequency steps (water at the outlet).
   - **The circuit fit averages over it:** the outlet fits leave about 8 % residual.
2. **The UC taps include ~13.4 mΩ·cm² the Gamry does not.** That is 44 µΩ of series resistance in the voltage-sense path (`dc_closure.sense_offset`). `uc_series_mohm_cm2` can subtract it; it is 0 by default, so Z is reported as the UC taps see it.
3. **To be sure about the chain, measure it.**
   - Sweep a known resistor (shunt or dummy load) through every FAMOS channel together with the Gamry: the delay and filter of each card, independent of the cell.
   - Measure a Gamry dummy cell for the Gamry's own chain.
   - The injected sweep of §4.3 is the per-channel version of this.
4. **Low-frequency scatter at high current.** At 150 A and 450 A the air-outlet segments scatter below ~20 Hz (up to ~10 % at the 450 A outlet). The outlet segments move together, which points to a non-stationary cell (water, air-flow drift), not to the computation. Quote R_mt there with the larger uncertainty (`docs/spectra_recheck.md`).
4a. **Preventing the high-current scatter in future measurements** (it is in the signal, so it must be prevented at the bench):
   - Hold the operating point until the cell voltage drifts < 1 mV/min before the sweep, and repeat one frequency (e.g. 1 kHz) at the start and the end to measure the drift.
   - Keep the air outlet from flooding at high current: higher cathode stoichiometry, lower inlet RH or slightly higher temperature for the EIS point.
   - Scale the excitation with the current. 7 A rms is 1.6 % of 450 A; ~15 A rms keeps the response near 12 mV (|Z| ≈ 0.55 mΩ) and doubles the outlet SNR.
   - Use more periods at the low frequencies, or repeat the sweep 2–3 times and compare; stop at 0.5–1 Hz at 450 A if the lowest points do not repeat.
   - Record the Gamry's AC current on a spare FAMOS channel and start both with one trigger: an AC current closure per step and exact step positions.
4b. **The aggregate is 7–18 % above the Gamry, at every frequency** (2612030: 45 A +8 %, 60 A +7 %, 150 A +13 %, 450 A +18 %; median |Z|). The phase agrees within ±3° up to 1 kHz.
   - It is the same in every build: the 12.10 full-band build only *looked* aligned in the Nyquist plane, because its high-frequency points were rotated onto the Gamry's phase (`gamry_card_phase`). Frequency by frequency it was still 8–17 % high, its points lying on the Gamry curve at about half the frequency.
   - About 13 mΩ·cm² of it is an offset: the UC taps carry ~47 µΩ of series resistance the Gamry sense leads do not (UC2 0.652 V vs Gamry Vdc 0.673 V at 450 A; item 2 above).
   - The rest (~+10 % at low frequency, at every current) is not explained by the data: DC current closure is −2.4 %, and the amplifier chain is flat to 0.03 % below 1 kHz. A simultaneous sweep of a known dummy load through both instruments settles it.
5. **Missing steps.** 2.4 Hz and 9.45 Hz are sometimes weak; the audit located them on every card (9.45 Hz keeps only ~2 whole periods). A step without a detectable tone is left out, not guessed.
6. **Datago.**
   - The condition mapping relies on five-file groups per start minute and on the protocol order.
   - The ADC slot order is approximated (UC, then segments, then Temp). The structural de-skew may therefore be a few µs off per channel; the channel-lag stage, which is measured from the data, absorbs that.
   - Prefer the Volumes `.DAT` files when both exist.
7. **Personal paths** (`uum5fe@bosch.com`) for `curr.csv`, `temp.csv`, the gain file, the module folder fallbacks and one Abgleich search root (§3).
8. **Tile statistics** treat segments 1–36 as their full 5 × 5 tile, even where the gen1 segment itself is smaller (segment 1: 7 pads).
9. **Units and statistics quirks.**
   - `R_hf_extra` in `plate_summary.csv` is in Ω·cm²; every other resistance there is in mΩ·cm².
   - `plate_statistics.csv` does not drop non-positive values; the manifest statistics do.
10. **One test fails on purpose:** `tests/test_card_alignment.py::test_noise_alone_produces_no_prominent_peak`. Pure noise reaches a prominence of 8.5 against the gate of 15. No false alignment is possible, but the test asks for a 2× margin and the margin is 1.8×. Raising the gate could hide real card offsets, so it was left as it is.
11. **No `bode.png`/`phase.png` files.** Bode and phase exist as panels of `gold/nyquist.html` and in the notebook viewer.

---

## 19. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "These modules are not from this build" | The zip was imported over an old folder. Delete the folder and import into an empty one. |
| A condition shows "SKIP" | `cache-only` with nothing cached, or no files found. Check the `.DAT` names or the datago printout of cell 5. |
| "No FAMOS data in datago … falling back to Volumes" | Normal when datago lacks the order. If neither source has it, cell 8 says "file discovery found nothing". |
| The plate map shows other numbers than expected | Wrong plate (gen1/gen2) or a stale cache. The plate is fixed to gen1 for FAMOS; use `rerun`. |
| "card alignment: FAIL" | A card was never time-aligned (weak UC correlation). Its segments are on the wrong windows. Check `bronze/card_alignment.csv`; the Gamry clock may corroborate the lag. |
| A spectrum stops early | Look at `silver/segment_reach.csv`, column `blocked_by`, which names the gate. |
| Gamry comparison missing | No `.dta` for the condition, or a build-token mismatch. The setup cell prints which Gamry folder and build it uses. |
| Results did not change after a code update | The cache key does not include the code version. Run once with `rerun`. |

---

## 20. Glossary

| Term | Meaning |
|---|---|
| Leepa / order | the test order number (e.g. 2612030); `RO2612030` in datago |
| Condition | the operating point, named by its current: 45A, 60A, 150A, 450A |
| Karte / card | one FAMOS acquisition card (16 channels); five per recording |
| UC1/UC2 | cell-voltage channels; UC2 is the reference in Z = K·A_U/A_seg |
| Abgleich | the bench calibration delivery: curr.csv, temp.csv, bode sweeps, temperature steps |
| ASR | area-specific resistance or impedance, Ω·cm² |
| HFR / R_ohmic | high-frequency (ohmic) resistance |
| DRT | distribution of relaxation times γ(τ) |
| lin-KK | linear Kramers–Kronig test of a spectrum's consistency |
| CRLB | Cramér–Rao lower bound: the best possible phasor uncertainty for a given SNR and length |
| Tier A/B/C/D | quality grade of a segment (D = inferred, no own data) |
| Tile | one of the 36 equal 5 × 5-pad blocks used for plate statistics |
| Bronze / silver / gold | the three processing layers: raw, corrected/modelled, plate field |

---
