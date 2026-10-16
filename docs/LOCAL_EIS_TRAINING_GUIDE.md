# Local EIS Evaluation Software — Training Guide

**Who this is for:** a colleague who will run, maintain or extend the local EIS
pipeline and has never seen it before. It starts from the physics and works
down to individual Python functions.

**Which version:** the `local_eis_final.zip` package. It has the sub-folder
layout `core/ pipeline/ checks/ analysis/ plotting/ readers/ tests/ docs/`. The
folder `databricks/local_eis/` in this repository is an *earlier, flat* version
of the same pipeline (all modules in one folder, no `checks/` or `plotting/`
package). Concepts carry over; file paths in this guide refer to the zip.

**How it relates to the other documents:** the zip ships its own reference
manual, `docs/PIPELINE_GUIDE.md`. That manual is a *lookup* document (every
file, every column, every setting). This guide is a *teaching* document: it
explains why things are the way they are, how the pieces connect, and which
programming ideas you need to read the code. Use this guide to learn and the
`PIPELINE_GUIDE.md` to look things up.

---

## Contents

**Part A — Background**
1. [The training plan (how to use this guide)](#1-the-training-plan)
2. [The physical setup: what is measured](#2-the-physical-setup)
3. [EIS in fifteen minutes](#3-eis-in-fifteen-minutes)
4. [The signal-processing ideas the code relies on](#4-signal-processing-ideas)

**Part B — Architecture**

5. [The big picture: one sentence, one diagram](#5-the-big-picture)
6. [How one run flows through the code](#6-how-one-run-flows-through-the-code)
7. [The data objects passed between layers](#7-the-data-objects)

**Part C — The stages in detail**

8. [Bronze: waveforms → raw impedance](#8-bronze)
9. [Silver: correct, gate, model, grade](#9-silver)
10. [Gold: the plate as a field](#10-gold)
11. [Checks: does the plate hold together?](#11-checks)
12. [The CSV path: a second pipeline](#12-the-csv-path)
13. [The runner notebook and the cache](#13-the-runner-notebook-and-the-cache)

**Part D — Every script explained**

14. [Script catalogue, folder by folder](#14-script-catalogue)

**Part E — Programming concepts**

15. [Python and programming concepts used in the pipeline](#15-python-and-programming-concepts)
16. [Numerical and scientific methods used](#16-numerical-and-scientific-methods)

**Part F — Working with it**

17. [Running it: Databricks, command line, tests](#17-running-it)
18. [Reading the outputs: what to open first](#18-reading-the-outputs)
19. [Troubleshooting](#19-troubleshooting)
20. [Known limitations and documentation inconsistencies](#20-known-limitations-and-documentation-inconsistencies)

**Part G — Practice**

21. [Hands-on exercises](#21-hands-on-exercises)
22. [Self-check quiz (with answers)](#22-self-check-quiz)
23. [Glossary](#23-glossary)

---

# Part A — Background

## 1. The training plan

The material is too much for one sitting. A workable split is five sessions of
about two hours each:

| Session | Topics | Sections | Hands-on |
|---|---|---|---|
| 1 | Hardware, EIS basics, why local EIS is hard | 2, 3, 4 | Look at a Nyquist plot from a finished run |
| 2 | Architecture, the run flow, the data objects | 5, 6, 7, 14 (skim) | Exercise 1 and 2 (run the self-test and the tests) |
| 3 | Bronze and silver in depth | 8, 9 | Exercise 3 (synthetic run), Exercise 4 (rejection ledger) |
| 4 | Gold, checks, CSV path, notebook, cache | 10–13, 17, 18 | Run the notebook on Databricks for one order |
| 5 | Python concepts, numerics, maintenance | 15, 16, 19, 20 | Exercise 5 and 6 (change a setting, add a test) |

At the end the colleague should be able to:
- run the notebook for a new order and judge whether the result is trustworthy;
- explain where each number on a heat map comes from;
- find the module responsible for any step and change it safely, with a test.

---

## 2. The physical setup

### 2.1 The fuel cell and the plate

- A **PEM fuel cell** (polymer electrolyte membrane) converts hydrogen and air into
  electricity. Its active area here is **304.92 cm²** (252.0 × 121.0 mm).
- Hydrogen and air flow along channels from one end of the cell to the other.
  The gas composition, humidity and temperature therefore **change along the
  flow direction**. Some regions dry out, some flood, some starve of oxygen.
- A whole-cell measurement only sees the *average*. To see *where* something is
  wrong you must measure **locally**.

The **Bosch R2-D2 measuring plate** sits on one side of the cell and makes local
measurement possible:

| Fact | Value |
|---|---|
| Pad grid | 45 columns × 20 rows = 900 pads of 5.60 × 6.05 mm (0.3388 cm² each) |
| Segments | 72 — each segment is a group of pads wired together |
| Segment areas (gen1) | 1.36 … 8.47 cm² (they are *not* equal) |
| Current measurement | each segment's current flows through vias acting as a shunt; the small voltage drop is amplified |
| Temperature | 4 sensors at x = 0, 84, 168, 252 mm along the flow |
| Two hardware revisions | **gen1** = green / "Kashyyyk", **gen2** = blue / "Naboo". Same pad grid, different grouping of pads into segments 37…72 |

> **Rule 1:** choosing the wrong plate generation does not crash. It silently
> draws the right numbers on the wrong squares. FAMOS data exists only for gen1.

### 2.2 The instruments

```
            Gamry potentiostat                       5 × imc FAMOS cards
     (drives the AC sweep, measures             (record raw waveforms, 16 ch each,
      the WHOLE cell → .DTA file)                typically 25 kHz → .DAT files)
                 │                                         │
                 ▼                                         ▼
     ┌─────────────────────────┐            ┌──────────────────────────────┐
     │   PEM fuel cell          │            │ per card:                    │
     │   + R2-D2 plate          │──────────▶ │  UC1/UC2  cell voltage       │
     │   (72 segments)          │            │  "1"…"72" segment currents   │
     └─────────────────────────┘            │  Temp_1…4 temperatures        │
                 ▲                           └──────────────────────────────┘
       test bench (MF4 log: currents,
       gas flows, pressures, temperatures)
```

- The **Gamry** runs a *galvanostatic* EIS sweep: it superimposes a small AC
  current on the DC load current and steps the frequency (e.g. from 12 kHz down
  to 0.15 Hz). It writes a `.DTA` file with the whole-cell impedance.
- The **FAMOS cards** listen at the same time. Each card has 16 channels and
  its **own clock**. Every card records a copy of the cell voltage, plus 12–15
  segment channels, plus possibly temperature channels.
- The **test bench** logs the operating point (current, flows, pressures) in an
  ASAM **MF4** file.

### 2.3 The calibration ("Abgleich")

The segment channel records a voltage, not a current. The plate was calibrated
on a bench before use. The delivery is called the **Abgleich**:

| File | Content | Used for |
|---|---|---|
| `coefficients/curr.csv` | 72 rows `c0;c1` | K(T) = c0 + 10⁻³·c1·T, in V per (A/cm²) |
| `coefficients/temp.csv` | 4 rows `c0;c1` | T = (u − c0)/c1 |
| `Step<k>_<T>Grad.csv` | raw DC sweeps at several plate temperatures | rebuilding/checking the two files above (`readers/abgleich.py`) |
| `bode/*.DTA` | Gamry sweeps of each segment's measuring chain, 1 Hz–100 kHz | the **chain gain file** (`readers/gamry_dta.py`) |

> **Rule 2:** K converts the segment voltage directly into a current *density*
> (A/cm²), because the calibration already contains the area. Never multiply by
> the area a second time.

### 2.4 Order, condition, card

- **Order / Leepa:** the test order number, e.g. `2612030` (in datago `RO2612030`).
- **Condition:** the operating point, named by its current: `45A`, `60A`,
  `150A`, `450A`. Each condition is a *separate* EIS experiment.
- **Card / Karte:** one FAMOS card. File names look like
  `Leepa_2612030_Current_45A_Test_01_Karte_3.DAT`.

---

## 3. EIS in fifteen minutes

### 3.1 Impedance

Apply a small sinusoidal current i(t) = I·cos(ωt) and measure the voltage
response u(t) = U·cos(ωt + φ). The **impedance** at that frequency is the
complex ratio

```
Z(f) = U_phasor / I_phasor = |Z| · e^{jφ}        ω = 2πf
```

A **phasor** is the complex number that holds the amplitude and phase of a sine
at a known frequency. Repeat for many frequencies and you get a **spectrum**.

### 3.2 Two ways of plotting a spectrum

- **Nyquist plot:** −Im Z against Re Z, one point per frequency. A fuel cell
  gives one or two arcs.
- **Bode plot:** |Z| and phase against log f.

```
 −Im Z
   │            charge-transfer arc     mass-transport arc
   │              ╭──────╮              ╭───────╮
   │            ╭╯        ╰╮          ╭╯         ╰╮
   │           ╱            ╲________╱             ╲
   └──────────●──────────────────────────────────────●──── Re Z
          R_ohmic   (high f, left)          (low f, right)  R_ohmic + R_pol
```

### 3.3 The quantities the pipeline reports

| Symbol | Meaning | Physical cause |
|---|---|---|
| **R_ohmic** (HFR) | where the spectrum meets the real axis at high frequency | membrane resistance + contacts. Rises when the membrane **dries** |
| **R_ct** | size of the faster arc | charge-transfer kinetics (catalyst) |
| **R_mt** | size of the slower arc | mass transport (oxygen reaching the catalyst). Rises with **flooding/starvation** |
| **R_pol** | R_ct + R_mt | total polarisation resistance |
| **L** | series inductance | cables, plate. Not electrochemistry |
| **j_dc** | DC current density in A/cm² | local load distribution |

### 3.4 Area-specific impedance (ASR)

The plate measures current *density*, so local impedance comes out in
**Ω·cm²** (files use mΩ·cm²). It is directly comparable between segments of
different size. The Gamry reports Ω for the whole cell; multiply by 304.92 cm²
to compare.

### 3.5 Segments in parallel

All segments share one cell voltage, so their *admittances* (1/Z) add:

```
Z_cell,ASR = A_cell / Σ_s ( A_s / Z_s )
```

This is a **harmonic, area-weighted mean**. It is dominated by low-impedance
segments, which is why a whole-cell measurement hides a flooded spot. It is also
the quantity the pipeline compares with the Gamry.

### 3.6 Equivalent circuits, CPE and ZARC

A real arc is flatter than an ideal RC semicircle. It is modelled by a **ZARC**
(resistor parallel to a constant-phase element):

```
Z_ZARC = R / (1 + (jωτ)^n)        0.5 ≤ n ≤ 1,  peak at f = 1/(2πτ)
```

The standard PEMFC circuit used here:

```
Z(f) = R_ohmic + jωL + R_ct/(1+(jωτ_ct)^n_ct) + R_mt/(1+(jωτ_mt)^n_mt)
```

### 3.7 DRT (distribution of relaxation times)

Instead of choosing a circuit, write the spectrum as many RC elements with fixed,
log-spaced time constants τ_m and solve for their weights γ_m:

```
Z(ω) = R_∞ + jωL + Σ_m γ_m / (1 + jωτ_m)
```

The plot of γ against τ shows *how many* processes exist and *where*. The
pipeline splits it: τ < 10⁻² s → R_ct, τ ≥ 10⁻² s → R_mt. Solving for γ is
ill-posed, so it needs **regularisation**. The pipeline uses a **Gaussian-process
prior** ("Bayesian DRT"), which also gives uncertainties.

### 3.8 Kramers–Kronig (KK) validation

A spectrum from a **linear, causal, stable, time-invariant** system obeys the
Kramers–Kronig relations. The **lin-KK test** (Boukamp) fits the RC-ladder model
above. If the relative residuals are larger than ~1–2 %, the measurement drifted,
was non-linear, or is corrupted. KK is blind to pure time delays.
**Z-HIT** (reconstructing |Z| from the phase) is not blind to them.

---

## 4. Signal-processing ideas

These five ideas explain about 80 % of the code.

### 4.1 A time delay rotates the phase, linearly in frequency

If the voltage and current channels are not sampled at the same instant, the
ratio picks up

```
Z_measured = Z_true · e^{−j 2π f Δt}
```

That factor is an **all-pass**: |Z| is unchanged, the phase changes by
−360°·f·Δt. At 1 kHz, a 100 µs delay is **36°**, which is a huge error. It
grows with frequency, so it destroys exactly the high-frequency end where
R_ohmic is read. Most of `bronze.py`, `silver.py`, `channel_lag.py` and
`gamry_sync.py` exist to find and remove such delays.

Sources of delay in this rig:

| Source | Size | Removed by |
|---|---|---|
| Cards armed at different times | up to seconds | card alignment (bronze) |
| Card clocks run at slightly different rates | 10–36 ppm | cancels inside one card; reported |
| Multiplexed ADC: channel k sampled k/(16·fs) later | 2.5 µs per slot at 25 kHz | structural skew model (silver) |
| Each segment's shunt/amplifier chain | −57…+76 µs | channel lag stage (silver) |
| Amplifier roll-off | −2.5° at 1 kHz, −11° at 4.5 kHz | chain gain file (bronze) |

### 4.2 Fitting a sine at a known frequency is linear least squares

```
y(t) ≈ a·cos(ωt) + b·sin(ωt) + c [+ d·t]      →   phasor A = a − j·b
```

With ω known, this is a linear system, solved with `numpy.linalg.lstsq`. This is
the IEEE-1057 "three-parameter sine fit". If the frequency is unknown too it
becomes non-linear (four-parameter fit, iterated). `tone_estimation.py` finds
one frequency per card and step by maximum likelihood over all channels.

### 4.3 Uncertainty from the Cramér–Rao bound

For N samples at linear SNR γ, the best possible relative uncertainty of a
phasor is

```
σ_rel ≈ sqrt( 1 / (N · γ) )
```

A long dwell (large N) compensates for a weak signal. The pipeline uses σ_rel
both as a **gate** (drop points with σ_rel > 0.60) and as a **weight** in the
fits.

### 4.4 Cross-correlation finds a lag

Every card records the same cell voltage. Slide card B's copy along card A's
and find where they match best (computed with the FFT). The position of the
peak is the lag. The pipeline accepts it only if the peak is **prominent**
(many robust standard deviations above the background), not just high.

### 4.5 Aliasing

A sampler at fs can only represent frequencies below fs/2. A 10 kHz tone sampled
at 11 kHz shows up as 1 kHz. The CSV logger path explicitly detects this
("UNDERSAMPLED").

---

# Part B — Architecture

## 5. The big picture

**In one sentence:** for every segment and every excitation frequency, compute

> **Z_seg(f) = K_seg(T) · A_U(f) / A_seg(f)**

(A_U = cell-voltage phasor, A_seg = segment-channel phasor, K = calibration),
make that ratio trustworthy, and draw the result as plate maps.

The code is organised as a **medallion architecture** (bronze → silver → gold),
a common pattern on Databricks. Each layer reads only the output of the layer
before it and writes its own folder.

```
 INPUTS                         PROCESSING                                    OUTPUTS
 ──────                         ──────────                                    ───────
 FAMOS .DAT (Volumes) ─┐
 or datago Delta table ┤   ┌──────────────────────────────────────────────┐
 curr.csv / temp.csv ──┤──▶│ BRONZE  find the steps, align cards, fit     │──▶ bronze/
 chain gain file ──────┤   │         phasors, Z = K·A_U/A_seg (no physics)│
 Gamry .DTA ───────────┘   └──────────────────────┬───────────────────────┘
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │ SILVER  de-skew, channel lag, 11 point gates, │──▶ silver/
                           │         DRT + ECM, R_ohmic, KK, tiers A/B/C   │
                           └──────────────────────┬───────────────────────┘
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │ GOLD    R_ct/R_mt split, spatial field,       │──▶ gold/
                           │         fault labels, 36-tile stats, maps     │
                           └──────────────────────┬───────────────────────┘
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │ CHECKS  Gamry vs aggregate, chain response,   │──▶ plausibility.csv,
                           │         plausibility verdicts                 │    gamry_comparison*
                           └──────────────────────┬───────────────────────┘
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │ RUNNER  cache on the Volume, drop-down        │──▶ interactive plots
                           │         viewers, ECM cells, pol. curves       │
                           └──────────────────────────────────────────────┘
```

Why layers?
- **Speed.** Reading tens of GB of `.DAT` is slow. Bronze does it once; silver
  and gold can be re-run from saved bronze files in seconds (`--reevaluate`).
- **Honesty.** Bronze contains only measured values. Every correction is in
  silver and can be switched off and compared.
- **Debuggability.** Each layer's CSVs can be opened and checked by hand.

**Two pipelines, not one.** `source_format = "famos"` runs bronze/silver/gold.
`source_format = "csv"` runs `pipeline/csv_pipeline.py`, a separate pipeline for
a single-clock logger. They share only the geometry, the calibration and the
physics formulas (§12).

---

## 6. How one run flows through the code

### 6.1 Call graph

```
Local EIS Pipeline Runner.py   (Databricks notebook)
│
├─ cell "Setup":  import eis_paths → sys.path; import + reload all modules;
│                 build check (_BUILD_MARKERS)
├─ cell "Widgets": order, conditions, segments, run mode; FIXED SETTINGS
├─ cell "Chain response": readers/gamry_dta.py → chain_gain_gen1.csv (once)
├─ cell "Datago Source": DatagoFamosFile shim, or Volumes .DAT
├─ cell "Run dashboard": cache inventory and run plan
└─ cell "Build Config + Run Pipeline":   for each condition:
      cfg = DEFAULT.replace(...)                         core/config.py
      pipeline/main.py: run_pipeline(cfg)
      │
      ├─ r2d2_geometry.use_plate(cfg.plate)              core/r2d2_geometry.py
      ├─ if csv → csv_pipeline.run_csv(cfg)              pipeline/csv_pipeline.py
      ├─ bronze.run(cfg)  → bronze.save()                pipeline/bronze.py
      │     ├─ discover_files, inventory_channels        eis_local.FamosFile
      │     ├─ PlateCalibration.load                     eis_local.py
      │     ├─ estimate_card_lags                        (cross-correlation)
      │     ├─ plate_temperatures                        r2d2_geometry
      │     ├─ consensus_schedule                        eis_local.detect_schedule,
      │     │                                            ladder_snap, gamry_sync, hf_schedule
      │     ├─ refine_frequencies                        tone_estimation
      │     ├─ card_gain.measure_reference               checks/card_gain.py
      │     └─ process_card  (per card)                  tone_estimation.phasors_at
      ├─ silver.run(br)   → silver.save()                pipeline/silver.py
      │     ├─ dc_share_faults                           neighbours.py
      │     ├─ fit_structural_skew  (per card)           eis_measurement_model
      │     ├─ channel_lag.estimate                      pipeline/channel_lag.py
      │     ├─ gamry_card_phase / card_hf_phase
      │     ├─ process_segment  (per segment)            gate_points, bayesian_drt,
      │     │                                            standard_eis, eis_validation
      │     └─ dc_closure, cell_aggregate_full, build_filled_spectra
      ├─ gold.run(sr)     → gold.save()                  pipeline/gold.py
      │     ├─ collect_parameters, split_processes
      │     ├─ fit_field (Gaussian process), label_faults
      │     └─ plate_stats.tile_stats, plate_figure maps  core/plate_stats.py, plotting/
      └─ whole_cell_and_plausibility
            ├─ gamry_compare.run                         checks/gamry_compare.py
            ├─ frequency_response.run                    checks/frequency_response.py
            └─ plausibility.check_run                    checks/plausibility.py
   → copy run folder to the Volume cache → PIPELINE_RESULTS[cond]
└─ result cells: Nyquist/Bode, Gamry vs aggregate, heat maps, ECM, summary,
   polarisation curves                                   plotting/viewers.py etc.
```

### 6.2 How modules find each other

All modules import each other by **plain name** (`import silver`,
`from config import DEFAULT`), even though they live in different folders. That
works because `eis_paths.py` inserts the root and the six sub-folders into
`sys.path` at import time. The notebook, `run.py`, `pipeline/main.py` and
`tests/conftest.py` all import `eis_paths` first.

Consequence: two modules with the same file name in different sub-folders would
clash. Keep names unique.

### 6.3 The single source of truth

| Question | The one place that answers it |
|---|---|
| Any tunable number | `core/config.py` (`Config` dataclass, `DEFAULT` instance) |
| Segment areas, centroids, pad membership | `core/r2d2_geometry.py` (via `utils.segment_areas(cfg)`) |
| Calibration K(T), T(u) | `pipeline/eis_local.py: PlateCalibration` |
| How a plate map looks | `plotting/plate_style.py` |
| Which settings form the cache key | `_CACHE_IDENTITY_KEYS` in the runner notebook |

---

## 7. The data objects

The layers hand Python objects to each other (and save them as CSV). Knowing
these objects is the fastest way into the code. All are `@dataclass`es.

```
Config (frozen)  ── every setting of the run
   │
   ▼
BronzeRun                                  pipeline/bronze.py
  .schedule   list[Step]          one entry per excitation step: freq, start, stop, amp, snr…
  .channels   {key: ChannelInfo}  every channel: card, ADC slot, kind (segment/reference/temp)
  .cards      {stem: CardInfo}    fs, n_ch, duration, reference channel
  .spectra    {seg: BronzeSpectrum}
  .lags       per-card alignment result
  .grid       geometric frequency grid
     BronzeSpectrum: freq[], Z_raw[] (Ω·cm²), snr_*[], thd[], drift[], n_per_step[],
                     on_grid[], channel_slot, ref_slot, K, K_imputed, T_degC, u_dc
   │
   ▼
SilverRun                                  pipeline/silver.py
  .spectra       {seg: SilverSpectrum}
  .skew          {card: SkewModel}        dt0, k_slot, applied
  .dc_closure    dict
  .cell_freq, .Z_cell                     the parallel aggregate
  .point_ledger  list[dict]               one row per point: kept? which gate?
  .unwired       segments with no channel
     SilverSpectrum: freq[], Z_corr[], Z_model[], sigma_rel[], R_ohmic(+sd), R_pol, L,
                     gamma[], tau_grid[], kk_res_max, zhit_dev, tier, flags[] …
   │
   ▼
GoldRun                                    pipeline/gold.py
  .records  {seg: SegmentRecord}   all 72: class (measured/inferred/bad/…), tier,
                                   values{param: v}, sd{param: sd}, fault
  .fields   {param: {value, sd, info}}
  .stats    plate statistics
```

`Step` is defined in `eis_local.py`; `ChannelLag` in `channel_lag.py`;
`CsvMeasurement` in `readers/csv_source.py` (the CSV path's equivalent of the
bronze input).

---

# Part C — The stages in detail

## 8. Bronze

File: `pipeline/bronze.py`, helped by `eis_local.py`, `tone_estimation.py`,
`ladder_snap.py`, `gamry_sync.py`, `hf_schedule.py`, `checks/card_gain.py`.

**Contract: bronze applies no physics corrections.** It writes measured values,
or values that follow from the calibration algebra. The only modifications are
the chain gain, the card voltage gain and an obvious polarity flip.

### 8.1 Steps in `bronze.run()`

1. **Find the files** — `discover_files(cfg)` tries every pattern in
   `config.FAMOS_PATTERNS`. If none match it falls back to "any .DAT", but still
   filters by condition and **refuses** a folder that would mix conditions
   (that once turned a 45 A request into a run over four conditions that never
   finished).
2. **Inventory the channels** — `inventory_channels` opens each card with
   `FamosFile`. `FamosFile` is a dispatcher: it returns a `FamosV1` (float32,
   all names in one `|CP` field) or a `FamosV2` (float64, one `|CN` block per
   channel) reader with the same interface. Channels are classified: digits →
   segment, `UC*` → voltage reference (`ref_channel = "UC2"`), `Temp*` →
   temperature. The **ADC slot** (position in the header) is recorded because
   silver's skew model needs it.
3. **Load the calibration** — `PlateCalibration.load(curr_cal, temp_cal)`.
   Without `curr.csv` the run stops: it is the only absolute scale. A missing
   row is imputed from the plate median and flagged `K_imputed`.
4. **Align the cards** — `estimate_card_lags`. Each card's UC2 is band-passed
   (0.5–300 Hz) and cross-correlated with the anchor card. The anchor is the card
   the others agree with best. A lag is applied only if |r| ≥ 0.5 **and** the peak
   prominence ≥ 15 robust σ, or if another card or the Gamry clock corroborates it.
5. **Plate temperature** — `plate_temperatures` converts every `Temp_*` channel
   on every card and interpolates linearly along x to each segment's centroid.
   Fallback 58.4 °C (with a warning: a 10 K error is ~4 % on every resistance).
6. **One consensus schedule** — `consensus_schedule`:
   - per card, `eis_local.detect_schedule` scans a log grid of trial frequencies,
     demodulates, finds the dwell window, fits the frequency (`fit4`);
   - steps from all cards are moved to a common time base and clustered (1 %);
   - a geometric grid (fixed points per decade) is fitted to the confident steps;
     a step is kept if ≥ 2 cards saw it **or** it lies on the grid;
   - `ladder_snap` snaps frequencies to exact rungs and repairs windows that break
     the time order;
   - `gamry_sync` puts the record on the Gamry's clock (mode `guide` in the
     notebook): exact frequencies, and misplaced/missing windows re-located and
     verified by two independent array tests (CFAR count and rank-1);
   - overlapping neighbour windows are cut at the midpoint.
7. **Exact frequency per card and step** — `refine_frequencies` →
   `tone_estimation.card_frequency`: a multichannel maximum-likelihood search
   (variable projection over all channels of the card).
8. **Card voltage gain** — `card_gain.measure_reference`: all cards record the
   same UC2, so each card's UC2 phasor ratio to the median card is its gain.
   Measured on AC steps 1–300 Hz with SNR ≥ 30 dB.
9. **Phasors and Z** — `process_card` for every segment and step:
   window → `phasors_at` (least squares on cos, sin, offset, ramp) →
   **Z = K(T)·A_ref/A_seg** → THD, stationarity, SNRs → divide by the chain gain
   and the card gain → polarity check on the lowest decade (`_fix_polarity`).
10. **Duplicates** — if two cards carry the same segment, keep the higher SNR.

### 8.2 Bronze outputs (`<run>/bronze/`)

`raw_spectra.csv`, `segment_meta.csv`, `schedule.csv`, `card_alignment.csv`,
`card_reference.csv`, `gamry_sync.csv`, `frequency_coverage.csv`,
`channels.csv`, `bronze_manifest.json`. `bronze.load(run_dir)` rebuilds a
`BronzeRun` from these files — that is what `--reevaluate` uses.

---

## 9. Silver

File: `pipeline/silver.py`, helped by `channel_lag.py`,
`eis_measurement_model.py`, `eis_validation.py`, `neighbours.py`,
`standard_eis.py`.

**Contract: nothing is deleted silently.** Every dropped point goes into the
rejection ledger with the name of the gate that dropped it.

### 9.1 Order of operations in `silver.run()`

1. **Channel faults** — `dc_share_faults`: a segment whose DC current is < 0.5×
   or > 2× its neighbours is left out (example: segment 33 at 18 %).
2. **ADC skew per card** — `fit_structural_skew`:
   `dt_seg = dt0 + k·(slot_seg − slot_ref)`, k fitted (nominal 1/(16·fs)),
   **dt0 fixed at 0** because a common delay is mathematically indistinguishable
   from a series inductance (L = −R0·dt).
3. **Channel lag per segment** — `channel_lag.estimate`: fit the phase of
   Z_seg / Z_plate-median between 50 Hz and the coherent limit with a pure delay
   τ. Iterated 3×. Accepted only if the phase really follows the model (residual
   ≤ 10°, |τ| ≤ 250 µs); otherwise flagged and demoted to tier C.
4. **Card phase above 1 kHz** — `gamry_card_phase` (anchors each card's
   high-frequency phase to the Gamry sweep) or `card_hf_phase` (to the plate).
5. **Per segment** — `process_segment`:
   - **point gates** (`gate_points`), in order; the first gate to reject a point
     owns it:

     | # | Gate | Rule |
     |---|---|---|
     | 1 | `not_finite` | f > 0 and Z finite |
     | 2 | `outside_band` | f_min ≤ f ≤ evaluated maximum |
     | 3 | `snr` | backstop only (−40 dB) |
     | 4 | `off_tone` | off by default |
     | 5 | `thd` | harmonic distortion ≤ 10 % (linearity) |
     | 6 | `drift` | stationarity test |
     | 7 | `magnitude` | 0.5 < \|Z\| < 800 mΩ·cm² |
     | 8 | `uncertainty` | CRLB σ_rel ≤ 0.60 — **the gate that really decides** |
     | 9 | `cycles` | ≥ 3 periods in the window |
     | 10 | `zmag_outlier` | log\|Z\| within 4.5 robust σ of a 7-point running median |
     | 11 | `passivity` | after timing corrections, Re Z > 0 for f ≥ 1 Hz |

     Fewer than 8 surviving points → segment dropped, with the verdict recorded.
     Note the order: the sign of Re Z is checked **after** de-skewing, because a
     delay can rotate a good point into Re Z < 0 and the correction rotates it back.
   - **corrections:** apply skew, channel lag, card phase, optional
     high-frequency response file;
   - **model:** Bayesian DRT (default `param_method = "drt"`) and, beside it, the
     standard circuit (`ecm_compare = True`, `standard_eis.fit_spectrum`);
   - **R_ohmic:** the high-frequency real-axis crossing, or extrapolated to
     Im Z = 0 from the top 6 points (`r_ohmic_method = "auto"`);
   - **validation:** `lin_kk` and `z_hit`;
   - **tier:** A (KK ≤ 2 % and CV(R_ohmic) ≤ 10 %), B (KK ≤ 6 % and CV ≤ 25 %),
     else C. Imputed calibration caps at B; unreliable channel lag forces C.
6. **Plate products** — DC closure (Σ j·A extrapolated to the plate), cell
   aggregate (segments in parallel), optional rebuilt segments from neighbours.

### 9.2 Silver outputs (`<run>/silver/`)

| File | Use it for |
|---|---|
| `spectra_clean.csv` | the corrected spectra (what the Nyquist plots show) |
| `segments_summary.csv` | one row per segment: tier, R_ohmic, R_pol, flags… |
| `point_rejections.csv` | **the ledger**: why each point was dropped |
| `segment_reach.csv` | highest kept frequency per segment and the gate that stopped it |
| `channel_lag.csv` | τ per segment and its status |
| `skew_model.csv`, `card_factors.csv` | per-card timing and scale |
| `cell_aggregate.csv` | the parallel sum compared with the Gamry |
| `spectra_reconstructed.csv` | segments rebuilt from neighbours |

---

## 10. Gold

File: `pipeline/gold.py`, with `core/plate_stats.py` and `plotting/`.

1. **Parameters per segment** — `collect_parameters`: R_ohmic, R_ct, R_mt, R_pol
   (`split_processes`), Re Z at 1 kHz (`re_at`, interpolated, never
   extrapolated), |Z| and phase at 100 Hz, j_dc, T, chain τ, σ_rel.
2. **Spatial field** — `fit_field`: Gaussian-process regression on the segment
   centroids (Matérn 5/2 kernel, 70 mm correlation along the flow, 35 mm across).
   **Measured segments keep their own value**; the field only fills gaps
   (class `inferred`, tier D), and refuses if more than 60 % would be guessed.
   The notebook runs with `infer_missing_segments=False`.
3. **Fault labels** — `label_faults`: ratios to the plate median
   (`config.FAULT_RULES`): drying, flooding, starvation, contact_loss.
4. **36-tile statistics** — `plate_stats.tile_stats`: the 45 × 20 pads split into
   9 × 4 tiles of 5 × 5 pads; segment k = 1…36 represents tile k. Equal-area
   mean/median/sd, so the denser edge segmentation does not bias the plate mean.
5. **Maps** — 2D interpolated plate maps `plate_<param>_interp.png`, plus
   `nyquist.png/.html`.

Outputs: `plate_summary.csv` (all 72 segments), `plate_statistics.csv`, maps,
`gold_manifest.json`.

---

## 11. Checks

`pipeline/main.py: whole_cell_and_plausibility()` runs after gold. **None of the
checks can stop a run** (each is wrapped in `try/except`): a diagnostic that can
end a run is a liability, not a safeguard.

| Check | Module | Question it answers |
|---|---|---|
| Gamry comparison | `checks/gamry_compare.py` | Do the 72 segments in parallel reproduce the whole-cell Gamry sweep? (shared band only, nothing extrapolated) |
| Chain response | `checks/frequency_response.py` | How does each segment's measuring chain respond, ex-situ (bench) and in-situ (vs plate median)? |
| Plausibility | `checks/plausibility.py` | Does the finished plate hold together as physics? |

Plausibility verdicts (`plausibility.csv`, each PASS / WARN / FAIL / n/a):
coverage by **area**, measured block, current closure, passivity, card
alignment, channel lag, card voltage gain, card impedance factor, R_ohmic
neighbour smoothness (against 200 random shuffles), R_ohmic trend along the
flow, R_s matched-band closure, aggregate vs whole cell, chain response
ex-situ, segment frequency response in-situ.

Why these are separate from the gates: a channel map off by one, a wrong area or
a swapped inlet passes every point-level gate, but shows up in the shape of the
finished plate.

Multi-run checks used by the notebook: `card_gain.py` and `dc_closure.py`
(plate current vs bench current over all conditions: I_plate = a·I_bench + b),
`segment_scale.py` (per-segment scale factors that persist across conditions),
`via_resistance.py`, `hf_response.py` (frozen high-frequency correction built
from several conditions, validated leave-one-current-out),
`card_diagnostics.py`.

---

## 12. The CSV path

File: `pipeline/csv_pipeline.py`, reader `readers/csv_source.py`.

**Why a separate pipeline:** FAMOS has five free-running clocks, so it must
*measure* the synchronisation. A CSV logger has one clock; running the
synchronisation stage there would estimate a quantity that does not exist and
return noise with a plausible-looking error bar.

What the CSV path does instead:

1. **Detect the dialect** (`detect_dialect`): `r2d2`/`r2d2_sweep` (the real
   logger: a folder of `metadata.csv` + `p1.csv, p2.csv …`, one file per
   frequency), `records`, `wide_time`, `long_time`, `freq` (already a spectrum),
   `gamry`. An unrecognised layout raises with the header attached.
2. **Fit phasors on the recorded timestamps** (`fit_phasor`,
   `fit_phasors_multi`) instead of resampling, because logger timestamps jitter.
   All tones of a multisine are fitted at once, so the residual is real noise.
3. **Add quantisation to the error budget** (`quantisation_step`, q²/12): a text
   file is quantised by its printed digits.
4. **Remove the channel scan analytically:** the R2-D2 logger's `timeshifts`
   row gives each column's acquisition instant inside a row (80 channels over
   ~87 µs). Segment 1 and the voltage taps are ~80 µs apart, which is 29° at
   1 kHz. Removed exactly, no fit.
5. **Do not apply `curr.csv` again** for the R2-D2 logger: its `s` columns are
   already A/cm² (the logger applied the coefficient set named in
   `metadata.csv`).
6. **Find the burst and refuse artefacts:** a point file is a burst with silent
   lead-in/out holding a ~999 Hz artefact. The tone is found on the segment
   average, the record is windowed to the burst, and a record whose tone is
   < 10× the noise floor is refused.
7. **Detect aliasing** (`nyquist_zone`): the phase ramp across the channel scan
   reveals the true analogue frequency. Points above Nyquist are flagged
   `UNDERSAMPLED`. Passing `cfg.csv_tones` replaces the inference.
8. **Validate, fit and aggregate** (`_finish_csv`): lin-KK, ZARC circuit with
   AICc choice of arc count, aggregate, outputs. Since the final version the
   Gamry comparison and plausibility checks also run on CSV results.

---

## 13. The runner notebook and the cache

### 13.1 Databricks notebook format

`Local EIS Pipeline Runner.py` is a Databricks notebook stored as a `.py` file:
`# Databricks notebook source` on line 1, `# COMMAND ----------` between cells,
`# DBTITLE 1,<title>` for cell titles, `# MAGIC %md` for markdown cells.
`dbutils.widgets.*` creates the input bar at the top; `spark.sql(...)` queries
the datago Delta tables; `displayHTML(...)` shows the viewers.

### 13.2 The cells, in order

1. restart Python (clean module reload) · 2. **Setup** (find folder, import,
reload, **build check**) · 3. **Widgets** + fixed settings · 4. Chain response ·
5. Datago source · 6. (SQL diagnostic) · 7. Run dashboard (cache vs run plan) ·
8. **Build config + run** (defines `PIPELINE_RESULTS`, prints plausibility) ·
9. Card gain and DC closure · 10. Frequency response · 11. HF response across
conditions · 12. Card diagnostics · 13. Via resistance · 14. Nyquist/Bode ·
15. Gamry vs aggregate · 16. Plate heat maps · 17. MF4 test-stand · 18. ECM
fitting · 19–22. ECM views/tables/maps · 23. Summary table · polarisation curves
(FAMOS and Gamry).

(The numbering differs slightly from `PIPELINE_GUIDE.md` §13 because cells were
added after it was written. Use the cell titles, not the numbers.)

`Local EIS Extra Analyses.py` runs the main notebook with `%run` and adds the
plate-numbering check, Lin-KK, Tikhonov DRT, DRT-informed ECM and the
comparison of the two ECM methods.

### 13.3 Widgets (what the user chooses)

Order ID, measurement format (famos/csv), CSV campaign, **Conditions**
(multi-select), Exclude segments, Rebuild from neighbours, Fill unmeasured
segments, **Run mode** (`cache` / `rerun` / `cache-only`), Write results to
cache. Band, SNR and gate settings are **fixed in code** (`BAND = 'complete'`,
`MIN_SNR_DB = 0`, `CHANNEL_LAG = 'correct'`, `GAMRY_SYNC = 'guide'`,
`HF_RESPONSE = 'auto'`).

### 13.4 The build check

`_BUILD_MARKERS` lists functions each module must contain in this version. If a
module comes from another folder or an older version, the notebook stops with
"These modules are not from this build". Cause: the zip was imported over an
old folder. Fix: import into an **empty** folder.

### 13.5 The cache

- Location: `/Volumes/…/EIS_Results/<leepa>/<cond>/mode_default/snr_0.0_<id>/`.
- `<id>` = 10-character hash of every result-changing setting
  (`_CACHE_IDENTITY_KEYS`), including the content of the gain file.
- **The code version is not in the hash.** After a code change, run once with
  `Run mode = rerun`.
- When you add a new result-changing setting, add its name to
  `_CACHE_IDENTITY_KEYS`, otherwise the cache serves stale results.

---

# Part D — Every script explained

## 14. Script catalogue

Each entry: **what** it does · **why** it exists · **key functions** · **used by**.

### 14.1 Top level

| File | What / why | Key items | Used by |
|---|---|---|---|
| `Local EIS Pipeline Runner.py` | The main Databricks notebook: widgets → run → checks → results. It is the only thing a normal user touches. | `_BUILD_MARKERS`, `_CACHE_IDENTITY_KEYS`, `PIPELINE_RESULTS`, `DatagoFamosFile` | the user |
| `Local EIS Extra Analyses.py` | Optional diagnostics kept out of the main notebook so it shows only results. `%run`s the runner. | Lin-KK, Tikhonov DRT, DRT-informed ECM, method comparison | the user |
| `eis_paths.py` | Puts `core/ pipeline/ checks/ analysis/ plotting/ readers/` on `sys.path` so modules import each other by plain name. | `add()`, `SUBDIRS` | notebook, `run.py`, `main.py`, `conftest.py` |
| `run.py` | Command line for any module: `python run.py <module> [args]` → that module's `main()`. | `importlib.import_module` | developers |
| `requirements.txt` | numpy, scipy, matplotlib (+ pandas, plotly). Unpinned on purpose (Databricks runtimes already ship them). | | local setup |

### 14.2 `core/` — shared foundations

| File | What / why | Key items |
|---|---|---|
| `config.py` | **Every tunable number in one place.** Before it existed, four files carried diverging copies of the same constants and results could not be reproduced. One frozen dataclass, one default instance, one way to override. Also physical constants, presentation constants, `FAULT_RULES`, `HEATMAP_LIMITS`, port layout. | `Config`, `DEFAULT`, `.replace()`, `.preset()`, `.from_json()`, `.from_env()`, `.from_cli()`, `f_hi()`, `FAMOS_PATTERNS` |
| `utils.py` | Everything shared by more than one layer: sine fits, the joint two-channel fit, CRLB uncertainty, outlier tests, delay helpers, geometric-grid fit, logging that survives a Windows cp1252 console, CSV/JSON writers. | `fit3`, `fit4`, `fit7_joint`, `crlb_phasor`, `sigma_rel_from_snr`, `zmag_outliers`, `apply_delay`, `geometric_grid_fit`, `get_logger`, `banner`, `write_table`, `write_json`, `segment_areas` |
| `r2d2_geometry.py` | **Both plate maps.** Pads → segments, areas, centroids, label positions, temperature interpolation. Plate-aware via `use_plate()`, which rebinds module-level names so every consumer sees the change. Self-check asserts 900 pads covered exactly once and areas sum to 304.92 cm². | `Segment`, `Plate`, `PLATES`, `use_plate`, `areas`, `centroids`, `segment_temperatures`, `self_check`, `plot_map` |
| `plate_stats.py` | Equal-area plate statistics over 36 tiles of 5×5 pads, so edge segmentation does not bias the mean. | `tile_of`, `tile_stats`, `write_statistics` |
| `plate_conditions.py` | Temperature (measured), pressure (interpolated) and humidity (modelled water balance) across the plate, each with a provenance label. | `condition_fields`, `cathode_stoichiometry`, `saturation_pressure_pa` |

### 14.3 `pipeline/` — the processing

| File | What / why | Key items |
|---|---|---|
| `main.py` | Orchestrator and CLI. `run_pipeline` runs bronze → silver → gold → checks and writes `run_manifest.json`. `reevaluate` reruns silver/gold from saved bronze CSVs. `self_test` checks everything that needs no data. | `run_pipeline`, `reevaluate`, `whole_cell_and_plausibility`, `self_test`, `build_parser` |
| `bronze.py` | Layer 1 (§8). Files → raw Z per segment, no physics. | `discover_files`, `inventory_channels`, `estimate_card_lags`, `consensus_schedule`, `refine_frequencies`, `process_card`, `run`, `save`, `load` |
| `eis_local.py` | The original standalone evaluator, now a library: FAMOS readers (v1/v2), calibration, gain file, sine fits, blind step detection, DC closure, aggregation. | `FamosFile`, `FamosV1`, `FamosV2`, `PlateCalibration`, `load_gain`, `gain_at`, `detect_schedule`, `Step`, `aggregate` |
| `tone_estimation.py` | One exact frequency per card and step by multichannel maximum likelihood (variable projection), because per-pair Gauss–Newton fits converged off-tone ~30 % of the time. | `card_frequency`, `phasors_at`, `stationarity_test` |
| `ladder_snap.py` | Snaps detected frequencies to the sweep's geometric ladder (integer points per decade); a 1 % frequency error costs ~16 dB of phasor SNR. Repairs windows that break time order. | `fit_ladder`, `snap_steps`, `repair_windows` |
| `hf_schedule.py` | Recovers the top decade of the sweep by stacking the segment channels (their tone is flat in f, unlike the cell voltage). CFAR and rank-1 tests verify a rung exists. | `polarity_aligned_reference`, `fit_ladder`, `cfar_channel_count`, `rank1_statistic`, `recover_schedule` |
| `gamry_sync.py` | Puts the FAMOS record on the Gamry clock: exact frequencies, misplaced/missing windows relocated and verified, windows confined between Gamry step ends, refused card lags corroborated. Modes `report`/`frequency`/`guide`. | `read_timeline`, `align`, `apply_frequencies`, `relocate`, `confine_windows`, `corroborate_card_lags` |
| `silver.py` | Layer 2 (§9). Corrections, gates, models, validation, tiers, ledger. | `fit_structural_skew`, `gate_points`, `process_segment`, `bayesian_drt`, `r_ohmic_intercept`, `gamry_card_phase`, `dc_share_faults`, `cell_aggregate_full`, `run`, `save` |
| `channel_lag.py` | Per-segment chain lag against the plate median, measured on every run; removed before R_ohmic is read. | `estimate`, `fit_tau`, `correction`, `plate_reference` |
| `eis_measurement_model.py` | Boukamp/Agarwal measurement model: fit Voigt+L, use the residual as KK compliance, aggregate ASR. | `fit_kk_model`, `fit_delay_and_model`, `aggregate_asr` |
| `eis_validation.py` | Instrument-independent quality tests: lin-KK with Schönleber's μ criterion, Z-HIT. | `lin_kk`, `z_hit`, `estimate_delay` |
| `neighbours.py` | Adjacency from shared pad edges; robust z-scores against the neighbour ring (gradient cancels, fault doesn't); rebuilds a segment from neighbours. | `adjacency`, `scalar_outliers`, `spectrum_outliers`, `fill_spectrum` |
| `standard_eis.py` | The textbook circuit fit (R_ohmic + L + two ZARCs) by complex non-linear least squares, as ZView/Echem Analyst do it; HF intercept; graphical R_pol. Always two arcs (its `choose_n_arcs` is a drop-in that does not choose). | `hf_intercept`, `capacitive_band`, `fit_spectrum`, `choose_n_arcs` |
| `gold.py` | Layer 3 (§10). Parameters, GP field, fault labels, tile stats, maps. | `split_processes`, `collect_parameters`, `re_at`, `fit_field`, `label_faults`, `run`, `save` |
| `csv_pipeline.py` | The separate CSV pipeline (§12). | `detect_tones`, `r2d2_point`, `nyquist_zone`, `deskew`, `segment_spectra`, `fit_ecm`, `run_csv` |

### 14.4 `checks/` — is it believable?

| File | What / why |
|---|---|
| `plausibility.py` | Plate-level physics verdicts → `plausibility.csv` (§11). Built for partially wired plates: coverage is judged by **area**, not segment count (segments 1–36 are 68.6 % of the area). |
| `gamry_compare.py` | Whole-cell Gamry sweep vs local aggregate, in the same units, on the shared band, with the bench operating point read from the MF4 log. Shapes of disagreement are diagnostic (offset → series R; scale → area/calibration; growing phase → delay/chain). |
| `frequency_response.py` | Ex-situ amplifier response (Abgleich bode) and in-situ segment response (vs plate median). |
| `hf_response.py` | Measures the residual high-frequency card/segment response across conditions, validates leave-one-current-out, writes a frozen correction file applied by silver. |
| `card_gain.py` | Card voltage-chain gain from UC2 phasors; card impedance factors; plate vs bench current. |
| `card_diagnostics.py` | Median Nyquist/Bode per card against the Gamry, card-to-Gamry transfer across currents: separates acquisition, chain and cell effects. |
| `dc_closure.py` | Σ segment currents vs bench current over all conditions; scale vs offset; voltage-sense offset. |
| `segment_scale.py` | Per-segment scale factors that persist across conditions (reported, not applied). |
| `via_resistance.py` | Via/shunt resistance vs temperature per card, from the Abgleich step files. |
| `validate_hf.py` | Synthetic self-validation of the high-frequency chain with known truth and skew present. |

### 14.5 `analysis/`

| File | What / why |
|---|---|
| `ecm_drt.py` | DRT-informed circuit fit ("method B"): arcs sorted by τ, CRLB weighting, optional transmission-line element for the catalyst layer, mains-frequency masking. |
| `polcurve.py` | Polarisation curves (voltage vs current density) from FAMOS (whole plate and three flow bands) and Gamry. |

### 14.6 `plotting/`

| File | What / why |
|---|---|
| `viewers.py` | **One plot with drop-downs** instead of a stack of figures: all figures embedded in one HTML block, switching is instant and works when saved. |
| `nyquist.py` | The per-segment Nyquist/Bode figure, coloured by position along the air path; Gamry overlay; same plot saved and shown. |
| `plate_figure.py` | Static plate maps in the style of the plate viewer (frame, ports, sensors), per segment or 2D interpolated. |
| `plate_plotly.py` | Interactive plate map: true segment outlines, value printed, hover, parameter drop-down. |
| `plate_maps.py` | Value maps with numbers written inside segments and robust colour limits. |
| `plate_style.py` | One look for every map: Jet colour scale, text sizes, orientation, port labels. |
| `figure_panels.py` | Splits a multi-panel Plotly figure into full-width separate figures. |
| `bench_plots.py` | MF4 bench channels as one interactive figure. |

### 14.7 `readers/`

| File | What / why |
|---|---|
| `gamry_dta.py` | Reads any Gamry `.DTA` (ZCURVE). Builds the chain gain file from the Abgleich bode sweeps. **Sign convention:** the file must carry gain = 1/H because the pipeline applies Z ← Z/gain. Cross-checks the bode index against `curr.csv` and refuses an inconsistent pairing (use `--shared` for gen2). |
| `abgleich.py` | Rebuilds `curr.csv`/`temp.csv` from the raw `Step*_<T>Grad.csv` files so a delivered calibration can be checked rather than trusted. |
| `csv_source.py` | Every CSV dialect into one `CsvMeasurement` shape; non-uniform phasor fits; quantisation step. |
| `_vendor/zstd.py` | Stand-in for the PyPI package `zstd` that `asammdf` imports, which has no Windows wheels for Python ≥ 3.11. Routes to `zstandard`. |

### 14.8 `tests/` and `docs/`

- `tests/`: ~457 pytest tests. `conftest.py` imports `eis_paths`.
  `make_synth_famos.py` and `make_synth_sweep.py` write synthetic FAMOS cards and
  a matching Gamry `.DTA` with **known** impedance and deliberately imposed
  timing faults, so the tests check that the pipeline *inverts* the
  measurement, not that it reproduces its own convention. `fixtures/r2d2_sample/`
  is 2 500 rows of a real logger file.
- `docs/`: `PIPELINE_GUIDE.md` (reference manual), `FIXES.md` (change log with
  numbers), analyses of specific orders, test plans, figures.

---

# Part E — Programming concepts

## 15. Python and programming concepts

Each concept: what it is, where it appears, why it was chosen.

### 15.1 Modules, packages and `sys.path`

A `.py` file is a **module**; `import silver` executes it once and caches it in
`sys.modules`. Python finds modules by searching the folders in `sys.path`.
`eis_paths.add()` inserts the sub-folders at the front, which is why
`import config` works from anywhere. This is a deliberate choice over a proper
package (`from local_eis.core import config`) because Databricks notebooks and
the older flat layout both import by plain name.

```python
# eis_paths.py
ROOT = Path(__file__).resolve().parent
SUBDIRS = ("core", "pipeline", "checks", "analysis", "plotting", "readers")
for p in [ROOT] + [ROOT / d for d in SUBDIRS]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
```

`importlib.reload(mod)` in the notebook re-executes a module after you edit it;
`importlib.import_module(name)` in `run.py` imports a module whose name is only
known at runtime.

### 15.2 Dataclasses

`@dataclass` generates `__init__`, `__repr__` and `__eq__` from annotated fields.
Used for every container (`Config`, `Step`, `BronzeSpectrum`, `SilverSpectrum`,
`SegmentRecord`, `ChannelLag` …, 44 in total).

```python
@dataclass(frozen=True)
class Config:
    plate: str = "gen1"
    f_min_hz: float = 0.15
    exclude_segments: frozenset[str] = field(default_factory=frozenset)

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

cfg = DEFAULT.replace(condition="45A", f_min_hz=0.2)   # new object; DEFAULT unchanged
```

- `frozen=True` makes the object **immutable**. A config that changes halfway
  through a run is how the old notebook ended up with two different frequency
  limits in one figure. To vary it you create a copy with `.replace()`.
- `field(default_factory=...)` gives each instance its own mutable default.
- `field(init=False)` (in `FamosV1`) marks attributes computed in
  `__post_init__` rather than passed in.
- `asdict()` turns the object into a dict for JSON (`Config.to_dict`).

### 15.3 Properties, class methods and alternative constructors

- `@property` makes a computed value look like an attribute:
  `BronzeSpectrum.j_dc`, `SkewModel.slot_ratio`, `Plate.title`.
- `@classmethod` builds an object from another source:
  `Config.from_json`, `Config.from_env`, `Config.from_cli`,
  `PlateCalibration.load`, `Segment.from_pads`.
- `FamosFile.__new__` is a **factory**: calling `FamosFile(path)` returns a
  `FamosV1` or `FamosV2` instance depending on file content. Callers never know
  which dialect they got (duck typing, §15.6).

### 15.4 Type hints and `from __future__ import annotations`

Signatures like `def run(cfg: Config = DEFAULT, log=None) -> BronzeRun:` and
`dict[str, BronzeSpectrum]`, `Path | None` document the data flow. They are not
enforced at runtime. The `__future__` import (in 86 files) makes annotations
lazy, so new syntax works on older Pythons and classes can reference each other.

### 15.5 Configuration layering and the command line

`argparse` builds every CLI (`main.build_parser`, `Config.build_argparser`).
Precedence is CLI > JSON file > environment variables (`EIS_*`) > dataclass
default. `Config._coerce` converts the strings coming from those sources into
typed fields by inspecting each field's annotation.

### 15.6 Duck typing and shims

Python cares about what an object *can do*, not its class. The notebook's
`DatagoFamosFile` reads waveforms from a Delta table but offers the same
methods (`.names`, `.fs`, `.channel(name)`, `.position(name)`) as the binary
reader. The notebook replaces `bronze.FamosFile` with it ("monkey-patching"),
and bronze works unchanged. The vendored `zstd.py` is the same idea at module
level.

### 15.7 Module-level state and `global`

`r2d2_geometry.use_plate()` rebinds module-level names with `global`:

```python
def use_plate(name):
    global ACTIVE_PLATE, SEGMENTS, LABEL_COL, LABEL_ROW
    p = plate(name)
    ACTIVE_PLATE, SEGMENTS = p, p.segments
    ...
```

Every consumer that does `import r2d2_geometry as geom` and then reads
`geom.SEGMENTS` sees the new plate. The catch: `from r2d2_geometry import
SEGMENTS` would copy the *old* binding and not see the change. That is why the
code always uses `geom.SEGMENTS`. `use_plate` must run before anything reads
the geometry, which is why `run_pipeline` calls it first.

### 15.8 Closures

A function defined inside another can use and modify the outer function's
variables. `silver.process_segment` uses this for the gates and the ledger:

```python
reason = np.array([""] * freq.size, dtype=object)

def gate(ok, name):
    newly = (reason == "") & ~ok     # first gate to reject a point owns it
    reason[newly] = name
    return ok

keep  = gate(np.isfinite(Z), "not_finite")
keep &= gate(mag < 800, "magnitude")
```

### 15.9 NumPy vectorisation and boolean masks

Loops over thousands of samples would be slow in pure Python. NumPy works on
whole arrays at once:
- boolean masks: `keep &= (freq >= f_min) & (freq <= f_max)`;
- complex arrays: `Z * np.exp(2j*np.pi*freq*dt)` de-skews a whole spectrum;
- `np.where`, `np.nanmedian`, `np.isfinite` handle missing values (`NaN`).
- `np.errstate(divide="ignore", invalid="ignore")` is a **context manager**
  (`with` block) that silences expected divide-by-zero warnings locally.

### 15.10 Memory-mapped files

`np.memmap` maps the `.DAT` file into memory without loading it. Reading one
channel touches only that channel's bytes:

```python
mm = np.memmap(path, dtype="<f4", mode="r", offset=header_len,
               shape=(n_samples, n_ch))
x = np.asarray(mm[:, names.index(name)], dtype=np.float64)
```

`"<f4"` = little-endian 32-bit float (v1). v2 files are float64, which is why
the dtype is read from the header, never assumed.

### 15.11 Regular expressions and binary headers

`re.search(r"\|CD,\d+,([\d.eE+-]+)", raw)` pulls the sample interval out of the
FAMOS header; `re.fullmatch(r"\d+", name)` decides that a channel named `"17"`
is a segment. The header is decoded as **latin-1** because channel names contain
umlauts. `identify_file.py`/`ddf_source.py` (earlier versions) read file
signatures from the first bytes rather than trusting extensions.

### 15.12 Error handling as a design principle

The code distinguishes three kinds of failure:
1. **Fatal input errors** → `raise SystemExit("…")` with an explanation (no
   calibration, no files, mixed conditions).
2. **Per-segment failures** → caught, logged, recorded, the run continues
   ("never lose the plate").
3. **Diagnostics** → wrapped in `try/except Exception` so a failing check can
   never end a run.

A recurring rule in the comments: *refuse rather than guess*. A reader that
half-succeeds on the wrong layout is worse than one that stops.

### 15.13 Logging

`utils.get_logger()` returns a `logging.Logger` with a custom stream handler
that replaces characters a Windows cp1252 console cannot print (box drawing,
µ, Ω, °). `utils.banner()` and `utils.section()` give the run log its structure.

### 15.14 Files, paths and serialisation

- `pathlib.Path` everywhere (`out / "silver" / "spectra_clean.csv"`).
- `utils.write_table` writes lists of dicts as CSV; `utils.write_json` writes
  manifests (converting NumPy types).
- `pandas.read_csv` in checks, viewers and the notebook.
- `hashlib.sha256` makes the config/input digests and the cache id.

### 15.15 Interactive output

Plotly figures (`go.Figure`, `make_subplots`) are serialised into one HTML
string (`viewers.selector_html`) with JavaScript drop-downs; Databricks shows it
with `displayHTML`. Matplotlib writes static PNGs with the non-GUI `Agg` backend.

### 15.16 Testing with pytest

- Plain functions named `test_*`, plain `assert`.
- **Fixtures**: `tmp_path` gives a fresh temporary folder (368 uses);
  `monkeypatch` swaps functions or attributes for one test (42 uses).
- `@pytest.mark.parametrize` runs one test over several inputs.
- `conftest.py` runs before every test file (here: `import eis_paths`).
- **Test philosophy:** plant a known defect (a lag, a scale error, a swapped
  channel) in synthetic data and assert that the check which should see it does,
  and that the others stay quiet.

### 15.17 Databricks specifics

| Concept | Where |
|---|---|
| `dbutils.widgets.dropdown/multiselect/text/get` | notebook widget bar |
| `spark.sql("SELECT …")` | datago Delta tables (`ps_xplatform_dev.rvadvtec_ops.*`) |
| Unity Catalog Volumes `/Volumes/…` | raw `.DAT`, Gamry folders, cache |
| `%run "./Local EIS Pipeline Runner"` | the extra-analyses notebook |
| `dbutils.library.restartPython()` | first cell: clean interpreter |
| `displayHTML` | the viewers |

---

## 16. Numerical and scientific methods

| Method | Used for | Where |
|---|---|---|
| Linear least squares (`np.linalg.lstsq`) | 3-parameter sine fit, lin-KK, Voigt fits | `utils`, `tone_estimation`, `eis_validation`, `eis_measurement_model` |
| Gauss–Newton iteration | 4- and 7-parameter sine fits (frequency unknown) | `utils.fit4`, `utils.fit7_joint` |
| Variable projection / concentrated maximum likelihood | one frequency per card and step | `tone_estimation.card_frequency` |
| Cramér–Rao lower bound | per-point uncertainty, gate and weight | `utils.crlb_phasor`, `sigma_rel_from_snr` |
| FFT cross-correlation, robust prominence (median/MAD) | card lags | `bronze._best_lag`, `_prominence` |
| Demodulation envelope, moving mean | finding dwell windows | `eis_local.demod_envelope`, `dwell_window` |
| Geometric-grid fit | sweep ladder | `utils.geometric_grid_fit`, `ladder_snap` |
| CFAR detection, rank-1 test | verifying a tone exists across channels | `hf_schedule` |
| Robust statistics (median, MAD, weighted median) | correlation prominence, \|Z\| outliers, polarity decision | `bronze._prominence`, `utils.zmag_outliers`, `bronze._fix_polarity` |
| Gaussian-process regression (Cholesky solve, marginal likelihood) | Bayesian DRT; spatial field over the plate | `silver.bayesian_drt`, `gold.fit_field` |
| Tikhonov (ridge) regularisation | DRT in the extra notebook | `Local EIS Extra Analyses.py` |
| Complex non-linear least squares with bounds (`scipy.optimize.least_squares`), multi-start | circuit fits | `standard_eis`, `analysis/ecm_drt`, `csv_pipeline` |
| AICc model selection | number of arcs (the pipeline's own `standard_eis` always fits two arcs) | `csv_pipeline.choose_n_arcs`, runner "ECM Fitting" cell |
| Jacobian standard errors | parameter uncertainties | `standard_eis`, `csv_pipeline.fit_ecm` |
| Lin-KK with μ criterion, Z-HIT | spectrum validation | `eis_validation` |
| Modified z-score (Iglewicz–Hoaglin) | neighbour outliers | `neighbours` |
| Linear interpolation in log f | gain curves, Re Z at 1 kHz | `utils.gain_at`, `gold.re_at` |

---

# Part F — Working with it

## 17. Running it

### 17.1 On Databricks (normal use)

1. Import the zip into an **empty** workspace folder (never on top of an old one).
2. Open `Local EIS Pipeline Runner`, attach an ML-runtime cluster.
3. Run the first three cells; set Order ID, Conditions, Run mode.
4. Run all. Read the "Run plan" (load from cache or compute, 5–10 min per
   condition) and the plausibility verdicts printed by the run cell.
5. Use the result cells (drop-downs for condition, parameter, view).

**Personal paths:** `curr.csv`, `temp.csv` and `chain_gain_gen1.csv` live under
`/Workspace/Users/uum5fe@bosch.com/`. A new owner copies them into their own
workspace and changes the paths in the notebook (search for `uum5fe`).

**Data sources:** datago first (Spark SQL on the Delta tables), Volumes `.DAT`
as fallback. Volumes ≈ 5 s per card, datago ≈ 30–60 s per card.

### 17.2 Command line (development)

```bash
cd local_eis
pip install -r requirements.txt pytest

python run.py main --self-test                        # no data needed
python -m pytest tests -q                             # ~457 tests, ~5 min
python run.py main --reevaluate RUN_DIR --out OUT     # silver/gold again from bronze CSVs
python pipeline/main.py --dat <Famos dir> --curr-cal curr.csv --temp-cal temp.csv \
       --gamry <Gamry dir> --leepa 2612030 --condition 45A --out out/45A
python run.py frequency_response RUN_DIR --bode <Abgleich>/bode
python run.py dc_closure RUN_45A RUN_60A ... -o OUT
python readers/gamry_dta.py <Abgleich>/bode --curr-cal <...>/curr.csv -o chain_gain.csv
```

Verified for this guide on the zip contents: `python -m pytest tests` gives
**453 passed, 3 skipped, 1 failed**. The failure is the documented, intentional
one (`test_card_alignment.py::test_noise_alone_produces_no_prominent_peak`,
see §20).

### 17.3 Shipping a new version

```bash
git archive --format=zip --prefix=local_eis/ HEAD:databricks/local_eis > Local_Eis.zip
```

Then import into an empty folder. If a new version needs a new function, add it
to `_BUILD_MARKERS`; if a new setting changes results, add it to
`_CACHE_IDENTITY_KEYS`.

---

## 18. Reading the outputs

```
<run>/                          e.g. EIS_Results/2612030/45A/mode_default/snr_0.0_<id>/
├── config_used.json            every setting of this run
├── run_manifest.json           stage summaries, elapsed time
├── plausibility.csv            ← open first
├── gamry_comparison.csv / _curves.csv / .png
├── frequency_response/
├── bronze/   raw_spectra, segment_meta, schedule, card_alignment, gamry_sync …
├── silver/   spectra_clean, segments_summary, point_rejections, segment_reach …
└── gold/     plate_summary, plate_statistics, plate_*_interp.png, nyquist.html
```

**Order in which to judge a run:**

1. `plausibility.csv` — any FAIL? Card alignment and current closure first.
2. `gold/plate_statistics.csv` — plate mean/median/CV over the 36 tiles.
3. `silver/segments_summary.csv` — tiers and flags per segment.
4. `silver/segment_reach.csv` — why a spectrum stops early (`blocked_by`).
5. `bronze/gamry_sync.csv` — were the windows where the Gamry says?
6. `gamry_comparison.png` — does the aggregate match the whole cell?

Units: impedances Ω·cm² inside the code, **mΩ·cm²** in output CSVs (exception:
`R_hf_extra` in `plate_summary.csv` is Ω·cm²).

---

## 19. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "These modules are not from this build" | zip imported over an old folder | delete folder, import into an empty one |
| Results unchanged after a code update | cache key ignores code version | Run mode = `rerun` |
| "card alignment: FAIL" | a card's lag was refused | check `bronze/card_alignment.csv`; Gamry clock may corroborate |
| All segments of one card fail the SNR/uncertainty gate | that card's windows sit on the wrong tone (misalignment) | same as above |
| Spectrum stops early | a gate removed the top points | `silver/segment_reach.csv` → `blocked_by` |
| Segment missing | not wired, excluded, DC-share fault, or < 8 points | `point_rejections.csv`, log line "segment N: dropped" |
| Map numbers look wrong | wrong plate or stale cache | gen1 for FAMOS; `rerun` |
| Gamry comparison missing | no `.DTA` or build-token mismatch | setup cell prints which Gamry folder/build it uses |
| `pip install asammdf` fails on Windows | `zstd` has no wheels for Python ≥ 3.11 | the vendored shim handles it; install `zstandard` |
| "incomplete FAMOS header" | unknown FAMOS dialect | `FamosFile` reports every attempt; check v1/v2 |

---

## 20. Known limitations and documentation inconsistencies

### 20.1 Limitations (from the code and `PIPELINE_GUIDE.md` §18)

1. **Top of the band.** Above ~1 kHz the card phase is anchored to the Gamry,
   so the HF crossing is not fully independent of it until an injected
   calibration sweep exists.
2. **Aggregate 7–18 % above the Gamry** at every frequency (order 2612030).
   About 13 mΩ·cm² is a voltage-sense series offset; the rest is unexplained.
   A simultaneous dummy-load sweep through both instruments would settle it.
3. **Low-frequency scatter at the 450 A air outlet** is in the signal (outlet
   segments scatter together), not in the processing.
4. **Datago:** condition mapping is a heuristic; ADC slot order is approximated
   (the channel-lag stage absorbs the error).
5. **gen2 geometry** is a reconstruction from drawings (`verified=False`); gen1
   is verified against the plant's pad matrix.
6. **Personal workspace paths** (`uum5fe@bosch.com`).
7. **One intentional test failure**: pure noise reaches prominence 8.5 against a
   gate of 15; the test wants a 2× margin, the margin is 1.8×. Raising the gate
   could hide real card offsets, so it was left.

### 20.2 Inconsistencies found while preparing this guide

Tell your colleague about these so they don't lose time:

| Where | Says | Code says |
|---|---|---|
| zip `README.md` "final version in one paragraph" | R_ct/R_mt computed "the standard way … (no DRT)" | `config.param_method = "drt"` (Bayesian DRT is primary); the circuit fit runs **beside** it (`ecm_compare = True`). `PIPELINE_GUIDE.md` §4 is correct. |
| `PIPELINE_GUIDE.md` §8.3, last paragraph | `"drt"` is "the earlier approach … kept for comparison only" | it is the default |
| `docs/FIXES.md`, `docs/GEN2_…md` | gen1 areas 0.678 … 8.470 cm² (×12.5) | `r2d2_geometry` self-check: 1.355 … 8.470 cm² (the gen1 map was corrected to the plant pad matrix later) |
| `PIPELINE_GUIDE.md` §13 cell numbers | 23 cells, numbering | cells were added (HF response, card diagnostics); use titles |
| Config default vs notebook | `eval_f_max_hz = 2400` | notebook sets `BAND = 'complete'` → up to 0.49·fs |
| Repo `databricks/local_eis/stage_*.py` | — | dev scripts with hard-coded `/home/claude/...` paths; `stage_gold.py` loads `silver.pkl` but `stage_silver.py` writes `silver_true.pkl`/`silver_equal.pkl` (older layout only) |

---

# Part G — Practice

## 21. Hands-on exercises

**Exercise 1 — Self-test (10 min).**
Run `python run.py main --self-test`. Find in the output: the area range of
gen1 and gen2, and the CSV reader checks. Question: why does gen2 have a smaller
area range than gen1?

**Exercise 2 — The test suite (15 min).**
Run `python -m pytest tests -q`. Open the failing test and the comment in
`config.py` above `align_min_prominence`. Explain in two sentences why the
threshold was not changed.

**Exercise 3 — A synthetic end-to-end run (30 min).**
Write three synthetic cards (18 segments, a descending sweep, card offsets and
multiplexer delay built in) plus a matching Gamry `.DTA`, then run the pipeline:

```bash
python tests/make_synth_sweep.py /tmp/synth
python pipeline/main.py --dat /tmp/synth --curr-cal /tmp/synth/curr.csv \
    --temp-cal /tmp/synth/temp.csv --gamry /tmp/synth \
    --leepa 9999999 --condition 450A --out /tmp/synth_out --no-html
```

It takes about 3–4 minutes. Expected log ending: `18/72 segments measured`,
`R_ohmic mean ≈ 54 mΩ·cm²`, card alignment PASS, passivity PASS, and
**R_ohmic smoothness FAIL**. Discuss why the FAIL is *correct* here: the
generator draws each segment's resistance at random, so the map really is
indistinguishable from random placement. That is exactly the pattern the check
exists to catch on real data (a channel-mapping error). Then:
- compare R_ohmic in `silver/segments_summary.csv` with the generator's values
  (`tests/make_synth_sweep.py`, `true_z`);
- re-run with `--skew none` into another folder and compare the in-situ phase
  spread in the plausibility output and R_ohmic.

**Exercise 4 — The rejection ledger (20 min).**
In a finished run, open `silver/point_rejections.csv` with pandas. Count the
rejected points per gate and per frequency decade. Which gate limits the top of
the band? Cross-check with `segment_reach.csv`.

```python
import pandas as pd
led = pd.read_csv("silver/point_rejections.csv")
print(led[led.kept == 0].groupby("reason").size().sort_values())
```

**Exercise 5 — Change a setting the right way (20 min).**
Make a config with a stricter uncertainty gate without editing `config.py`:
```python
from config import DEFAULT
cfg = DEFAULT.replace(sigma_rel_max=0.30)
```
Run `--reevaluate` with it and compare tiers. Then answer: what would you have
to add to the notebook so the cache does not serve the old result?
(`_CACHE_IDENTITY_KEYS` — `sigma_rel_max` is already there; check.)

**Exercise 6 — Write a test (30 min).**
Write a test that builds a "measured" spectrum with a known 20 µs skew,
`Z_meas = Z_true · exp(−j2πf · 20e-6)`, corrects it with
`utils.apply_delay(f, Z_meas, 20e-6)` (the function *undoes* a skew of `dt`,
i.e. multiplies by `exp(+j2πf·dt)`), and asserts the phase error is below 0.1°
at 1 kHz. Put it in `tests/test_my_first.py` and run it alone with
`pytest tests/test_my_first.py`.

**Exercise 7 — Read one module end to end (45 min).**
Read `pipeline/channel_lag.py` completely. Draw its data flow on paper: input
(which object), the reference, the fit, the acceptance rules, the output, and
where silver uses the result.

---

## 22. Self-check quiz

1. Why does choosing the wrong plate generation not raise an error?
2. Why is the area not multiplied in when computing j from the shunt voltage?
3. A 50 µs delay — what phase error at 2 kHz?
4. Why is the common delay `dt0` held at 0 in the skew model?
5. Why is the passivity gate applied after de-skewing, not before?
6. Which gate "really decides" whether a point is kept, and what does it measure?
7. What is the difference between the cell aggregate and the plate mean?
8. Why does the CSV path not run card alignment?
9. Why can a code change leave results unchanged in the notebook?
10. Where is the one place to change a threshold?

**Answers**

1. Both plates have 72 segments with valid areas; the numbers are just attached
   to different pads.
2. The calibration K already returns a current *density* (A/cm²).
3. 360° × 2000 × 50·10⁻⁶ = **36°**.
4. A common delay is degenerate with a series inductance (L = −R0·dt); the model
   already fits L, so fitting both asks the data for two numbers it contains as one.
5. A delay can rotate a good point into Re Z < 0; the de-skew rotates it back.
   Gating first deletes good data.
6. `uncertainty`: the Cramér–Rao σ_rel = √(1/(N·γ)), which combines SNR and dwell length.
7. The aggregate is the area-weighted *harmonic* mean (admittances add), what a
   whole-cell instrument measures; the plate mean is an arithmetic tile average.
8. One logger, one clock: there is no inter-card delay to measure.
9. The cache key contains every setting but not the code version → use `rerun`.
10. `core/config.py`, via `DEFAULT.replace(...)` in the notebook or on the CLI.

---

## 23. Glossary

| Term | Meaning |
|---|---|
| Abgleich | bench calibration delivery: `curr.csv`, `temp.csv`, bode sweeps, step files |
| ADC slot | position of a channel in the converter's scan order |
| Aggregate | segments combined in parallel: A_cell / Σ(A_s/Z_s) |
| ASR | area-specific resistance/impedance, Ω·cm² |
| Bronze / silver / gold | raw / corrected-and-modelled / plate-field layers |
| Card / Karte | one FAMOS acquisition card, 16 channels |
| CFAR | constant false-alarm-rate detector |
| Chain gain | frequency response of a segment's shunt + amplifier |
| Condition | operating point named by its current (45A …) |
| CPE / ZARC | constant-phase element / R parallel to a CPE |
| CRLB | Cramér–Rao lower bound on estimator variance |
| datago | Bosch Delta tables holding the raw measurements |
| DRT | distribution of relaxation times |
| ECM | equivalent-circuit model |
| gen1 / gen2 | green Kashyyyk / blue Naboo plate |
| HFR / R_ohmic | high-frequency (ohmic) resistance |
| Leepa / order | test order number |
| Ledger | `point_rejections.csv`: per point, kept or which gate dropped it |
| lin-KK | linear Kramers–Kronig validity test |
| MF4 | ASAM measurement file format of the test bench |
| Phasor | complex amplitude of a sine at a known frequency |
| Prominence | how many robust σ a correlation peak stands above the background |
| Tier A/B/C/D | segment quality grade (D = inferred, no own data) |
| Tile | one of 36 equal 5×5-pad blocks for plate statistics |
| UC1/UC2 | cell-voltage channels; UC2 is the reference |
| Z-HIT | Hilbert-transform test that also sees pure delays |
