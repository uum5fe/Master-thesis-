# Local EIS pipeline — Databricks

Turns the FAMOS recordings of the R2-D2 segmented plate into a local impedance
spectrum per segment, plate heat maps (R_ohmic, Re Z 1 kHz, R_ct, R_mt, R_pol,
j, T), a comparison with the Gamry whole-cell sweep and plausibility checks.

**Handing it to a colleague / choosing the plate:** [`SETUP_SHARED_FILES.md`](SETUP_SHARED_FILES.md)
— upload `EIS_config/` to the Volume once; pick gen1 or gen2 in the **Plate** widget.

**Start here:** [`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md) — the
complete guide: how to run it, where the data comes from, every module and
output file, the formulas, the cache, and the known limitations.

## The final version in one paragraph

Import the zip into an **empty** workspace folder, open
`01 Local EIS - Process`, choose order, conditions and run mode, run all;
look at the results in `02 Local EIS - Results`.

Where each kind of file comes from is set in `runner/settings.py`:
`GAMRY_FROM_DATAGO`, `BENCH_FROM_DATAGO` and `FAMOS_FROM_DATAGO` (default
`False`: the FAMOS cards are read from the Volume `.DAT` files), each with the
Volume as fallback; `DATAGO_FIRST = False` switches datago off for all three.
A FAMOS card taken from datago is rebuilt as a `.DAT` and used only when it
holds one signal per channel and, if the Volume has the card, the same
samples -- datago's import of the newer FAMOS-v2 cards returns UC2 under
every channel name, and such a card is read from the Volume instead.

R_ohmic, R_ct, R_mt and R_pol are computed the standard way: the
high-frequency intercept, then an equivalent-circuit fit
R_ohmic + L + (R_ct‖CPE) + (R_mt‖CPE) (`pipeline/standard_eis.py`; no DRT).
Every parameter is drawn as one 2D interpolated plate map.

The evaluation is fixed:
- band: **complete** (0.15 Hz – 0.49·fs, 12.25 kHz at 25 kHz), with the
  corrections of the raw-signal audit (guide §4); `BAND = 'trusted'` gives
  0.15 Hz – 1.2 kHz, where all cards agree uncorrected;
- maximum-likelihood sine fits, Cramér–Rao point uncertainties, Bayesian DRT
  for R_ct / R_mt / R_pol, the standard circuit fitted beside it;
- 0 dB bronze SNR gate; default gates.

There are no band, SNR or profile dropdowns: the evaluation is fixed in
`runner/settings.py`.

Full write-up of the gen2 plate and the CSV path:
[`docs/GEN2_PLATE_AND_CSV_PIPELINE.md`](docs/GEN2_PLATE_AND_CSV_PIPELINE.md).

## How to use it on Databricks

Upload this whole folder (keep the sub-folders). The work is split the way
the TOM `spp_eval` pipeline splits it: notebooks that **write** and notebooks
that only **read**, each with a banner saying which it is. All of them are
calls only -- the code is the `runner/` package.

```
FAMOS cards (datago / Volume) + Gamry sweeps + bench .mf4
  └─ 01 Local EIS - Process      bronze -> silver -> gold per order / condition / test
       ├─ EIS_Results/<order>/<condition>_Test<nn>/      the run folder (unchanged)
       └─ <TABLE_SCHEMA>.local_eis_*                      the same results as Delta tables
            ├─ 02 Local EIS - Results    one order's plots; every order from the tables
            └─ 03 Local EIS - Auto       job: evaluates what is not published yet
  09 Local EIS - Maintenance             removes one run (tables, optionally folder)
```

| Notebook | Banner | Does |
| --- | --- | --- |
| `01 Local EIS - Process` | 🛠 writes results and tables | widgets → `R.run()` → `R.fit_ecm()` → `R.publish()` → `R.summary()` |
| `02 Local EIS - Results` | 📊 read-only | saved order / conditions / test → frequency response, Nyquist / Bode, median Nyquist, heat maps, MF4, ECM; across orders: `R.show_runs()`, `R.show_trends()` |
| `03 Local EIS - Auto` | 🛠 writes (scheduled job) | `runner.auto.plan()` (dry run) / `runner.auto.run()`: every order / condition / test with FAMOS data not in the runs table (or published with other settings) |
| `08 Local EIS - Calibration` | 🛠 writes the calibration tables | publishes a plate's `curr.csv` / `temp.csv` / `chain_gain.csv` / `Abgleich/` as a new version in `local_eis_calibration_*`; runs take the version valid at their measurement date (`CALIBRATION_SOURCE`, `CALIBRATION_VERSION` in `runner/settings.py`) |
| `09 Local EIS - Maintenance` | ⚠️ destructive | removes one `run_key` from every table (and its folder, if asked); rebuilds all tables (`REBUILD`); no undo |

**Process** (`01`):

| cell | call | what it does |
| --- | --- | --- |
| Setup | `R = runner.Session(globals())` | finds the folder, imports the modules |
| Selection | `R.widgets()` | Order, Conditions, **Test** (Test_01 up-ramp, Test_02 down-ramp, ...), Plate, format, segments to exclude / rebuild, Run mode |
| Evaluate | `R.run()` | `cache`: use the saved result (restored from the tables when its folder is gone, `CACHE_FROM_TABLES`), run only what is missing; `rerun`: recompute and **overwrite** the saved result |
| ECM | `R.fit_ecm()` | standard and DRT-informed ECM, saved in the run folder (`ecm/`) |
| Publish | `R.publish()` | the run folders into the tables (`PUBLISH_TABLES`); a run's earlier rows are replaced |
| Summary | `R.summary()` | segments, Gamry sweep, failed / warned checks, FAMOS source per run |

**Results** (`02`) -- the selection lists what is SAVED, nothing is evaluated.
With `RESULTS_SOURCE = "tables"` (default) a published run is read from the
Delta tables (every file of it is in `local_eis_run_files`), so the Volume
folder is not needed; a run never published is read from its folder:

| cell | call | what it shows |
| --- | --- | --- |
| Selection | `R = runner.Results(globals())`, `R.widgets()` | a saved order folder (`2612090`, `2612091_gen2`, ...), its conditions and test |
| Frequency response | `R.show_frequency_response()` | ex-situ amplifiers / in-situ per condition |
| Nyquist / Bode | `R.show_nyquist()` | every segment + the Gamry sweep of the same test |
| Median Nyquist | `R.show_median_nyquist()` | plate median, card medians, aggregate, Gamry |
| Heat maps | `R.show_heatmaps()` | Condition · Parameter · View |
| MF4 bench log | `R.show_mf4()` | the WHOLE test, every channel, every Gamry sweep shaded |
| ECM fits | `R.show_ecm()` | Condition · Spectrum · Fit (standard / DRT-informed / both) |
| Runs | `R.show_runs()` | every published run (the runs table) |
| Trends | `R.show_trends()` | plate median ± sd of every gold parameter against current, one curve per order and test |

**Tables** (`runner/tables.py`, schema `TABLE_SCHEMA`, prefix `local_eis_`):
`runs`, `segment_meta`, `bronze_spectra`, `silver_spectra`,
`silver_reconstructed`, `silver_segments`, `gold_segments`, `gold_plate`,
`gold_cell`, `gamry_reference`, `gamry_comparison`, `checks`, `ecm`, and
`run_files` (every file of the run, byte for byte) -- each a
copy of one result file, every row keyed by `run_key` (= the run folder under
`EIS_Results`, e.g. `2612090/450A_Test01`), `order_id`, `plate`,
`orientation`, `condition`, `current_a`, `test_id`, `eval_mode`;
partitioned by `order_id`. Publishing a run replaces its rows (replaceWhere
on `run_key`), as a rerun replaces its folder. Example:

```sql
SELECT order_id, test_id, current_a, median AS R_ohmic
FROM ps_xplatform_dev.rvadvtec_dev.local_eis_gold_plate
WHERE parameter = 'R_ohmic' ORDER BY order_id, current_a
```

**Auto job** (`03`): create a Databricks Job on the notebook (e.g. nightly);
parameters `dry_run` (`yes` shows the plan only), `max_runs`, `orders`.
Plate and orientation per order: `AUTO_PLATE` / `AUTO_ORIENTATION` in
`runner/settings.py`, else the order's file names (Naboo = gen2, Kashyyyk =
gen1), else gen1 / standard. A run that fails is not published and is tried
again at the next execution.

Every plot is saved automatically as `EIS_Results/<order>/plots/<name>.html`
(plus the static PNGs and the ECM parameter tables under
`plots/<condition>_Test<nn>/`). Every result lives in one ordered folder:

    EIS_Results/2612086/60A_Test02/   bronze/ silver/ gold/ ecm/ run_identity.json

`run_identity.json` records the settings a result was made with; a saved
result made with other settings is still used under `cache`, and the run cell
says so -- choose `rerun` to renew it. (Results saved by runner v33 and
earlier, under `<cond>/mode_default/snr_..._<id>/`, are not read any more.)

**Which Gamry sweep.** Every current of an up/down ramp is measured twice
(RO2612086: HFR_103 at 08:39 and HFR_111 at 09:34, both 60 A). Datago and the
Volume both deliver every sweep; the pipeline takes the one whose run
overlaps the FAMOS recording's time stamps (`gamry_compare.select_sweep`),
then the test number. All plots draw the sweep that run was compared with.

**Which files from where.** FAMOS cards: the Volume `.DAT`, or datago with
`FAMOS_FROM_DATAGO = True` (checked card by card, the Volume as fallback).
Gamry sweeps and the bench log: datago first, the Volume as fallback
(`runner/settings.py`).

## Layout

| folder | what is in it |
| --- | --- |
| `01 Local EIS - Process.py`, `02 Local EIS - Results.py`, `03 Local EIS - Auto.py`, `09 Local EIS - Maintenance.py` | the notebooks: calls only, each with its banner |
| `runner/` | `settings.py` (paths, switches, fixed evaluation, tables, job), `datago.py`, `session.py` (widgets, run, saved results, ECM, publish, `show_*`), `results.py` (read-only session, across orders), `tables.py` (Delta tables), `auto.py` (incremental job), `views.py` (the figures) |
| `eis_paths.py` | puts the folders below on `sys.path` |
| `run.py` | command line: `python run.py <module> [args]` |
| `core/` | `config.py` (every tunable number), `utils.py`, `r2d2_geometry.py` (**both plate maps**), `plate_conditions.py` |
| `pipeline/` | `main.py` (`run_pipeline`, `reevaluate`), `bronze.py` `silver.py` `gold.py` (FAMOS path), `csv_pipeline.py`, and their signal processing: `eis_local.py`, `hf_schedule.py`, `ladder_snap.py`, `tone_estimation.py`, `gamry_sync.py`, `channel_lag.py`, `eis_measurement_model.py`, `eis_validation.py`, `neighbours.py`, `standard_eis.py` |
| `checks/` | `plausibility.py` and the per-run checks: `card_gain.py`, `dc_closure.py`, `frequency_response.py`, `hf_response.py`, `gamry_compare.py`, `validate_hf.py` |
| `analysis/` | `ecm_drt.py` (DRT and the DRT-informed ECM) |
| `plotting/` | `viewers.py` (the drop-down selector), `nyquist.py`, `plate_figure.py` (static maps), `plate_plotly.py`, `plate_style.py`, `plate_maps.py`, `figure_panels.py`, `bench_plots.py` |
| `readers/` | `gamry_dta.py` (Gamry `.DTA`, chain-response gain file), `abgleich.py`, `csv_source.py`, `_vendor/zstd.py` |
| `tests/` | `pytest tests` (+ `fixtures/`, `make_synth_famos.py`) |
| `docs/` | the guide (`PIPELINE_GUIDE.md`), write-ups, figures and the change log (`FIXES.md`); index in `docs/README.md` |

## The two things to get right before a run

**1. The plate.** `gen1` is the green Kashyyyk plate, `gen2` the blue Naboo
plate. Same 45×20 pad grid, same 72 segments, different grouping of pads into
segments 37…72 — and different areas for the interior segments that gave pad
rows to them. Choosing wrong does not fail; it draws the right numbers on the
wrong squares. The notebook's plate-map cell exists to be compared against the
coordinate drawing once per campaign.

**2. The source format.** `famos` is the five-card imc recording, which needs
the inter-card synchronisation stage. `csv` is the single-file logger, which
has one clock and therefore must *not* run it. They are two pipelines sharing
the geometry and the Abgleich, not two readers in front of one.

**3. The plate position.** If the plate was put into the cell turned 180°
(the segments that are inlet segments on the drawing sat at the outlet),
set **Plate position** to `rot180`. The ports stay fixed to the cell; the
maps and the inlet → outlet Nyquist colours follow the plate. Details in
[`SETUP_SHARED_FILES.md`](SETUP_SHARED_FILES.md) §4.

## The R2-D2 logger format

Point **CSV file / folder** at the sweep folder — the one holding
`metadata.csv` and `p1.csv, p2.csv, …`. Each point file is one frequency, so
the spectrum only exists once they are read together.

Two things about this format that are easy to miss and expensive to miss:

- The second header row, `timeshifts`, is the acquisition instant of each
  column **inside one row, in microseconds**. The logger scans 80 channels
  across 96 % of a sample period, so segment 1 (0 µs) and the cell-voltage taps
  (79–82 µs) are 80 µs apart — 29° at 1 kHz, and it is the *ratio* of those two
  channels that is the impedance. The delays are printed, so the correction is
  exact and needs no fit; it is applied automatically.
- The `s` columns are already a current density in A/cm² and the temperatures
  already in °C: the logger applies the coefficient set named in
  `metadata.csv`. `curr.csv` is deliberately **not** applied again.

A point file is a **burst**: the delivered one has 0.25 s of lead-in and 0.4 s
of lead-out with no excitation, and a persistent ~999 Hz artefact living in
them. The pipeline finds the tone on the segment-averaged spectrum (not on one
channel — `uc1` carries a larger 3488 Hz component), windows to the burst, and
refuses a record whose strongest common tone is under 10× the noise floor
rather than reporting an impedance measured against an artefact.

It also compares the tone in the record against the phase ramp the
channel scan measures, and reports a point whose analogue frequency is above
Nyquist. On the delivered `p1.csv` the record shows 923 Hz at fs = 11 001 Hz
while the scan says ~10 kHz: that point is an alias. Pass the sweep's own
frequency list in `cfg.csv_tones` (file order) to replace the inference with a
cross-check.

## Quick checks

```bash
python run.py main --self-test                       # geometry, estimators, CSV readers
python run.py main --reevaluate RUN_DIR --out OUT    # silver/gold again, no .DAT needed
python run.py frequency_response RUN_DIR --bode ABGLEICH/bode
python -m pytest tests                               # the test suite
```

## Running

```bash
# FAMOS, gen1 (unchanged)
python main.py --dat /Volumes/.../Famos --curr-cal cal/curr.csv \
    --temp-cal cal/temp.csv --leepa 2611976 --condition 45A --out out/45A

# CSV, gen2
python main.py --plate gen2 --source csv --csv /path/run.csv \
    --curr-cal cal/curr.csv --temp-cal cal/temp.csv \
    --gain cal/chain_gain.csv --out out/run
```

## The chain response is not optional

The Abgleich delivery ships 72 Gamry sweeps of the current-measurement chain in
`bode/`. Measured on both plates it is −2.5° of phase at 1 kHz, −11° at 4.5 kHz
and −24° at 10 kHz (all inside the complete band). That is the same order as the
acquisition skew the pipeline works hard to remove, and unlike a skew it moves
|Z| too. Build the file once:

```bash
python gamry_dta.py .../Abgleichdaten/Kashyyyk/bode \
    --curr-cal .../Abgleichdaten/Kashyyyk/coefficients/curr.csv \
    -o cal/chain_gain.csv
```

It cross-checks the bode index against the calibration rows first and refuses
a pairing that does not hang together. On the Naboo delivery it does not
(r = +0.41): use `--shared` there, which writes the index-free plate median and
still removes the common roll-off.

## Plate maps and bench parameters

Every plate heat map (gold `map_*.png/.html`, `plate_*.png`, the runner's
interactive map, the ECM maps and the numbering map) takes its look from
`plate_style.py`:

- **colours**: plotly's Jet scale, stop for stop, the same as the bench's
  intensity plots;
- **text**: the value large and bold, the segment number small above it;
- **view**: pad column 1 on the left -- O₂ out top left, H₂ out top right,
  H₂ in bottom left, air in bottom right. `config.PLATE_VIEW_MIRRORED = True`
  draws the plate from the other side instead; the whole drawing mirrors, so
  the ports stay beside the segments they feed. CSV coordinates stay in plate
  coordinates;
- **ports**: a column of three at each end -- gas outlet above the coolant,
  gas inlet below it: AIR OUT / COOLANT OUT / H₂ IN on the left, H₂ OUT /
  COOLANT IN / AIR IN on the right. `config.COOLANT_INLET_END = "xW"` (coolant
  in at the air-inlet end) is read from the MF4 log (the anode outlet follows
  a coolant-inlet dip, the cathode outlet does not) and the FAMOS plate
  sensors; confirm against the manifold drawing.

`plate_plotly.py` draws the interactive map (hover a segment for its value,
the dropdown switches parameter). `bench_plots.py` draws the MF4 test-bench
channels as one figure: a parameter dropdown, a legend that toggles traces,
hover values at the cursor, and the FAMOS plate sensors as °C reference lines.

## Per-segment checks that run on every measurement

**Current-chain lag (`channel_lag.py`, silver and the CSV path).**
`R_ohmic` is read at the top of the band. Each
segment's shunt/trace/amplifier chain can add a first-order time constant
(shunt L/R); on 2612030 up to ~110 µs, i.e. 37° at 1.2 kHz and an R_ohmic
~20 mΩ·cm² low — the 44 vs 66 mΩ·cm² between segments 19 and 14. Silver now
fits τ for every segment against the plate median, on exactly the points it
models, and removes it before R_ohmic is read. τ is accepted only if the phase
really follows −atan(ωτ) (rms residual ≤ 10°); anything else (a card read at
the wrong time, a scrambled channel) is left uncorrected, flagged and demoted
to tier C. `CHANNEL_LAG` in the runner / `cfg.channel_lag`: `correct`
(default), `report`, `off`.

Outputs: `silver/channel_lag.csv` (τ, residual, status per segment), new
columns in `segments_summary.csv`, `chain_tau_us` in `plate_summary.csv` and
in the interactive map's dropdown, and two plausibility checks: **channel
lag** and **card alignment** (FAIL when a card was never time-aligned).

On 2612030 (re-evaluated): plate CV 9.1 → 6.2 % at 60 A, segments
14/15/18/19/22/23 = 66.5/66.0/63.4/64.3/72.0/64.0 mΩ·cm² (were
66.5/53.3/63.4/44.0/69.9/45.8), tier C 50 → 4 segments. τ repeats across
45/60/150 A (r = 0.85–1.00).

What it cannot remove: a lag or gain that EVERY channel shares. Against the
Gamry sweep the plate still reads ~6.5 % high at low frequency and ~10
mΩ·cm² high at 1.2 kHz — calibration (curr.csv K) and voltage-sense
territory, for `cfg.gain_file` (now applied on the FAMOS path) and the
Abgleich, not for this stage.

**Re-evaluate an old run without its .DAT files.**

```bash
python main.py --reevaluate <run_dir> [--out <dir>] [--channel-lag report] [--gain g.csv]
```

`bronze.load()` rebuilds the bronze stage from `bronze/*.csv`; silver, gold,
the Gamry comparison and plausibility then run with the current code. Name the
output folder after the condition (`…/<leepa>/<cond>`) so the Gamry
comparison finds its sweep.

**Diagnostic across runs:** `python diagnose_channel_lag.py RUN [RUN ...] -o
out/ [--write-gain g.csv]` — the same estimator, τ side by side per run and
its repeatability.

## The Gamry clock (`gamry_sync.py`)

The FAMOS cards record the cell's response to the Gamry's OWN sweep: on
2612030 every FAMOS window from 0.3 Hz to 1.2 kHz ends at t_gamry + 99.24 s
(60 A; 106.94 s at 45 A), spread 0.44 s — the .dta's one-second stamps. So the
.dta for the condition (found under `gamry_dir` like the whole-cell
comparison) gives every step's exact frequency and time.

| `GAMRY_SYNC` / `cfg.gamry_sync` | what it does |
| --- | --- |
| `report` (CLI default) | measure the offset; `bronze/gamry_sync.csv`: per Gamry point ok / misplaced / missing |
| `frequency` | + the Gamry's exact frequencies (the ladder snap was up to 1 % off); a REFUSED card lag that the Gamry clock corroborates is applied (`bronze/gamry_card_sync.csv`) |
| `guide` (runner default) | + misplaced and missing windows re-located inside their Gamry slot, each verified by hf_schedule's CFAR and rank-1 tests — a prediction alone is never accepted, and a slot without the tone drops the wrong window |

What it found on 2612030 / 60 A: 1.5–4.7 kHz misplaced by 6–96 s (placed by
interpolation), 5.9–9.5 kHz never detected — the band the HFR arc closes in.
It refuses (and changes nothing) when fewer than 5 confidently detected steps
agree on one offset, as at 150 A where two cards were never aligned.
It needs the raw .DAT: `--reevaluate` cannot apply it to saved spectra.

## Chain calibration from the Abgleich bode sweeps

Set `ABGLEICH_DIR` in the "Chain response" cell and run it once: the gain file
is written to `CHAIN_GAIN_DEFAULT` (beside curr.csv), `GAIN_FILE` is set, and
every later session picks it up. It removes the roll-off every segment shares
(−2.5° at 1 kHz, −11° at 4.5 kHz). On the delivered sweeps the segments differ
by only ~2° at 4.5 kHz (~1 µs), far less than the in-situ lags (±40–110 µs),
so the in-situ stage (`CHANNEL_LAG`) stays on;
`diagnose_channel_lag.py RUN… --bode <bode/>` and the cell itself report how
much of the in-situ lag the ex-situ chain explains.
