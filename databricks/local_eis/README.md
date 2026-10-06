# Local EIS pipeline — Databricks

The bronze/silver/gold pipeline that runs in the Databricks workspace, plus the
CSV evaluation path and the two plate maps.

Full write-up of the gen2 plate and the CSV path:
[`../../docs/GEN2_PLATE_AND_CSV_PIPELINE.md`](../../docs/GEN2_PLATE_AND_CSV_PIPELINE.md).

## Layout

| file | what it is |
| --- | --- |
| `Local EIS Pipeline Runner.py` | the notebook. Widgets → run → Nyquist, heat maps, ECM, Gamry validation |
| `main.py` | `run_pipeline(cfg)`; routes to FAMOS or CSV by `cfg.source_format` |
| `config.py` | every tunable number, one definition each |
| `r2d2_geometry.py` | **both plate maps**, `use_plate("gen1"\|"gen2")` |
| `bronze.py` `silver.py` `gold.py` | the FAMOS path |
| `eis_local.py` `utils.py` `eis_measurement_model.py` `eis_validation.py` | shared estimators, KK, measurement model |
| `csv_source.py` | CSV reader, seven layouts incl. the R2-D2 logger, dialect auto-detection |
| `csv_pipeline.py` | the CSV evaluation path |
| `gamry_dta.py` | Gamry `.DTA` reader; builds the chain-response gain file |
| `abgleich.py` | reads the raw `Step*_<T>Grad.csv` bench files; refits and verifies `curr.csv`/`temp.csv` |
| `test_csv_pipeline.py` | end-to-end synthetic checks for the CSV path |

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

`gen2 + famos` is rejected: there is no FAMOS recording of the blue plate.

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
python main.py --self-test          # geometry (both plates), estimators, CSV readers
python r2d2_geometry.py             # print both maps, write segment_areas_<plate>.csv
python test_csv_pipeline.py         # end-to-end on synthetic data with known truth
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
`bode/`. Measured on both plates it is −11° of phase at 4.5 kHz — the top of
the default analysis band — and −24° at 10 kHz. That is the same order as the
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
`R_ohmic` is Re Z at the top of the kept band (~1.2 kHz on 2612030). Each
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
