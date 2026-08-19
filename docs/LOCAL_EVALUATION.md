# Running the full evaluation locally

Everything the Databricks pipeline does runs on a laptop. Verified end to end:
a synthetic 5-card FAMOS set (145 MB) through `Local_EIS_fixed/main.py` on a
plain Python 3.11 install, producing gold and silver output that the viewer
reads and fits circuits to. Nothing about the science needs Databricks — only
the notebook runner did.

## What the dashboard reads

The viewer is a *reader*. It never touches `.DAT`; it reads what the pipeline
wrote.

| File (under `<root>/<order id>/<condition>/`) | Needed? | What it unlocks |
| --- | --- | --- |
| `silver/spectra_clean.csv` | **required** | Nyquist, Bode, ECM fitting, click-a-segment |
| `gold/plate_summary.csv` | **required** | every heat map, the segment table, fault labels |
| `silver/cell_aggregate.csv` | optional | the area-weighted cell curve on the Nyquist axes |
| `silver/segments_summary.csv` | optional | card, SNR, THD, KK residual, dropped-point counts |
| `silver/skew_model.csv` | optional | the per-card acquisition-skew table on Overview |
| `gold/gold_manifest.json` | optional | DC closure, tier counts, inferred fraction, plate statistics |
| `silver/silver_manifest.json` | optional | tiers and skew, when gold's manifest is absent |
| `config_used.json` | optional | provenance: the settings that produced the run |
| `bronze/*` | not read | raw phasors; kept for re-running silver without re-reading `.DAT` |
| `gold/*.png`, `gold/*.html` | not read | the pipeline's own static figures; the viewer draws its own |

Missing an optional file costs that one panel and nothing else — the viewer
says what is absent rather than failing.

**One condition is one folder.** Several conditions of the same order sit side
by side, and that is what fills the Conditions tab:

```
<results root>\2611976\45A\gold\plate_summary.csv
<results root>\2611976\45A\silver\spectra_clean.csv
<results root>\2611976\60A\...
<results root>\2611976\150A\...
<results root>\2611976\450A\...
```

## What the pipeline needs to produce them

| Input | Needed? | Notes |
| --- | --- | --- |
| `Leepa_<order>_Current_<cond>_Test_01_Karte_<1..5>.DAT` | **required** | all cards of one condition; ~30 MB each in the synthetic test, far larger in a real campaign |
| `curr.csv` | **required** | per-segment shunt calibration, **72 lines** of `c0;c1` |
| `temp.csv` | recommended | per-sensor temperature calibration, **4 lines** of `c0;c1` |
| areas CSV | optional | `--areas`, overrides the per-segment areas |
| gain file | optional | `--gain`, ex-situ chain response |

`curr.csv` is not optional in any meaningful sense: it is the only absolute
scale left in the chain once the potentiostat is gone. Without it the numbers
come out in shunt volts per amp rather than in ohms, and every map is wrong by
an unknown factor.

## Running it

The pipeline is bundled in `local_eis/` — the same modules that ran on
Databricks, byte for byte, with only the notebook runner left behind. It is
driven by `run_evaluation.py`, which reads the paths from `.env`:

```powershell
python run_evaluation.py --list          # what recordings are there?
python run_evaluation.py --all           # process every condition found
python run_evaluation.py --condition 45A # or just one
python run_evaluation.py --self-test     # check the install, no data needed
```

Output goes straight to `<EIS_RESULTS_ROOT>\<order id>\<condition>`, which is
where the dashboard looks, so there is no copying step afterwards:

```powershell
python run_dashboard.py --open
```

Two commands, start to finish. Verified end to end: a synthetic five-card set
(145 MB) processed in 164 s, producing the full gold layer — 70/72 segments
measured, `gold_manifest.json`, every map — and read straight back by the
viewer.

Useful flags: `--dry-run` prints the exact command without running it;
`--equal-areas` is the deliberate simplification, recorded in the manifest;
`--no-png` skips the pipeline's own figures, which the viewer never reads;
`--stop-after silver` skips the DRT and spatial inference.

**Install matplotlib even with `--no-png`** — `gold.py` imports it at module
level. Without it the run completes bronze, silver and gold and then fails
while writing figures, leaving `gold_manifest.json` unwritten; the maps still
work but the Overview statistics are missing. `run_evaluation.py` warns before
starting rather than after ten minutes of work.

Expect minutes per condition: bronze reads every sample of every card. Silver
and gold then re-run from `bronze/` quickly.

### From inside the dashboard

With `EIS_ALLOW_INLINE_PIPELINE=1` in `.env`, selecting a raw recording offers
*Run pipeline on this selection* on the Overview tab. It runs the same command
as a background job in its own process, streaming the pipeline's own log as
progress, and shows the equivalent command line so it can be reproduced outside
the app.

### Using your own copy of the pipeline

Point `EIS_PIPELINE_DIR` at it. The bundled copy is used when that is unset, so
a newer pipeline from Databricks is either a straight file replacement into
`local_eis/` or one line in `.env`.

## Two pipelines, one viewer

| | `Local_EIS_fixed` (yours) | `eis/` (this repository) |
| --- | --- | --- |
| Output | `gold/`, `silver/`, `bronze/` CSVs | `impedance.parquet`, `segments.parquet` |
| Process split | DRT, cut at a fixed time constant → `R_ct`, `R_mt` | not computed |
| Unmeasured segments | inferred from a Gaussian-process field, drawn hatched | not inferred |
| Fault labels | drying / flooding / starvation / contact loss | not computed |
| Synchronisation | structural per-segment mux skew | measured inter-card skew and clock drift |
| Read by the viewer | yes | yes |

The viewer reads both, but only `Local_EIS_fixed` produces the gold layer. For
evaluation matching what you had on Databricks — `R_ct` and `R_mt` maps,
inferred segments, fault labels — run that pipeline. The in-app *Run pipeline*
button uses `eis/` and therefore produces the narrower output.

Both are strictly optional for the ECM tab: circuit fitting works from
`spectra_clean.csv` alone, and is the viewer's own computation either way.
