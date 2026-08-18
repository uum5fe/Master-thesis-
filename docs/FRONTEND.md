# Local EIS Viewer — frontend architecture and deployment

A Dash application that lets colleagues look at segment-resolved impedance
results without a notebook, without a personal file path and without the
pipeline having to be re-run for every question.

```
run_dashboard.py       start the dashboard locally; prints the link
app/
  app.py               Dash shell: sidebar selection, tabs, callbacks
  settings.py          every path, from the environment - no personal paths
  plates/
    registry.py        plate geometry as data; tiling self-check
    specs/*.json       one file per plate generation
  data/
    model.py           the canonical RunData every loader produces
    loaders.py         adapters: gold/silver, eis tables, generic CSV
    sources.py         discovery: Volumes results, raw FAMOS, datago
  services/
    figures.py         plate maps, Nyquist, Bode, residuals, comparisons
    ecm_service.py     equivalent-circuit fitting on top of eis.model.ecm
    runner.py          running the pipeline over raw .DAT, as a job
    store.py           caching and background jobs
  views/               one module per tab
```

## The three choices in the sidebar

**Data location** — a Volumes / file-system root, or the datago metadata
tables. **File format** — finished results (CSV / Parquet), or raw FAMOS
`.DAT` that still has to be processed. **Plate generation** — which segment
arrangement produced the data.

Everything else follows from those three. No module in `app/` contains a
hard-coded path.

## Environment variables

| Variable | Meaning |
|---|---|
| `EIS_RESULTS_ROOT` | Colon-separated roots of finished results, laid out `<root>/<order id>/<condition>/{gold,silver}/…` |
| `EIS_FAMOS_ROOT` | Colon-separated roots of raw `.DAT` recordings |
| `EIS_FAMOS_REGEX` | Extra filename pattern, if a campaign names its files differently |
| `EIS_DATAGO_METADATA_TABLE` | Unity Catalog table listing measurements |
| `EIS_DATAGO_PROPERTIES_TABLE` | Table carrying `measurement_type` |
| `EIS_DATAGO_MEASUREMENT_TYPE` | Defaults to `GALVEIS` |
| `DATABRICKS_WAREHOUSE_ID`, `DATABRICKS_HOST` | SQL warehouse the datago source queries through |
| `EIS_CURR_CAL`, `EIS_TEMP_CAL` | Calibration files, needed only to *run* the pipeline |
| `EIS_PLATE_SPEC_DIR` | Extra directory of plate specs, e.g. a Volume |
| `EIS_DEFAULT_PLATE` | Generation preselected in the dropdown |
| `EIS_ALLOW_INLINE_PIPELINE` | `1` lets the app process raw `.DAT` itself |
| `EIS_SCRATCH_RESULTS` | Where an inline run writes its output |
| `EIS_CACHE_DIR`, `EIS_TITLE`, `EIS_DEBUG` | Housekeeping |

## Running it

Locally, `run_dashboard.py` is the entry point:

```bash
pip install -r requirements.txt
python run_dashboard.py --results /path/to/results --open
```

| Flag | Effect |
|---|---|
| `--results DIR` | sets `EIS_RESULTS_ROOT` |
| `--famos DIR` | sets `EIS_FAMOS_ROOT` |
| `--plate-specs DIR` | sets `EIS_PLATE_SPEC_DIR` |
| `--port N` | listen on another port (default 8050) |
| `--open` | open the browser once the server is up |
| `--debug` | Dash debug mode with hot reload |

It puts the project root on the import path itself, so it works from any
working directory, and it prints the URL plus a count of discovered runs before
starting — a server that starts in silence looks like a server that did not
start. `python -m app.app` does the same thing without the flags.

Note that the bind address is not the address to visit: binding to `0.0.0.0` is
what lets a container be reached from outside, but `0.0.0.0` is not a
destination, so the link printed is always the loopback one.

As a Databricks App, from the repository root so that both `app` and `eis`
are importable:

```bash
databricks sync . /Workspace/Shared/local-eis-viewer
databricks apps deploy local-eis-viewer \
    --source-code-path /Workspace/Shared/local-eis-viewer
```

`app.yaml` carries the environment; fill in the catalog, schema and volume
names. Grant the app's **service principal** `SELECT` on the datago tables,
`READ VOLUME` on the results and FAMOS volumes, and `CAN USE` on the SQL
warehouse — the app never carries a personal token, which is what makes it
shareable in the first place.

Behind gunicorn, use **one worker with threads**:

```
gunicorn -b 0.0.0.0:8000 -w 1 --threads 8 --timeout 300 app.app:server
```

The run cache and the background fit jobs live in the process. Several workers
would each hold their own copy, and a fit started on one would be invisible to
the next request.

## Adding a plate generation (Gen 2, Gen 3, …)

A new generation is a JSON file, not a code change.

1. Copy `app/plates/specs/_gen2_template.json` to `gen2_<name>.json`, either
   in the repository or in a directory named by `EIS_PLATE_SPEC_DIR` (a Volume
   works, so a colleague can add one with no deployment). Files beginning with
   `_` are templates and are not offered as generations.
2. Give the pad grid — `pad_w_mm`, `pad_h_mm`, `n_cols`, `n_rows` — and then
   the segments, in either form:
   * `strips` + `bands` + `numbering`: the plate is cut into full-height
     vertical strips, each cut into bands of pad rows. Short, and it is how
     Gen 1 is written.
   * `segments`: one explicit rectangle per segment on the pad grid. Works for
     *any* arrangement, including ones that are not strip-based.
3. Optionally add `wiring` (segment → card and channel), `known_bad`
   (segment → why), `temp_sensor_x_mm` and `flow_channel_y_mm`.
4. Check it:

   ```bash
   python -m app.plates.registry
   ```

   or open the **Plate & sources** tab, which runs the same check. It verifies
   that the segments cover every pad exactly once and that their areas add up
   to the active area. A spec with a typo draws a heat map that looks entirely
   plausible and is wrong, so this check is not optional.

Because a segment is a union of whole pads, its area comes out exact and does
not depend on trusting a drawing scale.

## Processing raw `.DAT`

Selecting a FAMOS recording gives files, not spectra. Two ways to get from one
to the other:

* **Databricks Job (recommended).** Run `run_pipeline.py` on a proper cluster
  with `output_dir` inside `EIS_RESULTS_ROOT`. The result appears in the picker
  after pressing *Refresh sources*. An Apps container is sized to serve a UI,
  not to grind through gigabytes of recordings.
* **Inline, with `EIS_ALLOW_INLINE_PIPELINE=1`.** The Overview tab then offers
  a *Run pipeline* button. It runs as a background job with progress and
  reports what it cannot do — most importantly a missing shunt calibration,
  without which the impedances are in shunt volts per amp rather than in ohms.

## Result formats the viewer understands

| Layout | Files | Notes |
|---|---|---|
| `gold_silver` | `gold/plate_summary.csv`, `silver/spectra_clean.csv`, `silver/cell_aggregate.csv`, `*_manifest.json` | The bronze/silver/gold pipeline. Values already in mΩ·cm², °, A/cm². |
| `eis_tables` | `impedance.{parquet,csv}`, `segments.{parquet,csv}` | The packaged `eis` pipeline. One file can hold several conditions. |
| `generic_csv` | any single CSV of spectra | Columns matched by pattern; anything unmatched is reported, not guessed. |

Adding a fourth producer is one adapter function in `data/loaders.py`. The
views never see a producer-specific column name.

## Units

Fixed once, in `data/model.py`: impedance and resistance in **mΩ·cm²**,
frequency in **Hz**, lengths in **mm**, areas in **cm²**, current density in
**A/cm²**. The ECM fit runs internally in Ω·cm², where the parameters are
numerically comfortable, and converts back on the way out.

## A note on the ECM fit versus `R_ct` / `R_mt`

The pipeline's `R_ct` and `R_mt` come from cutting the distribution of
relaxation times at a fixed time constant. A fitted circuit is a different
quantity. The two disagreeing is informative rather than embarrassing, so the
ECM tab plots them against each other on a 1:1 line instead of quietly
replacing one with the other.

Fitting is weighted by the per-point uncertainty the pipeline stored, positive
parameters are fitted in log space, and by default the circuit is *selected*
from a ladder by corrected AIC rather than assumed. A parameter whose relative
error is too large is labelled *poorly determined* rather than presented as if
it were known.
