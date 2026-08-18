# Local EIS Viewer — start here

This archive is the complete architecture: the analysis pipeline (`eis/`) and
the Dash frontend (`app/`) that reads its results. It is the same content as
branch `claude/impedance-frontend-viz-not3ye` of the Master-thesis repository.

## Run it in five minutes, on your own machine

```bash
cd local-eis-viewer
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# point it at a folder of finished results, then start it
export EIS_RESULTS_ROOT=/path/to/results              # Windows: set EIS_RESULTS_ROOT=...
python -m app.app
# open http://127.0.0.1:8050
```

`EIS_RESULTS_ROOT` expects this layout — the same one the pipeline writes:

```
<root>/<order id>/<condition>/gold/plate_summary.csv
<root>/<order id>/<condition>/silver/spectra_clean.csv
<root>/<order id>/<condition>/silver/cell_aggregate.csv
```

So for your existing 45 A output, copy `results_true_areas/` to
`<root>/2611976/45A/` and it appears in the dropdowns immediately. Point the
root at several conditions (45A, 60A, 150A, 450A) to unlock the Conditions tab.

## Check the plate geometry before trusting any map

```bash
python -m app.plates.registry
```

Prints every plate generation it can find and verifies that its segments cover
every pad exactly once and that their areas sum to the active area. Gen 1
should report 72 segments, 900/900 pads, 304.92/304.92 cm², `[OK]`.

## Run the tests

```bash
pip install pytest && python -m pytest tests -q      # 63 tests
```

## Where things are

| Path | What it is |
| --- | --- |
| `app/app.py` | Dash shell: sidebar selection, tabs, callbacks |
| `app/settings.py` | Every path, from environment variables — no personal paths anywhere |
| `app/plates/registry.py` | Plate geometry as data, with the tiling self-check |
| `app/plates/specs/gen1_r2d2_72.json` | Gen 1: 72 segments, 15 strips, 900 pads |
| `app/plates/specs/_gen2_template.json` | Copy this to add Gen 2 / Gen 3 |
| `app/data/` | Canonical run model, result loaders, source discovery |
| `app/services/` | Figures, ECM fitting, pipeline jobs, caching |
| `app/views/` | One module per tab |
| `app.yaml` | Databricks App manifest — fill in catalog/schema/volume names |
| `eis/` | The analysis pipeline the frontend sits on |
| `docs/FRONTEND.md` | Deployment, full environment-variable list, adding a generation |
| `README.md` | The pipeline itself: synchronisation, calibration, validation |

## Deploying it for your colleagues

Read `docs/FRONTEND.md`. Short version: deploy **from the repository root**, so
that both `app` and `eis` are importable —

```bash
databricks sync . /Workspace/Shared/local-eis-viewer
databricks apps deploy local-eis-viewer \
    --source-code-path /Workspace/Shared/local-eis-viewer
```

— fill the catalog, schema and volume names into `app.yaml`, and grant the
app's **service principal** `READ VOLUME` on the data and `SELECT` on the
datago tables. The app never carries a personal token; that is what makes it
shareable rather than something only you can run.

## One thing to fix on the pipeline side

`silver/skew_model.csv` is written with an unquoted comma inside its free-text
`note` column ("...degenerate with series L, absorbed there"). Read naively,
pandas promotes the first column to the index and every value lands one column
to the left — `slot_us` reads 50 where it is really 54.7. The viewer's loader
detects and repairs this, but quoting the field in `utils.write_table` would
fix it at the source for everything else that reads those files.
