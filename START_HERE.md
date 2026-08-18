# Local EIS Viewer — start here

This archive is the complete architecture: the analysis pipeline (`eis/`) and
the Dash frontend (`app/`) that reads its results. It is the same content as
branch `claude/impedance-frontend-viz-not3ye` of the Master-thesis repository.

## Run it in five minutes, on your own machine

```bash
cd local-eis-viewer
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_dashboard.py --results /path/to/results --open
```

`run_dashboard.py` is the file to run. It prints the link before the server
starts:

```
====================================================================
  Local EIS Viewer
====================================================================
  Open this link:   http://127.0.0.1:8050
  (listening on 0.0.0.0:8050 — press Ctrl+C to stop)

  Found 2 run(s) across 1 order id(s): 2611976
====================================================================
```

`--open` launches your browser for you; drop it and click the link instead.
`--port 8060` if 8050 is taken. If it reports *no measurements found*, the
`--results` path is not laid out the way it expects — see below.

`EIS_RESULTS_ROOT` expects this layout — the same one the pipeline writes:

```
<root>/<order id>/<condition>/gold/plate_summary.csv
<root>/<order id>/<condition>/silver/spectra_clean.csv
<root>/<order id>/<condition>/silver/cell_aggregate.csv
```

So for your existing 45 A output, copy `results_true_areas/` to
`<root>/2611976/45A/` and it appears in the dropdowns immediately. Point the
root at several conditions (45A, 60A, 150A, 450A) to unlock the Conditions tab.

Raw recordings instead: `python run_dashboard.py --famos /path/to/Famos`.
Both flags can be given together, and both just set the corresponding
environment variable, so `EIS_RESULTS_ROOT=... python run_dashboard.py` works
equally well.

## If you started it in a notebook cell

The cell will never finish — a web server runs until you stop it, so a busy
cell *is* the running server. The link works within a second or two.

If the output lists a private address (`10.x.x.x`, `172.16–31.x.x`,
`192.168.x.x`) alongside `127.0.0.1`, the server is on a remote machine and
`127.0.0.1` in your browser points at your own laptop instead. On Databricks,
use the driver-proxy link the banner prints, or — for anything your colleagues
will use — deploy it as a Databricks App. On a plain remote VM, forward the
port: `ssh -L 8050:localhost:8050 you@the-vm`. See `docs/FRONTEND.md`.

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
| `run_dashboard.py` | **Run this to open the dashboard** |
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
