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

## Using it from VS Code

Open the folder that contains `run_dashboard.py`, `app/` and `eis/` as the
workspace root — not its parent. If you unzipped into
`Downloads\local-eis-viewer\local-eis-viewer\`, the inner folder is the one
to open.

Then either press **F5** and pick *Local EIS Viewer (dashboard)* — `.vscode/`
already has the configuration — or run this in the VS Code terminal:

```powershell
python run_dashboard.py --results C:\path\to\results --open
```

Do **not** press ▶ on `app/app.py`. That runs the file directly rather than as
part of the package, and it is what leads people to paste `sys.path` lines at
the top of it. Anything inserted above `from __future__ import annotations` is
an immediate `SyntaxError: from __future__ imports must occur at the beginning
of the file` — the message points at the future import, not at the line that
actually broke it. `run_dashboard.py` exists precisely so no such edit is ever
needed: it puts the project root on the import path itself.

If you have already edited `app/app.py`, restore it from this archive rather
than trying to repair it.

### "Terminal environment injection is disabled"

A VS Code notification, not an error — and it means VS Code has *found* your
`.env`, so the file is where it should be. Dismiss it: the application reads
`.env` itself and does not depend on VS Code or the shell putting anything into
the environment. The setting it offers is already enabled in `.vscode/settings.json`
for consistency, so the prompt should not return.

To confirm the file is actually being read, ask the application rather than the
editor:

```powershell
python run_dashboard.py --check
```

The first line of its report names the `.env` it read.

## If you started it in a notebook cell

The cell will never finish — a web server runs until you stop it, so a busy
cell *is* the running server. The link works within a second or two.

If the output lists a private address (`10.x.x.x`, `172.16–31.x.x`,
`192.168.x.x`) alongside `127.0.0.1`, the server is on a remote machine and
`127.0.0.1` in your browser points at your own laptop instead. On Databricks,
use the driver-proxy link the banner prints, or — for anything your colleagues
will use — deploy it as a Databricks App. On a plain remote VM, forward the
port: `ssh -L 8050:localhost:8050 you@the-vm`. See `docs/FRONTEND.md`.

## "No runs discovered" — find out why in one command

```powershell
python run_dashboard.py --check
```

It reports what it can and cannot see, and names the fix: whether a `.env` was
read at all (and whether one was saved as `.env.txt` by mistake), whether each
folder exists, how many `.DAT` files it holds and whether their names are
recognised, and whether the results folders are laid out the way the app
expects. In VS Code it is the *Check configuration* entry in the Run menu.

The four usual causes, in order:

1. **No `.env`** — or Notepad saved it as `.env.txt`. Explorer hides the
   extension by default; turn on *View > File name extensions* to see it.
2. **`.env` in the wrong folder** — it belongs beside `run_dashboard.py`. If
   you unzipped into `local-eis-viewer\local-eis-viewer\`, that is the inner
   folder.
3. **`EIS_RESULTS_ROOT` at the wrong level** — the app expects
   `<root>\<order id>\<condition>\gold\plate_summary.csv`. Pointing it
   straight at a folder that contains `gold` and `silver` gives one unnamed
   entry instead of a picker.
4. **`.DAT` filenames in an unfamiliar convention** — set `EIS_FAMOS_REGEX`.

## Where do I change the paths?

**In `.env`** — a file next to `run_dashboard.py`. It does not exist until you
make one, and the tool will make it for you:

```powershell
python run_dashboard.py --init
```

That writes `.env` and prints its full path. Creating a file whose name starts
with a dot is awkward on Windows — Explorer resists it and Notepad appends
`.txt`, which then *looks* right because Explorer hides the extension — so let
the tool do it.

You can fill it in at the same time:

```powershell
python run_dashboard.py --init --famos "\\bosch.com\DfsRB\...\Daten\2611976_16_07" --results "C:\Users\uum5fe\Lokale_EIS\results"
```

Otherwise open the created file in VS Code and set the paths by hand:

```ini
EIS_FAMOS_ROOT=C:\Users\uum5fe\OneDrive - Bosch Group\Local_Eis\2611976_16_07
EIS_RESULTS_ROOT=C:\Users\uum5fe\OneDrive - Bosch Group\Local_Eis\results
```

Windows paths go in exactly as Explorer shows them — no quotes, no doubled
backslashes, spaces are fine, and a `\\server\share` network path works too.
Restart the app after editing. The startup banner prints which `.env` it read,
so there is no doubt about which file is in force.

| What you want to change | Variable |
| --- | --- |
| Folder of raw FAMOS `.DAT` | `EIS_FAMOS_ROOT` |
| Folder of processed results (CSV/Parquet) | `EIS_RESULTS_ROOT` |
| A different `.DAT` naming convention | `EIS_FAMOS_REGEX` |
| Shunt / temperature calibration | `EIS_CURR_CAL`, `EIS_TEMP_CAL` |
| Let the app process `.DAT` itself | `EIS_ALLOW_INLINE_PIPELINE=1` |
| Where an inline run writes results | `EIS_SCRATCH_RESULTS` |
| Extra plate generations (Gen 2, Gen 3) | `EIS_PLATE_SPEC_DIR` |
| Port | `PORT` |

Subfolders are searched, so `EIS_FAMOS_ROOT` can point at the campaign folder
or at its parent — pointing at `...\Local_Eis` finds `2611976_16_07` and every
other campaign beside it, and each shows up under its own order id.

For a one-off that should not change the file, use a flag instead — it wins
over `.env`:

```powershell
python run_dashboard.py --famos "D:\some other campaign" --open
```

## Processing raw .DAT into results

The pipeline is bundled in `local_eis/` — the same bronze/silver/gold modules
that ran on Databricks. Nothing about the evaluation needs a cluster.

```powershell
python run_evaluation.py --list           # what recordings did it find?
python run_evaluation.py --all            # process them
python run_dashboard.py --open            # look at the results
```

For that you need, in `.env`:

```ini
EIS_FAMOS_ROOT=C:\Users\uum5fe\OneDrive - Bosch Group\Local_Eis\2611976_16_07
EIS_RESULTS_ROOT=C:\Users\uum5fe\OneDrive - Bosch Group\Local_Eis\results
EIS_CURR_CAL=C:\Users\uum5fe\OneDrive - Bosch Group\Local_Eis\cal\curr.csv
EIS_TEMP_CAL=C:\Users\uum5fe\OneDrive - Bosch Group\Local_Eis\cal\temp.csv
```

`curr.csv` is 72 lines of `c0;c1`, `temp.csv` is 4 lines of `c0;c1`. The shunt
calibration is not optional: it is the only absolute scale left in the chain
once the potentiostat is gone, and without it every impedance is in shunt volts
per amp rather than in ohms. `run_evaluation.py` refuses to start rather than
producing numbers with the wrong units.

Results are written to `<results root>\<order id>\<condition>\`, which is
exactly where the dashboard looks — no copying step. Allow minutes per
condition; bronze reads every sample of every card.

To run it from inside the dashboard instead, add `EIS_ALLOW_INLINE_PIPELINE=1`
and use *Run pipeline on this selection* on the Overview tab.

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
