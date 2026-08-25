# Local EIS Viewer — start here

This archive is the complete architecture: the analysis pipeline (`eis/`) and
the Dash frontend (`app/`) that reads its results. It is the same content as
branch `claude/impedance-frontend-viz-not3ye` of the Master-thesis repository.

## The three scripts, and which is which

| Run this | For |
| --- | --- |
| **`run_dashboard.py`** | Start the dashboard and view results |
| **`run_evaluation.py`** | Turn raw FAMOS `.DAT` into results, using the bundled `local_eis/` pipeline — the one that ran on Databricks |
| `run_pipeline.py` | The separate `eis/` pipeline. A different implementation with a different output shape; takes its paths as arguments, not from `.env`. **Not** the one you want for the Local EIS evaluation. |

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

## "Python konnte nicht gefunden werden" / "Python was not found"

Windows ships a placeholder at `python` that opens the Microsoft Store instead
of running anything. Two ways past it.

**Use the supplied launchers** — they try `.venv`, then the `py` launcher, then
`python`, and say what to do if none works:

```powershell
run_dashboard.cmd --open
run_evaluation.cmd --all --stage-local
```

**You may already have Python** and just not be reaching it. The current
python.org installer (the Python Install Manager) puts its shims in
`%LOCALAPPDATA%\Python\bin`, and the Store placeholder sits *earlier* on PATH
than that — so `python` fails while a perfectly good interpreter is installed.
Check, and use it directly:

```powershell
& "$env:LOCALAPPDATA\Python\bin\python.exe" --version
& "$env:LOCALAPPDATA\Python\bin\python.exe" run_evaluation.py --list
```

The `.cmd` launchers now probe that location too, so `run_evaluation.cmd`
works without any of this.

**Or fix the machine**, which is worth doing once:

1. Settings → Apps → Advanced app settings → **App execution aliases** → switch
   off `python.exe` and `python3.exe`. This is the direct fix: it removes the
   placeholder that is shadowing your real Python; or
2. Use the `py` launcher instead: `py -3 run_dashboard.py --open`; or
3. Re-install from python.org with **"Add python.exe to PATH"** ticked.

If a virtual environment exists at `.venv\` in this folder, the `.cmd`
launchers use it in preference to anything on PATH — which is also how to give
colleagues a fixed set of package versions:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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
EIS_CSV_ROOT=\\bosch.com\DfsRB\DfsDE\LOC\Fe\ILM\A_ILM_DSETD\Gruppenablage\EAT3\Charan\Lokale_EIS\csv_files
EIS_RESULTS_ROOT=C:\Users\uum5fe\OneDrive - Bosch Group\Local_Eis\results
```

Windows paths go in exactly as Explorer shows them — no quotes, no doubled
backslashes, spaces are fine, and a `\\server\share` network path works too.
Restart the app after editing. The startup banner prints which `.env` it read,
so there is no doubt about which file is in force.

| What you want to change | Variable |
| --- | --- |
| Folder of raw FAMOS `.DAT` | `EIS_FAMOS_ROOT` |
| Folder of raw R2-D2 **CSV sweeps** (the Gen 2 path) | `EIS_CSV_ROOT` |
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

## The Gen 2 plate and the CSV sweeps

There is **no FAMOS recording for the Gen 2 plate** — it is measured with the
R2-D2 CSV logger instead. Those sweeps are found through their own variable:

```ini
EIS_CSV_ROOT=\\bosch.com\DfsRB\DfsDE\LOC\Fe\ILM\A_ILM_DSETD\Gruppenablage\EAT3\Charan\Lokale_EIS\csv_files
```

or as a one-off flag:

```powershell
python run_dashboard.py --csv "\\bosch.com\...\Lokale_EIS\csv_files" --open
```

Point it at the folder that **holds the sweep folders**, not into one of them.
The search is recursive, so either works, but the parent gives you every sweep
at once:

```
csv_files\
    <sweep folder>\
        metadata.csv        <- the cell (Leepa:) and the coefficient set
        p1.csv, p2.csv, ... <- one file per frequency
    <another sweep>\
        ...
```

One sweep folder = one selectable run. It is listed under the cell named in
`metadata.csv`, not under the folder name, and the folder name becomes the
condition. **`metadata.csv` is required** — a folder of `p*.csv` without it is
not a run, because there is nothing to identify the cell with.

If nothing appears, ask the app rather than guessing:

```powershell
python run_dashboard.py --check
```

Section 3 of that report walks every sweep folder it found and prints the cell
and the number of frequency points in each, or says exactly what is missing.

## Seeing the raw signals (the Signals tab)

Same script — `python run_dashboard.py` — then the **Signals** tab. There is no
separate script for it.

It reads raw samples, so it only has something to draw when a **raw** run is
selected. If the tab is empty, check these in order:

1. **File format** in the left column must be `Raw recording — FAMOS .DAT`
   or `Raw sweep — R2-D2 CSV logger folder`. A finished pipeline result no
   longer contains the samples, so with `Pipeline results` selected the tab
   correctly says there is nothing to draw.
2. **A run must be selected** below it. If the run dropdown is empty, the data
   was never found — run `python run_dashboard.py --check` and read sections 2
   and 3.
3. **A segment and a frequency step** must both be picked. The step list is
   the dwells the pipeline actually located and kept, so it is empty if no
   dwell in the recording passed the SNR gate.

What the four panels show:

| Panel | What it is |
| --- | --- |
| 1 · the recording | Envelope of the whole record, with each located dwell shaded. Gaps are settling time, not lost data. |
| 2 · one dwell | Raw samples with the fitted sine through them. If the samples wander off the curve, the window is wrong or the step was not stationary. |
| 3 · are they simultaneous? | Calibrated voltage and current normalised and overlaid, with the offset in µs. Any horizontal shift is acquisition skew and goes straight into the impedance phase. |
| 4 · every tone | One row per dwell: frequency, amplitude, cycles, SNR. |

For FAMOS, each segment is paired with the cell-voltage copy on **its own
card** — the five cards free-run, so pairing across cards would put the whole
inter-card offset into the phase. For CSV there is one clock and no such
problem, so panel 3 instead shows the printed channel-scan offset as the phase
it costs **at the analogue frequency**.

## Calibration campaigns (no order id needed)

A folder of `Step<n>_<T>Grad.csv` plus `coefficients/` is a calibration
campaign. It belongs to a plate rather than to a measurement order, so the
viewer lists it by folder name — "Kashyyyk", "Naboo" — and never asks for an
order id.

```ini
EIS_CALIBRATION_ROOT=C:\Users\uum5fe\Lokale_EIS\Abgleichdaten
```

Leave it unset and the FAMOS and results roots are searched too. Then set
**File format → Calibration sweeps** and open the **Calibration** tab: shunt
sensitivity and its temperature coefficient per segment on the plate map,
linearity of every sweep, whether the shipped `curr.csv` matches the raw data,
drift between repeats of the same temperature, and the sensors against their
step labels. Clicking a segment shows the sweeps behind its number.

See `docs/CALIBRATION.md` for what the file format means and how the
coefficients relate to the sweeps.

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
