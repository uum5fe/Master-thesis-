# Local EIS dashboard

A dashboard over the bronze/silver/gold pipeline in `databricks/local_eis`.
It launches runs and browses their results; **every filesystem location comes
from `.env`**, so no module in this project contains a personal path.

This is a separate application from `dashboard/local-eis-viewer` (a Dash app
aimed at Databricks Apps). Nothing is shared between them, and running one does
not affect the other.

## Setup

```bash
cd dashboard/eis-dashboard
pip install -r requirements.txt
cp .env.example .env          # then edit the two paths under "REQUIRED"
streamlit run run.py
```

`python run.py` also works -- it re-execs itself under Streamlit on the port
from `EIS_DASHBOARD_PORT`.

## The `.env` contract

Two settings are the ones that change from machine to machine; everything else
has a working default.

| Variable | What it is |
|---|---|
| `EIS_DAT_DIR` | directory of FAMOS `.DAT` recordings -- the pipeline's `--dat` |
| `EIS_OUT_DIR` | where results are written; runs land in `<root>/<leepa>/<condition>/` |

Optional: `EIS_CURR_CAL`, `EIS_TEMP_CAL`, `EIS_AREAS`, `EIS_GAIN`,
`EIS_GAMRY_DIR`, `EIS_BENCH_LOG`, `EIS_PLATE`, `EIS_LEEPA`, `EIS_CONDITIONS`,
`EIS_PIPELINE_DIR`, `EIS_PYTHON`, `EIS_DASHBOARD_PORT`, `EIS_READ_ONLY`.
See `.env.example`, which documents each one.

**Anything already exported in the environment beats the file**, so a one-off
override is `set EIS_DAT_DIR=...` rather than editing `.env` and remembering to
change it back. `EIS_NO_DOTENV=1` ignores the file entirely, for a container
configured purely from the environment.

Windows paths need no escaping -- the whole rest of the line is the value:

```
EIS_DAT_DIR=C:\Users\me\OneDrive - Bosch Group\Famos
```

The **Setup** page shows what every variable resolved to and what is wrong with
it, rather than failing at import time, so a half-configured `.env` is
diagnosable in the browser.

## Pages

| Page | What it shows |
|---|---|
| Setup | resolved paths, per-setting problems, discovered conditions, existing runs |
| Run | launch the pipeline per condition, with the exact command shown; live log |
| Plate map | per-segment parameters over the plate geometry |
| Spectra | Nyquist / Bode, per segment and whole-cell |
| Diagnostics | card alignment, quality tiers, plausibility, rejected points |
| Files | every artifact the run wrote, with previews and downloads |

The dashboard never imports the pipeline; it shells out to `main.py` in
`EIS_PIPELINE_DIR`. That keeps the two dependency sets apart (the pipeline wants
scipy and matplotlib; this app does not) and means a run that dies takes a
subprocess with it, not the web server. Set `EIS_READ_ONLY=1` for a
browse-only deployment: the Run page is hidden and `EIS_PIPELINE_DIR` stops
being required.

## Why some segments show "band-limited" instead of a mass-transport value

`R_mt` is the sum of the DRT over `tau >= 10 ms`. The tau grid the pipeline
builds spans `1/(2*pi*f_max) .. 1/(2*pi*f_min)`, so that bucket only has bins in
it when the segment kept a point at or below

    1 / (2*pi*0.01 s) = 15.9 Hz

A segment whose low-frequency points were all dropped by the SNR gate has an
**empty** slow bucket. `numpy` sums an empty slice to `0.0`, so before this was
fixed such a segment reported `R_mt = 0` -- indistinguishable from a real,
measured zero, and read off the map as "no mass transport here".

`gold.split_processes` now returns NaN for a bucket the band never reached and
publishes `tau_max` beside it, and this dashboard draws those segments in their
own reserved colour and symbol with the count and segment numbers spelled out.
Because what decides it is the segment's own low-frequency SNR -- a wiring and
position property, not an operating-point one -- **the same segments come out
band-limited at every condition**, which is exactly the pattern to expect.

If you want those segments to carry a real number, the fix is in the
measurement, not the plotting: keep their sub-16 Hz points (longer dwell at the
bottom of the sweep, or a larger AC amplitude there), or accept a coarser split
by raising `tau_split_kinetic_s`.

## Tests

```bash
python -m pytest tests -q
```

`tests/make_fixture.py` also builds a browsable results tree on demand:

```bash
python tests/make_fixture.py /tmp/demo
EIS_NO_DOTENV=1 EIS_OUT_DIR=/tmp/demo EIS_READ_ONLY=1 streamlit run run.py
```
