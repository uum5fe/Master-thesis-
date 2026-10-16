# Setting the pipeline up for any user, and choosing gen1 / gen2

## 1. What used to break for a colleague

These files were read from one personal workspace folder
(`/Workspace/Users/uum5fe@bosch.com/`), so the notebook failed for everyone
else:

| File | Used for | Was read from |
|---|---|---|
| `curr.csv` | per-segment current calibration, **required** | `/Workspace/Users/uum5fe@bosch.com/curr.csv` |
| `temp.csv` | temperature-sensor calibration, **required** | `/Workspace/Users/uum5fe@bosch.com/temp.csv` |
| `chain_gain_gen1.csv` | measuring-chain response (applied to every Z) | `/Workspace/Users/uum5fe@bosch.com/` |
| Abgleich delivery (`coefficients/`, `bode/`, `Step*_<T>Grad.csv`) | building the chain gain file, via-resistance check | searched under `/Workspace/Users/uum5fe@bosch.com` |
| the pipeline folder itself | fallback when the notebook path is unknown | `/Workspace/Users/uum5fe@bosch.com/Local_EIS_pipeline` |

The same personal paths were in `pipeline/bronze.py` (stand-alone entry point).
None are left in the code.

## 2. Where they go now: one shared folder on the Volume

Upload the `EIS_config/` folder of this package to

```
/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/EIS_config/
    gen1/                        green / Kashyyyk   (filled in this package)
        curr.csv
        temp.csv
        chain_gain.csv           (was chain_gain_gen1.csv)
        Abgleich/                (the Kashyyyk delivery: coefficients/, bode/, Step*.csv)
    gen2/                        blue / Naboo       (put the Naboo files here)
        curr.csv
        temp.csv
        chain_gain.csv           (written by the Chain response cell)
        Abgleich/
```

Everyone who runs the notebook needs **read** access to this folder, and to
`Famos/`, `Gamry/` and `EIS_Results/` beside it (plus **write** access to
`EIS_Results/` for the result cache, and to `EIS_config/<plate>/` if they
build a chain gain file). These are the folders the notebook already used,
so the Volume permissions your colleagues need do not change.

### Lookup order (first hit wins)

1. `/Volumes/.../EIS_config/<plate>/` — the shared folder
2. `<this notebook's folder>/EIS_config/<plate>/` — so a freshly imported
   package runs for gen1 even before the upload
3. gen1 only: your own `/Workspace/Users/<you>/` (the old location)

A gen2 run never uses a gen1 file. The lookup is in `core/site_paths.py`; the
notebook prints which file it took. To use another shared folder, change
`EIS_CONFIG_ROOT` in the widgets cell (or set the environment variable
`EIS_CONFIG_ROOT` for command-line runs).

### What still lives on the Volume and needs no change

`Famos/` (raw .DAT), `Gamry/` (whole-cell .DTA and MF4 bench logs),
`EIS_Results/` (result cache, HF response file, saved viewers), datago tables.

## 3. The plate drop-down

The widget bar now has **Plate (gen1 green/Kashyyyk, gen2 blue/Naboo)**.
Choosing gen2:

- reads `curr.csv` / `temp.csv` / `chain_gain.csv` / `Abgleich/` from
  `EIS_config/gen2/`, and stops with a clear message if the required ones are
  missing;
- selects the gen2 pad map for the whole run: segment areas and centroids,
  the cell aggregate, the DC current closure, the 36-tile statistics, the
  plausibility checks, every plate heat map and the ECM maps (the map titles
  name the plate);
- keeps gen2 results apart from gen1 results: the plate is part of the cache
  key, the local run folder is `<order>_gen2/`, the HF response file is
  `EIS_Results/<order>/hf_response_gen2/`, and a gen1 result is never shown
  for a gen2 selection.

The FAMOS channel names (`1` … `72`) are the segment numbers printed on the
plate, so the same .DAT files are read the same way for both plates; only
the geometry and the calibration change.

Existing gen1 cache entries keep their key and are still found.

## 4. Cache mode re-ran every time — fixed

**Cause.** The cache key contains a hash of the *content* of the
high-frequency response file `EIS_Results/<order>/hf_response/hf_response.csv`.
The cell "High-frequency response across conditions" rebuilt that file on
every *Run all*, and each rebuild composes the old file with a new residual,
so the file changed every time. The next session computed a different key,
found no cache entry, and recomputed every condition, which rebuilt the
file again. It never settled.

**Fix.** The HF response is built once (when the order has none) and then
kept frozen. Set `HF_RESPONSE_REBUILD = True` in that cell for one deliberate
refinement; the next run then re-evaluates each condition once and is cached
again. The chain gain file is likewise built once per plate
(`CHAIN_GAIN_REBUILD = True` to rebuild).

**What to expect once.** The first *Run all* with this version still runs
every condition whose cached key was written with an older HF file. After
that, cache mode loads from the Volume ("FROM CACHE … pipeline not run").
