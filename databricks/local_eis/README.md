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
| `csv_source.py` | CSV reader, five layouts, dialect auto-detection |
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
