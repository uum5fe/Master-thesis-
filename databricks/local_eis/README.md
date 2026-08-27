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
| `famos_segments.py` | the real segment number of every FAMOS channel; writes a relabelled copy |
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

## FAMOS channels are named after the card slot, not the segment

A DASYLab card that records segments 64..79 writes its channels out as
`"0", "1", ... "15"` — the slot on the card, not the segment on the plate. The
samples are fine; only the labels are wrong, which is the dangerous kind of
wrong: a spectrum filed under segment 3 that belongs to segment 67 looks like
a perfectly good measurement.

`famos_segments.py` reads the real channel table out of the FAMOS keys and
puts the segment numbers back:

```bash
# what belongs to what -- the range is read from the "KANAL_6479" in the name
python famos_segments.py map 2026_08_27_KANAL_6479RO2612025_60A.DDF_1.DAT --stats

# say the range yourself when the name does not carry it
python famos_segments.py map FILE --segments 64-79 --csv map.csv

# a corrected copy, so the rest of the pipeline reads the right names
python famos_segments.py relabel FILE --out corrected.DAT
```

The pairing is positional and ascending — first channel in the header (lowest
byte offset in the frame) to lowest segment — and `--reverse` flips it. If the
range does not hold exactly as many segments as the file has channels, it
refuses to map rather than pairing the two lists off as far as they go.

`relabel` rewrites only the `|CN` name keys and copies every sample byte
through untouched, so the output is the same measurement under the right
names, and the input is never modified. The names come out as bare digits,
which is what `eis_local.FamosFile.segment_names` recognises.

Note that these 16-channel cards are float64 with the channel names in `|CN`
keys, while `eis_local.FamosFile` expects the gen1 dialect — float32 with the
names in `|CP`. Use `famos_segments.read_header` for the former.

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
