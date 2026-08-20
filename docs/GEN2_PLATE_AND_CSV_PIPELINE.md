# The gen2 plate and the CSV measurement path

Two additions to the Databricks local-EIS pipeline (`databricks/local_eis/`):

1. **The plate is now a choice.** `gen1` (green / Kashyyyk) and `gen2`
   (blue / Naboo) are both described, both verified, and selected from a
   dropdown in the runner notebook.
2. **The measurement file format is now a choice.** `famos` keeps the
   existing five-card path unchanged. `csv` is a *separate* pipeline, not a
   second reader in front of the old one, for reasons set out in §3.

Everything below is either measured from the delivered files or derived from
them, with the derivation shown. Where something is a reading rather than a
measurement it says so.

---

## 1. The gen2 plate

### 1.1 What actually changed

Both plates are 252.0 × 121.0 mm, both are a 45 × 20 grid of 5.60 × 6.05 mm
pads (900 pads, 304.92 cm²), both carry 72 segments and four temperature
sensors at x = 0, 84, 168, 252 mm. The drawings agree on all of that.

What changed is **which pads belong to which segment number** for the 36 edge
segments, 37…72 — and, as a consequence, the areas of the interior segments
that lost pad rows to them.

The 72 printed label pads were read off both coordinate drawings from their
own mm ticks (`x = (col−0.5)·5.60`, `y = (row−0.5)·6.05`), which reproduces
the gen1 map in `r2d2_geometry.py` exactly and is therefore a validated
method before it is applied to gen2:

| | gen1 (Kashyyyk) | gen2 (Naboo) |
| --- | --- | --- |
| Segments 1…36 | label cols 3,8,…,43 × rows 3,8,13,18 | identical |
| Segments 37…72 | six full-height edge strips at cols 1, 5, 10, 36, 41, 45; six segments each | scattered: cols 1, 2, 5, 10, 11, 18, 28, 35, 36, 41, 44, 45 |

On gen2, six of the edge labels sit on pads that belong to a *wide* strip on
gen1 — 49 @ (col 11, row 1), 51 @ (col 18, row 1), 57 @ (col 28, row 1),
55 @ (col 35, row 1) and their mirrors. So on gen2 those wide strips are cut
into six segments instead of four, and the extra two carry edge numbers along
the top and bottom edges of the plate.

### 1.2 The reconstruction, and why it is the only consistent one

The gen2 label set is **exactly mirror-symmetric** about the plate centre
under (col, row) → (46−col, 21−row): 37↔72, 38↔71, … 54↔55. All 18 pairs
check. That symmetry, plus "every label lies inside its own segment", plus
"the strips tile 45 columns and 20 rows exactly", pins the layout once the
column structure is taken to be unchanged:

- Strips 11–15, 16–20, 26–30, 31–35 need label rows {1, 3, 8, 13, 18, 20}
  inside six segments — only the row split 2,4,4,4,4,2 does that symmetrically.
- Strips 2–4, 5–6, 40–41, 42–44 need {1, 5, 9, 12, 16, 20} or
  {3, 5, 8, 13, 16, 18} inside six — only 3,3,4,4,3,3 does both.
- Strips 1, 7–9, 21–25, 37–39, 45 need four segments at {1, 8, 13, 20} or
  {3, 8, 13, 18} — 5,5,5,5.
- Strips 10 and 36 carry two labels only (rows 5 and 16) — 10,10.

The resulting segment counts per strip, **4,6,6,4,2,6,6,4,6,6,2,4,6,6,4**, are
themselves mirror-symmetric and sum to 72; the pad count is exactly 900.

`r2d2_geometry.self_check()` asserts all of it and passes for both plates:

```
plate           : R2-D2 gen2 (blue / Naboo)
segments        : 72
pads covered 1x : 900 / 900  (double 0, empty 0)
area sum        : 304.9200 cm2 (nominal 304.9200)
area range      : 1.6940 .. 8.4700 cm2  (x5.0)
label pads      : all inside their segment
PASS
```

### 1.3 Consequences worth knowing

| | gen1 | gen2 |
| --- | --- | --- |
| Segment area range | 0.678 … 8.470 cm² (×12.5) | 1.694 … 8.470 cm² (×5.0) |
| Distinct sizes | 8 | 9 |
| Largest segments | 20 × 8.47 cm² | 4 × 8.47 cm² (only strip 21–25) |

**Segments 1…36 are not identical in area between the plates.** They sit in
the same columns, but on gen2 the strips 11–15, 16–20, 26–30 and 31–35 gave
two pad rows each to the new edge segments, so segments 9…16 and 21…28 are
6.78 cm² on gen2 against 8.47 cm² on gen1. Read areas from the module; never
carry a number across.

### 1.4 Confidence

Firm: the column structure, the label pads, the tiling, the areas of the
interior segments. A reading rather than a measurement: the exact pad-row
boundary inside the strips carrying more than four segments — the labels
constrain each boundary to within one pad row. This is the same caveat the
gen1 module already carried. Verify once against the copper layer or the
KiCad board file and both maps become exact; everything downstream reads
areas from one place, so a correction is a one-line change.

The copper-layer PDFs shipped with the drawings do **not** settle it: the
pad grid is drawn uniformly on every layer and the segment boundaries are
defined by which pad connects to which shunt, which lives in the netlist.

---

## 2. What the Abgleich files say

`abgleich.py` reads the raw `Step<k>_<T>Grad.csv` bench files and rebuilds the
`curr.csv` / `temp.csv` coefficients from them, so a delivered calibration can
be checked rather than trusted.

Model confirmed, not assumed:

```
u_s  = R(T) · i_s          per segment, R linear in T
K(T) = c0 + 1e-3·c1·T      transfer in V/(A/cm²)      -> curr.csv
V    = c0 + c1·T           sensor line                 -> temp.csv
```

Measured on the delivered files:

| | Kashyyyk (gen1) | Naboo (gen2) |
| --- | --- | --- |
| Temperature steps | 6, 20…90 °C | 12, −40…90 °C |
| DC-sweep linearity | r² ≥ 0.999999 | r² ≥ 0.99971 |
| Copper TCR | 0.421 %/K (0.419–0.425) | 0.409 %/K (0.404–0.415) |
| Sensor lines refitted vs delivered | agree to 0.16 K | offset −1.77 K on all four |
| `R(T)/K(T)` implied constant | 2.9495 cm², CV 0.31 % | 3.0780 cm², CV 0.56 % |
| Outliers in that constant | none | segment 37 |

The last row is the sharpest single check on a calibration. `R(T)/K(T)` has to
be constant — in `T` because both are linear with the same TCR, and *across
segments* because every pad carries its own via, so a 25-pad segment has 25
vias in parallel and `R_via · A_seg` is invariant. That is why the Abgleich
returns a current **density** and why the geometric area must not be applied
a second time. Both plates satisfy it; a segment that did not would be a
segment whose calibration row does not belong to it.

The delivered gen2 `temp.csv` is uniformly offset by **−1.77 K** from a refit
of its own Step files. That is a deliberate offset correction of the same kind
as the `_mod` variant of the gen1 delivery (−0.02 V ≈ −1.8 K), not an error —
but it is a choice, and it should be a conscious one.

---

## 3. The measuring chain rolls off inside the analysis band

Each Abgleich delivery also ships `bode/<name>_100kHz_1Hz_500mA_#<n>.DTA` —
72 Gamry sweeps of the per-segment **current-measurement chain**, galvanostatic
at 500 mA rms, no DC bias. It is not electrochemistry: it is the shunt plus
its amplifier, measured ex-situ. `gamry_dta.py` reads them.

Median over 72 segments, both plates, normalised to the 1 Hz value:

| f | \|H\| | arg H | p5–p95 spread of the phase |
| --- | --- | --- | --- |
| 1 kHz | 0.9997 | −2.47° | 0.46° |
| **4.5 kHz** | **0.988** | **−11.21°** | 1.90° |
| 10 kHz | 0.940 | −24.30° | 3.75° |
| 100 kHz | 0.240 | −116.5° | 12.7° |

`config.f_max_hz` is 4500 Hz. **Eleven degrees of uncorrected phase at the top
of the band** is the same order as the acquisition skew the pipeline works
hard to measure and remove — and unlike a skew it is not all-pass, so it moves
|Z| as well. It biases exactly the decade that sets `R_Ω`.

`config.gain_file` has always had a slot for this correction and nothing has
ever filled it. `gamry_dta.write_gain_csv()` fills it. Note the sign
convention, which is documented in the module and is a named argument rather
than an assumption: with `H` the response normalised to 1 at low frequency,
`u_s = K·H·j`, so `Z_meas = Z_true/H`, and the pipeline applies `Z ← Z/gain`
— therefore the file must carry **gain = 1/H**. Getting it backwards doubles
the error instead of removing it.

### A finding to resolve before using the gen2 per-segment gain file

`cross_check_abgleich()` correlates each segment's low-frequency |Z| from the
bode sweep against `c0` from `curr.csv`. Both are the DC resistance of the
same per-segment path, so on a consistent delivery they track almost exactly:

| | correlation | ratio | ratio spread |
| --- | --- | --- | --- |
| Kashyyyk (gen1) | **+0.999** | 3.819 | 0.6 % |
| Naboo (gen2) | **+0.406** | 6.102 | 12 % |

gen1 is consistent. gen2 is not — and the Naboo `curr.csv` is also ~1.45×
lower in absolute terms than the Kashyyyk one while the two plates' bode
sweeps are nearly identical. Since the Naboo Abgleich is *internally*
self-consistent (§2), the disagreement is between the bode `#n` index and the
`curr.csv` row order, or between two different amplifier settings. This is a
question about the delivery, not something the code can resolve, so
`write_gain_csv` refuses the pairing and says why.

**Workaround, and it is a good one:** the segment-to-segment spread of the
chain phase is 1.9° at 4.5 kHz against a common −11.2°. Almost all of the
correction is in the median, and the median is index-free. Pass `shared=True`
to write one curve under segment `all`; the runner notebook does this
automatically when the cross-check fails.

---

## 4. The CSV path

### 4.1 Why it is a separate pipeline

| | FAMOS | CSV |
| --- | --- | --- |
| Clocks | five, free-running, armed separately | one |
| Measured inter-card offset | up to 5.7 s on the delivered set | does not exist |
| Synchronisation stage | mandatory, re-measured every run | **undefined** |

Running the synchronisation stage on a single-clock record does not waste
time so much as invent a quantity: there is no second clock to correlate
against, and what the estimator returns is noise with a plausible-looking
uncertainty. So the CSV path drops card alignment, drift regression, triplet
closure and the structural skew fit entirely — and picks up three problems
FAMOS does not have.

### 4.2 What replaces it

**1. Phasors on the recorded timestamps.** A FAMOS file declares one sample
interval and honours it. A CSV logger writes a timestamp per row and that
timestamp jitters. Resampling onto a uniform grid to make an FFT legal injects
interpolation error exactly where the phase matters most. A least-squares sine
fit needs the sample times to be *known*, not uniform, so
`csv_source.fit_phasor` uses the recorded timestamps whenever the measured
jitter exceeds 0.1 %. Verified end-to-end: with 20 % sample-interval jitter,
`R_Ω` still comes back to **0.06 %** of truth.

**2. All tones fitted at once.** Fitting one tone of a twelve-tone multisine
at a time charges the other eleven to the residual, so the "noise" it reports
is the rest of the excitation — a number that does not change when the actual
noise does, and is therefore useless as a weight. This was caught by the test
suite: σ was bit-identical between a clean record and a quantised one.
`csv_source.fit_phasors_multi` solves all tones in one design matrix; the
residual is then really noise, and σ tracks the printed resolution over five
orders of magnitude. It is also the leakage-free estimator, which is the whole
reason to use a designed multisine.

**3. Quantisation in the error budget.** The file is text, so the data is
quantised twice — once by the ADC and again by the number of digits printed.
`csv_source.quantisation_step` recovers the printed lattice from the data
itself and the variance `q²/12` is added in quadrature, so a coarse export
shows up as an error bar instead of as suspiciously clean data.

**4. Channel scan, if there is one.** If the logger walks the channel list
within a row, the segment and the cell voltage in one row are not
simultaneous. That is a real skew, but unlike the FAMOS case it is *known*
from the scan order and the row period rather than having to be measured —
`csv_scan_rate_hz` and `csv_channel_slots` remove it analytically. Left unset,
nothing is applied and the manifest records the assumption.

### 4.3 What is deliberately shared

The physics and the hardware description, because they belong to the plate and
not to the file format: the Abgleich (`j = u_s/K(T)`), the impedance
(`Z = U_cell/j`, area-free), the parallel aggregation
(`Z_cell = A/Σ(A_s/Z_s)`), the areas and centroids, the lin-KK validation —
and the dwell finder. `eis_local.detect_schedule` is reused verbatim for
stepped sweeps: finding a dwell is a property of the excitation, and the same
load bank drives both rigs.

### 4.4 Supported CSV layouts

`csv_source.detect_dialect` sniffs the file; `cfg.csv_dialect` overrides it.

| dialect | shape |
| --- | --- |
| `records` | `[t=1.23s;] s12: temp1=4.3V;…; i_s=0.5A;u_s=1.15V;` — the same syntax the Abgleich Step files use |
| `wide_time` | `time_s, u_cell, s1…s72 [, temp1…temp4]`, one row per instant |
| `long_time` | tidy `time_s, segment, u_s [, u_cell]` |
| `freq` | already a spectrum: `segment, freq_hz, z_re, z_im` (or \|Z\|/phase, or Gamry's `Zreal/Zimag/Zmod/Zphz`) |
| `gamry` | a folder of per-segment `.DTA` sweeps |

Column names are matched loosely — `s12`, `seg 12`, `u_s12`, `segment_12` all
resolve to segment 12. A layout that cannot be recognised raises with the
header attached rather than half-succeeding, because a reader that
half-succeeds on the wrong layout is how a plot ends up showing the
temperature channel as segment 3.

On the `freq` path the units are the only work, and `z_unit="auto"` picks
between Ω, Ω·cm² and mΩ·cm² from the magnitude and records what it chose.
Override it whenever you know: the guess is a convenience, not evidence.

### 4.5 The ECM fit

Rewritten for this path rather than reused from the notebook cell:

- **ZARCs parameterised by (R, τ, n)**, not by CPE admittance `Y₀`. With `Y₀`
  the three parameters of an arc trade against each other over orders of
  magnitude and the optimiser walks a curved valley; with τ each arc has a
  location, a size and a shape, and the two arcs of a PEMFC separate because
  their τ differ by decades. (`Y₀ = τⁿ/R` recovers the other form.)
- **Weighted by the propagated σ**, not by |Z|, when σ is known — and on this
  path it is, because the phasor fit reports it. χ²ᵥ then *means* something,
  and it is reported with a verdict rather than silently accepted.
- **Arc count chosen by AICc.** Adding an arc always lowers χ², and a two-arc
  fit to a one-arc spectrum splits a single relaxation into two coincident
  halves whose individual R and τ then mean nothing while the sum still looks
  fine. AICc charges the three extra parameters and stops that.
- **Standard errors** for every parameter from the Jacobian at the optimum.
- Starting values read off the spectrum (HF real part, LF intercept, the
  frequency where −Im Z peaks), not guessed.

### 4.6 Verification

`databricks/local_eis/test_csv_pipeline.py` synthesises a plate with a known
two-arc spectrum per segment, writes it out in each layout, runs the pipeline
and checks the numbers come back. The time-domain records are built from the
actual measurement equation (`u_s = K·U_cell/Z_s`, tone by tone), so a
pipeline that recovers Z has inverted it rather than reproduced its own
convention.

```
  time-domain multisine, uniform grid
    segment 1 R_ohmic       70.6436 mΩ·cm²  (true 70.6000,  0.06 %)  PASS
    segment 20 R_ohmic      80.9675 mΩ·cm²  (true 82.0000,  1.26 %)  PASS
    segment 72 R_ohmic     113.2700 mΩ·cm²  (true 113.2000, 0.06 %)  PASS
  same record, 20 % sample-interval jitter                            PASS
  text resolution as an error source   2.37e-05 vs 2.31e-10           PASS
  frequency-domain CSV (mΩ·cm²)                            0.00 %     PASS
  excitation discovery      mode=multisine, 12/12 tones recovered     PASS
  stepped sweep             mode=stepped_sweep, 10/10 dwells found    PASS
  plate selection           gen1 and gen2 differ on 60 segments       PASS
```

`python main.py --self-test` runs the geometry checks for both plates and the
`csv_source` round-trips alongside the existing synthetic validation.

---

## 5. Using it

In the runner notebook the two new dropdowns are **Plate generation**
(`gen1`/`gen2`) and **Measurement file format** (`famos`/`csv`), with
**CSV file / folder**, **CSV layout**, **CSV tones** and **Chain-response CSV**
alongside. `gen2 + famos` is rejected with an explanation rather than run,
because there is no FAMOS recording of the blue plate.

Two new cells sit before the pipeline run: one draws the selected plate map so
the numbering can be checked against the drawing once per campaign, and one
builds the chain-response gain file from an Abgleich `bode/` folder and
verifies the DC calibration while it is there.

From the CLI:

```bash
# gen2 plate, CSV logger, layout auto-detected
python main.py --plate gen2 --source csv --csv /path/to/run.csv \
    --curr-cal cal/curr.csv --temp-cal cal/temp.csv \
    --gain cal/chain_gain.csv --out out/run

# build the chain-response file first
python gamry_dta.py /path/to/Abgleichdaten/Naboo/bode \
    --curr-cal /path/to/Abgleichdaten/Naboo/coefficients/curr.csv \
    -o cal/chain_gain.csv --shared

# check a delivered calibration against its own Step files
python abgleich.py /path/to/Abgleichdaten/Naboo \
    --curr-cal .../curr.csv --temp-cal .../temp.csv
```

---

## 6. Open item

**No real CSV measurement file has been seen.** The reader covers the five
layouts above and auto-detects between them, and each is verified against
synthetic data built from the measurement equation — but a real file may use a
sixth layout, or one of these with a dialect quirk. If detection fails it
raises with the header attached and lists what it expected, which is the
failure mode to want; adding a sixth layout means adding one reader in
`csv_source.py` and changing nothing else. Send one real file and the
dialect can be pinned instead of sniffed.
