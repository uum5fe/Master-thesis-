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
| `r2d2` / `r2d2_sweep` | **the real logger format** — see §5 |
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

## 5. The R2-D2 logger format

A real sweep (`metadata.csv` + `p1.csv`, software V2.5, 20.04.2026) settled the
open item from §6 of the first pass. The format is now supported directly as
the `r2d2` / `r2d2_sweep` dialects.

### 5.1 Layout

```
<sweep folder>/
    metadata.csv          date, software version, which Abgleich coefficient
                          set was applied, the Leepa, and notes
    p1.csv, p2.csv, ...   ONE FILE PER FREQUENCY POINT
```

Each point file is tab-separated with **two** header rows:

```
timestamp    s1 ... s72   uc1 uc2 uc3 uc4   temp1 ... temp4
timeshifts   0.000000  1.100125  2.200017  ...  86.897984
2026.04.20 10:21:11,429062    1.936579   1.981619   ...
```

Measured on the delivered `p1.csv`: 80 channels, 19 417 rows, fs = 11 001.10 Hz,
1.765 s.

### 5.2 The `timeshifts` row is the whole game

`timeshifts` is the acquisition instant of each column **inside one row, in
microseconds**. The channels are 1.1 µs apart and span 86.898 µs against a
90.90 µs sample period — 96 % of it. The logger is a scanning multiplexer that
walks the whole plate once per row and only just finishes before the next one
starts.

Segment 1 is sampled at 0 µs; the cell-voltage taps at 79–82 µs. The impedance
is the *ratio* of those two channels, so **80 µs of skew sits directly in it**
— 29° at 1 kHz, more than a quadrant near the top of this rig's band. Nothing
in the file warns you. Read the rows as simultaneous samples and every phase
is wrong by an amount that grows with frequency, which is precisely the
signature of an electrochemical process and will be read as one.

Correcting it needs no fit at all: the delays are printed. `deskew()` applies
them, and `uc2 − uc1` is formed **in the phasor domain afterwards**, never as a
row-wise subtraction — uc1 and uc2 are themselves 1.1 µs apart.

### 5.3 What the columns hold — established, not assumed

**`s1…s72` are a current density in A/cm², not a shunt voltage.** They span
1.8× across the plate while the gen1 segment areas span 12.5×, and
`corr(s, area) = +0.25`; a *current* would have to track the area at ≈ +0.95.
The logger has already applied the coefficient set named in `metadata.csv`
(`Coefficients: Coruscant` on this delivery), which is also why `temp1…temp4`
arrive in °C rather than volts. **Applying `curr.csv` again would divide by K
twice** — the pipeline logs that it is skipping it and why.

**`uc1…uc4` are two differential pairs.** uc1 and uc3 sit near 0 V, uc2 and
uc4 near +0.59 V, so `uc2 − uc1` and `uc4 − uc3` are two independent readings of
the same 0.61 V cell. Both are computed; their mean is used and their
disagreement is reported. On `p1.csv` they differ by **28 %** in amplitude,
which is a lead-placement or contact difference and sets a floor on how well
any single reference defines Z.

Sanity check on the whole chain, gen1 areas: j = 1.37…2.46 A/cm² (median 2.03),
Σ j·A = **615.9 A** at U = 0.61 V. The DC map falls smoothly from ~2.3 A/cm²
at the inlet to ~1.4 A/cm² at the outlet — reactant depletion down the channel,
which is what the plate is for.

### 5.4 A point file is a burst, not a continuous recording

The delivered `p1.csv` runs 1.765 s. The excitation is only present from
≈ 0.25 s to ≈ 1.35 s. Before and after there is a persistent instrument
artefact near **999 Hz** sitting at about 6× the noise floor, while the
excitation runs at **380×**.

This matters more than it looks:

- Fitting the whole record inflates the residual — and therefore σ — by the
  silent fraction, for no gain in the numerator.
- Worse, a naive peak-pick on a *truncated* file locks onto the 999 Hz
  artefact and reports an impedance measured against it. Trimming the first
  1200 rows of the delivered file and analysing them gives a confident, wrong
  answer at 1000 Hz. That is not hypothetical: it is what the first version of
  this reader did, and it is why the fixture in
  `databricks/local_eis/fixtures/r2d2_sample/` is cut from *inside* the burst.

So the pipeline does two things before fitting anything:

1. **Finds the tone on the segments, not on one channel.** `uc1` carries a
   3488 Hz component larger than the excitation; the excitation is the one
   thing that drives all 72 segments coherently, so the segment-averaged
   amplitude spectrum reinforces it and averages down everything else.
2. **Windows to the burst.** `eis_local.demod_envelope` / `dwell_window` /
   `polish_window` already do this for the FAMOS sweeps — same job, same kind
   of signal — applied here to the normalised composite of all 72 segments,
   which has ≈ √72 times the SNR of any one channel. The peak is compared
   against the 10th percentile of the envelope rather than the median,
   because on this file the burst is 62 % of the record and a median test
   would conclude there was nothing to window to.

A record whose strongest common segment tone is under 10× the noise floor is
**refused with a reason** rather than analysed. A point file that is all
lead-in looks exactly like that.

### 5.5 The finding: the top of the sweep is undersampled

`p1.csv` contains a clean tone at **923.08 Hz** against fs = 11 001.10 Hz. Taken
at face value that is an ordinary EIS point. It is not.

Across one row the 80 channels sample the analogue waveform at 1.1 µs
intervals, so the phase difference between channel *k* and channel 0 is
`2π·f_analogue·τ_k` — **at the true frequency of the signal, whatever the
row-rate DFT reports**. A 923 Hz tone rotates the phase by 0.33 °/µs and would
produce 29° across the 87 µs scan. The measured rotation is ≈ 307°, an order of
magnitude more.

Scoring candidates by how well the 72 segment phasors collapse onto a common
phase after de-skewing (they must: one plate, one imposed current):

| de-skew at | resultant R |
| --- | --- |
| 923.08 Hz (the tone in the record) | **0.14** — no common phase at all |
| fs − 923.08 = 10 078.02 Hz, conjugated | **0.84** |
| fs + 923.08 = 11 924.19 Hz, its own sign convention | rejected |

So the analogue tone is near 10 kHz and the 923 Hz in the record is its alias,
folded by a sampler running at 11 kHz with nothing in front of it that stops
10 kHz. Consistent with `p1` being the *first* point of a sweep that starts
high — and with the Gamry chain response (§3), which is still at |H| = 0.94 at
10 kHz and passes the signal happily.

This is a property of the rig, not of the evaluation, and it is **reported
rather than silently worked around**. `nyquist_zone()` returns the frequency
the scan implies, the resultant at every candidate, and a verdict; the run log
says `UNDERSAMPLED` and the manifest records it per point.

**The limit of what the scan can do.** It resolves f only to about
1/scan_span ≈ 11 kHz — the same ambiguity as the aliasing itself. It can tell a
baseband point from an undersampled one, and it picks the zone here because the
conjugation parity breaks the tie, but it cannot replace *knowing* the
frequency. Pass the sweep's own frequency list in `cfg.csv_tones` (file order)
and it is used and cross-checked instead of inferred.

### 5.6 A single point file is still useful

One file is one frequency, so there is no spectrum to fit — lin-KK needs 6
points and the ECM 5. Rather than reporting zero segments, the pipeline says so
plainly and still writes the per-frequency table plus plate maps of |Z|, arg Z
and the DC current density at that frequency. Point `cfg.csv_path` at the whole
sweep folder to get spectra.

### 5.7 Verified

`test_csv_pipeline.py` builds a synthetic sweep in this exact format — one file
per frequency, a channel scan inside every row, one tone deliberately placed
above fs/2 so the alias path is exercised rather than assumed:

```
  R2-D2 logger sweep
    6 point files read, 0 failed                                  PASS
    6/6 analogue frequencies recovered (one above fs/2)           PASS
    undersampled points flagged: 1                                PASS
    segment 1 R_ohmic    70.5962 mΩ·cm²  (true 70.6000,  0.01 %)  PASS
    segment 72 R_ohmic  113.1790 mΩ·cm²  (true 113.2000, 0.02 %)  PASS
    scan span 87.3 us = 283° at 9000 Hz  (why the deskew is not optional)
```

Against the real fixture — 2500 rows of the delivered `p1.csv`, cut from
inside the burst:

```
  real fixture (2500 rows of a real p1.csv)
    dialect detected as a sweep / 80 channels / 72 segments        PASS
    scan spans ~96 % of a sample                                   PASS
    s columns read as A/cm2, temps as degC                         PASS
    coefficient set recorded, four uc taps                         PASS
    alias 923.1 Hz -> analogue 10078 Hz, zone 1, R 0.84 vs 0.14    PASS
    a record with no excitation is refused                         PASS
```

---

## 6. Using it

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

## 7. Open items

1. **The sweep's frequency list.** The channel scan identifies the Nyquist
   zone but resolves f only to about fs (§5.4). The rig commands the
   frequencies, so it knows them exactly — pass them in `cfg.csv_tones`, in
   file order, and the inference is replaced by a cross-check.

2. **Undersampling at the top of the sweep** (§5.4) is a measurement-setup
   question, not an evaluation one. Either the sampler runs faster than twice
   the highest tone, or an anti-alias filter goes in front of it, or the top
   points are treated as deliberate coherent undersampling with their
   frequencies declared. The pipeline handles the third case and flags the
   first two; it cannot make an aliased point unambiguous on its own.

3. **The two cell-voltage pairs disagree by 28 %** on the delivered file
   (§5.3). Both are used and the spread is reported, but it is worth tracing
   to the leads: it is a floor on the accuracy of every Z on that plate.

4. **`Coefficients: Coruscant`** names a third coefficient set, alongside
   Kashyyyk (gen1) and Naboo (gen2). Which physical plate that Leepa
   (`FC2600265-02`) carries decides which geometry the run should select, and
   the file does not say. The plate dropdown is explicit for exactly this
   reason.
