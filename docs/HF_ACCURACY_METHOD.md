# Making the high-frequency end of a locally-resolved spectrum trustworthy

**Problem.** Above a few hundred hertz the measured local impedance of a
segmented cell stops being dominated by the cell and starts being dominated by
the instrument. The errors are all *phase* errors that grow with frequency, so
none of them shows up in the magnitude, and a spectrum with tens of degrees of
error at 4 kHz looks entirely plausible until you notice the Nyquist arc bends
the wrong way.

**What was already handled.** The pipeline measures inter-card skew and clock
drift from the shared cell-voltage channel and aligns in the time domain to
better than 100 ns — 0.1° at 3.8 kHz. That work is necessary and it is not the
subject of this note.

**What this note is about.** With the timing budget met to a tenth of a degree,
the largest remaining high-frequency error was **twenty-six degrees**, and no
amount of synchronisation could have found it.

---

## 1. The error that timing cannot see: the via shunt is not a resistor

Each segment's current is measured as the voltage across a via shunt, and the
calibration file describes that shunt with two real numbers:

```
H(T) = c0 + 1e-3 * c1 * T          [V * cm^2 / A]
```

A real number. But the shunt is a physical loop, and a physical loop has
inductance. Working the numbers for this cell:

| quantity | value |
| --- | --- |
| `H` at 62 °C | 1.05 × 10⁻⁴ V·cm²/A |
| segment area | 4.235 cm² |
| **shunt resistance** `R = H/A` | **24.8 µΩ** |
| plausible loop inductance | 0.5 – 1 nH |
| **time constant** `τ = L/R` | **20 – 40 µs** |
| **phase at 3.8 kHz** | **26 – 46°** |

The shunt resistance is *tens of microohms*. Its own loop inductance therefore
has a time constant in the tens of microseconds — three orders of magnitude
larger than the 73 ns timing budget the synchronisation stage works so hard to
meet. Treating `H` as real puts that entire phase error onto every segment, in
exactly the band where the arc crosses the real axis.

This is invisible to every timing diagnostic in the pipeline, because it is not
a timing error. The cards are perfectly synchronised; the *transducer* is
complex and was modelled as real.

### The fix

Model the shunt as what it is and divide it out:

```
Z_true(f) = Z_measured(f) · (1 + jω·τ_shunt) · exp(−jω·τ_residual)
                          · H_aa(f; segment) / H_aa(f; cell voltage)
```

`eis/hf.py` — `InstrumentResponse`, `shunt_time_constant`. The anti-alias term
is there for completeness: only the *mismatch* between the two channels'
filters survives the ratio, worth about half a degree per per-cent of corner
tolerance, so a shared front end costs nothing and a mismatched one is not
nothing.

**Measured effect** (synthetic data with a known 0.6 nH shunt, everything else
perfect):

| | median error over the band | error at the top tone |
| --- | --- | --- |
| uncorrected | 1.46 % | **48.1 %** |
| corrected | 0.18 % | **0.11 %** |

A 450× improvement at the high-frequency end, from one physical constant.

### There is a second-order consequence worth knowing

When a card's cell-voltage channel dies, its timing has to be measured against
a *segment* channel instead — and a segment channel carries the shunt's phase.
The delay estimator reads 25 µs of shunt phase as 25 µs of card delay. In the
end-to-end demonstration this cost the proxy-timed card a factor of five:

| | median error on the proxy-timed card |
| --- | --- |
| shunt phase left in the proxy | 1.99 % |
| shunt de-embedded before timing | 0.39 % |

`eis/pipeline/bronze.py` — `_deembed_shunt`.

---

## 2. The residual delay: identifying what a single spectrum cannot

After the shunt correction, whatever phase error is left is a genuine delay.
Measuring it from the data is the classic trap, because **a delay and a series
inductance are degenerate over a finite band**:

```
Z · exp(jω·ε)  ≈  Z + jω·ε·Rs        (at HF, where |Z| ≈ Rs)
```

so a delay `ε` is indistinguishable from an extra inductance `ε·Rs`. This is
why an unbounded "Kramers–Kronig-optimal delay" search quietly eats the cell's
real inductance and returns a beautiful, wrong spectrum.

### The new idea: the segmented cell is not one spectrum, it is eighty

The degeneracy is unbreakable for a single spectrum. It is *not* unbreakable
for eighty spectra measured through one instrument, because the two quantities
have different scope:

- the residual delay `ε` is a property of the **acquisition card** — shared by
  every segment on it;
- the wiring inductance `L_k` is a property of **one segment's current path** —
  individual.

Fit `(Rs_k, L_k)` on the high-frequency band of every segment. A shared delay
maps

```
L̂_k  =  L_k  +  ε · Rs_k
```

so a non-zero delay appears as a **spurious proportionality between the fitted
inductance and the fitted ohmic resistance across the plate**, and the
regression slope *is* the delay:

```
ε  =  slope of  L̂_k  against  Rs_k
```

in closed form, from quantities that are themselves closed-form weighted least
squares. `eis/hf.py` — `identify_common_delay`.

**The identifying assumption, stated rather than hidden:** a segment's lead
inductance is uncorrelated with its local membrane resistance. The first is set
by via geometry, the second by hydration. It is also *testable*, and the
function reports the test — after correction the correlation must be gone and
the inductance scatter must shrink. A correction that does neither is rejected
as noise rather than applied.

**Two conditions the estimator checks and declines on:**

1. `Rs` has to actually vary across the plate by more than its own measurement
   uncertainty. When it does not, the regression is the ratio of two noises and
   the function says so instead of returning a confident number.
2. The answer is clipped to a bound taken from the *measured* skew. Inside the
   bound the estimate is a refinement; outside it, it would be absorbing real
   physics.

**Validated** on synthetic segments with independent `L_k` and a 30 % spread in
`Rs`: an imposed 400 ns is recovered as 445 ± 445 ns, the null case returns
45 ± 445 ns, the `L`-vs-`Rs` correlation goes 0.26 → 0.00, and a plate with no
`Rs` spread is correctly declined.

### Ranked alternatives, when they exist

| method | assumption | when to prefer it |
| --- | --- | --- |
| **reference anchor** — compare `Σ A_k/Z_k` against a cell-level potentiostat spectrum | none | whenever a reference spectrum exists; it is the only method that sees the delay common to *all* cards |
| **inductance–resistance decorrelation** | `L_k ⟂ Rs_k` | the default; needs ≥ 4 segments and real `Rs` spread |
| **pooled Kramers–Kronig scan** | a bound on `L` | fallback for a card timed from a segment proxy, where the delay is unknown to microseconds |

---

## 3. What the Kramers–Kronig fallback taught us, including a bug

The third method looked easy and was not. Both obvious formulations are wrong:

- fit the Voigt basis **with** a free `jωL` term and that term absorbs the
  rotation exactly — the residual is blind and the scan reports nothing;
- fit it **without** the term and the basis, being purely capacitive, cannot
  represent the cell's own inductance either — so the scan rotates until the
  *real* inductance is cancelled and reports a delay several times too large.

What identifies the delay is a **bound on the inductance**: a few nanohenries
of via geometry against the 84 nH that a 6 µs delay would fake on a 14 mΩ
segment. `eis/validate.py` — `_fit_voigt(max_inductance=...)`, an exact
single-box-constraint projection.

**And a bug this exposed.** While chasing a ±12 µs bias in the scan we found it
was not the degeneracy at all — it was the Lin-KK model order selection. The
Schönleber μ-criterion was implemented as "increase M until μ drops below
0.85", but **μ is not monotonic in M**: on real data it alternates, because an
even element count can place a pair of relaxations symmetrically where an odd
one cannot. Stopping at the first dip selected a four-element model:

| model order | μ | max model residual |
| --- | --- | --- |
| 3 | +0.904 | 7.6 % |
| **4 (chosen)** | **+0.562** | **13.2 %** |
| 12 | +0.925 | 0.011 % |
| 14 | +0.926 | 0.003 % |

Every Kramers–Kronig verdict in the pipeline was being computed from a fit with
**13 % model error**, so the "residual" it reported was the model's, not the
data's. The test could not have detected anything smaller than its own
inadequacy.

The fix evaluates every order, discards the ones μ flags as over-fitting, and
takes the smallest of the rest that gets within a factor of two of the best
achievable residual — parsimony among the models that fit, rather than the
first model that stops improving. Model residual on a clean spectrum: **13.2 %
→ 0.003 %**. With that fixed, the delay scan's bias collapsed from ±12 µs to
0.57 µs, which is exactly `L/Rs` — the irreducible degeneracy, and now reported
as the estimator's uncertainty rather than absorbed silently.

---

## 4. Measuring the top of the band instead of merely reaching it

A single Welch window length cannot serve a band spanning four decades. A
window long enough to hold eight periods of 1 Hz (80 000 samples at 10 kHz)
leaves a 40 s record with about ten averages — and throws away the thousands of
short windows the same record offers at 3 kHz, where the uncertainty matters
most because the excitation has rolled off.

The multi-resolution path gives each octave band the shortest window it can
afford:

```
nperseg(band) = 2^ceil(log2(min_periods · fs / f_low))
```

Two things follow, and both are the point:

- the number of averages **grows towards high frequency** instead of staying
  fixed, so `σ ∝ 1/√n_eff` falls where it was worst;
- the analysis window **shrinks towards high frequency**, so a drift or a
  transient smears fewer high-frequency windows and the ensemble gate can
  remove them individually.

The cost is coarser *absolute* frequency resolution at the top, which is
exactly where an impedance spectrum does not need it — what matters there is
relative resolution, and that is constant across the plan.

`eis/spectra.py` — `multiresolution_plan`, `impedance_multiresolution`.

---

## 5. Uncertainty that reflects what is actually known

Two changes, both of which turned out to matter more than expected.

**Timing uncertainty belongs in the phase uncertainty.** A card timed from a
segment proxy is good to ~10 µs; one timed from its own cell voltage is good to
nanoseconds. Before, both reported the same error bar at 3.8 kHz. Now:

```
σ_φ²  +=  (2π f σ_τ)²
```

**But a systematic must never weight a fit.** Folding the timing term into the
weights made the Lin-KK fit ignore the entire high-frequency band — where the
systematic is largest — and then report an enormous residual there. Spectra
therefore carry `sigma_rel` (total, for error bars and verdicts) and
`sigma_rel_random` (random only, for weights) separately.

Consequently the Kramers–Kronig verdict now uses **both** an absolute and a
statistical criterion, and a point must fail both to count:

```
|residual| > 1 %       AND       |residual| > 3σ
```

An honestly imprecise segment is no longer declared non-causal; an over-precise
one no longer passes on a systematic error.

---

## 6. In-plane crosstalk (available, off by default)

A segment collects some of its neighbours' current through the in-plane
conductivity of the diffusion medium, so the measured admittances are a
spatially *smoothed* version of the true ones. With a mixing matrix `M` built
from the segment adjacency, `Y_measured = M · Y_true`, and a regularised
inverse sharpens the map. `eis/hf.py` — `build_crosstalk_model`,
`deconvolve_crosstalk`.

`crosstalk_alpha` defaults to **0**. The mixing fraction has to come from a
characterisation of the plate, and inventing one would trade a known blur for
an unknown artefact.

---

## 7. Every segment stays on the plate

None of the above helps if the awkward segments are quietly dropped before
anyone sees them. So:

- nothing is rejected; every segment reaches the output with a machine-written
  `status`, a list of `flags` and a `quality` score in [0, 1];
- frequency points below the coherence gate are **marked, not deleted**, so all
  segments share one frequency grid — which is what makes the frequency-resolved
  plate map, the whole-plate admittance sum and the crosstalk deconvolution
  possible at all;
- a segment with no shunt calibration is scaled by unity, marked
  `no_calibration`, and still plotted — in shunt volts per amp, and the table
  says so;
- a card whose timing could not be verified still produces impedances, marked
  `timing_unverified`, with the unknown delay carried as uncertainty;
- plate maps draw every segment, desaturate the doubtful ones, and compute the
  colour scale from the trustworthy ones so a single broken channel cannot
  flatten the map.

The quality decision lives in the data, where it can be queried, rather than in
a list of segment numbers that never reached the table.

---

## 8. Where the numbers came from

Effects this chain corrects, and the literature that documents them:

- **In-plane conduction of the porous transport layer distorts spatially
  resolved current and impedance measurements**; the correction is a mixing
  matrix computed from transport-layer properties and inverted (§6).
  <https://iopscience.iop.org/article/10.1149/1945-7111/ae60aa>
- **Lateral current between segments ("crosstalk") falsifies spatial results**
  by decoupling the measured segment current from the local reaction current.
  <https://iopscience.iop.org/article/10.1149/1945-7111/ad9064>
- **Inductive artefacts in the upper frequency range originate in the
  measurement wiring**, and the high-frequency real-axis intercept is the sum
  of ohmic resistances only once that inductance is subtracted (§1, §2).
  <https://www.biologic.net/documents/eis-precautions-electrochemistry-battery-application-note-5/>
- **Instrument artefacts in impedance spectra are estimable and correctable as
  a complex response** rather than a scalar (§1).
  <https://www.nature.com/articles/s41598-020-80468-x>
- **Local high-frequency resistance distribution is the primary spatially
  resolved observable**, which is why the HFR fit is the number that reaches
  the map (§9).
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11647849/>
- **Spatially resolved EIS on automotive-scale cells** — the measurement this
  pipeline is built for.
  <https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/celc.202200069>

Method sources already used in the pipeline: Schönleber & Ivers-Tiffée for the
Lin-KK μ-criterion, Bendat & Piersol for the coherence-based uncertainty, and
Pintelon & Schoukens for the multisine best-linear-approximation framework.

---

## 9. And the number that reaches the map

`Rs` used to be the median of the highest few real parts. That estimator is
biased by the inductive tail it does not model, and it carries no uncertainty.
It is now a weighted fit over the top decade:

```
Z(f) = Rs + jωL + B·(jω)^(−n)
```

linear in its parameters, so it is one least-squares solve with an exact
covariance — no optimiser, no starting guess, no local minimum. `B` carries the
tail of the kinetic arc that is still present at the top of the band; without
it that tail leaks into `Rs` and biases the map everywhere the arc is large,
which is precisely where the interesting segments are.

`Rs` now arrives with a standard error, and the fitted `L` is what the delay
identification of §2 regresses on. `eis/hf.py` — `fit_hf_resistance`.

---

## Summary of measured effects

| change | effect |
| --- | --- |
| complex via-shunt response | error at the top tone **48.1 % → 0.11 %** |
| shunt de-embedded from the timing proxy | proxy-timed card **1.99 % → 0.39 %** |
| pooled KK delay on that card | **0.39 % → 0.22 %** |
| Lin-KK model order fixed | KK model residual **13.2 % → 0.003 %** |
| multi-resolution windows | > 4× the averages in the top decade, `σ` falls with `√n` |
| decorrelation identifier | 400 ns recovered as 445 ± 445 ns; declines when unidentifiable |
| every segment kept | 0 segments dropped; all classified and mapped |
