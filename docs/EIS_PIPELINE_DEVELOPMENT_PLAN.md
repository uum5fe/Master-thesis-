# Locally-Resolved EIS Pipeline — Bronze / Silver / Gold Development Plan

**Scope:** 80-segment PEM fuel cell, multi-card DAQ writing imc FAMOS `.DAT`, 10 kHz/channel, broadband galvanostatic excitation at several DC operating points.
**Goal:** rebuild the existing pipeline into a metrologically defensible, automated analysis platform producing per-segment Nyquist/Bode spectra, ECM parameters, and spatial maps.

> **Status.** This plan is implemented. The three tiers are `eis/pipeline/bronze.py`,
> `silver.py` and `gold.py`, driven by `main.py`; the audit findings A1–A13 below are
> all addressed.
>
> Two things turned out differently from the plan and are worth reading before
> the rest of it:
>
> 1. **The largest high-frequency error was not a timing error at all.** With
>    the §1.1 budget met to 0.1°, the via shunt's own loop inductance was still
>    worth **26°** at 3.8 kHz — the shunt is 25 µΩ, so half a nanohenry is a
>    20 µs time constant. No timing diagnostic can see it, because the cards
>    are correctly synchronised and the *transducer* is complex. §S5 item 5
>    ("shunt transfer function H(T), magnitude only") is wrong; it is a complex
>    element.
> 2. **The Δt-vs-L degeneracy of A5/§S5 is breakable**, but not the way §S5
>    suggested. The residual delay is common to a card while the wiring
>    inductance is individual, so regressing the fitted `L_k` on the fitted
>    `Rs_k` across the plate gives the delay in closed form as the slope.
>
> Both, plus the multi-resolution spectral estimate and a Lin-KK model-order bug
> that was making every KK verdict meaningless, are written up in
> **[HF_ACCURACY_METHOD.md](HF_ACCURACY_METHOD.md)**.
>
> Departures from the target layout of §2: the `spectra/`, `correct/`,
> `validate/` and `viz/` sub-packages are single modules (`eis/spectra.py`,
> `eis/validate.py`, `eis/viz.py`), the corrections live in `eis/hf.py` beside
> the identification that produces their constants, and the linked
> heatmap/spectrum dashboard of G3 is `eis/dashboard.py` — one self-contained
> HTML file with no Plotly dependency. DRT (G2, optional) is not implemented.

---

## 0. Why a rebuild — audit of the current pipeline

The existing code works end-to-end, but the accuracy chain has a small number of structural weaknesses. The three-tier plan below is organised around fixing them in dependency order.

| # | Finding | Where | Consequence |
| --- | --- | --- | --- |
| A1 | Cards are merged by **array index**, not by time. `time_s = arange(n_min)/fs` after truncating to the shortest file. | `Bronze_Raw_Data_ETL.consolidate_condition()` | Any start-trigger offset or clock-rate difference between cards becomes an uncorrected phase error. This is the root cause of everything downstream. |
| A2 | Absolute start time is discarded — the parser hardcodes `'start': 0.0`. | `eis_utils.load_channels_from_famos()` | The one piece of hard evidence about inter-file offset is thrown away before it can be used. |
| A3 | Timing is repaired *after the fact* with hardcoded constants: `DT_PER_CARD` (default −118 ns), `DT_INTERLEAVE_PER_CARD`, plus a self-calibration that can add a third correction. | `eis_utils.process_condition()` | Corrections can double-apply; the magnitude is not traceable to a measurement; the values are order-of-magnitude too small to fix a sample-level misalignment (see §1.1). |
| A4 | Inter-channel skew is inferred from **channel-index parity** (`uc_pos % 2 != seg_pos % 2`). | `process_condition()` | A guess about ADC core assignment, not a measurement. Wrong for any card whose channel map differs. |
| A5 | `estimate_dt_from_hf()` fits phase-vs-frequency at HF and attributes *all* residual phase to Δt. | `eis_utils` | Δt and cell inductance L are **first-order degenerate** (`e^{-jωΔt} ≈ 1 − jωΔt` behaves as `L = −Rs·Δt`). The estimator silently absorbs real physics into a timing constant. Also `arctan2` output is fitted without unwrapping. |
| A6 | Impedance uses the **H1 estimator** `Z = S_iv/S_ii`. | `process_condition()`, `Gold_Impedance_ETL.compute_impedance_welch()` | H1 is biased *low* by input noise: `E[Ĥ1] ≈ H·γ²` in the noisy-input limit. At γ² = 0.9 that is a systematic −10 % on |Z|, exactly in the low-coherence points you are trying to keep. |
| A7 | Tones are picked by **nearest Welch bin** (`argmin|f_b − t|`) from a Hann-windowed estimate. | `process_condition()` | Off-bin tones suffer scalloping/leakage: amplitude error up to −1.4 dB and phase error growing with off-bin fraction, unless the record is period-synchronous. |
| A8 | Sign of Z is fixed by **majority vote per card**, per run. | `process_condition()` | A wiring-polarity constant is being re-derived from data every run; one bad condition can flip a whole card. |
| A9 | `H_for_seg()` always evaluates `H(T)` at `T_FALLBACK`; the measured temperature channels are never used. | `eis_utils.H_for_seg()` | The documented temperature-compensated shunt calibration is inert. Copper TCR ≈ 0.39 %/K → a 10 K error is a 3.9 % error on every Rs. |
| A10 | No per-point uncertainty is produced; ECM fitting is unweighted. | `fit_ecm()` | Fit parameters have no confidence intervals; noisy LF points get the same leverage as clean HF ones. |
| A11 | `lin_kk_residual()` is a fixed 5-element Voigt fit, unregularised, unweighted, informational only. | `eis_utils` | Too coarse to certify data quality; cannot localise *where* in frequency the data fails. |
| A12 | Gold table is rebuilt with `DROP TABLE` + full overwrite; parameters are not recorded in the table. | `EIS_Analysis_Gold_Table.py` cell 4 | Not idempotent, not incremental, no provenance — a spectrum cannot be traced back to the parameters that produced it. |
| A13 | Manual exclusions live in plotting code (`EXCLUDE_SEGS_HEATMAP = [33]`). | `EIS_Pipeline.py` | Quality decisions are invisible to the data model. |

---

## 1. The core problem, quantified

### 1.1 Timing error budget

Phase error from a pure time offset is `φ = 2π·f·Δt`. Evaluated at `f_max = 3.8 kHz` and at `100 Hz`:

| Error source | Typical magnitude | φ @ 3.8 kHz | φ @ 100 Hz | Currently handled? |
| --- | --- | --- | --- | --- |
| Integer-sample start offset (1 sample @ 10 kHz) | 100 µs | **136.8°** | 3.6° | ❌ assumed zero |
| Card-internal channel/interleave offset | 50 µs | **68.4°** | 1.8° | ⚠️ parity guess |
| Free-running clock drift, 10 ppm over 120 s | up to 1.2 ms | **>4 wraps** | 43° | ❌ not modelled |
| Anti-alias filter group-delay mismatch | ~1 µs | 1.4° | 0.04° | ❌ |
| Analog front-end delay (INA/opamp chain) | ~120 ns | 0.16° | 0.004° | ✅ (`DT_PER_CARD`) |

**Reading of the table.** The only timing term the current pipeline corrects properly is the *smallest* one. A single-sample misalignment is three orders of magnitude larger than the −118 ns constant, and clock drift is unbounded in a 120 s record. Any nanosecond-level refinement is meaningless until the integer-sample and drift terms are demonstrably zero.

**Accuracy target.** To hold the imaginary-part error below 0.1° at `f_max`:

```
Δt_residual ≤ 0.1° / (360° · 3800 Hz) ≈ 73 ns
```

That is the number the Bronze tier must meet, and it must be *measured*, not assumed.

### 1.2 The asset that makes this solvable

Every card records its own copy of the cell voltage (`uc_XX_kN` in the Bronze schema). **The same physical signal is present on all cards simultaneously.** That converts inter-card synchronisation from an assumption into a direct measurement: cross-correlate `UC` on card *i* against `UC` on the reference card and read off the delay. No electrochemical assumption is involved, and the estimate is available for every recording, every condition, for free.

This is the single highest-value change in the whole plan.

---

## 2. Target repository layout

```
eis/
├── io/
│   ├── famos.py            # spec-complete FAMOS reader (header, |NT time, |CR scaling)
│   └── schema.py           # Delta schemas + provenance columns
├── sync/
│   ├── skew.py             # GCC-PHAT + cross-spectral phase-slope skew estimation
│   ├── drift.py            # block-wise delay regression → clock-rate offset
│   └── resample.py         # fractional-delay / rate-correction resampling
├── spectra/
│   ├── multisine.py        # period-synchronous DFT at designed tones
│   ├── welch_gated.py      # coherence-gated Welch (ensemble-level gating)
│   ├── estimators.py       # H1 / H2 / Hv (TLS) + γ²-based bias correction
│   └── uncertainty.py      # Bendat–Piersol σ(|Z|), σ(φ) propagation
├── correct/
│   ├── delay.py            # per-channel delay vector application
│   ├── polarity.py         # signed wiring map (config, not vote)
│   └── calibration.py      # H(T) with measured, spatially-mapped T
├── validate/
│   ├── linkk.py            # regularised Lin-KK + μ-criterion + residual bands
│   ├── stationarity.py     # split-record drift test
│   ├── linearity.py        # harmonic / non-excited-bin distortion test
│   └── invariants.py       # Σ(A_k/Z_k) vs cell-level reference
├── model/
│   ├── ecm.py              # weighted, bounded, multi-start ECM fitting + CIs
│   └── drt.py              # (optional) Tikhonov DRT
├── viz/
│   ├── plate.py            # heatmaps, geometry, RBF field
│   ├── dashboard.py        # linked heatmap ↔ spectrum
│   └── report.py           # per-measurement HTML/PDF report
└── jobs/                   # Databricks notebook wrappers (thin)
tests/
├── synthetic/              # generators with known Z, known Δt, known noise
└── test_*.py
```

Principle: **notebooks contain no algorithms.** Every numerical step lives in an importable, unit-tested module; notebooks are thin drivers over widgets. This is what makes the Gold-tier automation possible at all.

---

## 3. BRONZE TIER — Foundational Synchronisation and Calculation

### 3.1 Objective

Produce a *provably* time-aligned, physically-scaled multi-card dataset and a baseline impedance spectrum whose residual inter-channel timing error is measured and bounded below 100 ns.

### 3.2 Key implementation steps

#### B1 — Complete the FAMOS reader (`eis/io/famos.py`)

- Parse the full key set, not just `|CD`/`|CR`/`|CP`/`|CS`. Specifically recover:
  - **`|NT`** — absolute date/time of recording start. This is the first, coarse alignment evidence and is currently discarded (A2).
  - **`|CD`** — `dt` **and the x-axis offset/start value**, which may be non-zero per file.
  - **`|CR`** — transform flag, **scaling factor and offset**, and unit. Assert `factor == 1.0, offset == 0.0`; if not, apply them. Silently reading raw `float32` is only correct when the file happens to be unscaled.
  - **`|CG`/`|CP`** — component/packing description; use it to *verify* the derived `n_ch` and interleave stride rather than inferring from a regex on `|CR`.
- Cross-check every derived quantity: `n_samples · n_ch · 4 == data_bytes`; `len(ch_names) == n_ch`. Raise on mismatch instead of falling back to `dt = 1e-4, n_ch = 16` defaults.
- Return a typed `FamosFile` object carrying `t0_absolute`, `dt`, `channels`, `scaling`, and the raw memmap — never a bare dict.
- Keep the memmap-based lazy read (it is the right call for 1.2 M × 16 float32).

#### B2 — Measure inter-card skew from the shared UC channel (`eis/sync/skew.py`)

This replaces A1/A3 with a measurement.

```python
def estimate_skew(x_ref, x_test, fs, band=(20, 4000), nperseg=2**15):
    """Sub-sample delay of x_test relative to x_ref.

    Two independent estimators, cross-checked:
      1. GCC-PHAT  -> coarse, robust, unambiguous over ±N samples
      2. Cross-spectral phase slope -> sub-sample, high precision

    Returns (tau_seconds, ci_seconds, coherence_median).
    """
    # --- 1. GCC-PHAT: integer + parabolic-interpolated sub-sample peak
    X, Y = np.fft.rfft(x_ref), np.fft.rfft(x_test)
    R = X.conj() * Y
    R /= np.abs(R) + 1e-20                      # phase transform: whitens, sharpens peak
    cc = np.fft.irfft(R)
    k = int(np.argmax(np.abs(np.fft.fftshift(cc)))) - len(cc) // 2
    tau_coarse = k / fs

    # --- 2. Phase slope of the cross-spectrum, weighted by coherence
    f, Pxy = csd(x_ref, x_test, fs=fs, nperseg=nperseg)
    _, g2 = coherence(x_ref, x_test, fs=fs, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1]) & (g2 > 0.9)
    # de-rotate by the coarse estimate FIRST so unwrapping is safe
    phi = np.unwrap(np.angle(Pxy[m] * np.exp(1j * 2*np.pi * f[m] * tau_coarse)))
    w = g2[m] / (1 - g2[m] + 1e-9)              # Bendat-Piersol weighting
    slope, intercept = np.polyfit(2*np.pi*f[m], phi, 1, w=np.sqrt(w))
    return tau_coarse - slope, _slope_ci(...), np.median(g2[m])
```

Critical details:
- **De-rotate before unwrapping.** Fitting `arctan2` output directly (as `estimate_dt_from_hf` does) breaks the moment total phase exceeds ±π — which it does for anything above ~1 sample of skew.
- **Weight by `γ²/(1−γ²)`**, the correct inverse-variance weight for a phase estimate.
- **Require the intercept to be ≈ 0.** A non-zero intercept means the two channels differ by more than a delay (e.g. a filter mismatch) and the pure-delay model is inadequate — flag rather than silently fit.
- Run all card pairs, not just against a single reference, and check **consistency**: `τ_12 + τ_23 + τ_31 ≈ 0`. This closure residual is a free, powerful validity check.

#### B3 — Detect and correct clock drift (`eis/sync/drift.py`)

If the cards are not fed a shared sample clock, skew is a *function of time*.

- Split the record into K blocks (e.g. 10 s each), run B2 per block → `τ(t_k)`.
- Regress `τ` on `t`: intercept = static offset, **slope = fractional clock-rate error ε** (dimensionless, report in ppm).
- Gate: if `|ε| · T_record · f_max > 0.01 cycles`, a single constant delay correction is invalid and resampling is mandatory.
- Report `ε` per card in the Bronze metadata. A stable, repeatable `ε` across measurements is evidence of a hardware clocking topology issue worth fixing at source; a wandering `ε` indicates thermal drift.

#### B4 — Align in the time domain, before any FFT (`eis/sync/resample.py`)

- **Integer-sample** alignment by slicing (cheap, exact).
- **Fractional-sample** alignment by FFT phase ramp for whole-record correction, or a windowed-sinc / Farrow filter for streaming:
  ```python
  def fractional_delay(x, tau, fs):
      X = np.fft.rfft(x)
      f = np.fft.rfftfreq(len(x), 1/fs)
      return np.fft.irfft(X * np.exp(-1j * 2*np.pi * f * tau), n=len(x))
  ```
- **Rate correction** (only if B3 triggers): resample by `1/(1+ε)` with a polyphase resampler, then re-run B2 to confirm the residual drift is gone.
- **Why time-domain, not a post-hoc `Z *= exp(-j2πfΔt)`:** a phase rotation on the final `Z` fixes the phase but does *not* recover the coherence lost to misalignment inside each Welch window, does not fix tone detection (which runs on coherence), and cannot represent a time-varying delay at all. The current pipeline does exactly this post-hoc rotation, which is why its coherence gate has to be so permissive.

#### B5 — Build the per-channel delay vector honestly (`eis/correct/delay.py`)

Replace the parity heuristic (A4):
- Model total delay per channel as `τ_ch = τ_card + τ_channel_offset`, where `τ_channel_offset` is a **measured constant per (card, channel index)**.
- Obtain it from a dedicated **sync-calibration measurement**: feed one broadband signal (or the same multisine) into every input in parallel and measure each channel against a chosen reference. This produces a full delay table in one shot and is valid until the hardware changes.
- Store as versioned config (`config/delay_table_<hw_rev>.yaml`) with the measurement date and the estimated uncertainty per entry.
- Until that measurement exists, the pipeline must record `τ_channel_offset = unknown` and **propagate that as an uncertainty**, not substitute a guess.

#### B6 — Fix the polarity and calibration constants (A8, A9)

- Move sign conventions into an explicit per-(card, channel) polarity map derived once from the wiring, verified once against a reference instrument, and **asserted** at runtime rather than voted on.
- Make `H_for_seg(seg, T)` actually take `T`. Build a per-segment temperature by interpolating the sparse temperature sensors over the plate geometry (the same RBF machinery already used for heatmaps), with the constant fallback retained only as an explicitly-flagged degraded mode.

#### B7 — Baseline impedance and the Bronze data contract

- Baseline `Z(f)` on aligned data using the existing Welch CSD path — deliberately unchanged, so that Silver's improvements can be measured against it.
- Bronze Delta schema gains provenance and sync columns:

| Column | Purpose |
| --- | --- |
| `t0_absolute_us` | per-card recording start from `\|NT` |
| `skew_ns`, `skew_ci_ns` | measured per-card offset vs reference card + CI |
| `clock_ppm` | estimated per-card rate error |
| `skew_closure_ns` | τ12+τ23+τ31 consistency residual |
| `sync_method` | `phat+phaseslope` / `metadata_only` / `failed` |
| `pipeline_version`, `git_sha`, `param_hash` | reproducibility (fixes A12) |

- Write with `replaceWhere` on the partition, never `DROP TABLE`.

#### B8 — Tests

- **Synthetic round-trip:** generate two channels from a known `Z(f)`, impose a known delay (0.3, 1.7, 12.4 samples) and a known drift, run the full sync stack, assert recovery to < 20 ns and residual phase < 0.05° at `f_max`.
- **Closure test** on real data across all card triplets.
- **Null test:** two channels from the *same* card must return τ ≈ 0 ± CI.

### 3.3 Expected outcome

A Bronze table where time alignment is a **measured, reported, and bounded** quantity rather than an assumption.

- Every partition carries the skew, its confidence interval, the clock drift, and a pass/fail sync flag.
- Residual inter-card timing error demonstrably < 100 ns (target 73 ns → ≤ 0.1° at 3.8 kHz).
- Nyquist plots that no longer need a per-card magic constant to look physical; the HF end of the arc lands where it should without tuning.
- A hard quality gate: **a partition that fails sync closure never reaches Silver.**
- Baseline `Z(f)` per segment, reproducible from `(git_sha, param_hash)` alone.

---

## 4. SILVER TIER — High-Fidelity Signal Processing and Correction

### 4.1 Objective

Convert synchronised time series into impedance spectra with quantified, per-point uncertainty and an independent Kramers–Kronig certificate of validity.

### 4.2 Key implementation steps

#### S1 — Coherence-Gated Welch, done at the ensemble level (`eis/spectra/welch_gated.py`)

The current pipeline computes a Welch average over *all* windows and then discards frequency points whose final coherence is low. That is gating the output. The stronger form gates the **ensemble members**:

```python
def coherence_gated_csd(x, y, fs, nperseg, noverlap, gate):
    """Welch with per-window rejection.

    1. Split into L windows; compute per-window X_l(f), Y_l(f).
    2. Per-window quality metrics at each tone/band:
         - local SNR of x (excitation present?)
         - circular consistency of arg(X_l* Y_l) across l
         - amplitude z-score vs the ensemble median
    3. Reject windows that are outliers (load transient, arc, DC step).
    4. Average only retained windows -> Sxx, Syy, Sxy, gamma^2, n_d.
    5. Return the *retained count* n_d, which sets the uncertainty.
    """
```

Why it matters here specifically: a 120 s fuel-cell record is not stationary. A single load transient or a purge event contaminates the whole Welch average and cannot be recovered by output gating. Per-window rejection removes the contaminating ensemble members while keeping the rest.

Gating rules to implement:
- **Excitation-presence gate:** reject windows where the excitation amplitude at the designed tones is below a threshold (handles intermittent excitation — the same problem the current `select_best_uc()` 3-window max-RMS heuristic is working around).
- **Phase-consistency gate:** compute the circular variance of the per-window transfer-function phase; reject windows beyond `k·MAD`.
- **Stationarity gate:** reject windows whose DC level or RMS deviates from the record median beyond a threshold.
- Always report `n_d` (retained degrees of freedom) per frequency — every downstream uncertainty depends on it.

#### S2 — Period-synchronous DFT for designed multisines (`eis/spectra/multisine.py`)

If the excitation is a designed broadband multisine (which the tone structure strongly suggests), Welch is the wrong tool and costs accuracy (A7).

- Take the tone list from **configuration**, not from coherence-peak clustering. `detect_tones()` becomes a *verification* step ("are the designed tones where we expect them?"), not the source of truth.
- Choose the analysis window as an **integer number of excitation periods** (`N = M·fs/f_0`). Then a rectangular-window DFT at the tone bins is exact and leakage-free — no scalloping, no window-amplitude correction, no nearest-bin error.
- **Coherently average** the complex spectra across periods. The scatter across periods is a direct, assumption-free estimate of the measurement uncertainty at each tone.
- Keep the Welch path as a fallback for non-synchronous or unknown excitation; select automatically based on whether `fs/f_0` is rational and the record contains ≥ 4 full periods.

Expected gain: elimination of the leakage-induced amplitude/phase bias, and typically a large increase in effective SNR per tone because all excitation energy lands in exactly one bin.

#### S3 — Unbiased impedance estimators (`eis/spectra/estimators.py`)

Replace the bare H1 (A6):

| Estimator | Formula | Bias | Use when |
| --- | --- | --- | --- |
| H1 | `S_iv / S_ii` | biased **low** by noise on the current | output noise dominates |
| H2 | `S_vv / S_vi` | biased **high** by noise on the voltage | input noise dominates |
| Hv (TLS) | principal eigenvector of the 2×2 spectral matrix | unbiased for noise on both | **default** |

- Implement all three; report H1 and H2 as bracketing bounds and Hv as the estimate. The H2/H1 ratio is `1/γ²` — so the spread between them is a direct, interpretable noise diagnostic per frequency.
- Where the noise model is known, apply the `γ²`-based bias correction explicitly rather than relying on the coherence gate to hide the problem.

#### S4 — Per-point uncertainty (`eis/spectra/uncertainty.py`)

From Bendat & Piersol, with `n_d` retained windows from S1:

```
σ_|Z| / |Z|  =  sqrt( (1 - γ²) / (2 γ² n_d) )
σ_φ [rad]    =  sqrt( (1 - γ²) / (2 γ² n_d) )
```

- Attach `z_real_std`, `z_imag_std`, `gamma2`, `n_d` to **every row** of the Gold impedance table.
- These become the weights for Lin-KK (S6) and for ECM fitting (§5), and the error bars on every plot. This is what turns the pipeline from "a plot" into "a measurement".

#### S5 — Hardware artefact correction, in the right order

The corrections must be applied in a fixed, documented order, each traceable to a measurement:

| # | Correction | Source of the constant | Applied where |
| --- | --- | --- | --- |
| 1 | Integer + fractional sample alignment | measured (Bronze B2/B4) | time domain |
| 2 | Clock-rate resampling | measured (Bronze B3) | time domain |
| 3 | Per-channel delay `τ_channel_offset` | sync-calibration table (B5) | time domain |
| 4 | Anti-alias / digital filter phase response | **complex** `H_AA(f)` from hardware spec or calibration, divided out — *not* a scalar delay | frequency domain |
| 5 | Shunt transfer function `H(T)` | `curr.csv` + measured T (B6) | frequency domain, magnitude only |
| 6 | Polarity | wiring map (B6) | sign, config |
| 7 | Residual delay refinement | Lin-KK-optimal Δt (S6), **bounded to the B2 CI** | frequency domain |

Two points worth stating explicitly:

- **AA-filter mismatch is not a delay.** Near the corner frequency the group delay is frequency-dependent. Model it as a complex response and divide it out; approximating it as a constant Δt is exactly the kind of residual that Lin-KK will flag at HF.
- **Δt and inductance L are degenerate (A5).** To first order `Z·e^{-jωΔt} ≈ Z − jωΔt·Rs`, i.e. a delay is indistinguishable from an inductance `L = −Rs·Δt` over a limited band. Therefore:
  - Never estimate Δt from the cell's own HF phase alone. That estimator will happily absorb the real inductance of the segment current path.
  - Do use the structural constraint that **Δt is common to all segments on a card while L varies per segment** — a hierarchical fit (shared Δt per card, free L per segment) is weakly identifiable, *provided* L genuinely varies. Its common-mode component is still absorbed; report this limitation.
  - The clean solution is independent: the UC-vs-UC skew of Bronze B2 (no cell physics involved), plus a resistive-reference channel recorded alongside the cell. Recommend adding a precision shunt/resistor on one channel per card as a permanent hardware change — it makes Δt directly observable in every measurement.

#### S6 — Kramers–Kronig validation, properly (`eis/validate/linkk.py`)

Upgrade from the current fixed 5-element unweighted fit (A11) to the Schönleber/Ivers-Tiffée Lin-KK method:

- Basis: `Z_fit(f) = R_s + jωL + Σ_{k=1..M} R_k / (1 + jωτ_k)` with `τ_k` log-spaced over `[1/(2πf_max), 1/(2πf_min)]`.
- **Automatic order selection via the μ-criterion:** increase M until `μ = 1 − Σ|R_k^neg| / Σ|R_k^pos|` drops below ~0.85, i.e. stop just before the model begins over-fitting with oscillating negative resistances. This removes the arbitrary `M=5`.
- **Weighted** complex least squares using the S4 uncertainties (proportional weighting `1/|Z|²` as fallback).
- Output the **residual spectra** `Δ_re(f) = (Z_re − Ẑ_re)/|Z|` and `Δ_im(f)`, not a single scalar. The *shape* of the residual is the diagnostic:

| Residual signature | Diagnosis |
| --- | --- |
| Random, within ±1 %, both parts | Data is KK-compliant → causal, linear, stable |
| Systematic, monotonic in `Δ_im`, growing with f | **Uncorrected time delay** (the Voigt basis is minimum-phase and cannot represent `e^{-jωΔt}`) |
| Divergence at the low-frequency end | **Non-stationarity / drift** during the record |
| Both parts biased at a few isolated points | Leakage or a contaminated tone |

- Exploit signature #2: scan Δt to minimise the Lin-KK residual → a **KK-optimal Δt**, used only as a bounded refinement inside the Bronze-measured confidence interval (never as the primary estimate, because of the L-degeneracy).
- Emit a per-segment KK certificate: `kk_pass`, `kk_M`, `kk_mu`, `kk_max_residual_pct`, `kk_residual_shape_class`.

#### S7 — Independent validity checks (`eis/validate/`)

- **Stationarity:** split the record in half, compute `Z` on each, compare. A significant difference invalidates KK's stationarity premise regardless of what the KK residual says.
- **Linearity / distortion:** with an odd-random-phase multisine, some bins are deliberately left unexcited. Energy appearing at even harmonics and at unexcited odd bins separates *nonlinear distortion* from *noise* (Pintelon & Schoukens best-linear-approximation framework). Gate on THD. If the excitation is not currently designed this way, this is a high-value, low-cost change to the excitation recipe.
- **Physical invariant across segments:** the segments are electrically in parallel across a common cell voltage, so
  ```
  A_cell / Z_cell(f)  ≈  Σ_k  A_k / Z_k(f)
  ```
  Comparing the summed segment admittance against the reference potentiostat spectrum is a **whole-plate, frequency-resolved consistency check** — it validates the calibration chain, the polarity map, and the delay corrections all at once. The existing "Rs verification: parallel resistor formula vs Gamry" cell already does the DC-limit version of this by hand; promote it to an automated, frequency-resolved gate.
- **Segment-quality classification** replaces the scattered hardcoded exclusions (A13): every segment gets a machine-written status (`ok`, `low_coherence`, `kk_fail`, `nonstationary`, `channel_fault`) with the numeric reason, stored in the table. Nothing is silently dropped in plotting code.

### 4.3 Expected outcome

Silver produces impedance spectra that are **defensible as measurements**:

- Every `(segment, frequency)` row carries `Z`, `σ(Z)`, `γ²`, `n_d`, and its KK verdict.
- Systematic bias from the H1 estimator and from spectral leakage is removed, not gated around.
- Hardware artefacts are corrected from a documented, versioned calibration table, with the one genuinely non-identifiable parameter (Δt vs L) explicitly flagged rather than quietly resolved.
- A one-page automatic data-quality report per measurement: sync closure, KK pass rate, stationarity, linearity, plate-admittance-sum consistency.
- Failed data is *labelled*, not deleted — the reason is queryable in SQL.

---

## 5. GOLD TIER — Automation, Advanced Visualisation, and Physical Modelling

### 5.1 Objective

Turn the validated pipeline into a hands-off platform that ingests batches of measurements and returns physically-interpretable, spatially-resolved parameters with confidence intervals and interactive exploration.

### 5.2 Key implementation steps

#### G1 — Batch orchestration

- A single job takes a **manifest** (glob pattern, or a metadata-table query) and fans out over `(measurement, condition)` pairs. Databricks: one task with `for_each` over the discovered pairs, so partitions process in parallel rather than in the current sequential Python loop.
- **True incrementality on a content hash**, not just existence: reprocess when `(file_checksum, param_hash, git_sha)` changes. The current `condition in processed_pairs` check cannot detect that the *parameters* changed — which is why the analysis notebook has to `DROP TABLE`.
- Idempotent partition writes via `replaceWhere`; Delta time travel gives free result history.
- Per-partition status table: `queued / running / ok / failed(reason) / quarantined(quality)`. A batch never dies because one file is malformed.
- Structured logging and a run summary artefact per batch.

#### G2 — ECM fitting worth trusting (`eis/model/ecm.py`)

The current `fit_ecm` is a single-start, unweighted `least_squares` on a fixed 8-parameter model, returning no uncertainties. Upgrade:

- **Weighted objective** using the S4 uncertainties:
  ```
  χ²  =  Σ_f  [ (Z_re − Ẑ_re)² / σ_re²  +  (Z_im − Ẑ_im)² / σ_im² ]
  ```
  (Reduces to the standard proportional weighting when σ is unavailable.) This is the single biggest fit-quality improvement — it stops noisy LF points from dominating.
- **Parameterise in log space** for `Rs, Rct, Y0` (strictly positive, span decades) — dramatically better conditioning than box bounds on linear parameters.
- **Multi-start** from a physics-informed initial guess: `Rs` from the HF real-axis intercept, arc separation from the peak frequencies of `−Im(Z)`, CPE exponent from the arc depression angle. Keep the best χ².
- **Model selection, not a fixed circuit.** Fit a ladder and select by corrected AIC / BIC so the model complexity is justified by the data:

| Model | Elements | Physical reading |
| --- | --- | --- |
| M1 | `Rs + L` | ohmic + wiring inductance only |
| M2 | `Rs + L + (Rct‖CPE)` | single charge-transfer arc |
| M3 | `Rs + L + (Rct‖CPE)₁ + (Rct‖CPE)₂` | anode/contact + cathode ORR (current default) |
| M4 | `Rs + L + (Rct‖CPE) + Warburg / finite-length` | + mass transport, expected at high current density |

  Gas-transport limitation at 450 A is precisely where M3 will fail and M4 is needed — the model ladder makes that visible instead of forcing a bad two-arc fit.
- **Uncertainties:** parameter covariance from the Jacobian at the optimum (`J^T W J`)⁻¹, plus optional bootstrap over the retained Welch/period ensemble for parameters where the linearisation is poor. Report `Rs ± σ`, `Rct ± σ`, and the `Rs–Rct` correlation.
- **Fit-quality gate:** χ²_red, KK verdict, and parameter CI width all feed a single `fit_quality` flag. A parameter whose CI exceeds ~30 % of its value must not be painted on a heatmap as if it were known.
- **Optional: DRT** (`eis/model/drt.py`). Tikhonov-regularised distribution of relaxation times with GCV/L-curve regularisation choice. DRT is model-free and tells you *how many* processes are present and where their time constants sit — the principled way to choose between M2/M3/M4 rather than guessing. High value for a thesis: a DRT peak map across the plate shows *which* loss mechanism is spatially varying.

#### G3 — Interactive, linked visualisation (`eis/viz/dashboard.py`)

The existing plots are strong; what is missing is **linkage**.

- **Linked heatmap ↔ spectrum.** A single HTML page: plate heatmap on the left, Nyquist/Bode on the right. Clicking (or box-selecting) segments on the plate updates the spectrum panel; hovering a frequency on the Nyquist highlights the corresponding `|Z|(f)` heatmap. Implementation: Plotly `FigureWidget` + `on_click` for notebook use, or a self-contained HTML export with a small JS callback (`plotly_click` → `Plotly.restyle`) so the artefact is shareable without a running kernel.
- **Parameter selector** driving the heatmap: `Rs`, `Rct`, `R_total`, `CPE-n`, `|Z|@f`, `φ@f`, `χ²_red`, `γ²_median`, `τ_peak` (from DRT) — one code path, N maps.
- **Uncertainty is displayed, never hidden.** Error bars on Nyquist/Bode; on heatmaps, hatch or desaturate segments whose `fit_quality` is poor instead of dropping them silently (which is what `EXCLUDE_SEGS_HEATMAP = [33]` does today).
- **Frequency animation:** a slider over `f` driving the `|Z|(f)` and `φ(f)` plate maps — visually separates ohmic (spatially smooth, frequency-flat) from kinetic/transport (frequency-dependent, patterned along the flow field) losses. This is the most communicative single figure for the thesis.
- **Condition comparison:** small-multiple heatmaps across 45/60/150/450 A, plus difference maps (`ΔRs` between conditions) with a diverging colormap centred at zero.
- **Flow-field-aware layout:** keep the existing anode-in/cathode-in annotations and add mean-along-flow and mean-across-flow profile panels with confidence bands — the spatial trend along the channel is the physics of interest.
- Colour and accessibility: perceptually uniform sequential maps (`viridis`/`cividis`) for magnitudes, a symmetric diverging map for differences. Retire `jet`.

#### G4 — Automated reporting

- Per-measurement HTML report, generated by the batch job: metadata, sync certificate, KK pass rate, spectra gallery, parameter heatmaps, ECM table, flagged anomalies.
- Per-batch cross-measurement comparison (e.g. degradation across a test campaign): `Rs(t)`, `Rct(t)` per segment over successive measurements.
- Machine-readable export alongside the human-readable one (Parquet + a JSON summary), so downstream modelling never has to scrape a notebook.

#### G5 — Robustness and reproducibility

- Unit tests on synthetic data with known ground truth for every numerical module; regression tests pinning a reference measurement's `Rs` map within tolerance.
- Full provenance in every Gold row: `git_sha`, `param_hash`, `calibration_table_version`, `sync_method`.
- Config as versioned YAML with a schema, not module-level Python constants.

### 5.3 Expected outcome

A research platform rather than a notebook:

- Point it at a directory or a metadata query; it returns validated spectra, ECM parameters with confidence intervals, quality certificates, and a report — unattended, incrementally, in parallel.
- Physically-interpretable spatial maps: ohmic resistance (membrane hydration / contact), charge-transfer resistance (local kinetics), transport resistance (flooding / starvation along the channel), each with an explicit quality mask.
- A linked, interactive artefact where any anomalous region on the plate is one click away from its full spectrum and its fit residuals.
- Every published number traceable to a git SHA, a parameter hash, and a calibration table version.

---

## 6. Summary

| | Bronze | Silver | Gold |
| --- | --- | --- | --- |
| **Objective** | Provably time-aligned data + baseline `Z(f)` | Uncertainty-quantified, KK-certified spectra | Automated, interactive, physically-modelled platform |
| **Core technique** | GCC-PHAT + cross-spectral phase slope on the shared UC channel; fractional-delay resampling | Coherence-gated Welch / period-synchronous DFT; Hv estimator; regularised Lin-KK | Parallel batch orchestration; weighted ECM + model selection; linked Plotly dashboards |
| **Kills** | A1–A4, A8, A9 | A5–A7, A10, A11, A13 | A12 |
| **Key output** | `skew_ns ± CI`, `clock_ppm`, sync pass/fail | `σ(Z)`, `γ²`, `n_d`, KK certificate | ECM parameters ± CI, quality-masked heatmaps, reports |
| **Acceptance gate** | residual timing < 73 ns (≤ 0.1° @ 3.8 kHz); triplet closure < CI | ≥ 90 % of segments KK-pass at < 1 % residual; Σ(A/Z) matches reference within stated uncertainty | Batch of N files → zero manual steps; every number carries provenance |

### Recommended ordering of the first four work packages

1. **Complete the FAMOS reader** (B1) — everything else depends on trusting the bytes and recovering `|NT`.
2. **UC-vs-UC skew measurement** (B2) — highest value per unit effort in the entire plan; uses data you already have and immediately tells you how large the real problem is.
3. **Uncertainty propagation** (S4) — cheap, and it changes how every subsequent decision is judged.
4. **Sync-calibration measurement** (B5) — the one item that needs bench time; schedule it early because it gates the per-channel delay table.

### Two hardware/protocol recommendations

- **Add a resistive reference channel per card** (a precision shunt or resistor driven by the same excitation). It makes Δt directly observable in every measurement and permanently resolves the Δt-vs-inductance degeneracy that no amount of post-processing can break.
- **Design the excitation as an odd-random-phase multisine** with a record length that is an integer number of base periods. This enables leakage-free synchronous DFT *and* separates nonlinear distortion from noise — two significant accuracy gains for zero additional measurement time.
