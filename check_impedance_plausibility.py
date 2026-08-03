#!/usr/bin/env python3
"""Plausibility check for the local impedance of every segment.

Companion to ``check_current_density.py`` and deliberately identical in shape:
a summary header, a per-card bias table, a full per-segment table, then the
flag list for anything that was flagged.  That script gates the *input* to the
impedance calculation; this one gates the *output*.

Reads the pipeline's impedance table (and, when present, its per-segment table
of fitted circuit parameters), then screens each segment on the numbers a
reader of the thesis will actually quote: the ohmic resistance, the
charge-transfer resistance, how well-formed the arc is, and how many usable
frequency points it rests on.

Everything is reported as an area-specific resistance in **mOhm*cm^2**, which
is the form that compares between segments of different area and against
literature.  ``Rs`` is the high-frequency real-axis intercept and ``Rct`` the
diameter of the polarisation arc; both come from the fitted circuit when one is
available.  Without a fit they are read off the spectrum, and then ``Rs`` is
only trustworthy when the arc actually crosses ``Im(Z) = 0`` inside the measured
band.  When it does not - the usual case below about 4 kHz - the spectrum gives
an upper bound and nothing more, so recovering Rs itself needs a model.  The
checks below distinguish the two regimes instead of quoting a number the data
cannot support.

Checks
------
=========================  ====================================================
check                      what it catches
=========================  ====================================================
parallel_sum               segments recombined as parallel conductances vs a
                           cell-level reference - the whole-plate test
point_count                a spectrum resting on too few frequencies
magnitude                  Rs or Rct outside a physically plausible band
sign                       negative Rs or Rct, i.e. an unphysical fit
uniformity                 Rs/median outside the range a healthy plate shows
outlier                    robust median/MAD outlier against the population
neighbour                  a segment far from its spatial neighbours
arc_quality                a spectrum that is not a well-formed capacitive arc
kramers_kronig             a spectrum that is not causal, linear and stationary
fit_quality                a circuit fit with poor chi^2, unresolved
                           parameters, or an Rs inconsistent with the spectrum
                           it was fitted to
uncertainty                a segment whose own error bars are too wide to quote
card_bias                  one whole card offset from the others
=========================  ====================================================

Usage
-----
::

    # against the pipeline's own output
    python check_impedance_plausibility.py --from-out out/2611976

    # explicit tables, with a cell-level reference for the parallel sum
    python check_impedance_plausibility.py \\
        --impedance out/impedance.parquet --segments out/segments.parquet \\
        --rs-reference 62.0 --out out/plausibility --plots

    # generate a measurement, run the pipeline, then check it
    python check_impedance_plausibility.py --demo --plots
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eis.config import load_config  # noqa: E402

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_RANK = {OK: 0, WARN: 1, FAIL: 2, "SKIPPED": 0}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass
class Limits:
    """Every threshold the verdicts depend on, in one place."""

    # plausible area-specific resistances [mOhm*cm^2]
    rs_min: float = 10.0
    rs_max: float = 400.0
    rs_warn_min: float = 20.0
    rs_warn_max: float = 250.0

    rct_min: float = 1.0
    rct_max: float = 2000.0
    rct_warn_min: float = 5.0
    rct_warn_max: float = 1000.0

    # local-to-median ratio for a healthy plate
    uniformity_warn: tuple[float, float] = (0.75, 1.35)
    uniformity_fail: tuple[float, float] = (0.50, 2.00)

    # robust outlier score (|x - median| / (1.4826 * MAD))
    outlier_warn: float = 3.5
    outlier_fail: float = 6.0
    # floor on the robust scale, as a fraction of the median, so that a very
    # uniform plate does not flag every segment as an outlier
    outlier_mad_floor: float = 0.01

    # spatial neighbour deviation (relative)
    neighbour_warn: float = 0.20
    neighbour_fail: float = 0.45
    neighbour_radius_cm: float = 4.0

    # per-card median vs the plate median (relative)
    card_bias_warn: float = 0.08
    card_bias_fail: float = 0.20

    # arc quality score in [0, 1]
    arc_warn: float = 0.70
    arc_fail: float = 0.45

    # usable frequency points per segment
    points_warn: int = 12
    points_fail: int = 6
    # decades of frequency the spectrum should span
    decades_warn: float = 2.0

    # Kramers-Kronig max residual [%]
    kk_warn: float = 1.0
    kk_fail: float = 5.0

    # fitted Rs vs a measured real-axis intercept (relative)
    rs_fit_mismatch_warn: float = 0.05
    rs_fit_mismatch_fail: float = 0.15
    # when the arc does not close in band: min(Re(Z)) / fitted Rs
    rs_headroom_min: float = 0.95
    rs_headroom_warn: float = 1.60
    rs_headroom_fail: float = 2.20

    # circuit fit
    chi2_warn: float = 5.0
    chi2_fail: float = 50.0

    # own measurement uncertainty (median relative sigma on |Z|)
    sigma_warn: float = 0.05
    sigma_fail: float = 0.20

    # median coherence
    coherence_warn: float = 0.90
    coherence_fail: float = 0.70

    # parallel sum vs a cell-level reference (relative)
    sum_warn: float = 0.10
    sum_fail: float = 0.25


# ---------------------------------------------------------------------------
# Per-segment record
# ---------------------------------------------------------------------------

@dataclass
class SegmentCheck:
    segment: int
    card: int

    n_points: int = 0
    f_min_hz: float = np.nan
    f_max_hz: float = np.nan
    decades: float = np.nan

    rs_mohm_cm2: float = np.nan
    re_min_mohm_cm2: float = np.nan
    rs_measured_mohm_cm2: float = np.nan     # only when the arc crosses Im = 0
    rct_mohm_cm2: float = np.nan
    rp_mohm_cm2: float = np.nan
    rs_source: str = "spectrum"

    arc_quality: float = np.nan
    q_capacitive: float = np.nan
    q_closure: float = np.nan
    q_smooth: float = np.nan

    kk_max_residual_pct: float = np.nan
    kk_shape: str = ""
    chi2_reduced: float = np.nan
    ecm_model: str = ""
    ecm_unresolved: str = ""

    median_coherence: float = np.nan
    median_sigma_rel: float = np.nan

    uniformity: float = np.nan
    outlier_score: float = np.nan
    outlier_score_rct: float = np.nan
    neighbour_rel: float = np.nan

    verdict: str = OK
    flags: list[str] = field(default_factory=list)

    def raise_to(self, level: str, message: str) -> None:
        self.flags.append(f"{level}:{message}")
        if _RANK[level] > _RANK[self.verdict]:
            self.verdict = level


# ---------------------------------------------------------------------------
# Spectrum-derived quantities
# ---------------------------------------------------------------------------

def arc_metrics(f: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    """Describe how well the spectrum forms a capacitive arc.

    Returns ``q_capacitive``, ``q_closure``, ``q_smooth`` and their combination
    ``arc_quality``, all in [0, 1].  These are shape descriptors, not fits: a
    spectrum can fit a circuit and still be nonsense, and this catches the case
    where the arc never forms at all.
    """
    order = np.argsort(f)
    f, Z = np.asarray(f, float)[order], np.asarray(Z, complex)[order]
    neg_im = -Z.imag
    out = {
        "q_capacitive": np.nan, "q_closure": np.nan,
        "q_smooth": np.nan, "arc_quality": np.nan,
    }
    if len(f) < 4:
        return out

    peak = int(np.argmax(neg_im))
    span = float(neg_im[peak])

    # 1. Capacitive: below the peak frequency a fuel-cell arc sits under the
    #    real axis.  Points above the peak may legitimately turn inductive.
    below = neg_im[: peak + 1]
    out["q_capacitive"] = (
        float(np.mean(below > 0)) if below.size else 0.0
    )

    # 2. Closure: a complete arc comes back towards the real axis at the
    #    low-frequency end.  A spectrum truncated before the arc closes scores
    #    near zero - the Rct read off it is a lower bound, not a measurement.
    if span > 0:
        out["q_closure"] = float(np.clip((span - neg_im[0]) / span, 0.0, 1.0))
    else:
        out["q_closure"] = 0.0

    # 3. Smoothness: scatter about a locally smooth trend, in units of the arc
    #    height.  Catches a spectrum that is technically arc-shaped but noisy.
    if len(neg_im) >= 5 and span > 0:
        smoothed = np.convolve(neg_im, np.ones(3) / 3.0, mode="same")
        smoothed[0], smoothed[-1] = neg_im[0], neg_im[-1]
        scatter = float(np.median(np.abs(neg_im - smoothed))) / span
        out["q_smooth"] = float(np.clip(1.0 - 6.0 * scatter, 0.0, 1.0))
    else:
        out["q_smooth"] = 0.0

    parts = [out["q_capacitive"], out["q_closure"], out["q_smooth"]]
    out["arc_quality"] = float(np.mean(parts))
    return out


def rs_rct_from_spectrum(
    f: np.ndarray, Z: np.ndarray, n_hf: int = 6
) -> tuple[float, float]:
    """Rs from the high-frequency real-axis intercept, Rct as the arc diameter.

    Both in Ohm.

    Returns ``(rs, rct, rs_is_measured)``.  ``rs_is_measured`` says whether the
    arc actually crossed ``Im(Z) = 0`` inside the measured band, in which case
    ``rs`` is an interpolated measurement.  Otherwise it is only an upper
    bound - see the comment below - and downstream checks are weakened
    accordingly rather than pretending to a precision the data does not have.
    """
    order = np.argsort(f)
    f, Z = np.asarray(f, float)[order], np.asarray(Z, complex)[order]
    if len(f) < 3:
        return float("nan"), float("nan")

    take = min(max(n_hf, 3), len(f))
    re_hf, im_hf = Z.real[-take:], Z.imag[-take:]

    rs, measured = float("nan"), False
    sign_change = np.where(np.diff(np.sign(im_hf)) != 0)[0]
    if sign_change.size:                      # the arc crosses the real axis
        i = int(sign_change[-1])
        y0, y1 = im_hf[i], im_hf[i + 1]
        if y1 != y0:
            t = -y0 / (y1 - y0)
            rs = float(re_hf[i] + t * (re_hf[i + 1] - re_hf[i]))
            measured = True
    if not measured:
        # No crossing in band: the arc is still open at the top frequency, so
        # there is nothing to interpolate to.  Extrapolating a straight line in
        # the Re-Im plane is tempting and wrong - the arc is curved, and over a
        # realistic span it undershoots the true Rs by tens of percent.  The
        # honest statement is a bound: Rs cannot exceed the smallest real part
        # actually measured.  Recovering the value itself needs a model.
        rs = float(np.min(Z.real))

    n_lf = max(2, len(f) // 5)
    r_total = float(np.median(Z.real[:n_lf]))
    return rs, max(r_total - rs, 0.0), measured


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def resolve_inputs(args) -> tuple[Path, Path | None]:
    """Locate the impedance table, and the segment table when it exists."""
    if args.impedance:
        return Path(args.impedance), (Path(args.segments) if args.segments else None)
    if args.from_out:
        base = Path(args.from_out)
        for suffix in (".parquet", ".csv"):
            impedance = base / f"impedance{suffix}"
            if impedance.exists():
                segments = base / f"segments{suffix}"
                return impedance, (segments if segments.exists() else None)
        raise FileNotFoundError(f"no impedance table under {base}")
    raise SystemExit(
        "error: give --impedance, --from-out or --demo (see --help)"
    )


def collect_segments(
    impedance: pd.DataFrame, segments: pd.DataFrame | None,
    condition: str, area_cm2: float,
) -> dict[int, SegmentCheck]:
    """Build one record per segment for a single condition."""
    frame = impedance[impedance["condition"] == condition]
    asr = area_cm2 * 1e3                       # Ohm -> mOhm*cm^2

    per_segment = (
        segments[segments["condition"] == condition].set_index("segment")
        if segments is not None and "condition" in segments.columns
        else None
    )

    checks: dict[int, SegmentCheck] = {}
    for segment, group in frame.groupby("segment"):
        group = group.sort_values("frequency_hz")
        f = group["frequency_hz"].to_numpy(float)
        Z = (
            group["z_real_ohm"].to_numpy(float)
            + 1j * group["z_imag_ohm"].to_numpy(float)
        )
        card = int(group["card"].iloc[0]) if "card" in group.columns else 0
        check = SegmentCheck(segment=int(segment), card=card, n_points=len(f))
        if len(f):
            check.f_min_hz, check.f_max_hz = float(f.min()), float(f.max())
            if check.f_min_hz > 0:
                check.decades = float(np.log10(check.f_max_hz / check.f_min_hz))

        for key, value in arc_metrics(f, Z).items():
            setattr(check, key, value)

        if "coherence" in group.columns:
            check.median_coherence = float(np.median(group["coherence"]))
        if "sigma_rel" in group.columns:
            check.median_sigma_rel = float(np.median(group["sigma_rel"]))

        # Prefer the fitted circuit; fall back to reading the spectrum.
        rs_ohm, rct_ohm, rs_measured = rs_rct_from_spectrum(f, Z)
        check.re_min_mohm_cm2 = float(np.min(Z.real)) * asr if len(Z) else np.nan
        if rs_measured:
            check.rs_measured_mohm_cm2 = rs_ohm * asr
        check.rs_mohm_cm2 = rs_ohm * asr
        check.rs_source = "spectrum" if rs_measured else "spectrum_bound"
        check.rct_mohm_cm2 = rct_ohm * asr
        check.rs_source = "spectrum"

        if per_segment is not None and int(segment) in per_segment.index:
            row = per_segment.loc[int(segment)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for column, attribute in (
                ("kk_max_residual_pct", "kk_max_residual_pct"),
                ("kk_shape", "kk_shape"),
                ("ecm_chi2_reduced", "chi2_reduced"),
                ("ecm_model", "ecm_model"),
                ("ecm_poorly_determined", "ecm_unresolved"),
            ):
                if column in row.index and pd.notna(row[column]):
                    setattr(check, attribute, row[column])
            # The pipeline already writes these in mOhm*cm^2.
            if "ecm_Rs_mohm_cm2" in row.index and pd.notna(row["ecm_Rs_mohm_cm2"]):
                check.rs_mohm_cm2 = float(row["ecm_Rs_mohm_cm2"])
                check.rs_source = "ecm"
            if "ecm_rp_mohm_cm2" in row.index and pd.notna(row["ecm_rp_mohm_cm2"]):
                check.rp_mohm_cm2 = float(row["ecm_rp_mohm_cm2"])
                check.rct_mohm_cm2 = float(row["ecm_rp_mohm_cm2"])
            elif "ecm_Rct1_mohm_cm2" in row.index and pd.notna(row["ecm_Rct1_mohm_cm2"]):
                check.rct_mohm_cm2 = float(row["ecm_Rct1_mohm_cm2"])

        if not np.isfinite(check.rp_mohm_cm2):
            check.rp_mohm_cm2 = check.rct_mohm_cm2
        checks[int(segment)] = check

    return checks


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def apply_checks(
    checks: dict[int, SegmentCheck], cfg, limits: Limits,
    rs_reference_mohm_cm2: float | None,
) -> dict[str, object]:
    """Run every plausibility test and return the plate-level summary."""
    values = np.array(
        [c.rs_mohm_cm2 for c in checks.values() if np.isfinite(c.rs_mohm_cm2)]
    )
    summary: dict[str, object] = {}
    plate_flags: list[tuple[str, str]] = []
    if values.size == 0:
        return {"note": "no segment produced a finite Rs"}

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    mad_effective = max(mad, limits.outlier_mad_floor * abs(median))

    rct_values = np.array(
        [c.rct_mohm_cm2 for c in checks.values() if np.isfinite(c.rct_mohm_cm2)]
    )
    rct_median = float(np.median(rct_values)) if rct_values.size else np.nan
    rct_mad = (
        max(
            float(np.median(np.abs(rct_values - rct_median))) * 1.4826,
            limits.outlier_mad_floor * abs(rct_median),
        )
        if rct_values.size else np.nan
    )

    summary |= {
        "n_segments": len(values),
        "rs_median": median, "rs_mean": float(np.mean(values)),
        "rs_std": float(np.std(values)), "rs_mad": mad,
        "rs_mad_effective": mad_effective,
        "rs_min": float(values.min()), "rs_max": float(values.max()),
        "rct_median": rct_median,
        "rct_mean": float(np.mean(rct_values)) if rct_values.size else np.nan,
        "rct_std": float(np.std(rct_values)) if rct_values.size else np.nan,
        "rct_min": float(rct_values.min()) if rct_values.size else np.nan,
        "rct_max": float(rct_values.max()) if rct_values.size else np.nan,
    }

    # --- 1. per segment, independent of the population ---------------------
    for check in checks.values():
        # point count and span
        if check.n_points < limits.points_fail:
            check.raise_to(
                FAIL, f"only {check.n_points} usable frequency points"
            )
        elif check.n_points < limits.points_warn:
            check.raise_to(WARN, f"{check.n_points} usable frequency points")
        if np.isfinite(check.decades) and check.decades < limits.decades_warn:
            check.raise_to(
                WARN, f"spectrum spans {check.decades:.1f} decades "
                      f"({check.f_min_hz:.1f}-{check.f_max_hz:.0f} Hz)"
            )

        # sign and magnitude
        if not np.isfinite(check.rs_mohm_cm2):
            check.raise_to(FAIL, "no Rs could be determined")
        elif check.rs_mohm_cm2 <= 0:
            check.raise_to(
                FAIL, f"Rs = {check.rs_mohm_cm2:.1f} mOhm*cm2 is not physical"
            )
        elif not (limits.rs_min <= check.rs_mohm_cm2 <= limits.rs_max):
            check.raise_to(
                FAIL, f"Rs = {check.rs_mohm_cm2:.1f} mOhm*cm2 outside "
                      f"[{limits.rs_min}, {limits.rs_max}]"
            )
        elif not (limits.rs_warn_min <= check.rs_mohm_cm2 <= limits.rs_warn_max):
            check.raise_to(
                WARN, f"Rs = {check.rs_mohm_cm2:.1f} mOhm*cm2 near the "
                      f"plausible edge"
            )

        if np.isfinite(check.rct_mohm_cm2):
            if check.rct_mohm_cm2 < 0:
                check.raise_to(
                    FAIL, f"Rct = {check.rct_mohm_cm2:.1f} mOhm*cm2 is negative"
                )
            elif not (limits.rct_min <= check.rct_mohm_cm2 <= limits.rct_max):
                check.raise_to(
                    FAIL, f"Rct = {check.rct_mohm_cm2:.1f} mOhm*cm2 outside "
                          f"[{limits.rct_min}, {limits.rct_max}]"
                )
            elif not (limits.rct_warn_min <= check.rct_mohm_cm2 <= limits.rct_warn_max):
                check.raise_to(
                    WARN, f"Rct = {check.rct_mohm_cm2:.1f} mOhm*cm2 near the "
                          f"plausible edge"
                )

        # arc shape
        if np.isfinite(check.arc_quality):
            if check.arc_quality < limits.arc_fail:
                check.raise_to(
                    FAIL, f"arc quality {check.arc_quality:.2f} "
                          f"(capacitive {check.q_capacitive:.2f}, "
                          f"closure {check.q_closure:.2f}, "
                          f"smooth {check.q_smooth:.2f})"
                )
            elif check.arc_quality < limits.arc_warn:
                check.raise_to(
                    WARN, f"arc quality {check.arc_quality:.2f} "
                          f"(capacitive {check.q_capacitive:.2f}, "
                          f"closure {check.q_closure:.2f}, "
                          f"smooth {check.q_smooth:.2f})"
                )
            if check.q_closure < 0.15:
                check.raise_to(
                    WARN, "arc does not close at the low-frequency end: Rct is a "
                          "lower bound, not a measurement"
                )

        # Kramers-Kronig
        if np.isfinite(check.kk_max_residual_pct):
            if check.kk_max_residual_pct > limits.kk_fail:
                check.raise_to(
                    FAIL, f"KK residual {check.kk_max_residual_pct:.1f}%"
                          + (f" ({check.kk_shape})" if check.kk_shape else "")
                )
            elif check.kk_max_residual_pct > limits.kk_warn:
                check.raise_to(
                    WARN, f"KK residual {check.kk_max_residual_pct:.1f}%"
                          + (f" ({check.kk_shape})" if check.kk_shape else "")
                )
            if check.kk_shape == "delay_like":
                check.raise_to(
                    WARN, "KK residual has the signature of an uncorrected time "
                          "delay - re-check the synchronisation for this card"
                )
            elif check.kk_shape == "lf_divergent":
                check.raise_to(
                    WARN, "KK residual diverges at low frequency - the cell was "
                          "probably not stationary during the record"
                )

        # Is the fitted Rs consistent with the data it came from?  Two
        # regimes, because what the spectrum can say depends on whether the
        # arc closed inside the measured band.
        if check.rs_source == "ecm" and np.isfinite(check.rs_mohm_cm2):
            if np.isfinite(check.rs_measured_mohm_cm2):
                # The arc crossed Im = 0, so there is a real measurement to
                # compare against.
                mismatch = (
                    abs(check.rs_mohm_cm2 - check.rs_measured_mohm_cm2)
                    / max(check.rs_measured_mohm_cm2, 1e-12)
                )
                if mismatch > limits.rs_fit_mismatch_fail:
                    check.raise_to(
                        FAIL, f"fitted Rs {check.rs_mohm_cm2:.1f} disagrees with the "
                              f"measured intercept {check.rs_measured_mohm_cm2:.1f} "
                              f"mOhm*cm2 by {mismatch:.0%} - the circuit does not "
                              f"describe this spectrum"
                    )
                elif mismatch > limits.rs_fit_mismatch_warn:
                    check.raise_to(
                        WARN, f"fitted Rs is {mismatch:.0%} from the measured "
                              f"intercept ({check.rs_measured_mohm_cm2:.1f} mOhm*cm2)"
                    )
            elif np.isfinite(check.re_min_mohm_cm2) and check.rs_mohm_cm2 > 0:
                # The arc never closed, so only a bound is available: Rs must
                # sit just below the smallest real part measured.  A fitted Rs
                # far below it means the fit is extrapolating a long way past
                # the data; one above it is outright unphysical.
                headroom = check.re_min_mohm_cm2 / check.rs_mohm_cm2
                if headroom < limits.rs_headroom_min:
                    check.raise_to(
                        FAIL, f"fitted Rs {check.rs_mohm_cm2:.1f} exceeds the "
                              f"smallest measured Re(Z) "
                              f"({check.re_min_mohm_cm2:.1f} mOhm*cm2)"
                    )
                elif headroom > limits.rs_headroom_fail:
                    check.raise_to(
                        FAIL, f"fitted Rs {check.rs_mohm_cm2:.1f} sits "
                              f"{headroom:.1f}x below the smallest measured Re(Z) "
                              f"({check.re_min_mohm_cm2:.1f} mOhm*cm2) - the arc "
                              f"never closes in band and the fit is extrapolating "
                              f"far past the data"
                    )
                elif headroom > limits.rs_headroom_warn:
                    check.raise_to(
                        WARN, f"fitted Rs is {headroom:.2f}x below the smallest "
                              f"measured Re(Z); the arc does not close in band, so "
                              f"Rs rests on the model rather than on a measured "
                              f"intercept"
                    )

        # circuit fit
        if np.isfinite(check.chi2_reduced):
            if check.chi2_reduced > limits.chi2_fail:
                check.raise_to(
                    FAIL, f"circuit fit chi2_red = {check.chi2_reduced:.1f}"
                )
            elif check.chi2_reduced > limits.chi2_warn:
                check.raise_to(
                    WARN, f"circuit fit chi2_red = {check.chi2_reduced:.1f}"
                )
        # Only the parameters that get quoted matter here.  A stray inductance
        # is expected to be undetermined below a few kHz and flagging it just
        # buries the segments with a real problem.
        if isinstance(check.ecm_unresolved, str) and check.ecm_unresolved.strip():
            quoted = [
                name for name in check.ecm_unresolved.split(",")
                if name.strip().startswith(("Rs", "Rct", "Aw"))
            ]
            if quoted:
                check.raise_to(
                    WARN, f"circuit parameters not resolved by the data: "
                          f"{', '.join(quoted)}"
                )

        # the segment's own error bars
        if np.isfinite(check.median_sigma_rel):
            if check.median_sigma_rel > limits.sigma_fail:
                check.raise_to(
                    FAIL, f"median relative uncertainty {check.median_sigma_rel:.1%} "
                          f"on |Z| - too wide to quote"
                )
            elif check.median_sigma_rel > limits.sigma_warn:
                check.raise_to(
                    WARN, f"median relative uncertainty {check.median_sigma_rel:.1%} "
                          f"on |Z|"
                )
        if np.isfinite(check.median_coherence):
            if check.median_coherence < limits.coherence_fail:
                check.raise_to(
                    FAIL, f"median coherence {check.median_coherence:.2f}"
                )
            elif check.median_coherence < limits.coherence_warn:
                check.raise_to(
                    WARN, f"median coherence {check.median_coherence:.2f}"
                )

    # --- 2. population-relative --------------------------------------------
    for check in checks.values():
        if not np.isfinite(check.rs_mohm_cm2):
            continue
        check.uniformity = check.rs_mohm_cm2 / median if median else np.nan
        check.outlier_score = abs(check.rs_mohm_cm2 - median) / mad_effective
        if np.isfinite(check.rct_mohm_cm2) and np.isfinite(rct_mad) and rct_mad > 0:
            check.outlier_score_rct = abs(check.rct_mohm_cm2 - rct_median) / rct_mad

        lo_f, hi_f = limits.uniformity_fail
        lo_w, hi_w = limits.uniformity_warn
        if not (lo_f <= check.uniformity <= hi_f):
            check.raise_to(
                FAIL, f"Rs/median = {check.uniformity:.2f}, outside [{lo_f}, {hi_f}]"
            )
        elif not (lo_w <= check.uniformity <= hi_w):
            check.raise_to(WARN, f"Rs/median = {check.uniformity:.2f}")

        if check.outlier_score > limits.outlier_fail:
            check.raise_to(
                FAIL, f"Rs is a robust outlier, {check.outlier_score:.1f} MAD "
                      f"from the median"
            )
        elif check.outlier_score > limits.outlier_warn:
            check.raise_to(
                WARN, f"Rs is {check.outlier_score:.1f} MAD from the median"
            )

        if (
            np.isfinite(check.outlier_score_rct)
            and check.outlier_score_rct > limits.outlier_fail
        ):
            check.raise_to(
                WARN, f"Rct is {check.outlier_score_rct:.1f} MAD from the median "
                      f"({check.rct_mohm_cm2:.1f} vs {rct_median:.1f} mOhm*cm2)"
            )

    # --- 3. spatial neighbours ---------------------------------------------
    coords = cfg.geometry.segment_coords
    if coords:
        for check in checks.values():
            if check.segment not in coords or not np.isfinite(check.rs_mohm_cm2):
                continue
            x, y = coords[check.segment][:2]
            neighbours = [
                other.rs_mohm_cm2 for other in checks.values()
                if other.segment != check.segment
                and other.segment in coords
                and np.isfinite(other.rs_mohm_cm2)
                and np.hypot(coords[other.segment][0] - x, coords[other.segment][1] - y)
                <= limits.neighbour_radius_cm
            ]
            if len(neighbours) < 2:
                continue
            local = float(np.median(neighbours))
            if abs(local) < 1e-12:
                continue
            check.neighbour_rel = abs(check.rs_mohm_cm2 - local) / abs(local)
            if check.neighbour_rel > limits.neighbour_fail:
                check.raise_to(
                    FAIL, f"Rs is {check.neighbour_rel:.0%} away from its neighbours "
                          f"({local:.1f} mOhm*cm2)"
                )
            elif check.neighbour_rel > limits.neighbour_warn:
                check.raise_to(
                    WARN, f"Rs is {check.neighbour_rel:.0%} away from its neighbours"
                )
    else:
        summary["neighbour_check"] = (
            "skipped: no segment_coords configured, so there is no spatial layout "
            "to compare against"
        )

    # --- 4. per-card bias ---------------------------------------------------
    # A real plate has a spatial gradient in Rs, and cards usually map to
    # contiguous regions of it, so comparing a card's raw median against the
    # plate median flags physics as if it were a fault.  What distinguishes a
    # card-level calibration or timing error is that it produces a *step* at
    # the card boundary, while a gradient runs continuously through it.  So the
    # test is applied to the residual left after a smooth trend across the
    # plate has been removed.
    positions, resistances, keys = [], [], []
    coords_for_trend = cfg.geometry.segment_coords
    for check in checks.values():
        if not np.isfinite(check.rs_mohm_cm2):
            continue
        if coords_for_trend and check.segment in coords_for_trend:
            x, y = coords_for_trend[check.segment][:2]
        else:                       # no geometry: segment index is the best proxy
            x, y = float(check.segment), 0.0
        positions.append((x, y))
        resistances.append(check.rs_mohm_cm2)
        keys.append(check.segment)

    detrended: dict[int, float] = {}
    trend_removed = False
    if len(resistances) >= 6:
        design = np.column_stack([
            np.ones(len(positions)),
            [p[0] for p in positions],
            [p[1] for p in positions],
        ])
        try:
            coefficients, *_ = np.linalg.lstsq(design, np.asarray(resistances), rcond=None)
            residuals = np.asarray(resistances) - design @ coefficients
            detrended = dict(zip(keys, residuals))
            trend_removed = True
        except np.linalg.LinAlgError:
            pass
    if not trend_removed:
        detrended = {k: v - median for k, v in zip(keys, resistances)}
    summary["card_bias_detrended"] = trend_removed

    card_medians: dict[int, float] = {}
    card_verdicts: dict[int, str] = {}
    for card in sorted({c.card for c in checks.values()}):
        card_values = [
            c.rs_mohm_cm2 for c in checks.values()
            if c.card == card and np.isfinite(c.rs_mohm_cm2)
        ]
        card_residuals = [
            detrended[c.segment] for c in checks.values()
            if c.card == card and c.segment in detrended
        ]
        if not card_values or not card_residuals:
            continue
        card_median = float(np.median(card_values))
        card_medians[card] = card_median
        step = float(np.median(card_residuals))
        deviation = abs(step) / abs(median) if median else np.nan
        if deviation > limits.card_bias_fail:
            level, note = FAIL, "a card-level timing or calibration error, not physics"
        elif deviation > limits.card_bias_warn:
            level, note = WARN, "check this card's delay table and calibration"
        else:
            card_verdicts[card] = OK
            continue
        card_verdicts[card] = level
        plate_flags.append((
            level,
            f"card {card} sits {deviation:.0%} of the plate median Rs away from "
            f"the spatial trend - a step at the card boundary, not a gradient "
            f"({note})",
        ))
        # Context for its segments, not proof that any one of them is broken.
        for check in checks.values():
            if check.card == card:
                check.raise_to(
                    WARN, f"card {card} steps {deviation:.0%} off the plate trend"
                )
    summary["card_medians"] = card_medians
    summary["card_verdicts"] = card_verdicts

    # --- 5. parallel sum ----------------------------------------------------
    # The segments share one cell voltage, so they combine as parallel
    # conductances.  With every segment the same area this reduces to the
    # harmonic mean of the segment ASRs, which is what a cell-level instrument
    # should measure over the same area.
    conductances = [
        1.0 / c.rs_mohm_cm2 for c in checks.values()
        if np.isfinite(c.rs_mohm_cm2) and c.rs_mohm_cm2 > 0
    ]
    if conductances:
        plate_rs = len(conductances) / float(np.sum(conductances))
        summary["rs_parallel_mohm_cm2"] = plate_rs
        summary["rs_reference_mohm_cm2"] = rs_reference_mohm_cm2
        if rs_reference_mohm_cm2:
            ratio = plate_rs / rs_reference_mohm_cm2
            deviation = abs(ratio - 1.0)
            summary["parallel_sum_ratio"] = ratio
            if deviation > limits.sum_fail:
                verdict = FAIL
            elif deviation > limits.sum_warn:
                verdict = WARN
            else:
                verdict = OK
            summary["parallel_sum_verdict"] = verdict
            if verdict != OK:
                plate_flags.append((
                    verdict,
                    f"segments recombined in parallel give {plate_rs:.1f} "
                    f"mOhm*cm2 against a {rs_reference_mohm_cm2:.1f} mOhm*cm2 "
                    f"cell-level reference ({ratio:.3f}x). Either the shunt "
                    f"calibration carries a scale error or segments are missing "
                    f"from the sum - check the segment count first.",
                ))
        else:
            summary["parallel_sum_verdict"] = "SKIPPED"
    else:
        summary["parallel_sum_verdict"] = "SKIPPED"

    summary["plate_flags"] = plate_flags
    summary["plate_verdict"] = max(
        [level for level, _ in plate_flags] or [OK], key=lambda v: _RANK[v]
    )
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def to_frame(checks: dict[int, SegmentCheck]) -> pd.DataFrame:
    rows = []
    for check in sorted(checks.values(), key=lambda c: c.segment):
        rows.append({
            "segment": check.segment,
            "card": check.card,
            "verdict": check.verdict,
            "rs_mOhm_cm2": check.rs_mohm_cm2,
            "re_min_mOhm_cm2": check.re_min_mohm_cm2,
            "rs_measured_mOhm_cm2": check.rs_measured_mohm_cm2,
            "rct_mOhm_cm2": check.rct_mohm_cm2,
            "rp_mOhm_cm2": check.rp_mohm_cm2,
            "rs_source": check.rs_source,
            "uniformity_rs_over_median": check.uniformity,
            "outlier_MAD": check.outlier_score,
            "outlier_MAD_rct": check.outlier_score_rct,
            "neighbour_rel": check.neighbour_rel,
            "arc_quality": check.arc_quality,
            "q_capacitive": check.q_capacitive,
            "q_closure": check.q_closure,
            "q_smooth": check.q_smooth,
            "n_points": check.n_points,
            "f_min_hz": check.f_min_hz,
            "f_max_hz": check.f_max_hz,
            "decades": check.decades,
            "kk_max_residual_pct": check.kk_max_residual_pct,
            "kk_shape": check.kk_shape,
            "ecm_model": check.ecm_model,
            "chi2_reduced": check.chi2_reduced,
            "median_coherence": check.median_coherence,
            "median_sigma_rel": check.median_sigma_rel,
            "flags": "; ".join(check.flags),
        })
    return pd.DataFrame(rows)


def print_report(
    frame: pd.DataFrame, summary: dict[str, object],
    condition: str, measurement_id: str,
) -> None:
    line = "=" * 108
    print(f"\n{line}")
    print(f"  LOCAL IMPEDANCE PLAUSIBILITY - {measurement_id} / {condition}")
    print(line)

    if "n_segments" not in summary:
        print(f"  {summary.get('note', 'nothing to report')}")
        return

    print(f"  segments checked : {summary['n_segments']}")
    print(f"  Rs  median       : {summary['rs_median']:.2f} mOhm*cm2")
    print(f"  Rs  mean +/- std : {summary['rs_mean']:.2f} +/- {summary['rs_std']:.2f}"
          f"  (robust sigma {summary['rs_mad']:.2f})")
    print(f"  Rs  range        : {summary['rs_min']:.2f} .. {summary['rs_max']:.2f} "
          f"mOhm*cm2")
    if np.isfinite(summary.get("rct_median", np.nan)):        # type: ignore[arg-type]
        print(f"  Rct median       : {summary['rct_median']:.2f} mOhm*cm2")
        print(f"  Rct mean +/- std : {summary['rct_mean']:.2f} +/- "
              f"{summary['rct_std']:.2f}")
        print(f"  Rct range        : {summary['rct_min']:.2f} .. "
              f"{summary['rct_max']:.2f} mOhm*cm2")

    plate_rs = summary.get("rs_parallel_mohm_cm2")
    reference = summary.get("rs_reference_mohm_cm2")
    verdict = summary.get("parallel_sum_verdict")
    if plate_rs is not None and reference:
        ratio = summary["parallel_sum_ratio"]                 # type: ignore[index]
        print(f"\n  PARALLEL SUM     : {plate_rs:.2f} mOhm*cm2 recombined vs "
              f"{reference:.2f} mOhm*cm2 reference  ->  {ratio:.4f}x   [{verdict}]")
        print(f"                     (segments share one cell voltage, so they "
              f"combine as parallel conductances)")
    elif plate_rs is not None:
        print(f"\n  PARALLEL SUM     : {plate_rs:.2f} mOhm*cm2 recombined, no "
              f"cell-level reference given [SKIPPED]")
        print(f"                     (pass --rs-reference to turn this into a test)")

    card_medians = summary.get("card_medians") or {}
    card_verdicts = summary.get("card_verdicts") or {}
    if card_medians:
        basis = (
            "step away from the fitted spatial trend"
            if summary.get("card_bias_detrended")
            else "offset from the plate median"
        )
        print(f"\n  per-card median Rs [mOhm*cm2]   (judged on the {basis}):")
        for card, value in sorted(card_medians.items()):      # type: ignore[union-attr]
            deviation = value / summary["rs_median"] - 1.0    # type: ignore[operator]
            mark = card_verdicts.get(card, OK)                # type: ignore[union-attr]
            print(f"    card {card}: {value:.2f}  ({deviation:+.1%} vs plate median)"
                  f"  [{mark}]")

    if "neighbour_check" in summary:
        print(f"\n  note: {summary['neighbour_check']}")

    plate_flags = summary.get("plate_flags") or []
    print(f"\n  PLATE-LEVEL VERDICT: {summary.get('plate_verdict', OK)}")
    if plate_flags:
        for level, message in plate_flags:                    # type: ignore[misc]
            print(f"    [{level}] {message}")
    else:
        print("    no plate-wide problem detected")

    counts = frame["verdict"].value_counts().to_dict()
    print(f"\n  SEGMENT VERDICTS: OK={counts.get(OK, 0)}  WARN={counts.get(WARN, 0)}  "
          f"FAIL={counts.get(FAIL, 0)}")

    print(f"\n{line}")
    print(f"  {'seg':>4} {'card':>4} {'Rs [mOhm.cm2]':>14} {'Rct [mOhm.cm2]':>15} "
          f"{'Rs/med':>7} {'MAD':>6} {'nbr':>6} {'arc':>5} {'pts':>4} {'verdict':>7}")
    print("-" * 108)
    for _, row in frame.iterrows():
        neighbour = (
            f"{row['neighbour_rel']:.2f}" if np.isfinite(row["neighbour_rel"]) else "  -"
        )
        print(f"  {row['segment']:>4} {row['card']:>4} {row['rs_mOhm_cm2']:>14.2f} "
              f"{row['rct_mOhm_cm2']:>15.2f} "
              f"{row['uniformity_rs_over_median']:>7.3f} {row['outlier_MAD']:>6.1f} "
              f"{neighbour:>6} {row['arc_quality']:>5.2f} {row['n_points']:>4} "
              f"{row['verdict']:>7}")

    suspect = frame[frame["verdict"] != OK]
    if not suspect.empty:
        print(f"\n{line}")
        print("  FLAGGED SEGMENTS")
        print(line)
        for _, row in suspect.iterrows():
            print(f"  segment {row['segment']:>3} (card {row['card']}) "
                  f"[{row['verdict']}]")
            for flag in str(row["flags"]).split("; "):
                if flag:
                    print(f"      - {flag}")
    else:
        print("\n  every segment passed.")


def make_plots(frame: pd.DataFrame, summary: dict[str, object], cfg, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    colours = {OK: "#2f7d4f", WARN: "#d9a021", FAIL: "#b3312c"}
    bar_colours = [colours[v] for v in frame["verdict"]]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].bar(frame["segment"], frame["rs_mOhm_cm2"], color=bar_colours)
    reference = float(summary.get("rs_median", np.nan))     # type: ignore[arg-type]
    for factor, style, label in (
        (1.0, "-", "plate median"), (1.35, "--", "+35 %"), (0.75, "--", "-25 %")
    ):
        axes[0].axhline(reference * factor, color="0.35", ls=style, lw=1, label=label)
    axes[0].set_ylabel(r"$R_s$ [m$\Omega\cdot$cm$^2$]")
    axes[0].set_title("Ohmic resistance (green OK, amber WARN, red FAIL)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(frame["segment"], frame["rct_mOhm_cm2"], color=bar_colours)
    axes[1].set_xlabel("segment")
    axes[1].set_ylabel(r"$R_{ct}$ [m$\Omega\cdot$cm$^2$]")
    axes[1].set_title("Charge-transfer / polarisation resistance")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "impedance_bars.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].scatter(frame["rs_mOhm_cm2"], frame["rct_mOhm_cm2"],
                    c=bar_colours, s=32)
    axes[0].set_xlabel(r"$R_s$ [m$\Omega\cdot$cm$^2$]")
    axes[0].set_ylabel(r"$R_{ct}$ [m$\Omega\cdot$cm$^2$]")
    axes[0].set_title("Rs vs Rct")

    axes[1].scatter(frame["arc_quality"], frame["outlier_MAD"], c=bar_colours, s=32)
    axes[1].axvline(0.70, color="#d9a021", ls="--", lw=1, label="arc WARN")
    axes[1].axvline(0.45, color="#b3312c", ls="--", lw=1, label="arc FAIL")
    axes[1].axhline(3.5, color="#d9a021", ls=":", lw=1)
    axes[1].set_xlabel("arc quality")
    axes[1].set_ylabel("Rs outlier score [MAD]")
    axes[1].set_title("Shape vs population")
    axes[1].legend(fontsize=8)

    axes[2].scatter(frame["n_points"], frame["rs_mOhm_cm2"], c=bar_colours, s=32)
    axes[2].set_xlabel("usable frequency points")
    axes[2].set_ylabel(r"$R_s$ [m$\Omega\cdot$cm$^2$]")
    axes[2].set_title("Does Rs depend on how much data it rests on?")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "impedance_screening.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(path)

    try:
        from eis.viz import plot_plate

        suspect = {
            int(row["segment"]) for _, row in frame.iterrows() if row["verdict"] != OK
        }
        for column, title, unit, name in (
            ("rs_mOhm_cm2", "Ohmic resistance", r"$R_s$ [m$\Omega\cdot$cm$^2$]",
             "plate_rs.png"),
            ("rct_mOhm_cm2", "Charge-transfer resistance",
             r"$R_{ct}$ [m$\Omega\cdot$cm$^2$]", "plate_rct.png"),
        ):
            values = {
                int(row["segment"]): float(row[column])
                for _, row in frame.iterrows() if np.isfinite(row[column])
            }
            fig = plot_plate(values, cfg, title, unit, uncertain=suspect)
            if fig is not None:
                path = out_dir / name
                fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                written.append(path)
    except Exception as exc:
        print(f"  (plate maps skipped: {exc})")
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plausibility check for the local impedance of every segment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="YAML configuration file")
    p.add_argument("--from-out", help="pipeline output directory to read")
    p.add_argument("--impedance", help="impedance table (parquet or csv)")
    p.add_argument("--segments", help="per-segment table with KK and ECM results")
    p.add_argument("--condition", help="operating point; default is all found")
    p.add_argument("--segment-area", type=float, help="segment area [cm^2]")
    p.add_argument(
        "--rs-reference", type=float,
        help="cell-level ohmic ASR [mOhm*cm^2] for the parallel sum check",
    )
    p.add_argument("--out", help="write the per-segment CSV here")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--demo", action="store_true",
                   help="generate data, run the pipeline, then check the result")

    g = p.add_argument_group("thresholds")
    g.add_argument("--rs-min", type=float, help="lower plausible Rs [mOhm*cm^2]")
    g.add_argument("--rs-max", type=float, help="upper plausible Rs [mOhm*cm^2]")
    g.add_argument("--arc-warn", type=float, help="arc-quality WARN threshold")
    g.add_argument("--points-warn", type=int, help="minimum comfortable point count")
    g.add_argument("--sum-tolerance", type=float,
                   help="relative tolerance on the parallel sum before WARN")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.segment_area:
        cfg.geometry.segment_area_cm2 = args.segment_area

    if args.demo:
        import tempfile

        from eis.pipeline import run_measurement
        from tests.synthetic import simulate_measurement

        base = Path(args.out or tempfile.mkdtemp())
        raw = base / "demo_raw"
        truth = simulate_measurement(
            raw, measurement_id="DEMO", condition="150A",
            n_cards=4, segments_per_card=5, duration_s=30.0,
        )
        cfg.measurement_id, cfg.raw_dir = "DEMO", str(raw)
        cfg.output_dir = str(base / "pipeline")
        cfg.conditions = ["150A"]
        cfg.calibration.shunt_csv = str(raw / "curr.csv")
        cfg.calibration.temperature_csv = str(raw / "temp.csv")
        cfg.spectral.base_frequency_hz = 1.0
        cfg.spectral.excitation_tones_hz = list(truth.tones_hz)
        cfg.model.n_starts = 3
        print("synthetic demonstration: running the pipeline first ...")
        run_measurement(cfg, write=True, verbose=False)
        args.from_out = cfg.output_dir
        args.impedance = None

    impedance_path, segments_path = resolve_inputs(args)
    impedance = read_table(impedance_path)
    segments = read_table(segments_path) if segments_path else None
    print(f"\nimpedance table : {impedance_path}  ({len(impedance):,} rows)")
    if segments_path:
        print(f"segment table   : {segments_path}  ({len(segments):,} rows)")
    else:
        print("segment table   : not supplied - KK and circuit-fit checks are skipped")

    for column in ("condition", "segment", "frequency_hz", "z_real_ohm", "z_imag_ohm"):
        if column not in impedance.columns:
            print(f"error: the impedance table has no {column!r} column",
                  file=sys.stderr)
            return 2

    limits = Limits()
    if args.rs_min is not None:
        limits.rs_min = args.rs_min
    if args.rs_max is not None:
        limits.rs_max = args.rs_max
    if args.arc_warn is not None:
        limits.arc_warn = args.arc_warn
    if args.points_warn is not None:
        limits.points_warn = args.points_warn
    if args.sum_tolerance is not None:
        limits.sum_warn = args.sum_tolerance

    measurement_id = (
        str(impedance["measurement_id"].iloc[0])
        if "measurement_id" in impedance.columns and len(impedance)
        else (cfg.measurement_id or "?")
    )
    conditions = (
        [args.condition] if args.condition
        else sorted(impedance["condition"].unique())
    )

    worst = OK
    for condition in conditions:
        if condition not in set(impedance["condition"]):
            print(f"  skipping {condition}: not in the table")
            continue

        checks = collect_segments(
            impedance, segments, condition, cfg.geometry.segment_area_cm2
        )
        if not checks:
            print(f"  {condition}: no segments")
            continue
        summary = apply_checks(checks, cfg, limits, args.rs_reference)
        frame = to_frame(checks)
        print_report(frame, summary, condition, measurement_id)

        # Overall verdict: the worst of every segment, every plate-level
        # finding and the parallel sum, across every condition.  One FAIL
        # anywhere means the measurement needs looking at before it is used.
        for verdict in (
            list(frame["verdict"])
            + [summary.get("plate_verdict", OK)]
            + [summary.get("parallel_sum_verdict", OK)]
        ):
            if _RANK.get(str(verdict), 0) > _RANK[worst]:
                worst = str(verdict)

        if args.out:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"impedance_plausibility_{condition}.csv"
            frame.to_csv(path, index=False)
            print(f"\n  wrote {path}")
            if args.plots:
                for figure in make_plots(frame, summary, cfg, out_dir / condition):
                    print(f"  figure: {figure}")
        elif args.plots:
            for figure in make_plots(
                frame, summary, cfg, Path("./plausibility") / condition
            ):
                print(f"  figure: {figure}")

    print(f"\noverall: {worst}")
    if worst == FAIL:
        print("  a single FAIL anywhere means this measurement needs investigation "
              "before any downstream analysis.")
    return 1 if worst == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
