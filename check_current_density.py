#!/usr/bin/env python3
"""Plausibility check for the local DC current density of every segment.

Reads the raw FAMOS files of one operating point, converts each segment's
via-shunt voltage into a local current density, and puts every segment through
a series of independent checks.  Nothing here estimates impedance - this is the
sanity gate you run *before* trusting any EIS result, because a segment whose
current density is wrong produces an impedance that is wrong in a way that
looks perfectly reasonable on a Nyquist plot.

Units
-----
Current density is A/cm^2.  Two resistances are derived from it and reported in
milliohms, both **operating-point (DC) resistances**, not the ohmic Rs from an
impedance spectrum:

    ASR_k = U_cell / j_k          [Ohm*cm^2]  -> reported in mOhm*cm^2
    R_k   = U_cell / I_k          [Ohm]       -> reported in mOhm

``R_k`` is the DC resistance seen by one segment alone.  Because the segments
are in parallel across a common cell voltage, ``R_k`` is roughly ``N`` times the
whole-cell resistance; ``ASR_k`` removes the area and is the number to compare
between segments and against literature.

Checks
------
=========================  ====================================================
check                      what it catches
=========================  ====================================================
current_sum                total of all segments vs the nominal cell current -
                           the strongest single test of the calibration chain
sign                       a segment carrying reverse current
magnitude                  j outside a physically plausible band
uniformity                 j/median outside the range a healthy cell shows
outlier                    robust median/MAD outlier against the population
neighbour                  a segment far from its spatial neighbours
card_bias                  one whole card offset from the others - wiring or
                           calibration, not electrochemistry
dead_channel               near-zero variance, or a stuck value
clipping                   samples pinned at the converter's range
drift                      DC level not stationary across the record
calibration                missing, non-physical or out-of-family H(T)
temperature                sensor outside its plausible range
=========================  ====================================================

Usage
-----
::

    python check_current_density.py --raw-dir /data/Famos \\
        --measurement-id 2611976 --condition 150A \\
        --shunt-csv cal/curr.csv --temp-csv cal/temp.csv \\
        --nominal-current auto --plots --out out/plausibility

    # try it on generated data first
    python check_current_density.py --demo --plots
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eis.calibrate import (  # noqa: E402
    build_temperature_field, load_shunt_calibration, load_temperature_calibration,
)
from eis.config import load_config  # noqa: E402
from eis.io.famos import classify_channel, discover_files, open_famos  # noqa: E402

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_RANK = {OK: 0, WARN: 1, FAIL: 2}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass
class Limits:
    """Every threshold the verdicts depend on, in one place."""

    # plausible local current density [A/cm^2]
    j_min: float = 0.02
    j_max: float = 3.0
    j_warn_min: float = 0.05
    j_warn_max: float = 2.5

    # local-to-median ratio for a healthy cell
    uniformity_warn: tuple[float, float] = (0.70, 1.40)
    uniformity_fail: tuple[float, float] = (0.40, 2.00)

    # robust outlier score (|j - median| / (1.4826 * MAD))
    outlier_warn: float = 3.5
    outlier_fail: float = 6.0

    # spatial neighbour deviation (relative)
    neighbour_warn: float = 0.25
    neighbour_fail: float = 0.50
    neighbour_radius_cm: float = 4.0

    # per-card mean vs the global mean (relative)
    card_bias_warn: float = 0.10
    card_bias_fail: float = 0.25

    # sum of segment currents vs the nominal cell current (relative)
    sum_warn: float = 0.05
    sum_fail: float = 0.15

    # DC stability across the record (relative spread of thirds)
    drift_warn: float = 0.05
    drift_fail: float = 0.15

    # AC content relative to DC - large values mean the "DC" is not a level
    ac_dc_warn: float = 0.50

    # dead channel: AC RMS below this fraction of the population median
    dead_fraction: float = 0.05

    # clipping: fraction of samples pinned at the extremes
    clip_warn: float = 1e-4
    clip_fail: float = 1e-3

    # floor on the robust scale, as a fraction of the median j, so that a very
    # uniform plate does not flag every segment as an outlier
    outlier_mad_floor: float = 0.005

    # shunt factor H(T) sanity [V*cm^2/A] and family spread
    h_min: float = 1e-6
    h_max: float = 1e-1
    h_family_warn: float = 0.25


# ---------------------------------------------------------------------------
# Per-segment record
# ---------------------------------------------------------------------------

@dataclass
class SegmentCheck:
    segment: int
    card: int
    channel: str
    n_samples: int = 0

    # raw
    v_dc_v: float = np.nan
    v_ac_rms_v: float = np.nan
    v_min_v: float = np.nan
    v_max_v: float = np.nan
    clipped_fraction: float = 0.0

    # calibration
    temperature_c: float = np.nan
    shunt_H: float = np.nan

    # results
    j_dc: float = np.nan             # A/cm^2
    i_dc_a: float = np.nan           # A
    j_ac_rms: float = np.nan
    drift_rel: float = np.nan
    uniformity: float = np.nan
    outlier_score: float = np.nan
    neighbour_rel: float = np.nan
    asr_mohm_cm2: float = np.nan
    r_mohm: float = np.nan

    verdict: str = OK
    flags: list[str] = field(default_factory=list)

    def raise_to(self, level: str, message: str) -> None:
        self.flags.append(f"{level}:{message}")
        if _RANK[level] > _RANK[self.verdict]:
            self.verdict = level


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def nominal_current_from_condition(condition: str) -> float | None:
    """Read the operating current out of a label such as ``150A``."""
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*A\b", condition, re.IGNORECASE)
    return float(match.group(1).replace(",", ".")) if match else None


def collect_segments(card_files: dict[int, str], cfg, limits: Limits):
    """Read every segment, cell-voltage and temperature channel of one condition."""
    checks: dict[int, SegmentCheck] = {}
    sensor_readings: dict[str, float] = {}
    u_cell_dc: list[float] = []
    fs = None

    temp_cal = (
        load_temperature_calibration(cfg.calibration.temperature_csv)
        if cfg.calibration.temperature_csv else None
    )
    sensor_index = 0

    for card, path in sorted(card_files.items()):
        famos = open_famos(path, card=card)
        fs = famos.fs
        if famos.header.scaling_note:
            print(f"  ! card {card}: {famos.header.scaling_note}")

        for name in famos.channel_names:
            role, key = classify_channel(name)

            if role == "segment":
                trace = famos.channel(name)
                check = SegmentCheck(
                    segment=int(key), card=card, channel=name, n_samples=len(trace)
                )
                # Median, not mean: robust to a transient or a purge event that
                # would otherwise shift the reported operating point.
                check.v_dc_v = float(np.median(trace))
                check.v_ac_rms_v = float(np.std(trace))
                check.v_min_v = float(trace.min())
                check.v_max_v = float(trace.max())

                # Clipping: samples sitting at the extreme value of the record.
                span = check.v_max_v - check.v_min_v
                if span > 0:
                    tolerance = 1e-6 * max(abs(check.v_max_v), abs(check.v_min_v), 1e-12)
                    pinned = (
                        np.isclose(trace, check.v_max_v, atol=tolerance)
                        | np.isclose(trace, check.v_min_v, atol=tolerance)
                    )
                    check.clipped_fraction = float(pinned.sum()) / len(trace)

                # DC stationarity across thirds of the record.
                thirds = np.array_split(trace, 3)
                levels = np.array([np.median(t) for t in thirds])
                scale = max(abs(float(np.median(levels))), 1e-15)
                check.drift_rel = float(np.ptp(levels) / scale)

                checks[int(key)] = check

            elif role == "cell_voltage":
                trace = famos.channel(name)
                level = float(np.median(trace))
                if abs(level) > 1e-3:                 # ignore a dead UC channel
                    u_cell_dc.append(level)

            elif role == "temperature" and temp_cal is not None:
                if sensor_index < len(temp_cal.c0):
                    volts = float(np.mean(famos.channel(name)))
                    sensor_readings[f"card{card}:{name}"] = temp_cal.to_celsius(
                        sensor_index, volts
                    )
                sensor_index += 1

        del famos

    return checks, sensor_readings, u_cell_dc, fs


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def apply_checks(
    checks: dict[int, SegmentCheck], cfg, limits: Limits,
    nominal_current_a: float | None, u_cell_v: float | None,
) -> dict[str, object]:
    """Run every plausibility test and return the whole-cell summary."""
    values = np.array(
        [c.j_dc for c in checks.values() if np.isfinite(c.j_dc)], dtype=float
    )
    summary: dict[str, object] = {}
    if values.size == 0:
        return {"note": "no segment produced a finite current density"}

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    mean = float(np.mean(values))

    # A very uniform plate has a tiny MAD, and then every segment sits hundreds
    # of "sigma" from the median on differences that are physically irrelevant.
    # Floor the scale at a fraction of the median so the outlier test measures
    # something meaningful rather than the width of the noise.
    mad_effective = max(mad, limits.outlier_mad_floor * abs(median))

    summary |= {
        "n_segments": len(values),
        "j_median": median, "j_mean": mean,
        "j_min": float(values.min()), "j_max": float(values.max()),
        "j_std": float(np.std(values)),
        "j_mad": mad,
        "j_mad_effective": mad_effective,
    }
    plate_flags: list[tuple[str, str]] = []

    # --- 1. per-segment, independent of the population --------------------
    ac_values = np.array(
        [c.j_ac_rms for c in checks.values() if np.isfinite(c.j_ac_rms)]
    )
    ac_median = float(np.median(ac_values)) if ac_values.size else 0.0

    for check in checks.values():
        if not np.isfinite(check.j_dc):
            check.raise_to(FAIL, "no current density (calibration or channel missing)")
            continue

        # sign
        if nominal_current_a is not None and nominal_current_a != 0:
            if np.sign(check.j_dc) != np.sign(nominal_current_a):
                check.raise_to(
                    FAIL, f"reverse current: j={check.j_dc:+.3f} A/cm2 against a "
                          f"{nominal_current_a:+.0f} A operating point"
                )
        elif check.j_dc < 0:
            check.raise_to(FAIL, f"negative current density {check.j_dc:.3f} A/cm2")

        magnitude = abs(check.j_dc)
        if not (limits.j_min <= magnitude <= limits.j_max):
            check.raise_to(
                FAIL, f"|j|={magnitude:.3f} A/cm2 outside "
                      f"[{limits.j_min}, {limits.j_max}]"
            )
        elif not (limits.j_warn_min <= magnitude <= limits.j_warn_max):
            check.raise_to(WARN, f"|j|={magnitude:.3f} A/cm2 near the plausible edge")

        # dead / stuck channel
        if ac_median > 0 and check.j_ac_rms < limits.dead_fraction * ac_median:
            check.raise_to(
                FAIL, f"channel looks dead: AC RMS is "
                      f"{check.j_ac_rms / max(ac_median, 1e-30):.1%} of the population"
            )
        if check.v_max_v == check.v_min_v:
            check.raise_to(FAIL, "constant signal (stuck channel)")

        # clipping
        if check.clipped_fraction > limits.clip_fail:
            check.raise_to(
                FAIL, f"{check.clipped_fraction:.2%} of samples pinned at the range end"
            )
        elif check.clipped_fraction > limits.clip_warn:
            check.raise_to(
                WARN, f"{check.clipped_fraction:.3%} of samples pinned at the range end"
            )

        # DC stationarity
        if check.drift_rel > limits.drift_fail:
            check.raise_to(
                FAIL, f"DC level moves {check.drift_rel:.1%} across the record"
            )
        elif check.drift_rel > limits.drift_warn:
            check.raise_to(
                WARN, f"DC level moves {check.drift_rel:.1%} across the record"
            )

        # excitation relative to the operating point
        if magnitude > 0 and check.j_ac_rms / magnitude > limits.ac_dc_warn:
            check.raise_to(
                WARN, f"AC/DC = {check.j_ac_rms / magnitude:.2f}: the DC level is "
                      f"not a well-defined operating point"
            )

        # calibration
        if not np.isfinite(check.shunt_H):
            check.raise_to(FAIL, "no shunt calibration for this segment")
        elif not (limits.h_min <= check.shunt_H <= limits.h_max):
            check.raise_to(
                FAIL, f"shunt factor H={check.shunt_H:.3e} V*cm2/A is non-physical"
            )

    # --- 2. population-relative -------------------------------------------
    for check in checks.values():
        if not np.isfinite(check.j_dc):
            continue
        check.uniformity = check.j_dc / median if median else np.nan
        check.outlier_score = (
            abs(check.j_dc - median) / mad_effective if mad_effective > 0 else 0.0
        )

        lo_f, hi_f = limits.uniformity_fail
        lo_w, hi_w = limits.uniformity_warn
        if not (lo_f <= check.uniformity <= hi_f):
            check.raise_to(
                FAIL, f"j/median = {check.uniformity:.2f}, outside [{lo_f}, {hi_f}]"
            )
        elif not (lo_w <= check.uniformity <= hi_w):
            check.raise_to(WARN, f"j/median = {check.uniformity:.2f}")

        if check.outlier_score > limits.outlier_fail:
            check.raise_to(
                FAIL, f"robust outlier, {check.outlier_score:.1f} MAD from the median"
            )
        elif check.outlier_score > limits.outlier_warn:
            check.raise_to(
                WARN, f"{check.outlier_score:.1f} MAD from the median"
            )

    # --- 3. spatial neighbours (only with real geometry) -------------------
    coords = cfg.geometry.segment_coords
    if coords:
        for check in checks.values():
            if check.segment not in coords or not np.isfinite(check.j_dc):
                continue
            x, y = coords[check.segment][:2]
            neighbours = [
                other.j_dc for other in checks.values()
                if other.segment != check.segment
                and other.segment in coords
                and np.isfinite(other.j_dc)
                and np.hypot(coords[other.segment][0] - x, coords[other.segment][1] - y)
                <= limits.neighbour_radius_cm
            ]
            if len(neighbours) < 2:
                continue
            local = float(np.median(neighbours))
            if abs(local) < 1e-12:
                continue
            check.neighbour_rel = abs(check.j_dc - local) / abs(local)
            if check.neighbour_rel > limits.neighbour_fail:
                check.raise_to(
                    FAIL, f"{check.neighbour_rel:.0%} away from its neighbours "
                          f"({local:.3f} A/cm2)"
                )
            elif check.neighbour_rel > limits.neighbour_warn:
                check.raise_to(
                    WARN, f"{check.neighbour_rel:.0%} away from its neighbours"
                )
    else:
        summary["neighbour_check"] = (
            "skipped: no segment_coords configured, so there is no spatial layout "
            "to compare against"
        )

    # --- 4. per-card bias --------------------------------------------------
    # Compared against the plate *median*: one bad card must not drag the
    # reference it is being judged against.
    card_means: dict[int, float] = {}
    card_verdicts: dict[int, str] = {}
    for card in sorted({c.card for c in checks.values()}):
        card_values = [
            c.j_dc for c in checks.values() if c.card == card and np.isfinite(c.j_dc)
        ]
        if not card_values:
            continue
        card_mean = float(np.median(card_values))
        card_means[card] = card_mean
        deviation = abs(card_mean - median) / abs(median) if median else np.nan
        if deviation > limits.card_bias_fail:
            level, note = FAIL, "wiring or calibration, not electrochemistry"
        elif deviation > limits.card_bias_warn:
            level, note = WARN, "check this card's calibration"
        else:
            card_verdicts[card] = OK
            continue
        card_verdicts[card] = level
        plate_flags.append(
            (level, f"card {card} median j is {deviation:+.0%} from the plate "
                    f"median ({note})")
        )
        # A whole-card offset is context for its segments, not proof that any
        # individual one is broken - so it never raises a segment past WARN.
        for check in checks.values():
            if check.card == card:
                check.raise_to(
                    WARN, f"card {card} is {deviation:.0%} off the plate median"
                )
    summary["card_means"] = card_means
    summary["card_verdicts"] = card_verdicts

    # --- 5. total current --------------------------------------------------
    total = float(
        np.nansum([c.i_dc_a for c in checks.values() if np.isfinite(c.i_dc_a)])
    )
    summary["total_current_a"] = total
    summary["nominal_current_a"] = nominal_current_a
    if nominal_current_a:
        ratio = total / nominal_current_a
        summary["current_sum_ratio"] = ratio
        deviation = abs(ratio - 1.0)
        if deviation > limits.sum_fail:
            summary["current_sum_verdict"] = FAIL
        elif deviation > limits.sum_warn:
            summary["current_sum_verdict"] = WARN
        else:
            summary["current_sum_verdict"] = OK
        # This is a statement about the *plate*, not about any one segment.
        # Stamping it onto all 80 records would turn every row red and bury the
        # segments that are genuinely faulty, so it stays at plate level.
        if deviation > limits.sum_warn:
            plate_flags.append((
                summary["current_sum_verdict"],            # type: ignore[arg-type]
                f"segment currents sum to {total:.2f} A against a nominal "
                f"{nominal_current_a:.0f} A ({ratio:.3f}x). Either the shunt "
                f"calibration carries a scale error, or segments are missing "
                f"from the sum - check the segment count before blaming any "
                f"individual channel."
            ))
    else:
        summary["current_sum_verdict"] = "SKIPPED"

    summary["u_cell_v"] = u_cell_v
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
            "channel": check.channel,
            "verdict": check.verdict,
            "j_dc_A_cm2": check.j_dc,
            "i_dc_A": check.i_dc_a,
            "asr_mOhm_cm2": check.asr_mohm_cm2,
            "r_mOhm": check.r_mohm,
            "uniformity_j_over_median": check.uniformity,
            "outlier_MAD": check.outlier_score,
            "neighbour_rel": check.neighbour_rel,
            "drift_rel": check.drift_rel,
            "j_ac_rms_A_cm2": check.j_ac_rms,
            "v_dc_V": check.v_dc_v,
            "v_ac_rms_V": check.v_ac_rms_v,
            "clipped_fraction": check.clipped_fraction,
            "temperature_C": check.temperature_c,
            "shunt_H_V_cm2_per_A": check.shunt_H,
            "flags": "; ".join(check.flags),
        })
    return pd.DataFrame(rows)


def print_report(
    frame: pd.DataFrame, summary: dict[str, object], limits: Limits,
    condition: str, measurement_id: str,
) -> None:
    line = "=" * 100
    print(f"\n{line}")
    print(f"  LOCAL CURRENT DENSITY PLAUSIBILITY - {measurement_id} / {condition}")
    print(line)

    if "n_segments" not in summary:
        print(f"  {summary.get('note', 'nothing to report')}")
        return

    u_cell = summary.get("u_cell_v")
    print(f"  segments checked : {summary['n_segments']}")
    print(f"  j  median        : {summary['j_median']:.4f} A/cm2")
    print(f"  j  mean +/- std  : {summary['j_mean']:.4f} +/- {summary['j_std']:.4f}"
          f"  (robust sigma {summary['j_mad']:.4f})")
    print(f"  j  range         : {summary['j_min']:.4f} .. {summary['j_max']:.4f} A/cm2")
    if u_cell is not None:
        print(f"  cell voltage     : {u_cell:.4f} V (DC)")

    total = summary.get("total_current_a")
    nominal = summary.get("nominal_current_a")
    verdict = summary.get("current_sum_verdict")
    if nominal:
        ratio = summary["current_sum_ratio"]                       # type: ignore[index]
        print(f"\n  CURRENT SUM      : {total:.2f} A measured vs {nominal:.1f} A "
              f"nominal  ->  {ratio:.4f}x   [{verdict}]")
        print(f"                     (the strongest single test of the shunt "
              f"calibration chain)")
    else:
        print(f"\n  CURRENT SUM      : {total:.2f} A measured, no nominal value given "
              f"[SKIPPED]")

    card_means = summary.get("card_means") or {}
    card_verdicts = summary.get("card_verdicts") or {}
    if card_means:
        print("\n  per-card median j [A/cm2]:")
        for card, value in sorted(card_means.items()):     # type: ignore[union-attr]
            deviation = value / summary["j_median"] - 1.0  # type: ignore[operator]
            mark = card_verdicts.get(card, OK)             # type: ignore[union-attr]
            print(f"    card {card}: {value:.4f}  ({deviation:+.1%} vs plate median)"
                  f"  [{mark}]")

    if "neighbour_check" in summary:
        print(f"\n  note: {summary['neighbour_check']}")

    plate_flags = summary.get("plate_flags") or []
    print(f"\n  PLATE-LEVEL VERDICT: {summary.get('plate_verdict', OK)}")
    if plate_flags:
        for level, message in plate_flags:                 # type: ignore[misc]
            print(f"    [{level}] {message}")
    else:
        print("    no plate-wide problem detected")

    counts = frame["verdict"].value_counts().to_dict()
    print(f"\n  SEGMENT VERDICTS: OK={counts.get(OK, 0)}  WARN={counts.get(WARN, 0)}  "
          f"FAIL={counts.get(FAIL, 0)}")

    print(f"\n{line}")
    print(f"  {'seg':>4} {'card':>4} {'j [A/cm2]':>10} {'I [A]':>8} "
          f"{'ASR [mOhm.cm2]':>15} {'R [mOhm]':>10} {'j/med':>7} {'MAD':>6} "
          f"{'verdict':>7}")
    print("-" * 100)
    for _, row in frame.iterrows():
        print(f"  {row['segment']:>4} {row['card']:>4} {row['j_dc_A_cm2']:>10.4f} "
              f"{row['i_dc_A']:>8.3f} {row['asr_mOhm_cm2']:>15.2f} "
              f"{row['r_mOhm']:>10.2f} {row['uniformity_j_over_median']:>7.3f} "
              f"{row['outlier_MAD']:>6.1f} {row['verdict']:>7}")

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
    """Bar chart, distribution, and a plate map of the current density."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    colours = {OK: "#2f7d4f", WARN: "#d9a021", FAIL: "#b3312c"}

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), height_ratios=[2, 1])
    axes[0].bar(
        frame["segment"], frame["j_dc_A_cm2"],
        color=[colours[v] for v in frame["verdict"]],
    )
    reference = float(summary.get("j_median", np.nan))         # type: ignore[arg-type]
    for factor, style, label in (
        (1.0, "-", "plate median"),
        (1.4, "--", "+40 %"),
        (0.7, "--", "-30 %"),
    ):
        axes[0].axhline(reference * factor, color="0.35", ls=style, lw=1, label=label)
    axes[0].set_xlabel("segment")
    axes[0].set_ylabel(r"$j_{\rm DC}$ [A/cm$^2$]")
    axes[0].set_title("Local DC current density (green OK, amber WARN, red FAIL)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(
        frame["segment"], frame["asr_mOhm_cm2"],
        color=[colours[v] for v in frame["verdict"]],
    )
    axes[1].set_xlabel("segment")
    axes[1].set_ylabel(r"$U_{\rm cell}/j$ [m$\Omega\cdot$cm$^2$]")
    axes[1].set_title("Operating-point area-specific resistance (not the ohmic Rs)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "current_density_bars.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(frame["j_dc_A_cm2"].dropna(), bins=max(8, len(frame) // 4),
                 color="#3b6ea5", edgecolor="white")
    axes[0].axvline(reference, color="0.2", ls="--", lw=1.2)
    axes[0].set_xlabel(r"$j_{\rm DC}$ [A/cm$^2$]")
    axes[0].set_ylabel("segments")
    axes[0].set_title("Distribution")
    axes[1].scatter(frame["j_dc_A_cm2"], frame["outlier_MAD"],
                    c=[colours[v] for v in frame["verdict"]], s=30)
    axes[1].axhline(3.5, color="#d9a021", ls="--", lw=1, label="WARN")
    axes[1].axhline(6.0, color="#b3312c", ls="--", lw=1, label="FAIL")
    axes[1].set_xlabel(r"$j_{\rm DC}$ [A/cm$^2$]")
    axes[1].set_ylabel("robust outlier score [MAD]")
    axes[1].set_title("Outlier screening")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "current_density_distribution.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(path)

    try:
        from eis.viz import plot_plate

        values = {
            int(row["segment"]): float(row["j_dc_A_cm2"])
            for _, row in frame.iterrows() if np.isfinite(row["j_dc_A_cm2"])
        }
        suspect = {
            int(row["segment"]) for _, row in frame.iterrows()
            if row["verdict"] != OK
        }
        fig = plot_plate(
            values, cfg, "Local DC current density", r"$j$ [A/cm$^2$]",
            uncertain=suspect, cmap="viridis",
        )
        if fig is not None:
            path = out_dir / "current_density_plate.png"
            fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            written.append(path)
    except Exception as exc:                      # geometry is optional
        print(f"  (plate map skipped: {exc})")
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg, condition: str, card_files: dict[int, str], limits: Limits,
        nominal_current_a: float | None):
    print(f"\nreading {len(card_files)} card file(s) for {condition} ...")
    checks, sensor_readings, u_cell_list, fs = collect_segments(card_files, cfg, limits)
    if not checks:
        raise RuntimeError("no segment channels found in these files")
    print(f"  {len(checks)} segment channels, fs = {fs:.0f} Hz")

    u_cell = float(np.median(u_cell_list)) if u_cell_list else None
    if u_cell is None:
        print("  ! no usable cell-voltage channel: the mOhm columns cannot be filled")

    # --- temperature -> H(T) ------------------------------------------------
    segments = sorted(checks)
    if cfg.calibration.use_measured_temperature and sensor_readings:
        field_ = build_temperature_field(
            sensor_readings, None, cfg.geometry.segment_coords, segments,
            cfg.calibration.temperature_fallback_c,
            tuple(cfg.calibration.temperature_valid_range_c),
        )
    else:
        from eis.calibrate import TemperatureField

        field_ = TemperatureField(
            per_segment_c={s: cfg.calibration.temperature_fallback_c for s in segments},
            fallback_c=cfg.calibration.temperature_fallback_c, degraded=True,
            note="no temperature calibration supplied",
        )
    print(f"  temperature: {field_.mean_c:.1f} degC"
          f"{'  [DEGRADED: ' + field_.note + ']' if field_.degraded else ''}")

    shunt = (
        load_shunt_calibration(cfg.calibration.shunt_csv)
        if cfg.calibration.shunt_csv else None
    )
    if shunt is None:
        raise RuntimeError(
            "a shunt calibration file is required: without H(T) there is no "
            "current density, only a voltage"
        )

    # --- convert ------------------------------------------------------------
    area = cfg.geometry.segment_area_cm2
    for check in checks.values():
        check.temperature_c = field_.at(check.segment)
        if not shunt.has(check.segment):
            check.raise_to(FAIL, "segment absent from the shunt calibration file")
            continue
        H = shunt.H(check.segment, check.temperature_c)
        check.shunt_H = H
        if not np.isfinite(H) or H == 0:
            continue
        check.j_dc = check.v_dc_v / H
        check.j_ac_rms = check.v_ac_rms_v / abs(H)
        check.i_dc_a = check.j_dc * area
        if u_cell is not None and abs(check.j_dc) > 1e-12:
            check.asr_mohm_cm2 = (u_cell / check.j_dc) * 1e3
            check.r_mohm = (u_cell / check.i_dc_a) * 1e3

    # --- family check on H --------------------------------------------------
    h_values = np.array(
        [c.shunt_H for c in checks.values() if np.isfinite(c.shunt_H)]
    )
    if h_values.size >= 4:
        h_median = float(np.median(h_values))
        for check in checks.values():
            if not np.isfinite(check.shunt_H) or h_median <= 0:
                continue
            deviation = abs(check.shunt_H - h_median) / h_median
            if deviation > limits.h_family_warn:
                check.raise_to(
                    WARN, f"shunt factor {deviation:.0%} from the family median "
                          f"({check.shunt_H:.3e} vs {h_median:.3e})"
                )

    if field_.degraded:
        for check in checks.values():
            check.raise_to(
                WARN, "H(T) evaluated at the fallback temperature; every current "
                      "density carries that systematic error"
            )

    summary = apply_checks(checks, cfg, limits, nominal_current_a, u_cell)
    return to_frame(checks), summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plausibility check for the local DC current density of every "
                    "segment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="YAML configuration file")
    p.add_argument("--raw-dir", help="directory holding the FAMOS .DAT files")
    p.add_argument("--measurement-id", default="", help="measurement / order ID")
    p.add_argument("--condition", help="operating point; default is all found")
    p.add_argument("--shunt-csv", help="per-segment c0;c1 file (required)")
    p.add_argument("--temp-csv", help="per-sensor c0;c1 file")
    p.add_argument("--segment-area", type=float, help="segment area [cm^2]")
    p.add_argument(
        "--nominal-current", default="auto",
        help="total cell current [A] for the sum check; 'auto' reads it from the "
             "condition label, 'none' disables the check",
    )
    p.add_argument("--out", help="write the per-segment CSV here")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--demo", action="store_true",
                   help="generate synthetic data and check that instead")

    g = p.add_argument_group("thresholds")
    g.add_argument("--j-min", type=float, help="lower plausible j [A/cm^2]")
    g.add_argument("--j-max", type=float, help="upper plausible j [A/cm^2]")
    g.add_argument("--uniformity-warn", type=float, nargs=2, metavar=("LO", "HI"))
    g.add_argument("--sum-tolerance", type=float,
                   help="relative tolerance on the current sum before WARN")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.demo:
        import tempfile
        from tests.synthetic import simulate_measurement

        raw = Path(args.out or tempfile.mkdtemp()) / "demo_raw"
        simulate_measurement(raw, measurement_id="DEMO", condition="150A",
                             n_cards=4, segments_per_card=5, duration_s=10.0)
        cfg.measurement_id, cfg.raw_dir = "DEMO", str(raw)
        cfg.calibration.shunt_csv = str(raw / "curr.csv")
        cfg.calibration.temperature_csv = str(raw / "temp.csv")
        print("synthetic demonstration: 4 cards x 5 segments at a 150 A operating "
              "point")

    if args.raw_dir:
        cfg.raw_dir = args.raw_dir
    if args.measurement_id:
        cfg.measurement_id = args.measurement_id
    if args.shunt_csv:
        cfg.calibration.shunt_csv = args.shunt_csv
    if args.temp_csv:
        cfg.calibration.temperature_csv = args.temp_csv
    if args.segment_area:
        cfg.geometry.segment_area_cm2 = args.segment_area

    if not cfg.raw_dir or cfg.raw_dir == ".":
        print("error: --raw-dir is required (or use --demo)", file=sys.stderr)
        return 2
    if not cfg.calibration.shunt_csv:
        print("error: --shunt-csv is required; without H(T) there is no current "
              "density, only a voltage", file=sys.stderr)
        return 2

    limits = Limits()
    if args.j_min is not None:
        limits.j_min = args.j_min
    if args.j_max is not None:
        limits.j_max = args.j_max
    if args.uniformity_warn:
        limits.uniformity_warn = tuple(args.uniformity_warn)   # type: ignore[assignment]
    if args.sum_tolerance is not None:
        limits.sum_warn = args.sum_tolerance

    found = discover_files(
        cfg.raw_dir, cfg.acquisition.discovery_regex, cfg.measurement_id or None
    )
    if not found:
        print(f"error: no files under {cfg.raw_dir} match the discovery pattern",
              file=sys.stderr)
        return 2
    conditions = [args.condition] if args.condition else sorted(found)

    worst = OK
    for condition in conditions:
        if condition not in found:
            print(f"  skipping {condition}: no files")
            continue

        if args.nominal_current == "auto":
            nominal = nominal_current_from_condition(condition)
            if nominal is None:
                print(f"  ! could not read a nominal current from {condition!r}; "
                      f"the sum check is skipped")
        elif args.nominal_current.lower() == "none":
            nominal = None
        else:
            nominal = float(args.nominal_current)

        frame, summary = run(cfg, condition, found[condition], limits, nominal)
        print_report(frame, summary, limits, condition, cfg.measurement_id or "?")

        for verdict in list(frame["verdict"]) + [summary.get("plate_verdict", OK)]:
            if _RANK.get(verdict, 0) > _RANK[worst]:
                worst = verdict                            # type: ignore[assignment]

        if args.out:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"current_density_{condition}.csv"
            frame.to_csv(path, index=False)
            print(f"\n  wrote {path}")
            if args.plots:
                for figure in make_plots(frame, summary, cfg, out_dir / condition):
                    print(f"  figure: {figure}")
        elif args.plots:
            for figure in make_plots(frame, summary, cfg, Path("./plausibility")):
                print(f"  figure: {figure}")

    print(f"\noverall: {worst}")
    return {OK: 0, WARN: 0, FAIL: 1}[worst]


if __name__ == "__main__":
    raise SystemExit(main())
