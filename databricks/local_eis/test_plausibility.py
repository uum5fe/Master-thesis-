"""Plausibility checks: do they catch the faults they claim to catch?

A check that passes everything is decoration, so every test here plants a
specific defect and asks whether the check that should see it does -- and,
just as important, whether the checks that should stay quiet do.
"""

from __future__ import annotations

import numpy as np
import pytest

import plausibility as P
import r2d2_geometry


PLATE = r2d2_geometry.plate("gen1")
AREAS = {int(k): s.area_cm2 for k, s in PLATE.segments.items()}
POS = {int(k): (s.cx_mm, s.cy_mm) for k, s in PLATE.segments.items()}
A_TOTAL = sum(AREAS.values())


# ---------------------------------------------------------------------------
# 36 of 72 is not half the plate
# ---------------------------------------------------------------------------

def test_the_two_36_segment_blocks_are_not_halves() -> None:
    """The whole reason this module exists.

    If they were halves this would be a much easier problem; they are 69 % and
    31 %, so which 36 you have decides how much the result can say.
    """
    first = P.coverage([str(n) for n in range(1, 37)])
    last = P.coverage([str(n) for n in range(37, 73)])

    assert first.value == pytest.approx(0.686, abs=0.005)
    assert last.value == pytest.approx(0.314, abs=0.005)
    assert first.value + last.value == pytest.approx(1.0, abs=1e-9)


def test_coverage_says_the_count_would_have_misled() -> None:
    """Both blocks are 50 % by count. Neither is 50 % by area."""
    c = P.coverage([str(n) for n in range(1, 37)])
    assert "50.0 %" in c.detail, "must show what counting would have claimed"
    assert "68.6 %" in c.detail, "and what the area actually is"


def test_the_perimeter_block_is_flagged_as_the_weaker_one() -> None:
    interior = P.describe_block([str(n) for n in range(1, 37)])
    perimeter = P.describe_block([str(n) for n in range(37, 73)])

    assert interior.verdict == P.PASS
    assert "INTERIOR" in interior.detail
    assert perimeter.verdict == P.WARN
    assert "PERIMETER" in perimeter.detail


def test_a_scattered_set_is_not_reported_as_a_wiring_block() -> None:
    """What the current runs actually produce: whatever survived the gates.

    That is a different situation from a cabled block and must not be
    described as one.
    """
    scattered = [str(n) for n in (1, 2, 5, 9, 13, 21, 34, 55, 60, 71)]
    c = P.describe_block(scattered)
    assert c.verdict == P.WARN
    assert "whatever survived" in c.detail


def test_coverage_fails_rather_than_warns_when_there_is_almost_nothing() -> None:
    assert P.coverage([str(n) for n in range(37, 45)]).verdict == P.FAIL
    assert P.coverage([]).verdict == P.FAIL


# ---------------------------------------------------------------------------
# current closure
# ---------------------------------------------------------------------------

def _uniform_j(segments, j=0.5):
    return {str(n): j for n in segments}


def test_a_uniform_plate_closes_on_the_setpoint() -> None:
    """Extrapolating by AREA from the interior block must land on the load."""
    segs = list(range(1, 37))
    j = 0.5                                    # A/cm2
    setpoint = j * A_TOTAL
    c = P.current_closure(_uniform_j(segs, j),
                          {str(n): AREAS[n] for n in segs}, setpoint)
    assert c.verdict == P.PASS
    assert abs(c.value) < 1e-9


def test_extrapolating_by_segment_count_would_have_failed_here() -> None:
    """The trap this check is built to avoid.

    Counting says the interior block is half the plate, so it scales by 72/36
    = 2.0 where the truth is 1/0.686 = 1.46. Counting therefore OVER-states
    the full-plate current by 37 % on a perfectly good measurement -- and on
    the perimeter block it errs the other way by the same mechanism.
    """
    segs = list(range(1, 37))
    j = 0.5
    i_meas = sum(j * AREAS[n] for n in segs)
    by_area = i_meas * A_TOTAL / sum(AREAS[n] for n in segs)
    by_count = i_meas * 72 / 36

    assert by_area == pytest.approx(j * A_TOTAL, rel=1e-9)
    assert by_count / by_area == pytest.approx(1.371, abs=0.01), (
        "counting over-states the interior block's full-plate current by 37 %")

    # the perimeter block, same error, opposite sign
    peri = list(range(37, 73))
    i_peri = sum(j * AREAS[n] for n in peri)
    peri_area = i_peri * A_TOTAL / sum(AREAS[n] for n in peri)
    peri_count = i_peri * 72 / 36
    assert peri_count / peri_area == pytest.approx(0.629, abs=0.01), (
        "and under-states the perimeter block's by 37 %")


def test_a_calibration_error_is_caught() -> None:
    segs = list(range(1, 37))
    j = 0.5
    setpoint = j * A_TOTAL
    # every shunt reading 40 % high
    c = P.current_closure(_uniform_j(segs, 1.4 * j),
                          {str(n): AREAS[n] for n in segs}, setpoint)
    assert c.verdict == P.FAIL
    assert c.value == pytest.approx(0.40, abs=0.01)


def test_no_setpoint_is_not_a_pass() -> None:
    """Nothing to compare against means the check did not run, not that it
    succeeded -- the distinction a silent `return True` would destroy."""
    segs = list(range(1, 37))
    c = P.current_closure(_uniform_j(segs), {str(n): AREAS[n] for n in segs},
                          None)
    assert c.verdict == P.NA
    assert "--i-setpoint" in c.detail


def test_the_closure_check_names_its_assumption() -> None:
    segs = list(range(37, 73))
    c = P.current_closure(_uniform_j(segs), {str(n): AREAS[n] for n in segs},
                          0.5 * A_TOTAL)
    assert "same mean current density" in c.rests_on


# ---------------------------------------------------------------------------
# the shape checks: what a wrong channel map cannot fake
# ---------------------------------------------------------------------------

def _smooth_field(segments, seed=0, noise=0.002):
    """R_ohmic falling along x, as a hydrating membrane gives."""
    rng = np.random.default_rng(seed)
    return {str(n): 0.12 - 0.0002 * POS[n][0] + noise * rng.standard_normal()
            for n in segments}


def test_a_correct_map_reads_as_spatially_organised() -> None:
    c = P.neighbour_smoothness(_smooth_field(range(1, 37)))
    assert c.verdict == P.PASS
    assert c.value < 0.6


def test_a_scrambled_channel_map_is_caught() -> None:
    """The failure this check exists for.

    Every point passes every gate in silver; only the arrangement is wrong,
    and nothing that looks at one segment at a time can see it.
    """
    segs = list(range(1, 37))
    field = _smooth_field(segs)
    values = list(field.values())
    rng = np.random.default_rng(3)
    rng.shuffle(values)
    scrambled = dict(zip(field, values))

    c = P.neighbour_smoothness(scrambled)
    assert c.verdict == P.FAIL
    assert "channel-to-segment mapping" in c.detail


def test_smoothness_needs_enough_segments_to_mean_anything() -> None:
    c = P.neighbour_smoothness({"1": 0.1, "2": 0.11, "3": 0.1})
    assert c.verdict == P.NA


def test_a_flat_map_is_flagged_not_passed() -> None:
    """A plate with a real humidity gradient should show one; no gradient at
    all is more likely a geometry or flow-direction mistake than physics."""
    flat = {str(n): 0.1 + 1e-6 * (n % 3) for n in range(1, 37)}
    c = P.flow_trend(flat)
    assert c.verdict == P.WARN
    assert "essentially flat" in c.detail


def test_a_real_gradient_is_reported_with_its_direction() -> None:
    c = P.flow_trend(_smooth_field(range(1, 37)))
    assert c.verdict == P.PASS
    assert "falls" in c.detail and c.value < -0.5


# ---------------------------------------------------------------------------
# passivity
# ---------------------------------------------------------------------------

class _Spec:
    def __init__(self, Z):
        self.Z_corr = np.asarray(Z, complex)


def test_a_passive_cell_passes() -> None:
    f = np.logspace(-1, 3, 20)
    Z = 0.06 + 0.3 / (1 + 1j * 2 * np.pi * f * 2e-3)
    assert P.passivity({"1": _Spec(Z), "2": _Spec(Z)}).verdict == P.PASS


def test_a_flipped_sign_is_caught() -> None:
    f = np.logspace(-1, 3, 20)
    Z = 0.06 + 0.3 / (1 + 1j * 2 * np.pi * f * 2e-3)
    c = P.passivity({"1": _Spec(Z), "7": _Spec(-Z)})
    assert c.verdict == P.FAIL
    assert "7" in c.detail


# ---------------------------------------------------------------------------
# the aggregate against a whole-cell sweep
# ---------------------------------------------------------------------------

def _cell(freq, r_ohm=0.06, r_ct=0.30, tau=2e-3, n=0.85):
    w = 2 * np.pi * np.asarray(freq, float)
    return r_ohm + r_ct / (1.0 + (1j * w * tau) ** n)


def test_an_area_specific_aggregate_needs_no_rescaling() -> None:
    """The point that makes a partial plate comparable at all.

    Both sides are already ohm.cm2, so a 69 %-coverage aggregate of a uniform
    plate matches the whole-cell sweep exactly -- no factor of 1/0.686
    anywhere.
    """
    f = np.logspace(-0.5, 3, 30)
    c = P.aggregate_vs_reference(f, _cell(f), f, _cell(f), area_fraction=0.686)
    assert c.verdict == P.PASS
    assert abs(c.value) < 1e-9


def test_the_tolerance_widens_as_the_coverage_falls() -> None:
    """A third of the plate is allowed to disagree more than two thirds is.

    Same 25 % deviation, two coverages, two verdicts -- otherwise the check
    would either be too strict on the perimeter block or too slack on the
    interior one.
    """
    f = np.logspace(-0.5, 3, 30)
    dev = 1.25
    wide = P.aggregate_vs_reference(f, dev * _cell(f), f, _cell(f), 0.314)
    narrow = P.aggregate_vs_reference(f, dev * _cell(f), f, _cell(f), 0.686)

    assert wide.verdict == P.PASS, "31 % coverage tolerates 25 %"
    assert narrow.verdict != P.PASS, "69 % coverage should not"


def test_a_gross_area_error_fails_at_any_coverage() -> None:
    f = np.logspace(-0.5, 3, 30)
    for frac in (0.314, 0.686, 1.0):
        c = P.aggregate_vs_reference(f, 3.0 * _cell(f), f, _cell(f), frac)
        assert c.verdict == P.FAIL, f"coverage {frac}"


def test_no_overlap_is_reported_as_not_applicable() -> None:
    lo = np.logspace(-1, 0, 10)
    hi = np.logspace(3, 4, 10)
    c = P.aggregate_vs_reference(lo, _cell(lo), hi, _cell(hi), 0.686)
    assert c.verdict == P.NA
    assert "no overlap" in c.detail


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def test_a_report_is_only_as_good_as_its_worst_check() -> None:
    rep = P.Report([
        P.Check("a", P.PASS, ""), P.Check("b", P.WARN, ""),
        P.Check("c", P.PASS, ""),
    ])
    assert rep.verdict == P.WARN
    rep.checks.append(P.Check("d", P.FAIL, ""))
    assert rep.verdict == P.FAIL


def test_not_applicable_outranks_pass_but_not_warn() -> None:
    """An unevaluated check must not be able to make a run look clean, and
    must not be able to make a warned run look worse than it is."""
    assert P.Report([P.Check("a", P.PASS, ""),
                     P.Check("b", P.NA, "")]).verdict == P.NA
    assert P.Report([P.Check("a", P.WARN, ""),
                     P.Check("b", P.NA, "")]).verdict == P.WARN


def test_an_empty_report_is_not_a_pass() -> None:
    assert P.Report([]).verdict == P.NA
