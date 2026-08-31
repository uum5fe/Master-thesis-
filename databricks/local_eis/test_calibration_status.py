"""Was the calibration actually applied, or only passed on the command line?

Every way this can be half-done is silent and every one of them changes the
numbers, so the answer is computed and written into the manifest rather than
inferred from the presence of a file.
"""

from __future__ import annotations

import numpy as np
import pytest

import bronze as B
import r2d2_geometry as geom
from config import DEFAULT
from eis_local import PlateCalibration, T_FALLBACK_C


def full_cal(n=None, n_temp=4):
    n = n or geom.N_SEGMENTS
    return PlateCalibration(
        seg_c0={str(i): 0.5 for i in range(1, n + 1)},
        seg_c1={str(i): 1.0 for i in range(1, n + 1)},
        temp_c0={f"temp{i}": 0.0 for i in range(1, n_temp + 1)},
        temp_c1={f"temp{i}": 0.01 for i in range(1, n_temp + 1)})


def segments_at(t):
    return {s: t for s in geom.SEGMENTS}


def test_a_complete_calibration_reports_complete() -> None:
    rep = B.calibration_report(
        full_cal(), DEFAULT.replace(curr_cal="curr.csv", temp_cal="temp.csv"),
        sensor_T={"temp1": 58.0, "temp2": 61.0}, T_seg=segments_at(59.5))
    assert rep.current_ok and rep.temperature_ok
    assert not rep.problems
    assert rep.summary()["temperature"]["source"] == "measured"


def test_a_short_current_file_names_the_imputed_segments() -> None:
    """Missing rows do not drop the segment -- they silently rescale it.

    K is a scalar, so the SHAPE of those spectra is untouched and nothing
    looks wrong; only the absolute level moves, and only on some segments.
    A map then mixes measured and imputed levels with nothing on the figure
    to say which is which.
    """
    rep = B.calibration_report(
        full_cal(n=60), DEFAULT.replace(curr_cal="curr.csv"),
        sensor_T={"temp1": 58.0}, T_seg=segments_at(58.0))
    assert not rep.current_ok
    assert rep.summary()["current"]["segments_on_plate_median"] == (
        geom.N_SEGMENTS - 60)
    assert any("plate-median" in p for p in rep.problems)


def test_a_missing_current_calibration_is_a_problem_not_a_note() -> None:
    """It is the only absolute scale left once the potentiostat is gone."""
    cal = PlateCalibration({}, {}, {"temp1": 0.0}, {"temp1": 0.01})
    rep = B.calibration_report(cal, DEFAULT, {"temp1": 58.0},
                               segments_at(58.0))
    assert rep.problems and not rep.current_ok
    assert "absolute scale" in rep.problems[0]


def test_no_temperature_file_is_reported_as_the_fallback_constant() -> None:
    rep = B.calibration_report(
        full_cal(n_temp=0), DEFAULT.replace(curr_cal="curr.csv"),
        sensor_T={}, T_seg=segments_at(T_FALLBACK_C))
    assert not rep.temperature_ok
    assert rep.summary()["temperature"]["source"] == "fallback constant"
    assert any("fallback constant" in n for n in rep.notes)


def test_a_loaded_temperature_cal_with_no_usable_sensor_is_a_problem() -> None:
    """The worst case: the file is there, so it looks calibrated, and is not.

    This is what a channel-name mismatch produces -- 'Temp_1' against a
    calibration keyed 'temp1' -- and the run continues on the fallback
    constant with a temp-cal path printed in the header.
    """
    rep = B.calibration_report(
        full_cal(), DEFAULT.replace(curr_cal="curr.csv", temp_cal="temp.csv"),
        sensor_T={}, T_seg=segments_at(T_FALLBACK_C),
        rejected={"Temp_1": "no calibration row for 'temp1'"})
    assert not rep.temperature_ok
    assert any("every sensor" in p or "NO sensor" in p for p in rep.problems)
    assert rep.summary()["temperature"]["sensors_rejected"]["Temp_1"]


def test_the_gradient_is_reported_because_that_is_what_it_is_for() -> None:
    t = dict(zip(geom.SEGMENTS, np.linspace(55.0, 65.0, geom.N_SEGMENTS)))
    rep = B.calibration_report(
        full_cal(), DEFAULT.replace(curr_cal="c.csv", temp_cal="t.csv"),
        sensor_T={"temp1": 55.0, "temp4": 65.0}, T_seg=t)
    assert rep.t_min_c == pytest.approx(55.0)
    assert rep.t_max_c == pytest.approx(65.0)


def test_the_report_round_trips_to_the_manifest() -> None:
    import json
    rep = B.calibration_report(
        full_cal(), DEFAULT.replace(curr_cal="c.csv", temp_cal="t.csv"),
        sensor_T={"temp1": 58.0}, T_seg=segments_at(58.0))
    back = json.loads(json.dumps(rep.summary()))
    assert back["current"]["complete"] is True
