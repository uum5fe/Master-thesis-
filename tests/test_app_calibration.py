r"""Reading and evaluating a calibration campaign.

The fixtures are written in the exact shape of the real Step files - the
``s12:`` prefix, tab separators, ``temp<n>=<V>V`` and repeated
``i_s=<A>;u_s=<V>`` pairs, CRLF line endings - so a change to the parser that
would break on the real data breaks here instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data import calibration as cal          # noqa: E402
from app.data.sources import CalibrationSource   # noqa: E402

#: The chain constant: shipped coefficient = through-origin sensitivity / K.
K = 3.0


def _write_campaign(root: Path, name: str, temps=(20, 60, 20),
                    n_segments: int = 4, coefficients: bool = True) -> Path:
    """A campaign whose true sensitivity is known, so the maths can be checked."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    # H(T) = h0 + h1*T, deliberately different per segment.
    h0 = {s: 1.0 + 0.1 * s for s in range(1, n_segments + 1)}
    h1 = {s: 0.004 + 0.0002 * s for s in range(1, n_segments + 1)}
    # temp.csv: V = c0 + c1*T  ->  T = (V - c0)/c1
    tc = [(3.0, 0.0125), (2.9, 0.0120), (3.1, 0.0130), (3.2, 0.0135)]

    for index, temperature in enumerate(temps, start=1):
        lines = []
        for segment in range(1, n_segments + 1):
            volts = ";".join(f"temp{k+1}={c0 + c1 * temperature:.6f}V"
                             for k, (c0, c1) in enumerate(tc))
            H = h0[segment] + h1[segment] * temperature
            sweep = "\t".join(
                f"i_s={i:.6f}A;u_s={H * i:.6f}V"
                for i in (0.0, 0.25, 0.5, 0.75, 1.0))
            lines.append(f"s{segment}:\t{volts};\t{sweep};")
        (folder / f"Step{index}_{temperature}Grad.csv").write_bytes(
            ("\r\n".join(lines) + "\r\n").encode("latin-1"))

    if coefficients:
        coeff = folder / "coefficients"
        coeff.mkdir(exist_ok=True)
        coeff.joinpath("curr.csv").write_text("\n".join(
            f"{h0[s]/K:.6f};{h1[s]*1000/K:.6f}" for s in range(1, n_segments + 1)))
        coeff.joinpath("temp.csv").write_text("\n".join(
            f"{c0:.6f};{c1:.6f}" for c0, c1 in tc))
    return folder


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parses_the_real_line_shape(tmp_path):
    folder = _write_campaign(tmp_path, "Naboo")
    step = cal.parse_step(folder / "Step1_20Grad.csv")
    assert step.index == 1 and step.nominal_c == 20
    assert set(step.sweeps) == {"1", "2", "3", "4"}
    i, u = step.sweeps["1"]
    assert len(i) == 5 and i[0] == 0.0 and i[-1] == 1.0
    assert len(step.sensor_v["1"]) == 4


def test_a_negative_temperature_in_the_name_is_read(tmp_path):
    folder = _write_campaign(tmp_path, "Naboo", temps=(-40,))
    step = cal.parse_step(folder / "Step1_-40Grad.csv")
    assert step.nominal_c == -40


def test_grad_is_a_temperature_and_the_sensors_agree(tmp_path):
    """The label is degrees Celsius, not an angle - temp.csv proves it."""
    folder = _write_campaign(tmp_path, "Kashyyyk", temps=(20, 60, 90))
    run = cal.load_campaign(folder)
    for step in run.steps:
        assert step.measured_c == pytest.approx(step.nominal_c, abs=0.01)


# ---------------------------------------------------------------------------
# the fit convention
# ---------------------------------------------------------------------------

def test_sensitivity_is_fitted_through_the_origin():
    i = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    u = 2.0 * i + 0.5                       # a deliberate offset
    # A free straight line would return 2.0; through the origin it must not.
    assert cal.sensitivity(i, u) != pytest.approx(2.0, rel=1e-3)
    assert cal.sensitivity(i, u) == pytest.approx(float(np.sum(u * i) / np.sum(i * i)))


def test_a_perfect_sweep_has_no_linearity_error():
    i = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert cal.linearity_error_pct(i, 2.0 * i) == pytest.approx(0.0, abs=1e-9)


def test_a_kinked_sweep_is_caught():
    i = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    u = 2.0 * i
    u[2] *= 1.1                              # 10 % off at mid-scale
    assert cal.linearity_error_pct(i, u) > 3.0


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_reconstruction_recovers_the_known_coefficients(tmp_path):
    folder = _write_campaign(tmp_path, "Naboo", temps=(20, 60, 90))
    run = cal.load_campaign(folder)
    table = cal.segment_table(run).set_index("segment")

    for segment in ("1", "2", "3", "4"):
        s = int(segment)
        assert table.loc[segment, "H_c0"] == pytest.approx(1.0 + 0.1 * s, rel=1e-6)
        assert table.loc[segment, "H_c1"] == pytest.approx(
            (0.004 + 0.0002 * s) * 1000, rel=1e-6)
        # The ratio to the shipped coefficient is the chain constant, the same
        # for every segment - that constancy is the check.
        assert table.loc[segment, "ratio_c0"] == pytest.approx(K, rel=1e-6)


def test_the_chain_constant_is_reported_not_assumed(tmp_path):
    folder = _write_campaign(tmp_path, "Naboo", temps=(20, 60, 90))
    info = cal.summary(cal.load_campaign(folder))
    assert info["chain_constant"] == pytest.approx(K, rel=1e-6)
    assert info["coefficients_consistent"] is True


def test_one_mismatched_segment_breaks_the_consistency_check(tmp_path):
    folder = _write_campaign(tmp_path, "Naboo", temps=(20, 60, 90))
    coeff = folder / "coefficients" / "curr.csv"
    rows = coeff.read_text().splitlines()
    c0, c1 = rows[2].split(";")
    rows[2] = f"{float(c0) * 1.05:.6f};{c1}"          # segment 3 calibrated wrong
    coeff.write_text("\n".join(rows))

    run = cal.load_campaign(folder)
    table = cal.segment_table(run).set_index("segment")
    assert abs(table.loc["3", "ratio_c0_dev_pct"]) > 3.0
    assert cal.summary(run)["coefficients_consistent"] is False


def test_repeat_drift_is_zero_for_a_stable_plate(tmp_path):
    folder = _write_campaign(tmp_path, "Naboo", temps=(20, 60, 20))
    drift = cal.repeat_drift(cal.load_campaign(folder))
    assert not drift.empty
    assert drift["drift_pct"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_no_repeat_means_no_drift_table(tmp_path):
    folder = _write_campaign(tmp_path, "Naboo", temps=(20, 60, 90))
    assert cal.repeat_drift(cal.load_campaign(folder)).empty


def test_missing_coefficients_are_reported_not_fatal(tmp_path):
    folder = _write_campaign(tmp_path, "Bare", coefficients=False)
    run = cal.load_campaign(folder)
    assert run.steps and any("curr.csv" in w for w in run.warnings)
    table = cal.segment_table(run)
    assert "H_c0" in table.columns          # the reconstruction still works
    assert "ratio_c0" not in table.columns  # there is nothing to compare against


def test_step_table_shows_nominal_against_measured(tmp_path):
    folder = _write_campaign(tmp_path, "Naboo", temps=(20, 60))
    frame = cal.step_table(cal.load_campaign(folder))
    assert list(frame["nominal_C"]) == [20, 60]
    assert frame["offset_C"].abs().max() < 0.01
    assert "temp4_C" in frame.columns


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_campaigns_are_found_by_folder_name_without_an_order_id(tmp_path):
    _write_campaign(tmp_path / "R2D2_green", "Kashyyyk")
    _write_campaign(tmp_path / "R2D2_blue", "Naboo")
    refs = CalibrationSource([tmp_path]).scan()
    names = sorted(r.measurement_id for r in refs)
    assert names == ["Kashyyyk", "Naboo"]
    # No order id anywhere: a campaign belongs to a plate, not to an order.
    assert all(r.condition == "(calibration)" for r in refs)
    assert all("step" in r.describe() for r in refs)


def test_a_folder_without_steps_is_not_a_campaign(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("nothing here")
    assert CalibrationSource([tmp_path]).scan() == []
