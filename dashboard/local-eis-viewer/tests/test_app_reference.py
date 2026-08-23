"""The whole-cell check, as the tab reads it.

The tab renders from the CSVs the pipeline writes, so these tests write those
CSVs the way the pipeline does and ask whether the screen says the true thing
about them -- including when the answer is "this cannot be measured here".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from app.views import reference


def _pipeline():
    root = Path(__file__).resolve().parents[1] / "local_eis"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gamry_compare
    return gamry_compare


A_CELL = 304.92


def _cell(freq, r_ohm=60e-3, r_ct=300e-3, tau=2e-3, n=0.85, L=2.6e-7):
    w = 2 * np.pi * np.asarray(freq, float)
    return 1j * w * L + r_ohm + r_ct / (1.0 + (1j * w * tau) ** n)


@pytest.fixture()
def written(tmp_path):
    """Run the real comparison and let it write its real output files."""
    GC = _pipeline()
    freq = np.logspace(np.log10(0.3), np.log10(30000), 61)
    sweeps = []
    for amps, scale in ((60, 1.00), (450, 1.05)):
        sweeps.append(GC.CellSweep(
            path=Path(f"ref_CurrVal_{amps}.dta"), freq=freq,
            Z_ohm=_cell(freq) / A_CELL, current_a=float(amps), started=None))
    # 60 A: identical.  450 A: 5 % high, and cut off below the HFR crossing.
    comps = [
        GC.compare(freq, _cell(freq), sweeps[0], A_CELL),
        GC.compare(freq[freq <= 200], 1.05 * _cell(freq[freq <= 200]),
                   sweeps[1], A_CELL),
    ]
    GC.write_outputs(comps, tmp_path)
    return tmp_path


def test_the_pipeline_writes_what_the_tab_reads(written):
    assert (written / "gamry_comparison.csv").is_file()
    assert (written / "gamry_comparison_curves.csv").is_file()

    rows = reference._read_summary(written)
    assert [r["condition"] for r in rows] == ["60A", "450A"]

    curves = reference._read_curves(written)
    assert set(curves) == {"60A", "450A"}
    for d in curves.values():
        assert d["f"].size == d["zl"].size == d["zr"].size > 0


def test_an_exact_match_reads_as_no_difference(written):
    rows = {r["condition"]: r for r in reference._read_summary(written)}
    assert abs(float(rows["60A"]["mag_rel_median_pct"])) < 0.1
    assert abs(float(rows["60A"]["phase_diff_max_deg"])) < 0.1
    # and the HFR agrees, because that band reaches the crossing
    assert abs(float(rows["60A"]["hfr_rel_pct"])) < 1.0


def test_a_scale_error_is_reported_as_a_magnitude_difference(written):
    rows = {r["condition"]: r for r in reference._read_summary(written)}
    assert float(rows["450A"]["mag_rel_median_pct"]) == pytest.approx(5.0, abs=0.2)
    assert abs(float(rows["450A"]["phase_diff_max_deg"])) < 0.1


def test_an_unmeasurable_hfr_is_shown_as_a_dash_not_a_number(written):
    """The 450 A band stops at 200 Hz, below the intercept.

    A table that printed the low-frequency crossing here would be quoting the
    polarisation resistance as the ohmic one. The tab has to show that the
    number does not exist, and say why.
    """
    rows = {r["condition"]: r for r in reference._read_summary(written)}
    assert reference._fmt(rows["450A"]["hfr_local_mohm_cm2"]) == "—"
    assert reference._fmt(rows["450A"]["hfr_rel_pct"], 1) == "—"
    assert "above the evaluated band" in rows["450A"]["notes"]
    # the reference itself still has one, because its sweep does reach it
    assert reference._fmt(rows["450A"]["hfr_ref_mohm_cm2"]) != "—"


def test_the_tab_says_what_to_do_when_no_comparison_was_run():
    """An empty tab must name the flag that fills it."""
    assert "--gamry" in reference._HINT
    assert "--bench-log" in reference._HINT
