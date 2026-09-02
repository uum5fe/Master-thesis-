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


# ---------------------------------------------------------------------------
# finding the comparison for the selected run
# ---------------------------------------------------------------------------
#
# The tests above read the CSVs directly, which left the step that turns a
# SELECTION into a folder untested -- and that is exactly where the tab
# crashed in the field, on a RunRef attribute that does not exist. These drive
# the real callback through a real catalogue so that path cannot rot again.


def _results_tree(tmp_path):
    """A minimal but real results layout: <root>/<id>/<condition>/silver.

    The root is a subdirectory of tmp_path, never tmp_path itself: layout
    detection claims a folder that merely contains CSVs, so a stray file at
    the root would be read as one big run and the per-condition folders would
    never be reached.
    """
    run = tmp_path / "results" / "2611976" / "45A"
    (run / "silver").mkdir(parents=True)
    (run / "gold").mkdir(parents=True)
    import pandas as pd
    pd.DataFrame({"freq_hz": [1.0, 10.0],
                  "z_re_mohm_cm2": [100.0, 80.0],
                  "z_im_mohm_cm2": [-10.0, -20.0]}).to_csv(
        run / "silver" / "cell_aggregate.csv", index=False)
    pd.DataFrame({"segment": ["1"], "R_ohmic": [0.06]}).to_csv(
        run / "gold" / "plate_summary.csv", index=False)
    return run


def _catalog_for(tmp_path, monkeypatch):
    from app.data.sources import Catalog
    from app.settings import Settings
    from app.views import reference as ref_view

    catalog = Catalog(Settings(results_roots=[str(tmp_path / "results")],
                               famos_roots=[], csv_roots=[])).refresh(
        kinds=("results",))
    monkeypatch.setattr(ref_view.store, "current_catalog", lambda: catalog)
    return catalog


def test_the_selected_run_resolves_to_its_comparison_folder(tmp_path,
                                                            monkeypatch):
    run = _results_tree(tmp_path)
    catalog = _catalog_for(tmp_path, monkeypatch)
    assert catalog.runs, "the results tree was not discovered"
    sel = {"kind": "results", "measurement_id": "2611976", "condition": "45A"}

    # Nothing written yet: no folder, and no exception.
    assert reference._comparison_dir(sel) is None

    (run / "gamry_comparison.csv").write_text("condition\n45A\n")
    assert reference._comparison_dir(sel) == run


def test_a_campaign_level_comparison_is_found_from_a_condition_run(
        tmp_path, monkeypatch):
    """The pipeline may write one comparison for the whole campaign."""
    run = _results_tree(tmp_path)
    _catalog_for(tmp_path, monkeypatch)
    (run.parent / "gamry_comparison.csv").write_text("condition\n45A\n")
    sel = {"kind": "results", "measurement_id": "2611976", "condition": "45A"}
    assert reference._comparison_dir(sel) == run.parent


def test_the_tab_renders_without_a_comparison_instead_of_raising(
        tmp_path, monkeypatch, written):
    """The regression: the callback must survive a run that has no comparison.

    It previously reached for a RunRef field that does not exist, so selecting
    any run at all raised AttributeError and the tab showed a Dash error
    instead of the hint telling you which flag to pass.
    """
    _results_tree(tmp_path)
    _catalog_for(tmp_path, monkeypatch)
    sel = {"kind": "results", "measurement_id": "2611976", "condition": "45A"}

    status, nyq, res, table = reference.render(sel)
    assert table is None
    assert not nyq.data and not res.data
    # And the message is actionable: it names one of the two ways to get a
    # comparison, rather than only reporting that there is not one. Which of
    # them depends on how far the run got, so either is acceptable.
    text = str(status)
    assert "--gamry" in text or "EIS_GAMRY_ROOT" in text, text[:200]

    # And an empty selection is not a crash either.
    assert reference.render({})[3] is None
    assert reference.render(None)[3] is None


def test_the_tab_draws_the_comparison_that_is_there(tmp_path, monkeypatch,
                                                    written):
    """With the files present, both figures and the table come back."""
    import shutil
    run = _results_tree(tmp_path)
    for name in ("gamry_comparison.csv", "gamry_comparison_curves.csv"):
        shutil.copy(written / name, run / name)
    _catalog_for(tmp_path, monkeypatch)
    sel = {"kind": "results", "measurement_id": "2611976", "condition": "45A"}

    _status, nyq, res, table = reference.render(sel)
    assert table is not None
    # reference + local for each of the two conditions in the fixture
    assert len(nyq.data) == 4
    assert len(res.data) == 4
    assert "Z′" in nyq.layout.xaxis.title.text


# ---------------------------------------------------------------------------
# computing the comparison instead of demanding a re-run
# ---------------------------------------------------------------------------

def _run_with_aggregate_and_sweeps(tmp_path):
    """A finished run, and sweeps on a different branch. No comparison CSV."""
    import pandas as pd
    GC = _pipeline()

    results = tmp_path / "Daten" / "EIS_Results" / "2611976" / "45A"
    gamry = tmp_path / "EIS_Daten_Gamry_Tom"
    (results / "silver").mkdir(parents=True)
    (results / "gold").mkdir(parents=True)
    gamry.mkdir(parents=True)

    pd.DataFrame({"segment": ["1"], "R_ohmic": [0.06]}).to_csv(
        results / "gold" / "plate_summary.csv", index=False)

    freq = np.logspace(np.log10(0.3), np.log10(4500), 45)
    Z = _cell(freq)
    pd.DataFrame({"freq_hz": freq,
                  "z_re_mohm_cm2": 1e3 * Z.real,
                  "z_im_mohm_cm2": 1e3 * Z.imag}).to_csv(
        results / "silver" / "cell_aggregate.csv", index=False)

    wide = np.logspace(np.log10(0.3), np.log10(30000), 61)
    _write_gamry(gamry / "cell_CurrVal_45.dta", wide, _cell(wide) / A_CELL)
    return results, gamry


def _write_gamry(path, freq, Z_ohm):
    head = ["EXPLAIN", "TAG\tGALVEIS",
            "STARTTIME\tLABEL\t16.07.2026 07:47:03\tStart Time",
            "ZCURVE\tTABLE",
            "\tPt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\tIdc\tVdc\tIERange",
            "\t#\ts\tHz\tohm\tohm\tV\tohm\tdeg\tA\tV\t#"]
    g = lambda v: f"{v:.6E}".replace(".", ",")
    for i, (f, z) in enumerate(zip(freq, Z_ohm), start=1):
        head.append("\t".join(["", str(i), "1", g(f), g(z.real), g(z.imag), "1",
                               g(abs(z)), g(np.degrees(np.angle(z))),
                               g(4e-3), g(0.77), "13"]))
    path.write_text("\n".join(head) + "\n", encoding="latin-1")


def test_the_comparison_is_computed_when_none_was_written(tmp_path,
                                                          monkeypatch):
    """A run made without --gamry still has everything the comparison needs.

    Re-processing gigabytes to obtain numbers that take a fraction of a second
    to compute is not a reasonable thing to ask for.
    """
    from app.data.sources import Catalog
    from app.settings import Settings

    results, gamry = _run_with_aggregate_and_sweeps(tmp_path)
    assert not (results / "gamry_comparison.csv").exists()

    settings = Settings(results_roots=[str(tmp_path / "Daten" / "EIS_Results")],
                        gamry_roots=[str(gamry)], famos_roots=[], csv_roots=[])
    catalog = Catalog(settings).refresh(kinds=("results",))
    monkeypatch.setattr(reference.store, "current_catalog", lambda: catalog)
    monkeypatch.setattr(reference, "SETTINGS", settings)

    sel = {"kind": "results", "measurement_id": "2611976", "condition": "45A",
           "plate_key": "gen1_r2d2_72"}
    status, nyq, res, table = reference.render(sel)

    assert table is not None
    assert len(nyq.data) == 2                    # reference and local
    assert "computed now" in str(status)
    # identical physics in, so the difference must be nil
    rows, _curves, _problem = reference._live_comparison(sel)
    assert abs(float(rows[0]["mag_rel_median_pct"])) < 0.5


def test_a_missing_sweep_for_this_condition_says_which_ones_exist(tmp_path,
                                                                  monkeypatch):
    from app.data.sources import Catalog
    from app.settings import Settings

    results, gamry = _run_with_aggregate_and_sweeps(tmp_path)
    (gamry / "cell_CurrVal_45.dta").rename(gamry / "cell_CurrVal_450.dta")

    settings = Settings(results_roots=[str(tmp_path / "Daten" / "EIS_Results")],
                        gamry_roots=[str(gamry)], famos_roots=[], csv_roots=[])
    catalog = Catalog(settings).refresh(kinds=("results",))
    monkeypatch.setattr(reference.store, "current_catalog", lambda: catalog)
    monkeypatch.setattr(reference, "SETTINGS", settings)

    _rows, _curves, problem = reference._live_comparison(
        {"kind": "results", "measurement_id": "2611976", "condition": "45A",
         "plate_key": "gen1_r2d2_72"})
    assert "No sweep at 45A" in problem
    assert "450A" in problem, "it must say what it did find"


def test_an_extrapolated_hfr_is_shown_but_marked_as_one():
    """Several delivered sweeps stop below the real-axis crossing.

    A dash in every HFR cell of the whole campaign tells the reader nothing;
    the extrapolation tells them something, as long as it never pretends to be
    the measurement.
    """
    cells = reference._hfr_cells({
        "hfr_local_mohm_cm2": "nan", "hfr_ref_mohm_cm2": "nan",
        "hfr_rel_pct": "nan",
        "hfr_local_fit_mohm_cm2": "59.90", "hfr_ref_fit_mohm_cm2": "57.05",
        "hfr_fit_rel_pct": "5.0"})
    assert cells == {"HFR local": "~59.90", "HFR ref": "~57.05",
                     "ΔHFR [%]": "~5.0"}


def test_a_measured_hfr_is_never_overwritten_by_the_extrapolation():
    cells = reference._hfr_cells({
        "hfr_local_mohm_cm2": "62.26", "hfr_ref_mohm_cm2": "59.30",
        "hfr_rel_pct": "5.0",
        "hfr_local_fit_mohm_cm2": "59.90", "hfr_ref_fit_mohm_cm2": "57.05",
        "hfr_fit_rel_pct": "4.9"})
    assert cells == {"HFR local": "62.26", "HFR ref": "59.30",
                     "ΔHFR [%]": "5.0"}


def test_a_measured_reference_is_not_compared_against_an_extrapolation():
    """Only the local side lacks an intercept -- the campaign's 45 A case.

    Printing the reference's measured 71.53 beside the local extrapolation
    would charge the extrapolation's bias to the local side and read as a real
    disagreement, so the row switches to the like-for-like pair together.
    """
    cells = reference._hfr_cells({
        "hfr_local_mohm_cm2": "nan", "hfr_ref_mohm_cm2": "71.53",
        "hfr_rel_pct": "nan",
        "hfr_local_fit_mohm_cm2": "68.40", "hfr_ref_fit_mohm_cm2": "69.10",
        "hfr_fit_rel_pct": "-1.0"})
    assert cells["HFR ref"] == "~69.10", "not the measured 71.53"
    assert cells["HFR local"] == "~68.40"
    assert cells["ΔHFR [%]"] == "~-1.0"


def test_a_comparison_csv_without_the_fit_columns_still_renders():
    """Results written before the fallback existed have no such column."""
    cells = reference._hfr_cells({"hfr_local_mohm_cm2": "nan",
                                  "hfr_ref_mohm_cm2": "nan",
                                  "hfr_rel_pct": "nan"})
    assert set(cells.values()) == {"—"}


def test_the_selected_plate_is_the_one_actually_used(tmp_path, monkeypatch):
    """`selection` carries the key as "plate_key" (see app.py's `_selection`
    callback) -- this tab used to read `selection.get("plate")`, which is
    never present, so it silently fell back to registry.default_key() no
    matter which plate the sidebar had selected. That is invisible on this
    campaign, where gen1 and gen2 happen to share one total area, but
    operating.py reads the SAME key for per-segment centroids, where the two
    plates genuinely differ -- so it is worth pinning down here.
    """
    results, gamry = _run_with_aggregate_and_sweeps(tmp_path)
    from app.data.sources import Catalog
    from app.settings import Settings

    settings = Settings(results_roots=[str(tmp_path / "Daten" / "EIS_Results")],
                        gamry_roots=[str(gamry)], famos_roots=[], csv_roots=[])
    catalog = Catalog(settings).refresh(kinds=("results",))
    monkeypatch.setattr(reference.store, "current_catalog", lambda: catalog)
    monkeypatch.setattr(reference, "SETTINGS", settings)

    seen: list[str] = []
    real_get = reference.registry.get
    monkeypatch.setattr(reference.registry, "get",
                        lambda key: seen.append(key) or real_get(key))

    reference._live_comparison(
        {"kind": "results", "measurement_id": "2611976", "condition": "45A",
         "plate_key": "gen2_r2d2_naboo_72"})
    assert seen == ["gen2_r2d2_naboo_72"], (
        "the plate the sidebar selected must be the one looked up, not "
        "always the registry default")


# ---------------------------------------------------------------------------
# matching a run to its reference sweep
# ---------------------------------------------------------------------------

class _Sweep:
    def __init__(self, current_a: float) -> None:
        self.current_a = float(current_a)

    @property
    def condition(self) -> str:
        i = int(round(self.current_a))
        return (f"{i}A" if abs(self.current_a - i) < 1e-9
                else f"{self.current_a:g}A")


SWEEPS = [_Sweep(45), _Sweep(60), _Sweep(150), _Sweep(450)]


def test_the_exact_condition_still_matches():
    assert reference.match_sweep(SWEEPS, "150A").current_a == 150.0


def test_a_processing_suffix_does_not_hide_the_setpoint():
    """Splitting a mixed-rate plate names the folder 45A_percard.

    It is the same cell at the same 45 A; the suffix records how the LOCAL
    data was processed, which the reference instrument knows nothing about.
    """
    for label in ("45A_percard", "45A_g1_50kHz", "45A_100kHz", "45 A"):
        match = reference.match_sweep(SWEEPS, label)
        assert match is not None and match.current_a == 45.0, label


def test_450A_is_not_matched_to_the_45A_sweep():
    """The setpoint is parsed as a number, never compared as a prefix."""
    assert reference.match_sweep(SWEEPS, "450A_percard").current_a == 450.0
    assert reference.match_sweep([_Sweep(45)], "450A_percard") is None


def test_a_setpoint_with_no_sweep_is_refused_not_approximated():
    assert reference.match_sweep(SWEEPS, "300A_10kHz") is None
    assert reference.match_sweep(SWEEPS, "OCV") is None
    assert reference.match_sweep([], "45A") is None
