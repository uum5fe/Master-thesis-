"""Loaders, discovery and the ECM service.

The fixtures are written in the shapes the real producers write, so a change to
a column name in either pipeline fails here rather than in the browser.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data import loaders
from app.data.sources import FamosSource, ResultsSource, _condition_sort_key
from app.services import ecm_service


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _synthetic_spectrum(segment: str, r_s: float = 80.0, r_ct: float = 250.0,
                        tau: float = 2e-3, n_points: int = 30):
    """A single-arc spectrum in mOhm*cm2, the units the app carries."""
    f = np.logspace(-1, 3.5, n_points)
    w = 2 * np.pi * f
    z = r_s + r_ct / (1 + 1j * w * tau)
    return pd.DataFrame({
        "segment": segment, "freq_hz": f,
        "z_re_mohm_cm2": z.real, "z_im_mohm_cm2": z.imag,
        "sigma_rel": 0.01,
    })


@pytest.fixture
def gold_silver_run(tmp_path):
    run = tmp_path / "2611976" / "45A"
    (run / "gold").mkdir(parents=True)
    (run / "silver").mkdir(parents=True)

    spectra = pd.concat([_synthetic_spectrum(s) for s in ("1", "2")],
                        ignore_index=True)
    spectra["zmodel_re_mohm_cm2"] = spectra["z_re_mohm_cm2"]
    spectra["zmodel_im_mohm_cm2"] = spectra["z_im_mohm_cm2"]
    spectra.to_csv(run / "silver" / "spectra_clean.csv", index=False)

    pd.DataFrame({
        "segment": ["1", "2", "33"],
        "class": ["measured", "measured", "bad"],
        "tier": ["A", "B", ""],
        "cx_mm": [14.0, 14.0, 0.0], "cy_mm": [15.1, 45.4, 0.0],
        "area_cm2": [5.082, 5.082, 0.678],
        "fault": ["", "flooding", ""],
        "flags": ["", "many_points_dropped", ""],
        # Empty strings are how the pipeline writes "not finite".
        "R_ohmic": [80.0, 92.0, ""],
        "R_ohmic_sd": [0.4, 0.6, ""],
        "R_ct": [250.0, 265.0, ""],
    }).to_csv(run / "gold" / "plate_summary.csv", index=False)

    pd.DataFrame({"freq_hz": [1.0, 10.0], "z_re_mohm_cm2": [300.0, 120.0],
                  "z_im_mohm_cm2": [-20.0, -40.0]}).to_csv(
        run / "silver" / "cell_aggregate.csv", index=False)

    (run / "gold" / "gold_manifest.json").write_text(json.dumps({
        "n_total": 72, "n_measured": 2, "n_inferred": 0, "n_bad": 1,
        "dc_closure": {"ok": True, "deviation_pct": -2.6},
    }))
    pd.DataFrame({"card": ["Karte_1"], "slot_us": [54.7], "applied": [1]}).to_csv(
        run / "silver" / "skew_model.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# layout detection and loading
# ---------------------------------------------------------------------------

def test_detects_the_gold_silver_layout(gold_silver_run):
    assert loaders.detect_layout(gold_silver_run / "2611976" / "45A") == "gold_silver"


def test_gold_silver_loads_into_canonical_columns(gold_silver_run):
    run = loaders.load(gold_silver_run / "2611976" / "45A", "2611976", "45A",
                       "gen1_r2d2_72")
    assert run.loader == "gold_silver"
    assert run.segment_names == ["1", "2", "33"]
    assert {"freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2"} <= set(run.spectra.columns)
    assert "z_re_model_mohm_cm2" in run.spectra.columns      # renamed from zmodel_*
    assert run.classes()["33"] == "bad"
    assert run.cell is not None and len(run.cell) == 2
    assert run.meta["gold"]["n_bad"] == 1


def test_empty_strings_become_missing_not_text(gold_silver_run):
    run = loaders.load(gold_silver_run / "2611976" / "45A")
    values = run.value("R_ohmic")
    assert values == {"1": 80.0, "2": 92.0}          # segment 33 dropped, not "" kept
    assert run.sd("R_ohmic")["1"] == pytest.approx(0.4)


def test_mappable_params_are_numeric_and_ordered(gold_silver_run):
    run = loaders.load(gold_silver_run / "2611976" / "45A")
    params = run.mappable_params()
    assert params[:2] == ["R_ohmic", "R_ct"]         # known parameters first
    assert "R_ohmic_sd" not in params                # uncertainties are not maps
    assert "fault" not in params                     # text is not a map


def test_eis_tables_layout(tmp_path):
    directory = tmp_path / "2611976"
    directory.mkdir()
    pd.DataFrame({
        "measurement_id": ["2611976"] * 4, "condition": ["45A", "45A", "60A", "60A"],
        "segment": ["1", "1", "1", "1"], "frequency_hz": [1.0, 10.0, 1.0, 10.0],
        "z_real_mohm_cm2": [300.0, 120.0, 310.0, 125.0],
        "z_imag_mohm_cm2": [-20.0, -40.0, -22.0, -41.0],
        "sigma_rel": [0.01] * 4,
    }).to_csv(directory / "impedance.csv", index=False)
    pd.DataFrame({
        "measurement_id": ["2611976"] * 2, "condition": ["45A", "60A"],
        "segment": ["1", "1"], "status": ["ok", "rejected"],
        "rs_hf_mohm_cm2": [80.0, 82.0], "temperature_c": [58.0, 59.0],
    }).to_csv(directory / "segments.csv", index=False)

    assert loaders.detect_layout(directory) == "eis_tables"
    run = loaders.load(directory, "2611976", "45A", "gen1_r2d2_72")
    assert len(run.spectra) == 2                      # the other condition filtered out
    assert run.value("R_ohmic") == {"1": 80.0}
    assert run.classes()["1"] == "measured"           # 'ok' translated


def test_generic_csv_column_matching(tmp_path):
    path = tmp_path / "export.csv"
    pd.DataFrame({"Segment": [1, 1], "Frequency": [1.0, 10.0],
                  "Z_real": [300.0, 120.0], "Z_imag": [-20.0, -40.0]}).to_csv(
        path, index=False)
    run = loaders.load_generic_csv(path)
    assert list(run.spectra["freq_hz"]) == [1.0, 10.0]
    assert run.spectra["z_re_mohm_cm2"].tolist() == [300.0, 120.0]


def test_unreadable_columns_are_reported_not_guessed(tmp_path):
    path = tmp_path / "odd.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    run = loaders.load_generic_csv(path)
    assert run.spectra.empty
    assert any("could not identify" in w for w in run.warnings)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_results_source_finds_measurement_and_condition(gold_silver_run):
    refs = ResultsSource([gold_silver_run]).scan()
    assert [(r.measurement_id, r.condition, r.layout) for r in refs] == \
        [("2611976", "45A", "gold_silver")]


def test_famos_discovery_groups_cards(tmp_path):
    for card in range(1, 6):
        (tmp_path / f"Leepa_2611976_Current_45A_Test_01_Karte_{card}.DAT").touch()
    (tmp_path / "Leepa_2611976_Current_450A_Test_01_Karte_1.DAT").touch()
    refs = FamosSource([tmp_path]).scan()
    by_condition = {r.condition: r for r in refs}
    assert set(by_condition) == {"45A", "450A"}
    assert len(by_condition["45A"].files) == 5


def test_order_id_naming_convention_also_matches(tmp_path):
    (tmp_path / "RO2611976-01_Current_45A_Test_01_Karte_1.DAT").touch()
    refs = FamosSource([tmp_path]).scan()
    assert refs and refs[0].measurement_id == "2611976"


def test_conditions_sort_by_current_not_alphabetically():
    assert sorted(["450A", "45A", "150A", "60A"], key=_condition_sort_key) == \
        ["45A", "60A", "150A", "450A"]


# ---------------------------------------------------------------------------
# ECM service
# ---------------------------------------------------------------------------

def test_fit_recovers_known_circuit_parameters():
    spectrum = _synthetic_spectrum("1", r_s=80.0, r_ct=250.0, tau=2e-3)
    result = ecm_service.fit_segment(spectrum, "1", model="R_L_1RQ")
    assert result.ok
    # Fitting runs in Ohm*cm2; resistances come back in mOhm*cm2 for display.
    table = result.table().set_index("parameter")["value"]
    assert table["Rs"] == pytest.approx(80.0, rel=0.02)
    assert table["Rct1"] == pytest.approx(250.0, rel=0.02)
    assert table["tau1"] == pytest.approx(2e-3, rel=0.10)
    assert table["R_pol"] == pytest.approx(250.0, rel=0.02)


def test_model_selection_does_not_buy_complexity_it_cannot_justify():
    spectrum = _synthetic_spectrum("1")
    result = ecm_service.fit_segment(spectrum, "1", model="auto")
    assert result.fit.model in ("R_L_1RQ", "R_L_2RQ")
    assert "R_L" in result.ladder                     # the ladder was actually fitted
    assert result.ladder["R_L"].aicc > result.fit.aicc


def test_band_limits_are_honoured():
    spectrum = _synthetic_spectrum("1", n_points=40)
    result = ecm_service.fit_segment(spectrum, "1", model="R_L_1RQ",
                                     f_min=1.0, f_max=100.0)
    assert result.f.min() >= 1.0 and result.f.max() <= 100.0


def test_too_few_points_is_reported_not_fitted():
    spectrum = _synthetic_spectrum("1", n_points=3)
    result = ecm_service.fit_segment(spectrum, "1", model="R_L_2RQ")
    assert not result.ok
    assert "usable points" in result.note


def test_fit_run_produces_a_mergeable_table(gold_silver_run):
    run = loaders.load(gold_silver_run / "2611976" / "45A", plate_key="gen1_r2d2_72")
    table, fits = ecm_service.fit_run(run, model="R_L_1RQ", n_starts=2)
    assert set(table["segment"]) == {"1", "2"}
    assert "ecm_Rs" in table.columns and "ecm_Rs_sd" in table.columns
    assert "ecm_Rs" in ecm_service.ecm_params(table)

    ecm_service.attach_to_segments(run, table)
    assert "ecm_Rs" in run.segments.columns
    assert run.value("ecm_Rs")["1"] == pytest.approx(80.0, rel=0.05)
    # Merging twice must not duplicate columns.
    ecm_service.attach_to_segments(run, table)
    assert list(run.segments.columns).count("ecm_Rs") == 1


# ---------------------------------------------------------------------------
# reading files the pipeline actually writes
# ---------------------------------------------------------------------------

def test_unquoted_comma_in_a_trailing_text_column_does_not_shift_the_table(tmp_path):
    """The pipeline writes skew_model.csv with an unquoted comma in `note`.

    Read naively, pandas promotes the first column to the index and every value
    lands one column to the left - a table that is entirely wrong and entirely
    plausible. The loader has to recover the row intact.
    """
    path = tmp_path / "skew_model.csv"
    path.write_text(
        "card,basis,dt0_us,slot_us,applied,note\n"
        "Karte_1,parity,0,54.6875,1,"
        "step fitted; common offset not fitted (degenerate with series L, "
        "absorbed there)\n"
        "Karte_2,parity,0,50.0,1,step nominal (fitted value rejected)\n"
    )
    frame = loaders._read_table(path)

    assert list(frame.columns) == ["card", "basis", "dt0_us", "slot_us",
                                   "applied", "note"]
    assert len(frame) == 2
    row = frame.iloc[0]
    assert row["card"] == "Karte_1"
    assert row["basis"] == "parity"
    assert row["slot_us"] == pytest.approx(54.6875)
    assert row["applied"] == 1
    assert row["note"].endswith("absorbed there)")     # nothing truncated
    assert isinstance(frame.index, pd.RangeIndex)


def test_well_formed_csv_still_reads_normally(tmp_path):
    path = tmp_path / "clean.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(path, index=False)
    frame = loaders._read_table(path)
    assert frame["a"].tolist() == [1, 2]
    assert frame["b"].tolist() == ["x", "y"]


def test_case_insensitive_file_systems_do_not_double_count_cards(tmp_path, monkeypatch):
    """Windows matches *.DAT and *.dat identically; each card must count once.

    The symptom is a plate that reports twice as many card files as exist, and
    a staging step that copies every one of them twice.
    """
    real = [tmp_path / f"Leepa_2611976_Current_45A_Test_01_Karte_{c}.DAT"
            for c in range(1, 6)]
    for path in real:
        path.touch()

    original = Path.rglob

    def case_insensitive_rglob(self, pattern):
        # Stand in for Windows: both spellings return the same five files.
        if pattern.lower() == "*.dat":
            return iter(real)
        return original(self, pattern)

    monkeypatch.setattr(Path, "rglob", case_insensitive_rglob)
    refs = FamosSource([tmp_path]).scan()

    assert len(refs) == 1
    assert len(refs[0].files) == 5
    assert len(set(refs[0].files)) == 5


def test_a_case_sensitive_file_system_keeps_both_spellings(tmp_path):
    """On Linux, Karte_1.DAT and karte_1.dat really are two different files."""
    (tmp_path / "Leepa_2611976_Current_45A_Test_01_Karte_1.DAT").touch()
    (tmp_path / "Leepa_2611976_Current_45A_Test_01_Karte_2.dat").touch()
    refs = FamosSource([tmp_path]).scan()
    assert len(refs[0].files) == 2
