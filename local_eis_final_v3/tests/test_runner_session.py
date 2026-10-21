"""The runner package behind the notebook: selection, saved results, plots.

Runs without Databricks: a stand-in for dbutils.widgets, no Spark, and the
viewers written to files instead of displayHTML.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import runner
from runner import settings as S
from runner import views
import viewers

ROOT = Path(__file__).resolve().parents[1]


class _W:
    def __init__(self, vals):
        self.v = dict(vals)
        self.removed = []

    def get(self, n):
        if n not in self.v:
            raise KeyError(n)
        return self.v[n]

    def dropdown(self, n, d, ch, label=""):
        self.v.setdefault(n, d)

    multiselect = dropdown

    def text(self, n, d, label=""):
        self.v.setdefault(n, d)

    def remove(self, n):
        self.removed.append(n)
        self.v.pop(n, None)


class _DB:
    def __init__(self, vals):
        self.widgets = _W(vals)


@pytest.fixture()
def volume(tmp_path):
    famos = tmp_path / "Famos"
    famos.mkdir()
    for cond, tests in (("60A", ("01", "02")), ("450A", ("01",)), ("45A", ("02",))):
        for t in tests:
            for k in (1, 2):
                (famos / f"Leepa_RO2612086_Current_{cond}_Test_{t}_Karte_{k}.DAT"
                 ).write_bytes(b"x" * 10)
    (famos / "Leepa_RO2612030_Current_150A_Test_01_Karte_1.DAT").write_bytes(b"x")
    gamry = tmp_path / "Gamry"
    gamry.mkdir()
    (gamry / "20261020_0809_RO2612086-01_V26_109_lokale_EIS.mf4").write_bytes(b"")
    return tmp_path


def _session(volume, **w):
    shown = []
    env = {"dbutils": _DB(w), "spark": None, "displayHTML": shown.append}
    R = runner.Session(env, famos_root=volume / "Famos",
                       gamry_root=volume / "Gamry",
                       results_root=volume / "EIS_Results",
                       config_root=ROOT / "EIS_config",
                       scratch=volume / "scratch")
    return R, shown


def test_orders_conditions_and_tests_come_from_the_file_names(volume):
    R, _ = _session(volume)
    assert R.orders() == ["2612030", "2612086"]
    ct = R.conditions_and_tests("2612086")
    assert list(ct) == ["45A", "60A", "450A"]           # by current
    assert ct["60A"] == ["01", "02"] and ct["45A"] == ["02"]


def test_the_selection_and_the_ordered_result_folder(volume):
    R, _ = _session(volume, leepa_id="2612086", conditions="60A,45A",
                    test_id="02")
    sel = R.widgets()
    assert sel["conditions"] == ["45A", "60A"]
    assert sel["test_id"] == "02"
    assert sel["gamry_version"] == "V26_109"
    assert R.saved_dir("60A") == volume / "EIS_Results" / "2612086" / "60A_Test02"
    assert R.label("60A") == "60A Test02"


def test_all_means_every_condition(volume):
    R, _ = _session(volume, leepa_id="2612086", conditions="ALL")
    assert R.widgets()["conditions"] == ["45A", "60A", "450A"]


def test_old_widgets_are_removed_and_there_is_no_cache_write_switch(volume):
    R, _ = _session(volume, leepa_id="2612086", cache_write="no")
    R.widgets()
    assert "cache_write" in R.dbutils.widgets.removed
    assert S.RUN_MODES == ("cache", "rerun")


def _fake_result(d: Path, cond="60A"):
    (d / "silver").mkdir(parents=True)
    f = np.geomspace(0.2, 5000, 30)
    rows = []
    for seg, rs in (("15", 0.05), ("16", 0.06), ("22", 0.055)):
        z = 1000 * (rs + 0.2 / (1 + 1j * 2 * np.pi * f * 3e-3))
        rows += [{"segment": seg, "freq_hz": ff, "z_re_mohm_cm2": zz.real,
                  "z_im_mohm_cm2": zz.imag, "sigma_rel": 0.01}
                 for ff, zz in zip(f, z)]
    pd.DataFrame(rows).to_csv(d / "silver" / "spectra_clean.csv", index=False)
    z = 1000 * (0.055 + 0.2 / (1 + 1j * 2 * np.pi * f * 3e-3))
    pd.DataFrame({"freq_hz": f, "z_re_mohm_cm2": z.real,
                  "z_im_mohm_cm2": z.imag}).to_csv(
        d / "silver" / "cell_aggregate.csv", index=False)
    pd.DataFrame({"segment": [15, 16, 22], "R_ohmic_mohm_cm2": [50, 60, 55]}
                 ).to_csv(d / "silver" / "segments_summary.csv", index=False)
    (d / "bronze").mkdir()
    pd.DataFrame({"segment": [15, 16, 22],
                  "card": ["L_2612086_Current_60A_Test_02_Karte_2"] * 2
                  + ["L_2612086_Current_60A_Test_02_Karte_4"]}).to_csv(
        d / "bronze" / "segment_meta.csv", index=False)
    g = 1000 * (0.05 + 0.19 / (1 + 1j * 2 * np.pi * f * 3e-3))
    pd.DataFrame({"condition": cond, "freq_hz": f, "z_re_ref_mohm_cm2": g.real,
                  "z_im_ref_mohm_cm2": g.imag, "in_local_band": 1}).to_csv(
        d / "gamry_reference_full.csv", index=False)
    pd.DataFrame({"reference": ["V26_109_HFR_111_CurrVal_60.dta"]}).to_csv(
        d / "gamry_comparison.csv", index=False)


def test_cache_mode_uses_the_saved_result_and_says_when_settings_differ(volume, capsys):
    R, _ = _session(volume, leepa_id="2612086", conditions="60A", test_id="02")
    R.widgets()
    d = R.saved_dir("60A")
    _fake_result(d)
    (d / "run_identity.json").write_text(json.dumps(
        {"digest": "something-else", "settings": {"min_snr_db": 3.0}}))
    out = R.run()
    assert out == {"60A": d}
    assert "OTHER settings" in capsys.readouterr().out


def test_rerun_replaces_the_saved_folder(volume):
    R, _ = _session(volume)
    src, dst = volume / "new", volume / "saved"
    (src / "silver").mkdir(parents=True)
    (src / "silver" / "a.csv").write_text("new")
    (dst / "silver").mkdir(parents=True)
    (dst / "silver" / "stale.csv").write_text("old")
    assert R._save(src, dst) == dst
    assert (dst / "silver" / "a.csv").read_text() == "new"
    assert not (dst / "silver" / "stale.csv").exists()


def test_the_plots_are_drawn_and_saved_from_a_saved_result(volume):
    R, shown = _session(volume, leepa_id="2612086", conditions="60A",
                        test_id="02")
    R.widgets()
    _fake_result(R.saved_dir("60A"))
    for show in (R.show_nyquist, R.show_median_nyquist, R.show_ecm):
        p = show()
        assert p.is_file()
    assert len(shown) == 3
    assert (R.order_dir() / "plots" / "60A_Test02" / "ecm_aggregate.png").is_file()
    assert (R.saved_dir("60A") / "ecm" / "ecm_parameters.csv").is_file()


def test_the_gamry_drawn_is_the_one_the_run_was_compared_with(volume):
    d = volume / "r"
    _fake_result(d)
    g, name = views.gamry_of_run(d)
    assert name == "V26_109_HFR_111_CurrVal_60.dta" and len(g) == 30


def test_the_ecm_viewer_switches_fits_without_duplicating_figures(volume):
    d = volume / "r"
    _fake_result(d)
    fits = views.ecm_fits(d, say=lambda *a: None)
    assert set(fits) == {"Aggregate", "15", "16", "22"}
    e = fits["15"]
    assert e["std"]["ok"] and e["drt"]["ok"]
    assert e["std"]["Rs"] == pytest.approx(50, rel=0.05)
    fig = views.ecm_figure(e, "x")
    assert {t.meta for t in fig.data} == {"data", "std", "drt"}
    html = views.ecm_viewer({("60A", "Segment 15"): fig})
    P = viewers.unpack_payload(html)
    assert list(P["data"]) == [f"60A{viewers.SEP}Segment 15"]
    for o in views.FIT_OPTIONS:
        assert f"<option>{o}</option>" in html


def test_every_viewer_is_pure_ascii_and_repairs_a_misread_minus():
    """The fragment must survive a page decoded as Windows-1252."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=[1, 2], y=[-50, 5], name="−Z'' [mΩ·cm²]"))
    for html in (viewers.selector_html({("45A", "Nyquist"): fig},
                                       ["Condition", "Plot"], title="Ω test °C"),
                 views.ecm_viewer({("45A", "Aggregate"): fig})):
        assert all(ord(c) < 128 for c in html)
        assert "unmangle" in html and "MutationObserver" in html
        assert "&#937;" in html or "ecm" in html.lower()


def test_the_notebook_is_only_calls():
    nb = (ROOT / "Local EIS Pipeline Runner.py").read_text(encoding="utf-8")
    code = [l for l in nb.splitlines()
            if l.strip() and not l.startswith("#")]
    assert len(code) < 40, len(code)
    for call in ("R.widgets()", "R.run()", "R.show_frequency_response()",
                 "R.show_nyquist()", "R.show_median_nyquist()",
                 "R.show_heatmaps()", "R.show_mf4()", "R.show_ecm()"):
        assert call in nb
    assert not (ROOT / "Local EIS Extra Analyses.py").exists()


def test_the_whole_bench_log_marks_every_sweep():
    import bench_plots as bp
    t = np.linspace(0, 6000, 500)
    ch = [bp.Channel("I_S", "Stack current", "A", "Electrical", t, t / 10)]
    fig = views.bench_figure_whole(ch, "x", sweeps=[("60A #103", 1800, 1950),
                                                    ("60A #111", 5100, 5240)])
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == 2 and rects[1].x0 == pytest.approx(85.0)
    assert fig.data[0].x[-1] == pytest.approx(100.0)     # minutes, whole log
