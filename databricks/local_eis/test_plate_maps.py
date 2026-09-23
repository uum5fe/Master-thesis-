"""Plate maps with the value in every segment, on a scale that shows spread.

The HFR map was unreadable for uniformity because a flat plate with one or
two outliers, drawn min..max, spends almost all of its colour range on the
outliers. These tests pin the three things that fix it: the value is printed
inside every segment, the colour scale is robust to a single outlier, and the
uniformity statistics are the numbers they claim to be.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

import plate_maps                                           # noqa: E402
import r2d2_geometry as geom                                # noqa: E402


RUNNER = Path(__file__).with_name("Local EIS Pipeline Runner.py")


def _flat_plate_with_one_outlier():
    rng = np.random.default_rng(3)
    vals = {int(s): 65.0 + 1.5 * rng.standard_normal() for s in geom.SEGMENTS}
    vals[12] = 50.0
    return vals


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def test_every_segment_shows_its_value() -> None:
    vals = _flat_plate_with_one_outlier()
    fig, ax = plate_maps.draw_value_map(vals, "HFR (Rs)", "mΩ·cm²")
    texts = _texts(ax)
    for s, v in vals.items():
        assert f"{v:.1f}" in texts, f"segment {s} value missing"
        assert str(s) in texts, f"segment {s} number missing"
    plt.close(fig)


def test_a_single_outlier_does_not_set_the_colour_scale() -> None:
    vals = _flat_plate_with_one_outlier()
    lo, hi = plate_maps.robust_limits(vals)
    assert lo > 55.0, "the one 50 mΩ·cm² segment stretched the scale"
    assert hi - lo < 10.0
    fig, ax = plate_maps.draw_value_map(vals, "HFR (Rs)", "mΩ·cm²")
    # the outlier is still on the map, printed as measured, and the figure
    # says the colour scale was cut below it
    assert "50.0" in _texts(ax)
    assert "beyond it" in ax.get_title(loc="left")
    plt.close(fig)


def test_min_max_scale_is_still_available() -> None:
    vals = _flat_plate_with_one_outlier()
    lo, hi = plate_maps.robust_limits(vals, pct=None)
    assert lo == pytest.approx(min(vals.values()))
    assert hi == pytest.approx(max(vals.values()))


def test_uniformity_numbers() -> None:
    vals = {1: 60.0, 2: 62.0, 3: 64.0, 4: float("nan"), 5: None}
    st = plate_maps.uniformity_stats(vals)
    assert st["n"] == 3
    assert st["mean"] == pytest.approx(62.0)
    assert st["median"] == pytest.approx(62.0)
    assert st["sd"] == pytest.approx(2.0)
    assert st["cv_pct"] == pytest.approx(100 * 2.0 / 62.0)
    assert (st["min"], st["max"]) == (60.0, 64.0)


def test_rebuilt_segments_are_marked() -> None:
    vals = {int(s): 65.0 for s in geom.SEGMENTS}
    vals[33] = 67.25
    fig, ax = plate_maps.draw_value_map(vals, "HFR (Rs)", "mΩ·cm²",
                                        classes={"33": "substituted"})
    assert "67.2*" in _texts(ax) or "67.3*" in _texts(ax)
    assert "1 rebuilt" in ax.get_title(loc="left")
    plt.close(fig)


def test_unmeasured_segments_are_drawn_grey_with_number_only() -> None:
    vals = {int(s): 65.0 for s in geom.SEGMENTS if s != "36"}
    fig, ax = plate_maps.draw_value_map(vals, "HFR (Rs)", "mΩ·cm²")
    assert "36" in _texts(ax)
    plt.close(fig)


# ---------------------------------------------------------------------------
# the heat-map cell, executed
# ---------------------------------------------------------------------------


def _cell(marker: str) -> str:
    for chunk in RUNNER.read_text().split("\n# COMMAND ----------\n"):
        if marker in chunk:
            return chunk
    raise AssertionError(f"cell {marker!r} not found")


def test_the_heatmap_cell_draws_labelled_maps_without_plate_viewer(
        tmp_path, monkeypatch) -> None:
    pytest.importorskip("IPython")
    import IPython.display as ipd
    rng = np.random.default_rng(0)
    rows = [dict(segment=int(s), **{"class": "measured"},
                 R_ohmic=65 + rng.standard_normal(),
                 R_ct=110 + 5 * rng.standard_normal(),
                 R_mt=30 + 3 * rng.standard_normal(),
                 R_pol=140 + 5 * rng.standard_normal())
            for s in geom.SEGMENTS]
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    pd.DataFrame(rows).to_csv(gold_dir / "plate_summary.csv", index=False)

    shown = []
    monkeypatch.setattr(ipd, "display", lambda obj: shown.append(obj))
    ns = {"plate_maps": plate_maps, "MIN_SNR_DB": 5.0,
          "EVALUATION_MODE": "default", "displayHTML": lambda h: None,
          "_widget": lambda *a, default='': "2612030",
          "selected_conditions": lambda: ["150A"],
          "result_dir": lambda leepa, cond: (tmp_path, "this session's run"),
          "describe_source": lambda cond, d, prov: f"  {cond}: {prov}"}
    exec(_cell("INTERACTIVE PLATE HEATMAPS"), ns)
    for col in ("R_ohmic", "R_ct", "R_mt", "R_pol"):
        assert (gold_dir / f"plate_{col}.png").is_file(), col
    assert len(shown) == 4
    rs_ax = shown[0].axes[0]
    assert "HFR (Rs)" in rs_ax.get_title(loc="left")
    assert f"{rows[0]['R_ohmic']:.1f}" in _texts(rs_ax)
    for f in shown:
        plt.close(f)
