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
