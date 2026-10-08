"""viewers: one plot chosen from drop-downs; the plate viewer offers every
parameter in both views; repeated pieces are stored once."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

import r2d2_geometry as geom
import viewers

go = pytest.importorskip("plotly.graph_objects")


def _summary(tmp_path, cond="45A"):
    segs = sorted(geom.ACTIVE_PLATE.segments, key=int)
    rows = ["segment,class,R_ohmic,R_ct,j_dc,T_degC"]
    for s in segs:
        n = int(s)
        rows.append(f"{s},{'measured' if n % 9 else 'reconstructed'},"
                    f"{50 + n % 7},{120 + n},{1.4 + 0.001 * n},{60 + n / 50}")
    p = tmp_path / f"{cond}.csv"
    p.write_text("\n".join(rows))
    return p


def _data(h):
    return json.loads(re.search(r"var DATA = (\{.*?\});\n", h, re.S).group(1))


def _shared(h):
    return json.loads(re.search(r"var SHARED = (\[.*?\]);\n", h, re.S).group(1))


def test_selector_has_one_dropdown_per_dimension_and_every_combination():
    figs = {(c, v): go.Figure(go.Scatter(x=[1, 2], y=[i, i + 1]))
            for i, c in enumerate(("45A", "450A")) for v in ("Nyquist", "Bode")}
    h = viewers.selector_html(figs, ["Condition", "Plot"])
    assert h.count("<select") == 2
    assert set(_data(h)) == {viewers.SEP.join(k) for k in figs}
    assert "Plotly.react" in h and "cdn.plot.ly" in h


def test_html_items_and_missing_combinations():
    h = viewers.selector_html({("a", "x"): "<b>table</b>",
                               ("b", "y"): go.Figure()}, ["One", "Two"])
    d = _data(h)
    assert d[viewers.SEP.join(("a", "x"))] == {"html": "<b>table</b>"}
    assert "Not available for this combination" in h


def test_plate_maps_offer_every_parameter_in_both_views(tmp_path):
    h = viewers.plate_maps({"45A": _summary(tmp_path), "450A":
                            _summary(tmp_path, "450A")}, title="t")
    d = _data(h)
    params = {k.split(viewers.SEP)[1] for k in d}
    assert params == {"HFR (Rs)", "R_ct (charge transfer)", "Current density",
                      "Temperature"}
    views = {k.split(viewers.SEP)[2] for k in d}
    assert views == {viewers.VIEW_2D, viewers.VIEW_SEG}
    assert len(d) == 2 * 4 * 2
    # the 2D view: a shaded image, squares on the segments, a colour bar
    f2d = d[viewers.SEP.join(("45A", "HFR (Rs)", viewers.VIEW_2D))]["fig"]
    sh = _shared(h)
    types = [(sh[t["__ref__"]] if "__ref__" in t else t).get("type")
             for t in f2d["data"]]
    assert "image" in types and "scatter" in types
    # pieces repeated across maps (here: identical data) are stored once
    assert len(sh) >= 1 and h.count('"__ref__"') > len(sh)


def test_by_condition_splits_panels_into_a_second_dropdown():
    from plotly.subplots import make_subplots
    figs = {}
    for c in ("45A", "60A"):
        f = make_subplots(rows=1, cols=3,
                          subplot_titles=["Nyquist", "|Z|(f) Bode", "Phase(f)"])
        for col in (1, 2, 3):
            f.add_trace(go.Scatter(x=[1, 2], y=[1, 2]), row=1, col=col)
        figs[c] = f
    h = viewers.by_condition(figs)
    assert h.count("<select") == 2
    assert {k.split(viewers.SEP)[1] for k in _data(h)} == {"Nyquist", "|Z|(f) Bode",
                                                   "Phase(f)"}
