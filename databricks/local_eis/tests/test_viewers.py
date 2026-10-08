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
    return viewers.unpack_payload(h)["data"]


def _shared(h):
    return viewers.unpack_payload(h)["shared"]


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
    P = viewers.unpack_payload(h)
    keys = [k.split(viewers.SEP) for k in P["items"]]
    assert {k[1] for k in keys} == {"HFR (Rs)", "R_ct (charge transfer)",
                                    "Current density", "Temperature"}
    assert len(keys) == 2 * 4
    # the View drop-down offers both views; the browser builds the one chosen
    assert h.count("<select") == 3
    assert viewers.VIEW_2D in h and viewers.VIEW_SEG in h
    it = P["items"][viewers.SEP.join(("45A", "HFR (Rs)"))]
    assert len(it["v"]) == len(geom.ACTIVE_PLATE.segments)
    nr, nc, _ = it["gs"]
    assert len(it["g"]) == nr * nc
    # geometry once: every pad has its owner segment
    assert len(P["G"]["owner"]) == geom.N_ROWS
    # small enough for a notebook cell
    assert len(h) < 60_000


def test_the_quantised_field_decodes_to_the_values(tmp_path):
    h = viewers.plate_maps({"45A": _summary(tmp_path)})
    it = viewers.unpack_payload(h)["items"][viewers.SEP.join(("45A",
                                                              "R_ct (charge transfer)"))]
    nr, nc, _ = it["gs"]
    g = np.cumsum(np.array(it["g"]).reshape(nr, nc), axis=1)
    z = it["vmin"] + g / 250 * (it["vmax"] - it["vmin"])
    v = np.array(list(it["v"].values()))
    assert z.min() >= min(v.min(), it["vmin"]) - 1e-9
    assert z.max() <= max(v.max(), it["vmax"]) + 1e-9


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
