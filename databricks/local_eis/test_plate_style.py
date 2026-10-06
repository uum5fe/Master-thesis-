"""One look for every plate map: Jet colours, value over number, the view
direction, and the gas / coolant ports on the ends they really are."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

import config                                               # noqa: E402
import plate_figure                                         # noqa: E402
import plate_maps                                           # noqa: E402
import plate_style as style                                 # noqa: E402
import r2d2_geometry as geom                                # noqa: E402


def _vals():
    return {s: 60.0 + int(s) % 9 for s in geom.ACTIVE_PLATE.segments}


@pytest.fixture(params=[False, True], ids=["plate-view", "mirrored"])
def view(request, monkeypatch):
    monkeypatch.setattr(config, "PLATE_VIEW_MIRRORED", request.param)
    return request.param


def test_the_scale_is_plotly_jet_stop_for_stop():
    pytest.importorskip("plotly")
    import plotly.colors as pc
    # plotly.py ships the same six colours, evenly spaced; plotly.js places
    # them at 0, .125, .375, .625, .875, 1 -- the reference plots' scale
    assert [c for _, c in style.JET_STOPS] == list(pc.sequential.Jet)
    assert [p for p, _ in style.JET_STOPS] == [0, .125, .375, .625, .875, 1]
    cm = style.mpl_cmap()
    assert np.allclose(cm(0.0)[:3], (0, 0, 131 / 255))
    assert np.allclose(cm(1.0)[:3], (128 / 255, 0, 0))
    for t in (0.2, 0.5, 0.8):                # between stops, same as plotly
        ref = pc.sample_colorscale(style.plotly_colorscale(), [t])[0]
        ref = [float(x) / 255 for x in ref[ref.index("(") + 1:-1].split(",")]
        assert np.allclose(cm(t)[:3], ref, atol=0.01)


def test_every_map_uses_the_same_scale_by_default():
    for meta in config.PARAM_META.values():
        assert meta["cmap"] == style.HEATMAP_CMAP
    for _, _, ramp, _ in plate_figure.FIELDS.values():
        assert ramp == style.HEATMAP_CMAP


def test_mirroring_reverses_the_x_axis_and_the_flow_note(view):
    assert style.xlim(0, 10) == ((10, 0) if view else (0, 10))
    note = style.flow_note()
    if view:
        assert "air left to right" in note and "H2 right to left" in note
        assert "mirrored" in note
    else:
        assert "air right to left" in note and "H2 left to right" in note


def test_coolant_ports_follow_config(monkeypatch):
    monkeypatch.setattr(config, "COOLANT_INLET_END", "x0")
    ends = style.end_streams()
    assert ("COOLANT IN" in [s[0] for s in ends["x0"]]
            and "COOLANT OUT" in [s[0] for s in ends["xW"]])
    # the air inlet is at the far end of a counter-flow plate
    assert "AIR IN" in [s[0] for s in ends["xW"]]
    monkeypatch.setattr(config, "COOLANT_INLET_END", "xW")
    assert "COOLANT IN" in [s[0] for s in style.end_streams()["xW"]]


def test_mirrored_view_puts_the_air_inlet_on_the_left(monkeypatch):
    monkeypatch.setattr(config, "PLATE_VIEW_MIRRORED", True)
    assert style.screen_side("xW") == "left"
    monkeypatch.setattr(config, "PLATE_VIEW_MIRRORED", False)
    assert style.screen_side("xW") == "right"


def _font_sizes(ax, number: str, value: str):
    sizes = {t.get_text(): t.get_fontsize() for t in ax.texts}
    return sizes[number], sizes[value]


def test_plate_figure_value_larger_than_number_in_both_views(view):
    vals = _vals()
    fig = plate_figure.draw_flow_plate(vals, "R_ohmic")
    ax = fig.axes[0]
    n_sz, v_sz = _font_sizes(ax, "18", f"{vals['18']:.1f}")
    assert v_sz > n_sz
    lo, hi = ax.get_xlim()
    assert (lo > hi) == view
    texts = [t.get_text() for t in ax.texts]
    assert "COOLANT IN" in texts and "COOLANT OUT" in texts
    plt.close(fig)


def test_plate_figure_is_drawn_in_jet():
    vals = _vals()
    fig = plate_figure.draw_flow_plate(vals, "R_ct", limits=(60.0, 68.0))
    # the lowest value is Jet's dark blue, the highest its dark red
    faces = [tuple(np.round(p.get_facecolor()[:3], 3))
             for p in fig.axes[0].patches]
    assert tuple(np.round(style.mpl_cmap()(0.0)[:3], 3)) in faces
    assert tuple(np.round(style.mpl_cmap()(1.0)[:3], 3)) in faces
    plt.close(fig)


def test_plate_maps_value_larger_and_ends_labelled(view):
    vals = _vals()
    fig, ax = plate_maps.draw_value_map(vals, "HFR (Rs)", "mΩ·cm²")
    n_sz, v_sz = _font_sizes(ax, "18", f"{vals['18']:.1f}")
    assert v_sz > n_sz
    joined = " ".join(t.get_text() for t in ax.texts)
    for lab in ("AIR IN", "AIR OUT", "H₂ IN", "COOLANT IN", "COOLANT OUT"):
        assert lab in joined
    lo, hi = ax.get_xlim()
    assert (lo > hi) == view
    plt.close(fig)


def test_gold_map_is_the_plate_figure_with_row_one_on_top():
    import gold
    from config import DEFAULT
    recs = {s: gold.SegmentRecord(segment=s, cls="measured", tier="A",
                                  cx_mm=g.cx_mm, cy_mm=g.cy_mm,
                                  area_cm2=g.area_cm2,
                                  values={"R_ohmic": 0.06 + int(s) * 1e-4})
            for s, g in geom.SEGMENTS.items()}
    fig = gold.plate_heatmap(recs, "R_ohmic", DEFAULT)
    ax = fig.axes[0]
    lo, hi = ax.get_ylim()
    assert lo > hi, "gold drew the plate upside down again"
    plt.close(fig)


def test_numbering_map_follows_the_view(tmp_path, view):
    p = geom.plot_map(tmp_path / "m.png")
    assert p.is_file() and p.stat().st_size > 0


def test_interactive_plate_hover_dropdown_and_view(view):
    pytest.importorskip("plotly")
    import plate_plotly
    vals = _vals()
    fds = [plate_plotly.Field("R_ohmic", "HFR (Rs)", "mΩ·cm²", vals),
           plate_plotly.Field("T_degC", "Temperature", "°C",
                              {s: 58 + v / 100 for s, v in vals.items()},
                              decimals=2)]
    fig = plate_plotly.interactive_plate(fds, title="t")
    heats = [t for t in fig.data if t.type == "heatmap"]
    assert len(heats) == 2
    assert [list(c) for c in heats[0].colorscale] == \
        [[p, c] for p, c in style.JET_STOPS]
    # every pad of a segment carries that segment's number for the hover
    z, cd = np.array(heats[0].z, float), np.array(heats[0].customdata)
    for c, r in geom.SEGMENTS["18"].pads:
        assert cd[r - 1, c - 1][0] == "18"
        assert z[r - 1, c - 1] == pytest.approx(vals["18"])
    assert len(fig.layout.updatemenus[0].buttons) == 2
    rng = list(fig.layout.xaxis.range)
    assert (rng[0] > rng[1]) == view
    ann = " ".join(a.text for a in fig.layout.annotations)
    assert "COOLANT IN" in ann and "AIR IN" in ann


def test_each_end_is_a_column_gas_out_above_coolant_gas_in_below(monkeypatch):
    monkeypatch.setattr(config, "PLATE_VIEW_MIRRORED", False)
    monkeypatch.setattr(config, "COOLANT_INLET_END", "xW")
    fig = plate_figure.draw_flow_plate(_vals(), "R_ohmic")
    pos = {t.get_text(): t.get_position() for t in fig.axes[0].texts}
    W = geom.PLATE_W_MM
    right = ("H₂ OUT", "COOLANT IN", "AIR IN")
    left = ("AIR OUT", "COOLANT OUT", "H₂ IN")
    for column, outside in ((right, lambda x: x > W), (left, lambda x: x < 0)):
        xs = [pos[lab][0] for lab in column]
        ys = [pos[lab][1] for lab in column]
        assert all(outside(x) for x in xs), column
        assert max(xs) - min(xs) < 1e-6, f"{column} not stacked in one column"
        assert ys[0] < ys[1] < ys[2], f"{column} not top-to-bottom (y down)"
    plt.close(fig)
