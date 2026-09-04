"""The realistic plate tab, and the geometry underneath it.

`app.plates.registry` answers "which plate spec is this run on"; this geometry
answers "where is segment 34 on the copper, and what shape is it". The two are
independent transcriptions of the same pad map, which is exactly why both are
worth testing: a typo in either draws a heat map that looks entirely plausible
and is wrong, and nothing downstream can tell.
"""

from __future__ import annotations

import pytest

from app.plates.geometry import (PLATE, Field, plate_figure, synthetic_fields,
                                 to_pad_matrix, write_html)


# ---------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------


def test_the_segments_tile_the_pad_grid_exactly_once():
    """The check the README asks for whenever the pad matrix changes.

    A segment is the SET OF PADS carrying its number, so if the 72 sets do not
    partition the 900 pads then some pad is counted twice or not at all, and
    every area -- and therefore every A/cm2 and every mOhm*cm2 -- is wrong.
    """
    check = PLATE.self_check()
    assert check["tiles_exactly"]
    assert check["area_closes"]


def test_the_areas_match_the_plate_the_rest_of_the_app_uses():
    from app.plates import registry

    spec = registry.get("gen1_r2d2_72")
    total = sum(s.area_cm2 for s in PLATE.segments.values())
    assert len(PLATE.segments) == spec.n_segments == 72
    assert total == pytest.approx(spec.cell_area_cm2, abs=1e-6)
    assert total == pytest.approx(304.92, abs=1e-6)


def test_both_numberings_describe_the_same_copper():
    """`wired` is the harness numbering, `topleft` a reading order over the
    same segments. A renumbering that is not a bijection would silently move
    values onto the wrong segments.
    """
    ren = PLATE.renumbering()
    assert len(ren) == 72
    assert sorted(ren) == list(range(1, 73))
    assert sorted(ren.values()) == list(range(1, 73))


def test_a_pad_matrix_paints_every_pad():
    values = {s.wired: float(s.wired) for s in PLATE.segments.values()}
    m = to_pad_matrix(values)
    assert m.shape == (20, 45)
    assert m.size == 900
    # every pad carries the value of the segment that owns it
    for seg in PLATE.segments.values():
        assert (m == float(seg.wired)).sum() == seg.n_pads


# ---------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------


def test_the_plotly_figure_draws_one_trace_per_segment():
    f = synthetic_fields()
    fig = plate_figure({s.wired: f["j"][s.wired] for s in PLATE.segments.values()},
                       label="Current density", unit="A/cm2",
                       colorscale="Inferno")
    assert len(fig.data) >= len(PLATE.segments)


def test_a_missing_segment_is_drawn_rather_than_dropped():
    """A hole in a heat map reads to the eye as a cold spot, which is worse
    than an honest gap -- so an unmeasured segment must still be drawn.
    """
    f = synthetic_fields()
    partial = {s.wired: f["j"][s.wired]
               for s in list(PLATE.segments.values())[:40]}
    fig = plate_figure(partial, label="j", unit="A/cm2")
    assert len(fig.data) >= len(PLATE.segments)


def test_the_standalone_html_is_self_contained(tmp_path):
    """`write_html` is the output that needs no server and no network: it has
    to survive being emailed and opened offline.
    """
    f = synthetic_fields()
    out = write_html([Field("j", "Current density", "A/cm2", "inferno", 3,
                            f["j"])],
                     tmp_path / "plate.html", title="test")
    text = (tmp_path / "plate.html").read_text(encoding="utf-8")
    assert out
    assert text.lstrip().lower().startswith("<!doctype html")
    assert "<script" in text                      # the controls are inline
    assert "src=\"http" not in text               # and nothing is fetched
    assert "cdn" not in text.lower().split("<style")[0]


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------


def test_the_tab_is_registered_and_renders():
    from app.app import build_app

    app = build_app()
    labels = [t.label for t in app.layout.children[3].children[0].children]
    assert "Plate (realistic)" in labels

    from app.views import plate_realistic
    assert plate_realistic.layout() is not None


def test_the_tab_callback_is_reachable_for_its_tab_value():
    """The tab list and the render callback are two separate places; a tab
    whose value is missing from the callback's dict raises KeyError on click,
    which is a blank page rather than an error message.
    """
    from app.app import build_app

    app = build_app()
    values = [t.value for t in app.layout.children[3].children[0].children]
    assert "tab-plate3d" in values

    key = next(k for k in app.callback_map if "tab-body" in k)
    # Dash wraps the registered function to do its own serialisation; the
    # thing under test is the plain function it wrapped.
    render = app.callback_map[key]["callback"].__wrapped__
    # EVERY tab, not just the new one: the bar and the callback's dict are two
    # separate places, and a tab missing from the dict raises KeyError on
    # click, which the user sees as a blank page rather than an error.
    for value in values:
        assert render(value) is not None, value
