"""The plate registry is the piece a new generation depends on, so it is the
piece with the tightest tests: a spec that does not tile the plate exactly once
produces a heat map that looks plausible and is wrong.
"""

from __future__ import annotations

import json

import pytest

from app.plates import registry


def test_gen1_is_available():
    plates = registry.available()
    assert "gen1_r2d2_72" in plates


def test_gen1_matches_the_drawing():
    geom = registry.get("gen1_r2d2_72")
    assert geom.n_segments == 72
    assert geom.n_cols * geom.n_rows == 900
    assert geom.cell_area_cm2 == pytest.approx(304.92, abs=1e-6)

    areas = geom.areas()
    # Unequal by a factor of 12.5 - the whole reason the geometry is explicit.
    assert min(areas.values()) == pytest.approx(0.6776, abs=1e-4)
    assert max(areas.values()) == pytest.approx(8.470, abs=1e-3)
    assert sum(areas.values()) == pytest.approx(geom.cell_area_cm2, abs=1e-9)


def test_gen1_tiles_the_pad_grid_exactly_once():
    check = registry.get("gen1_r2d2_72").self_check()
    assert check["tiles_exactly"], check["uncovered_pads"][:5]
    assert check["area_closes"]
    assert check["pads_covered"] == check["pads_total"] == 900


def test_no_segment_is_excluded_before_the_data_is_seen():
    """33 and 59 are measured like every other segment.

    They were carried as permanently bad, which meant every later campaign
    inherited that verdict without re-testing it and the map grew two holes -
    and a hole reads to the eye as a cold spot. A run that really has no
    response there now says so through a low-SNR flag on that run, which is a
    statement about the run rather than about the plate for ever.
    """
    geom = registry.get("gen1_r2d2_72")
    assert geom.known_bad == {}
    assert {"33", "59"} <= set(geom.segments)
    # and they carry a real area, so they weigh correctly in every sum
    assert all(geom.area_cm2(s) > 0 for s in ("33", "59"))


def test_equal_areas_is_a_simplification_not_a_correction():
    geom = registry.get("gen1_r2d2_72")
    equal = geom.equal_areas()
    assert len(set(equal.values())) == 1
    # The total is preserved; only the distribution changes.
    assert sum(equal.values()) == pytest.approx(geom.cell_area_cm2)


def _write(tmp_path, name, spec):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(spec))
    return path


BASE = {
    "key": "gen_test", "name": "test plate",
    "pad_w_mm": 10.0, "pad_h_mm": 10.0, "n_cols": 4, "n_rows": 2,
}


def test_explicit_segments_load(tmp_path):
    spec = BASE | {"segments": [
        {"name": "1", "col0": 1, "col1": 2, "row0": 1, "row1": 2},
        {"name": "2", "col0": 3, "col1": 4, "row0": 1, "row1": 2},
    ]}
    geom = registry.load_spec(_write(tmp_path, "explicit", spec))
    assert geom.n_segments == 2
    assert geom.self_check()["tiles_exactly"]
    assert geom.area_cm2("1") == pytest.approx(4.0)      # 2 x 2 pads of 1 cm2
    assert geom.centroid_mm("1") == (10.0, 10.0)


def test_strip_form_expands_to_the_same_thing(tmp_path):
    spec = BASE | {
        "bands": {"half": [[1, 1], [2, 2]]},
        "strips": [{"cols": [1, 2], "bands": "half"},
                   {"cols": [3, 4], "bands": "half"}],
        "numbering": [{"strips": [0, 1], "start": 1}],
    }
    geom = registry.load_spec(_write(tmp_path, "strips", spec))
    assert geom.n_segments == 4
    assert geom.self_check()["tiles_exactly"]


def test_a_gap_in_the_spec_is_reported_not_hidden(tmp_path):
    spec = BASE | {"segments": [
        {"name": "1", "col0": 1, "col1": 2, "row0": 1, "row1": 2},
    ]}
    geom = registry.load_spec(_write(tmp_path, "gappy", spec))
    check = geom.self_check()
    assert not check["tiles_exactly"]
    assert len(check["uncovered_pads"]) == 4
    assert not check["area_closes"]


def test_an_overlap_is_reported(tmp_path):
    spec = BASE | {"segments": [
        {"name": "1", "col0": 1, "col1": 3, "row0": 1, "row1": 2},
        {"name": "2", "col0": 3, "col1": 4, "row0": 1, "row1": 2},
    ]}
    check = registry.load_spec(_write(tmp_path, "overlap", spec)).self_check()
    assert not check["tiles_exactly"]
    assert check["overlapping_pads"]


def test_out_of_range_segment_is_rejected(tmp_path):
    spec = BASE | {"segments": [
        {"name": "1", "col0": 1, "col1": 9, "row0": 1, "row1": 2},
    ]}
    with pytest.raises(registry.PlateSpecError):
        registry.load_spec(_write(tmp_path, "oob", spec))


def test_extra_spec_directory_is_picked_up(tmp_path, monkeypatch):
    spec = BASE | {"key": "gen9_extra", "segments": [
        {"name": "1", "col0": 1, "col1": 4, "row0": 1, "row1": 2},
    ]}
    _write(tmp_path, "gen9_extra", spec)
    monkeypatch.setenv(registry.ENV_SPEC_DIR, str(tmp_path))
    registry.reload()
    try:
        assert "gen9_extra" in registry.available()
        assert "gen1_r2d2_72" in registry.available()      # built-ins survive
    finally:
        registry.reload()


def test_template_is_not_offered_as_a_generation():
    # Files starting with '_' are templates, not plates.
    assert not any(k.startswith("_") for k in registry.available())


# ---------------------------------------------------------------------------
# a layout that has not been checked must say so
# ---------------------------------------------------------------------------

def test_gen2_tiles_the_pad_grid_exactly_once():
    """The Gen-2 layout is now a reconstruction that closes, not a copy of Gen 1.

    It was previously shipped as the Gen-1 arrangement under a Gen-2 name with
    an "unverified" flag. That draws a plausible map of the wrong plate. The
    layout here is read from the 72 printed label pads on Coordinates(blue).pdf
    and is pinned by three constraints at once: mirror symmetry of the label
    set about the plate centre, every label inside its own segment, and exact
    tiling of the 45x20 pad grid.
    """
    geom = registry.get("gen2_r2d2_naboo_72")
    assert geom.n_segments == 72
    assert geom.verified is True
    check = geom.self_check()
    assert check["tiles_exactly"], check["uncovered_pads"][:5]
    assert check["area_closes"]
    assert check["pads_covered"] == check["pads_total"] == 900


def test_gen2_is_a_different_plate_from_gen1():
    """The two generations must not silently agree.

    Segments 1..36 keep their columns but four wide strips gave two pad rows
    each to a new edge segment, so 9..16 and 21..28 are smaller here. Carrying
    an area across generations is the mistake this guards against.
    """
    g1 = registry.get("gen1_r2d2_72")
    g2 = registry.get("gen2_r2d2_naboo_72")
    a1, a2 = g1.areas(), g2.areas()
    differing = [s for s in a1 if abs(a1[s] - a2[s]) > 1e-9]
    assert len(differing) >= 40
    for s in ("9", "16", "21", "28"):
        assert a1[s] == pytest.approx(8.470, abs=1e-3)
        assert a2[s] == pytest.approx(6.776, abs=1e-3)
    # The notes must still name the one thing that is a reading rather than a
    # measurement, and what would settle it.
    assert "KiCad" in g2.notes
    assert "within one pad row" in g2.notes


def test_gen2_shares_the_pad_grid_that_the_drawing_confirms():
    gen1, gen2 = registry.get("gen1_r2d2_72"), registry.get("gen2_r2d2_naboo_72")
    for attr in ("pad_w_mm", "pad_h_mm", "n_cols", "n_rows"):
        assert getattr(gen2, attr) == getattr(gen1, attr)
    assert gen2.self_check()["tiles_exactly"]
    assert gen2.cell_area_cm2 == pytest.approx(304.92, abs=1e-6)


def test_gen1_stays_verified():
    assert registry.get("gen1_r2d2_72").verified is True


def test_verified_defaults_to_true_for_a_spec_that_says_nothing(tmp_path):
    spec = BASE | {"segments": [
        {"name": "1", "col0": 1, "col1": 4, "row0": 1, "row1": 2}]}
    assert registry.load_spec(_write(tmp_path, "quiet", spec)).verified is True
