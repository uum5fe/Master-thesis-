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


def test_known_faults_travel_with_the_geometry():
    geom = registry.get("gen1_r2d2_72")
    assert set(geom.known_bad) == {"33", "59"}


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
