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


def test_gen1_areas_come_from_the_plant_pad_map():
    """The areas are counted from get_900_matrix, not inferred from a shape.

    The numbers here are the ones that changed when the real pad map arrived:
    the smallest segment is 4 pads, not the 2 a strips-and-bands reconstruction
    predicted, and the spread is 6.2x rather than 12.5x.
    """
    geom = registry.get("gen1_r2d2_72")
    assert geom.n_segments == 72
    assert geom.n_cols * geom.n_rows == 900
    assert geom.cell_area_cm2 == pytest.approx(304.92, abs=1e-6)
    assert geom.verified

    areas = geom.areas()
    pad = geom.pad_area_cm2
    assert min(areas.values()) == pytest.approx(4 * pad, abs=1e-6)
    assert max(areas.values()) == pytest.approx(25 * pad, abs=1e-6)
    assert sum(areas.values()) == pytest.approx(geom.cell_area_cm2, abs=1e-9)
    # Every area is a whole number of pads: nothing here is a fitted value.
    for name, a in areas.items():
        assert abs(a / pad - round(a / pad)) < 1e-9, name


def test_most_gen1_segments_are_not_rectangles():
    """This is what the earlier reconstruction got wrong.

    Describing the plate as strips cut into rectangular bands tiled the grid
    and reproduced every printed label pad, so it looked right, and it still
    put 60 of the 72 areas wrong. The staircases are the reason.
    """
    geom = registry.get("gen1_r2d2_72")
    staircases = [n for n, s in geom.segments.items() if not s.is_rectangle]
    assert len(staircases) == 40
    # Segment 1 is the one that was wrong by more than a factor of two.
    assert geom.segments["1"].n_pads == 7
    assert geom.area_cm2("1") == pytest.approx(7 * geom.pad_area_cm2, abs=1e-9)


def test_a_staircase_is_drawn_as_its_true_outline():
    """A bounding box would colour pads owned by the neighbour.

    Segment 37 is the top-left corner: three pads of row 1, two of row 2, one
    of row 3. Its bounding box is 3x3, so drawing the box would claim three
    pads that belong to segments 1 and 38.
    """
    geom = registry.get("gen1_r2d2_72")
    seg = geom.segments["37"]
    assert seg.n_pads == 6
    assert (seg.col1 - seg.col0 + 1) * (seg.row1 - seg.row0 + 1) == 9

    xs, ys = geom.outline_mm("37")
    assert xs[0] == xs[-1] and ys[0] == ys[-1], "outline is not closed"
    assert None not in xs, "a staircase is one ring, not several"
    # Shoelace area of the traced polygon must equal the pad count exactly.
    area = 0.5 * abs(sum(xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
                         for i in range(len(xs) - 1)))
    assert area / 100.0 == pytest.approx(seg.n_pads * geom.pad_area_cm2, abs=1e-9)


def test_every_label_lands_on_a_pad_the_segment_owns():
    """A label is only meaningful on a tile that belongs to the segment.

    On a staircase the middle of the bounding box can sit in the neighbour, so
    the label point is the pad centroid instead. That this matters is not
    hypothetical: on this plate at least one segment's box centre lands on a
    pad another segment owns, and the centroid never does.
    """
    geom = registry.get("gen1_r2d2_72")

    def pad_at(x, y):
        return (int(x // geom.pad_w_mm) + 1, int(y // geom.pad_h_mm) + 1)

    outside_by_box = outside_by_centroid = 0
    for name, seg in geom.segments.items():
        assert pad_at(*geom.label_point_mm(name)) in seg.pads, name
        x0, y0, x1, y1 = geom.bounds_mm(name)
        if pad_at(0.5 * (x0 + x1), 0.5 * (y0 + y1)) not in seg.pads:
            outside_by_box += 1
        if pad_at(*geom.centroid_mm(name)) not in seg.pads:
            outside_by_centroid += 1
    # Neither of the obvious choices is safe on this plate, which is why the
    # label point snaps to an owned pad rather than using either.
    assert outside_by_box > 0
    assert outside_by_centroid > 0


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
    """Closing is necessary, and it is not sufficient.

    The Gen-2 layout is read from the 72 printed label pads on
    Coordinates(blue).pdf and pinned by mirror symmetry of the label set, every
    label inside its own segment, and exact tiling of the 45x20 pad grid. The
    Gen-1 reconstruction satisfied all three and was still wrong on 60 of 72
    areas, so this test asserts only what it can: that the layout closes. The
    verified flag is checked elsewhere, and it is False.
    """
    geom = registry.get("gen2_r2d2_naboo_72")
    assert geom.n_segments == 72
    check = geom.self_check()
    assert check["tiles_exactly"], check["uncovered_pads"][:5]
    assert check["area_closes"]
    assert check["pads_covered"] == check["pads_total"] == 900


def test_gen2_is_a_different_plate_from_gen1():
    """The two generations must not silently agree.

    Carrying an area across generations is the mistake this guards against.
    """
    g1 = registry.get("gen1_r2d2_72")
    g2 = registry.get("gen2_r2d2_naboo_72")
    a1, a2 = g1.areas(), g2.areas()
    differing = [s for s in a1 if abs(a1[s] - a2[s]) > 1e-9]
    assert len(differing) >= 40
    # Both still cover the same physical plate.
    assert sum(a1.values()) == pytest.approx(sum(a2.values()), abs=1e-9)


def test_gen2_is_marked_unverified_until_its_pad_map_arrives():
    """Gen 2 is still a rectangle reconstruction, and that shape is now known
    to be wrong on this hardware.

    Gen 1 was built from the same kind of evidence, tiled the grid, matched
    every printed label -- and disagreed with the plant's pad map on 60 of 72
    areas. Claiming Gen 2 is verified on that same basis would be repeating a
    mistake we have already seen fail, so it says so instead.
    """
    g2 = registry.get("gen2_r2d2_naboo_72")
    assert not g2.verified
    assert "UNVERIFIED" in g2.describe()
    # It must say what would settle it.
    assert "get_900_matrix" in g2.notes
    assert "pad_matrix" in g2.notes
    # Every segment in it is a rectangle -- the very thing under suspicion.
    assert all(s.is_rectangle for s in g2.segments.values())


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
