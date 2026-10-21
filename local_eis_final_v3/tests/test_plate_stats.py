"""Plate mean / median over the 36 representative segments: segment 1..36
stands for its 5x5-pad tile, the edge segments 37..72 inside the tiles are
not counted."""
import numpy as np
import pytest

import plate_stats
import r2d2_geometry as geom


@pytest.mark.parametrize("plate", ["gen1", "gen2"])
def test_the_36_tiles_cover_the_plate_once_with_the_label_in_the_centre(plate):
    p = geom.plate(plate)
    seen = set()
    for k in plate_stats.REPRESENTATIVE:
        c0, c1, r0, r1 = plate_stats.tile_of(k)
        assert (c1 - c0 + 1, r1 - r0 + 1) == (5, 5)
        assert (p.label_col[k], p.label_row[k]) == (c0 + 2, r0 + 2)
        seen |= {(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)}
    assert len(seen) == geom.N_COLS * geom.N_ROWS == 900


def test_gen1_labels_are_the_drawings_red_boxes():
    xy = geom.label_xy("gen1")
    assert np.allclose(xy["37"], (2.8, 3.025))
    assert np.allclose(xy["1"], (14.0, 15.125))
    assert np.allclose(xy["49"], (53.2, 3.025))
    assert np.allclose(xy["40"], (2.8, 69.575))
    assert np.allclose(xy["67"], (249.2, 3.025))
    assert np.allclose(xy["33"], (238.0, 15.125))
    p = geom.plate("gen1")
    for k, s in p.segments.items():         # every label pad is its own
        assert (p.label_col[k], p.label_row[k]) in s.pads


def test_tile_1_covers_the_corner_segments():
    assert plate_stats.covered("1", "gen1") == ["37", "38", "43", "44"]


def test_edge_segments_do_not_enter_the_mean_or_median():
    vals = {str(k): 50.0 + (k % 3) for k in range(1, 37)}
    vals.update({str(k): 500.0 for k in range(37, 73)})
    st = plate_stats.tile_stats(vals)
    ref = np.array([vals[str(k)] for k in range(1, 37)])
    assert st["n"] == 36 and st["missing"] == []
    assert st["mean"] == pytest.approx(ref.mean())
    assert st["median"] == pytest.approx(np.median(ref))
    assert st["max"] < 500


def test_a_missing_or_rebuilt_tile_is_left_out_and_counted():
    vals = {str(k): 60.0 for k in range(1, 37)}
    vals["33"] = float("nan")
    vals["5"] = 90.0
    st = plate_stats.tile_stats(vals, classes={"5": "substituted"})
    assert st["n"] == 34 and st["missing"] == ["5", "33"]
    assert st["median"] == 60.0
    assert "34 of 36 tiles" in plate_stats.summary_line(
        vals, {"5": "substituted"})


def test_statistics_table_from_a_plate_summary(tmp_path):
    import pandas as pd
    rows = [{"segment": k, "class": "measured", "R_ohmic": 40.0 + k,
             "R_pol": 100.0} for k in range(1, 73)]
    csv = tmp_path / "plate_summary.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    out = plate_stats.write_statistics(csv)
    st = pd.read_csv(out).set_index("parameter")
    assert st.loc["R_ohmic", "n_tiles"] == 36
    assert st.loc["R_ohmic", "median"] == pytest.approx(np.median(
        [40.0 + k for k in range(1, 37)]))
    assert st.loc["R_pol", "mean"] == pytest.approx(100.0)
