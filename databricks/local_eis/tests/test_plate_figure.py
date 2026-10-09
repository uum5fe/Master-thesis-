"""plate_figure: the viewer-style plate map draws, and uses the fixed scale."""
import numpy as np

import config
import plate_figure
import r2d2_geometry as geom


def _values(v0=60.0):
    return {s: v0 + int(s) % 7 for s in geom.ACTIVE_PLATE.segments}


def test_draws_every_resistance_on_its_fixed_scale(tmp_path):
    import matplotlib.pyplot as plt
    for p in ("R_ohmic", "R_ct", "R_mt", "R_pol"):
        fig = plate_figure.draw_flow_plate(_values(), p, title="test",
                                           classes={"33": "substituted"})
        cb = fig.axes[1]
        lo, hi = config.heatmap_limits(p)
        assert np.isclose(cb.get_ylim()[0], lo) and np.isclose(cb.get_ylim()[1], hi)
        fig.savefig(tmp_path / f"{p}.png", dpi=40)
        plt.close(fig)
        assert (tmp_path / f"{p}.png").stat().st_size > 0


def test_missing_segments_and_nan_do_not_break_the_drawing():
    import matplotlib.pyplot as plt
    vals = {"1": 50.0, "2": float("nan"), "3": "", "4": None}
    fig = plate_figure.draw_flow_plate(vals, "R_ohmic")
    plt.close(fig)


def test_write_condition_maps_reads_a_plate_summary(tmp_path):
    csv = tmp_path / "plate_summary.csv"
    rows = ["segment,class,R_ohmic,R_ct,R_mt,R_pol"]
    rows += [f"{s},measured,60,120,80,200" for s in geom.ACTIVE_PLATE.segments]
    csv.write_text("\n".join(rows) + "\n")
    out = plate_figure.write_condition_maps(csv, tmp_path, "2612030 / 45A", dpi=40)
    assert set(out) == {"R_ohmic", "R_ct", "R_mt", "R_pol"}


def test_interpolated_render_marks_each_segment_at_its_drawing_position(tmp_path):
    """Every measured segment is outlined as its label pad -- the red number
    box of the plate drawing, 5.60 x 6.05 mm -- with its number inside."""
    import matplotlib.pyplot as plt
    geom.use_plate("gen1")
    vals = _values()
    fig = plate_figure.draw_flow_plate(vals, "R_ohmic", render="interpolated")
    ax = fig.axes[0]
    assert len(ax.images) == 1                         # the interpolated field
    boxes = {}
    from matplotlib.patches import Rectangle
    for p in ax.patches:
        if (isinstance(p, Rectangle)
                and np.isclose(p.get_width(), geom.PAD_W_MM)
                and np.isclose(p.get_height(), geom.PAD_H_MM)
                and p.get_facecolor()[3] == 0):
            x, y = p.get_xy()
            boxes[(round(x + geom.PAD_W_MM / 2, 3),
                   round(y + geom.PAD_H_MM / 2, 3))] = p
    assert len(boxes) == len(vals)
    # the drawing: segment 37 at (2.8, 3.025), 1 at (14, 15.125),
    # 72 at (249.2, 117.975), 60 at (198.8, 117.975)
    for xy in ((2.8, 3.025), (14.0, 15.125), (249.2, 117.975),
               (198.8, 117.975)):
        assert xy in boxes
    numbers = {t.get_text(): t.get_position() for t in ax.texts
               if t.get_text() in vals}
    assert set(numbers) == set(vals)
    assert np.allclose(numbers["37"], (2.8, 3.025))
    plt.close(fig)


def test_write_condition_maps_both_styles(tmp_path):
    csv = tmp_path / "plate_summary.csv"
    rows = ["segment,class,R_ohmic"] + [f"{s},measured,{v}"
                                        for s, v in _values().items()]
    csv.write_text("\n".join(rows))
    out = plate_figure.write_condition_maps(csv, tmp_path, params=("R_ohmic",),
                                            dpi=30, render="both")
    assert set(out) == {"R_ohmic", "R_ohmic_interp"}
