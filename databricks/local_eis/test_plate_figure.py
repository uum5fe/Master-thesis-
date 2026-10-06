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
