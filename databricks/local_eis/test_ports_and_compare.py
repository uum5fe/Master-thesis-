"""The four corner ports, and comparing two operating points along them.

The plate's gases enter and leave at the corners and cross:

    top-left  O2 out              H2 out  top-right
    bottom-left  H2 in             O2 in  bottom-right

so hydrogen runs bottom-left to top-right and oxygen runs bottom-right to
top-left -- opposite directions across the plate.  A single "flow axis with
an inlet end" cannot represent that: whichever way it points, it is right
for one gas and mirrored for the other.
"""

from __future__ import annotations

import numpy as np
import pytest

import compare_conditions as C
import plate_conditions as pc
import r2d2_geometry as geom


CENT = geom.centroids()


# ---------------------------------------------------------------------------
# the port map
# ---------------------------------------------------------------------------

def test_the_two_gases_run_in_opposite_directions() -> None:
    """The property the old single-axis model could not express."""
    h2 = pc.flow_coordinate(CENT, "H2")
    o2 = pc.flow_coordinate(CENT, "O2")
    xs = np.array([CENT[k][0] for k in h2])
    r_h2 = np.corrcoef(xs, [h2[k] for k in h2])[0, 1]
    r_o2 = np.corrcoef(xs, [o2[k] for k in o2])[0, 1]
    assert r_h2 > 0.5, "hydrogen should advance with increasing x"
    assert r_o2 < -0.5, "oxygen should advance with decreasing x"


def test_the_corners_are_where_the_user_says_they_are() -> None:
    h2_in, h2_out = pc.gas_path(gas="H2")
    o2_in, o2_out = pc.gas_path(gas="O2")
    assert (h2_in, h2_out) == ("bottom-left", "top-right")
    assert (o2_in, o2_out) == ("bottom-right", "top-left")


def test_top_is_minimum_y_because_the_origin_is_the_top_left_pad() -> None:
    """Easy to get backwards, and it mirrors every map when it is."""
    ys = [c[1] for c in CENT.values()]
    assert pc.corner_xy(CENT, "top-left")[1] == pytest.approx(min(ys))
    assert pc.corner_xy(CENT, "bottom-left")[1] == pytest.approx(max(ys))


def test_the_flow_coordinate_runs_zero_to_one_from_inlet_to_outlet() -> None:
    for gas in ("H2", "O2"):
        xi = pc.flow_coordinate(CENT, gas)
        assert min(xi.values()) == pytest.approx(0.0)
        assert max(xi.values()) == pytest.approx(1.0)


def test_a_diagonal_path_reduces_to_the_old_axis_when_it_is_one() -> None:
    """Ports on opposite edges rather than corners: same answer as before."""
    cent = {str(i): (float(x), 60.0) for i, x in enumerate(np.linspace(0, 252, 10),
                                                          start=1)}
    ports = (pc.Port("O2", "in", "bottom-left"),
             pc.Port("O2", "out", "top-right"))
    new = pc.flow_coordinate(cent, "O2", ports)
    old = pc._flow_coordinate(cent, axis="x", inlet="min")
    for k in cent:
        assert new[k] == pytest.approx(old[k], abs=1e-9)


def test_the_cathode_path_is_the_default_because_water_is_made_there() -> None:
    fields = pc.condition_fields(
        CENT, geom.areas(),
        pc.PortState(current_a=450.0, t_in_c=70.0, t_out_c=62.0,
                     p_in_bara=1.5, p_out_bara=1.39, dew_in_c=55.0,
                     air_dry_flow_nl_min=8.0))
    assert any("O2: bottom-right to top-left" in n
               for n in fields["humidity"].notes)


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------

def _summary(values):
    """A fake plate_summary keyed by segment."""
    return {seg: {"R_ohmic": 70.0, "R_ct": 40.0, "R_mt": v, "R_pol": 60.0,
                  "j_dc": 0.5, "T_degC": 60.0}
            for seg, v in values.items()}


def test_flooding_at_the_oxygen_outlet_is_recognised() -> None:
    """R_mt rising towards the O2 outlet is what liquid water looks like."""
    xi = pc.flow_coordinate(CENT, "O2")
    a = _summary({s: 10.0 for s in CENT})
    b = _summary({s: 10.0 * (1.0 + 3.0 * xi[s]) for s in CENT})
    result = C.compare(a, b, "45A", "450A", CENT)
    verdict = C.flooding_verdict(result, "O2")
    assert verdict["ok"]
    assert verdict["outlet_over_inlet"] > 1.2
    assert "accumulating towards the outlet" in verdict["reads_as"]


def test_a_uniform_rise_is_not_called_flooding() -> None:
    """Every resistance grows with current; that alone says nothing local."""
    a = _summary({s: 10.0 for s in CENT})
    b = _summary({s: 25.0 for s in CENT})
    verdict = C.flooding_verdict(C.compare(a, b, "45A", "450A", CENT), "O2")
    assert verdict["ok"]
    assert "uniform rise" in verdict["reads_as"]


def test_a_gradient_along_the_wrong_gas_is_not_mistaken_for_flooding() -> None:
    """The reason the port map matters at all.

    A pattern that rises along the HYDROGEN path is flat-to-falling along
    the oxygen path.  Judged against the oxygen coordinate -- the one the
    cathode water balance runs on -- it must not read as flooding.
    """
    xi_h2 = pc.flow_coordinate(CENT, "H2")
    a = _summary({s: 10.0 for s in CENT})
    b = _summary({s: 10.0 * (1.0 + 3.0 * xi_h2[s]) for s in CENT})
    verdict = C.flooding_verdict(C.compare(a, b, "45A", "450A", CENT), "O2")
    assert verdict["rank_correlation_with_path"] < 0
    assert "uniform rise" in verdict["reads_as"]


def test_both_coordinates_are_carried_on_every_row() -> None:
    a = _summary({s: 10.0 for s in CENT})
    result = C.compare(a, a, "45A", "450A", CENT)
    row = result["rows"][0]
    assert "xi_H2" in row and "xi_O2" in row
    assert row["R_mt_ratio"] == pytest.approx(1.0)
    assert row["T_degC_delta"] == pytest.approx(0.0)


def test_the_profile_is_binned_inlet_to_outlet() -> None:
    xi = pc.flow_coordinate(CENT, "O2")
    a = _summary({s: 10.0 for s in CENT})
    b = _summary({s: 10.0 * (1.0 + 3.0 * xi[s]) for s in CENT})
    bins = C.along_path(C.compare(a, b, "45A", "450A", CENT), "R_mt", "O2")
    assert len(bins) >= 4
    medians = [x["median"] for x in bins]
    assert medians[-1] > medians[0], "the profile must run inlet to outlet"
