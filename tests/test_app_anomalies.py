"""Finding the segment that misbehaves, without flagging the gradient.

The trap this guards is the reason the tab exists. A plate has a real spatial
gradient -- reactant is consumed along the channel, so R_p rises from inlet to
outlet -- and ranking segments against the PLATE mean flags the entire outlet,
which is physics rather than a fault. Worse, it hides the genuinely faulty
segment when that segment sits somewhere the plate is already extreme.

So the tests below are a pair, and the first one matters more than the second:
a detector that finds the planted fault but also flags forty innocent segments
is useless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local_eis"))

import neighbours as nb            # noqa: E402
import r2d2_geometry as geom       # noqa: E402


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------


def test_neighbours_are_segments_that_actually_touch():
    adj = nb.adjacency()
    assert len(adj) == 72
    # adjacency is symmetric, and nothing is its own neighbour
    for name, ring in adj.items():
        assert name not in ring
        for other in ring:
            assert name in adj[other]
    sizes = [len(v) for v in adj.values()]
    assert min(sizes) >= 2 and max(sizes) <= 10


def test_adjacency_comes_from_pads_not_from_centroid_distance():
    """The gen1 edge segments are staircases.

    A centroid-distance rule pairs a staircase with whatever is near its
    centre of mass rather than with what it touches, which on this plate is
    not the same set.
    """
    adj = nb.adjacency()
    cents = geom.centroids()
    mismatched = 0
    for name, ring in adj.items():
        cx, cy = cents[name]
        near = sorted(cents, key=lambda o: (cents[o][0] - cx) ** 2
                      + (cents[o][1] - cy) ** 2)[1:len(ring) + 1]
        if set(near) != set(ring):
            mismatched += 1
    assert mismatched > 0, (
        "if k-nearest-by-centroid agreed with true edge adjacency everywhere, "
        "the pad-set computation would be pointless")


# ---------------------------------------------------------------------------
# The pair that matters
# ---------------------------------------------------------------------------


def _gradient():
    """Inlet-to-outlet gradient: every segment differs from the plate mean,
    and not one of them is a fault."""
    return {k: 0.30 - 0.0008 * cx for k, (cx, _cy) in geom.centroids().items()}


def test_a_pure_spatial_gradient_flags_nothing():
    assert nb.scalar_outliers(_gradient(), param="rs") == []


def test_a_global_ranking_would_rank_by_position_not_by_fault():
    """What the local comparison buys, demonstrated rather than asserted.

    On a plate carrying nothing but a gradient, ranking segments by their
    departure from the PLATE mean sorts them by where they sit: the worst
    offenders are all at one end. That ranking is the gradient. The local
    comparison returns nothing at all, which is the right answer.
    """
    values = _gradient()
    cents = geom.centroids()
    v = np.array(list(values.values()))
    # SIGNED, not absolute: a linear gradient is extreme at both ends, so
    # ranking by |z| picks both and says nothing. The "worst offenders" a
    # reader would actually look at are the highest values.
    z_global = {k: (x - v.mean()) / v.std() for k, x in values.items()}
    worst10 = sorted(z_global, key=lambda k: -z_global[k])[:10]

    xs = [cents[k][0] for k in worst10]
    span = max(c[0] for c in cents.values()) - min(c[0] for c in cents.values())
    assert (max(xs) - min(xs)) < 0.35 * span, (
        "a global ranking should pick out one end of the plate")

    assert nb.scalar_outliers(values, param="rs") == []


def test_one_planted_fault_is_the_segment_that_comes_back():
    values = _gradient()
    values["34"] *= 1.9
    found = nb.scalar_outliers(values, param="rs")
    assert [f.segment for f in found] == ["34"]
    assert found[0].z > 5
    assert found[0].direction == "high"
    assert "34" not in found[0].ring


def test_two_adjacent_faults_do_not_hide_each_other():
    """The reason the spread is a MAD and not a standard deviation.

    Faults come in adjacent pairs -- a blocked channel affects both segments
    under it. One faulty neighbour must not inflate the ring's spread enough
    to conceal the segment under test.
    """
    adj = nb.adjacency()
    a = "34"
    b = sorted(adj[a], key=lambda s: int(s))[0]
    values = _gradient()
    values[a] *= 1.9
    values[b] *= 1.9
    found = {f.segment for f in nb.scalar_outliers(values, param="rs")}
    assert a in found and b in found


def test_a_corner_segment_is_reported_with_its_caveat():
    adj = nb.adjacency()
    small = min(adj, key=lambda k: len(adj[k]))
    values = _gradient()
    values[small] *= 2.5
    found = [f for f in nb.scalar_outliers(values, param="rs")
             if f.segment == small]
    assert found
    if found[0].n_ring < nb.MIN_RING:
        assert "barely estimated" in found[0].note


# ---------------------------------------------------------------------------
# Where in the spectrum
# ---------------------------------------------------------------------------


def _plate_spectra(fault=None, kind="ohmic"):
    f = np.logspace(-1, 3, 25)
    out = {}
    for name, (cx, _cy) in geom.centroids().items():
        rs = 70.0 + 0.02 * cx
        rp = 200.0 + 0.05 * cx
        if name == fault:
            if kind == "ohmic":
                rs *= 2.0            # offset at every frequency
            else:
                rp *= 2.2            # a bigger arc, mid band
        z = rs + rp / (1 + 1j * 2 * np.pi * f * 2e-3)
        out[name] = (f, z)
    return out


def test_an_ohmic_fault_shows_up_across_the_whole_band():
    sf = [s for s in nb.spectrum_outliers(_plate_spectra("34", "ohmic"))
          if s.segment == "34"]
    assert sf
    assert np.nanmedian(sf[0].dev_db) > 1.0
    assert sf[0].n_ring >= 2


def test_a_kinetic_fault_shows_up_in_the_mid_band_not_the_top():
    sf = [s for s in nb.spectrum_outliers(_plate_spectra("34", "kinetic"))
          if s.segment == "34"]
    assert sf
    f, d = sf[0].freq_hz, sf[0].dev_db
    low = np.nanmedian(d[f < 10])
    high = np.nanmedian(d[f > 300])
    assert low > high, "a charge-transfer fault must not look like an ohmic one"


def test_a_healthy_plate_produces_no_large_spectral_departure():
    worst = max(abs(np.nan_to_num(s.worst()[1]))
                for s in nb.spectrum_outliers(_plate_spectra()))
    assert worst < 0.5


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------


def test_the_tab_is_registered_and_renders():
    from app.app import build_app

    app = build_app()
    labels = [t.label for t in app.layout.children[3].children[0].children]
    values = [t.value for t in app.layout.children[3].children[0].children]
    assert "Neighbour anomalies" in labels
    assert "tab-anomalies" in values

    key = next(k for k in app.callback_map if "tab-body" in k)
    render = app.callback_map[key]["callback"].__wrapped__
    assert render("tab-anomalies") is not None


def test_only_one_plate_map_tab_remains():
    """The lattice view is gone; the realistic one is the plate map."""
    from app.app import build_app

    labels = [t.label for t in build_app().layout.children[3].children[0].children]
    assert labels.count("Plate map") == 1
    assert "Plate (realistic)" not in labels


def test_analyse_returns_rows_the_table_can_render():
    values = _gradient()
    values["34"] *= 1.9
    result = nb.analyse({"rs": values})
    assert result["n_segments_flagged"] == 1
    row = result["rows"][0]
    for key in ("segment", "param", "value", "ring_median", "z", "direction",
                "n_ring", "neighbours"):
        assert key in row
    assert row["segment"] == "34"
