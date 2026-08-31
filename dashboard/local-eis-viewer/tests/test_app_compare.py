"""Comparing two operating points along the gas paths.

The plate's four ports are at the corners and the two circuits cross:
hydrogen bottom-left to top-right, oxygen bottom-right to top-left.  So a
profile drawn against x is right for one gas and mirrored for the other --
and the corner where flooding starts, the oxygen outlet at top-left, would
be drawn at the dry end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.model import RunData
from app.plates import registry
from app.views import compare


def _pc():
    root = Path(__file__).resolve().parents[1] / "local_eis"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import plate_conditions
    return plate_conditions


GEOM = registry.get(registry.default_key())
SELECTION = {"kind": "results", "measurement_id": "2612025",
             "condition": "45A", "plate_key": GEOM.key}


def _run(condition, r_mt):
    """A RunData carrying one scalar per segment, plus a token spectrum."""
    segs = sorted(GEOM.centroids(), key=int)
    frame = pd.DataFrame({
        "segment": segs,
        "R_mt": [r_mt[s] for s in segs],
        "R_ohmic": [70.0] * len(segs),
        "R_ct": [40.0] * len(segs),
    })
    freq = np.geomspace(1.0, 1000.0, 12)
    spectra = pd.DataFrame({
        "segment": np.repeat(segs, len(freq)),
        "freq_hz": np.tile(freq, len(segs)),
        "z_real": np.tile(70.0 + 40.0 / (1 + (freq / 10) ** 2), len(segs)),
        "z_imag": np.tile(-40.0 * (freq / 10) / (1 + (freq / 10) ** 2), len(segs)),
    })
    return RunData(measurement_id="2612025", condition=condition,
                   plate_key=GEOM.key, segments=frame, spectra=spectra)


@pytest.fixture
def flooded(monkeypatch):
    """45 A flat, 450 A rising towards the OXYGEN outlet."""
    PC = _pc()
    xi = PC.flow_coordinate(GEOM.centroids(), "O2")
    runs = {"45A": _run("45A", {s: 10.0 for s in xi}),
            "450A": _run("450A", {s: 10.0 * (1 + 3.0 * xi[s]) for s in xi})}
    monkeypatch.setattr(compare, "_load",
                        lambda selection, condition: runs[condition])
    return runs


@pytest.fixture
def uniform(monkeypatch):
    runs = {"45A": _run("45A", {s: 10.0 for s in GEOM.centroids()}),
            "450A": _run("450A", {s: 25.0 for s in GEOM.centroids()})}
    monkeypatch.setattr(compare, "_load",
                        lambda selection, condition: runs[condition])
    return runs


def test_flooding_at_the_oxygen_outlet_is_called_what_it_is(flooded) -> None:
    _fig, _prof, _spec, status = compare.render(SELECTION, "450A", "R_mt", "O2")
    text = str(status)
    assert "Concentrated at the outlet" in text
    assert "liquid water" in text


def test_a_uniform_rise_is_not_called_flooding(uniform) -> None:
    """Every resistance grows with current; that alone is not local."""
    _fig, _prof, _spec, status = compare.render(SELECTION, "450A", "R_mt", "O2")
    text = str(status)
    assert "uniform rise" in text
    assert "working harder" in text


def test_the_same_data_reads_differently_along_the_other_gas(flooded) -> None:
    """The reason the port map matters.

    A pattern that rises along the oxygen path is flat-to-falling along the
    hydrogen path, because the two run in opposite directions.  Both
    readings are of the same numbers; only the coordinate differs.
    """
    _f, _p, _s, on_o2 = compare.render(SELECTION, "450A", "R_mt", "O2")
    _f, _p, _s, on_h2 = compare.render(SELECTION, "450A", "R_mt", "H2")
    assert "Concentrated at the outlet" in str(on_o2)
    assert "Concentrated at the outlet" not in str(on_h2)


def test_the_profile_axis_names_the_corners(flooded) -> None:
    """A reader has to be able to tell which end of the axis is the inlet."""
    _fig, prof, _spec, _status = compare.render(SELECTION, "450A", "R_mt", "O2")
    title = prof.layout.xaxis.title.text
    assert "bottom-right" in title and "top-left" in title


def test_both_conditions_are_drawn_on_the_spectra_panel(flooded) -> None:
    _fig, _prof, spec, _status = compare.render(SELECTION, "450A", "R_mt", "O2")
    names = {tr.name for tr in spec.data}
    assert names == {"45A", "450A"}


def test_a_missing_second_condition_explains_itself(monkeypatch) -> None:
    def _boom(selection, condition):
        raise FileNotFoundError("no results for 450A")
    monkeypatch.setattr(compare, "_load", _boom)
    _f, _p, _s, status = compare.render(SELECTION, "450A", "R_mt", "O2")
    assert "could not load both conditions" in str(status)


def test_the_tab_is_registered_in_the_app() -> None:
    from app.app import build_app
    app = build_app()
    labels = []

    def walk(node):
        for child in (getattr(node, "children", None) or []):
            if isinstance(child, (list, tuple)):
                for c in child:
                    walk(c)
            else:
                if getattr(child, "label", None):
                    labels.append(child.label)
                walk(child)
    walk(app.layout)
    assert "Compare conditions" in labels
