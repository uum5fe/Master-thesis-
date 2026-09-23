"""The widgets cell, run cold.

The cell resolves the order, then discovers everything that depends on it:
the condition list, the Gamry folder, the build token. It used to read the
order at the BOTTOM, after those blocks, which on a fresh kernel meant LEEPA
did not exist when they ran.

That failed silently where it mattered most. The condition discovery sits in
a try/except, so the NameError was swallowed and CONDITIONS fell back to a
hard-coded list -- the dropdown never read the disk on the first run of a
session, and worked on the second because LEEPA was left over from the first.
A notebook that behaves differently on the second run is a notebook nobody
can reason about, so the order is pinned here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import gamry_compare
import r2d2_geometry as geom


RUNNER = Path(__file__).with_name("Local EIS Pipeline Runner.py")
VOLUME = "/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev"


class _Widgets:
    """Databricks widgets, starting empty as they do on a cold kernel."""

    def __init__(self):
        self.vals: dict[str, str] = {}

    def dropdown(self, name, default, choices, label=None):
        self.vals.setdefault(name, default)

    def text(self, name, default, label=None):
        self.vals.setdefault(name, default)

    def get(self, name):
        if name not in self.vals:
            raise Exception(f"no widget named {name}")
        return self.vals[name]

    def remove(self, name):
        self.vals.pop(name, None)


class _DBUtils:
    def __init__(self):
        self.widgets = _Widgets()


def _widgets_cell() -> str:
    for chunk in RUNNER.read_text().split("\n# COMMAND ----------\n"):
        if "WIDGETS  —  Order ID" in chunk:
            return chunk
    raise AssertionError("the widgets cell is gone or was renamed")


@pytest.fixture()
def volume(tmp_path):
    """A stand-in Volume: two conditions on disk, one Gamry campaign."""
    (tmp_path / "Famos").mkdir()
    (tmp_path / "Gamry").mkdir()
    for cond in ("45A", "450A"):
        for card in (1, 2):
            (tmp_path / "Famos" /
             f"Leepa_RO2612030_Current_{cond}_Test_1_Karte_{card}.DAT").touch()
    (tmp_path / "Gamry" / "V26_092_HFR_101_CurrVal_45.dta").touch()
    (tmp_path / "Gamry" /
     "20260907_RO2612030-01_V26_092_lokale_EIS.mf4").touch()
    return tmp_path


def _run_cell(volume, preset: dict | None = None) -> dict:
    db = _DBUtils()
    if preset:
        db.widgets.vals.update(preset)
    ns = {"Path": Path, "dbutils": db, "spark": None, "geom": geom,
          "gamry_compare": gamry_compare, "__name__": "__main__"}
    src = _widgets_cell().replace(VOLUME, str(volume))
    exec(compile(src, "widgets_cell", "exec"), ns)
    return ns


# ---------------------------------------------------------------------------


def test_the_cell_runs_on_a_cold_kernel(volume) -> None:
    """No leftover state, no widgets yet. This raised NameError."""
    ns = _run_cell(volume)
    assert ns["LEEPA"] == "2612030"


def test_the_conditions_come_from_the_disk_not_the_fallback(volume) -> None:
    """The silent half of the bug.

    The fallback list is ['450A', '60A', '45A', '150A']. The disk has two
    conditions. Getting four back means the discovery never ran.
    """
    ns = _run_cell(volume)
    assert ns["CONDITIONS"] == ["450A", "45A"]
    assert "60A" not in ns["CONDITIONS"]


def test_the_gamry_build_is_resolved_for_the_selected_order(volume) -> None:
    ns = _run_cell(volume)
    assert ns["GAMRY_VERSION"] == "V26_092"
    assert Path(ns["GAMRY_DIR"]).name == "Gamry"


def test_the_search_roots_follow_the_order_rather_than_being_baked(volume) -> None:
    """A list built once from whatever LEEPA held is wrong twice: it cannot
    be built before the order is known, and it keeps the previous order's
    folders when the widget changes."""
    ns = _run_cell(volume)
    roots = ns["gamry_search_roots"]("9999999")
    assert any("9999999" in str(r) for r in roots)
    assert not any("2612030" in str(r) for r in roots)


def test_choosing_a_different_order_changes_what_is_discovered(volume) -> None:
    for cond in ("60A",):
        (volume / "Famos" /
         f"Leepa_RO2611976_Current_{cond}_Test_1_Karte_1.DAT").touch()
    ns = _run_cell(volume, {"leepa_id": "2611976"})
    assert ns["LEEPA"] == "2611976"
    assert ns["CONDITIONS"] == ["60A"]


def test_the_segment_widgets_parse(volume) -> None:
    ns = _run_cell(volume, {"exclude_segments": "59",
                            "substitute_segments": "33",
                            "fill_gaps": "yes"})
    assert ns["EXCLUDE_SEGMENTS"] == frozenset({"59"})
    assert ns["SUBSTITUTE_SEGMENTS"] == frozenset({"33"})
    assert ns["FILL_GAPS"] is True


def test_an_empty_volume_says_so_rather_than_inventing_conditions(volume,
                                                                  capsys):
    """A fallback that does not announce itself is how the first bug hid."""
    for f in (volume / "Famos").iterdir():
        f.unlink()
    ns = _run_cell(volume)
    out = capsys.readouterr().out
    assert ns["CONDITIONS"] == ["450A", "60A", "45A", "150A"]
    assert "fallback" in out
