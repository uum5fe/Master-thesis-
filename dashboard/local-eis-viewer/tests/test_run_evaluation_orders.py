"""Selecting one order id out of a campaign folder holding several.

The FAMOS root is now the parent that holds one folder per order, so the
order id is the thing that picks a measurement -- and the same order is
written three ways on disk (folder `2612025_27_08`, cards
`RO2612025-01_Current_...`, catalogue key `2612025`). Whichever one someone
copies has to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_evaluation as RE


@pytest.mark.parametrize("written", [
    "2612025",            # the catalogue key
    "RO2612025",          # the card prefix
    "RO2612025-01",       # the card prefix with the station suffix
    "ro2612025-01",       # ... typed in lower case
    "2612025_27_08",      # the folder name
])
def test_every_spelling_of_one_order_selects_it(written):
    assert RE.normalise_order_id(written) == "2612025"


def test_the_station_suffix_is_dropped_not_absorbed():
    """`-01` is a station, not part of the number.

    Stripping non-digits before removing it welds them together --
    "RO2612025-01" becomes "261202501", which matches nothing and looks like
    a path problem rather than a filter one.
    """
    assert RE.normalise_order_id("RO2612025-01") != "261202501"
    assert RE.normalise_order_id("RO2612025-01") == "2612025"


def test_different_orders_stay_different():
    """Tolerance must not collapse two orders into one."""
    ids = {RE.normalise_order_id(x)
           for x in ("RO2611976-01", "RO2612025-01", "RO2611959-01")}
    assert ids == {"2611976", "2612025", "2611959"}


# ---------------------------------------------------------------------------
# "it is not listing" -- say why, per folder
# ---------------------------------------------------------------------------

def _patterns():
    from app.data.sources import famos_patterns
    return famos_patterns()


def _folders(lines):
    """The survey prepends a machine-readable count; drop it."""
    return [l for l in lines if not l.startswith("__folders__")]


def test_a_folder_of_ddf_is_named_with_the_tool_for_it(tmp_path):
    """The case that prompted this: an order recorded by DASYLab.

    FamosSource matches on file name and drops everything else silently, so
    the order simply did not appear and no line of output pointed at it.
    """
    d = tmp_path / "RO2612025_27_08"
    d.mkdir()
    (d / "Messung_450A.ddf").write_bytes(b"\x00" * 16)

    text = "\n".join(_folders(RE.survey_roots([tmp_path], _patterns())))
    assert "RO2612025_27_08" in text
    assert "1 x .ddf" in text
    assert "ddf_source.py probe" in text
    assert "Messung_450A.ddf" in text, "name a real file, not a placeholder"


def test_unmatched_cards_are_shown_with_a_sample_name(tmp_path):
    """A renamed card set is a different problem from a missing one, and the
    name is what tells them apart -- so it has to be in the output."""
    d = tmp_path / "RO2611959_03_07"
    d.mkdir()
    for k in (1, 2, 3):
        (d / f"EIS_2611959_60A_card{k}.DAT").write_bytes(b"")

    text = "\n".join(_folders(RE.survey_roots([tmp_path], _patterns())))
    assert "EIS_2611959_60A_card1.DAT" in text
    assert "EIS_FAMOS_REGEX" in text


def test_a_folder_that_did_produce_a_run_is_not_reported(tmp_path):
    """The survey lists what went WRONG. A working folder in that list would
    train the reader to ignore it."""
    d = tmp_path / "RO2611976_16_07"
    d.mkdir()
    for k in (1, 2):
        (d / f"RO2611976-01_Current_45A_Test_01_Karte_{k}.DAT").write_bytes(b"")

    assert _folders(RE.survey_roots([tmp_path], _patterns())) == []


def test_results_and_figures_are_not_mistaken_for_failed_recordings(tmp_path):
    """The root also holds output. Reporting every csv and png folder as a
    problem would bury the two lines that matter."""
    d = tmp_path / "EIS_Results" / "2611976" / "45A" / "gold"
    d.mkdir(parents=True)
    (d / "plate_summary.csv").write_text("segment\n1\n")
    (d / "map.png").write_bytes(b"\x89PNG")

    assert _folders(RE.survey_roots([tmp_path], _patterns())) == []


def test_the_count_matches_the_folders_reported(tmp_path):
    for name, fname in (("a", "x.ddf"), ("b", "y.ddf")):
        (tmp_path / name).mkdir()
        (tmp_path / name / fname).write_bytes(b"\x00")

    lines = RE.survey_roots([tmp_path], _patterns())
    count = int(lines[0].split(":")[1])
    assert count == 2
