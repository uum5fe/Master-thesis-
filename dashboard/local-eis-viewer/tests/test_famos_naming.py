"""Card file names, as they actually appear on the share.

Every pattern in _FAMOS_PATTERNS was written against a REAL name from the
campaign folder, and each is pinned here with that name. A name that matches
nothing is a measurement that silently does not exist, so the cost of a
regression is an order quietly missing from every listing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.data.sources import (_normalise_condition, _order_id_from_parts,
                              famos_patterns)


def resolve(folder: str, name: str):
    """What scan() would file this card under: (order id, condition)."""
    path = Path(folder) / name
    hit = next((m for m in (p.search(name) for p in famos_patterns()) if m),
               None)
    if hit is None:
        return None
    mid = ((hit.groupdict().get("measurement_id") or "").strip()
           or _order_id_from_parts(path))
    if not mid:
        return None
    return mid, _normalise_condition(hit.group("condition"))


@pytest.mark.parametrize("folder,name,expected", [
    # the one that prompted this: FAMOS converted out of DASYLab
    ("Daten/2612025_27_08/FAMOS",
     "2026_08_27_KANAL_0_15-RO2612025_45A.DDF_1.DAT", ("2612025", "45A")),
    ("Daten", "Karte_1_Leepa_2611921_Measurement_01_Test_1_45A.DAT",
     ("2611921", "45A")),
    ("Daten/2611959_03_07/Kopie",
     "Karte_2_Messung_leepa_2611959_Versuch_test01_450A.DAT",
     ("2611959", "450A")),
    ("Daten/famos", "RO2611922-01_KARTE_1_LEEPA__MEASUREMENT_01_TEST_1_150A.DAT",
     ("2611922", "150A")),
    # no order number in the name -- it comes from the folder
    ("Daten/2611959_03_07", "Channel_1_60A.dat", ("2611959", "60A")),
    # the scheme that already worked, still works
    ("Daten/2611976_16_07", "RO2611976-01_Current_45A_Test_01_Karte_1.DAT",
     ("2611976", "45A")),
])
def test_a_real_card_name_resolves_to_its_order_and_condition(
        folder, name, expected):
    assert resolve(folder, name) == expected


def test_a_name_with_no_condition_is_refused(): 
    """Guessing an operating point would file a recording under the wrong
    current, which is worse than not finding it."""
    assert resolve("Daten/2026_07_15",
                   "Karte_1_Messung_sync_Versuch_3.DAT") is None


def test_a_name_with_no_order_anywhere_is_refused():
    """`Channel_1_150.dat` in a folder called `DAT` names no order, so there
    is nothing to file it under. Inventing an id would hide a real
    measurement behind a label nobody would search for."""
    assert resolve("Daten/DAT", "Channel_1_150.dat") is None


def test_a_date_folder_is_not_mistaken_for_an_order():
    """2026_07_15 is seven digits' worth of date. Requiring an unbroken run
    of seven keeps it out."""
    assert _order_id_from_parts(Path("Daten/2026_07_15/x.DAT")) == ""


def test_the_order_is_found_from_a_folder_above_the_cards():
    """The id sits on the campaign folder while cards may be a level down."""
    assert _order_id_from_parts(
        Path("Daten/2611959_03_07/Kopie/x.DAT")) == "2611959"


@pytest.mark.parametrize("written,expected", [
    ("150", "150A"), ("150A", "150A"), ("45 A", "45A"), ("60a", "60A"),
])
def test_a_condition_written_two_ways_becomes_one(written, expected):
    """One scheme drops the unit on some files and keeps it on others. Two
    spellings would split one operating point into two half-populated
    conditions, neither of which is evaluable."""
    assert _normalise_condition(written) == expected


def test_the_same_card_in_two_folders_is_counted_once(tmp_path):
    """A working copy beside the original is the same physical recording.

    Feeding bronze both makes it believe it has twice the coverage it has.
    """
    from app.data.sources import FamosSource

    name = "2026_08_27_KANAL_0_15-RO2612025_45A.DDF_1.DAT"
    for folder in ("2612025_27_08/FAMOS", "2612025_Famos"):
        d = tmp_path / folder
        d.mkdir(parents=True)
        (d / name).write_bytes(b"")

    runs = FamosSource([tmp_path]).scan()
    assert len(runs) == 1
    assert runs[0].measurement_id == "2612025"
    assert len(runs[0].files) == 1, "the copy must not count as a second card"
