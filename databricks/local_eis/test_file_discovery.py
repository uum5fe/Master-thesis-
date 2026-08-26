"""Which card files belong to THIS run?

A campaign folder holds every condition. Picking up all of them turns a
request for one condition into a run over four, at every stage, over a network
share -- which does not finish, and looks like a hang rather than a mistake.
That is what these tests exist to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import bronze
from config import DEFAULT

CONDITIONS = ("45A", "60A", "150A", "450A")


def _campaign(root: Path, template: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        for card in range(1, 6):
            (root / template.format(cond=cond, card=card)).write_bytes(b"x")
    return root


def _conditions_of(files) -> list[str]:
    return sorted({f.name.split("_Current_")[1].split("_")[0] for f in files})


@pytest.mark.parametrize("template", [
    "Leepa_2611976_Current_{cond}_Test_01_Karte_{card}.DAT",
    "RO2611976-01_Current_{cond}_Test_01_Karte_{card}.DAT",
    "2611976_Current_{cond}_Test_01_Karte_{card}.DAT",
])
def test_one_condition_is_read_whatever_the_naming_convention(tmp_path,
                                                              template):
    """The RO form is the one that was silently falling through."""
    folder = _campaign(tmp_path / "campaign", template)
    cfg = DEFAULT.replace(dat_dir=folder, leepa="2611976", condition="45A")
    files = bronze.discover_files(cfg)
    assert len(files) == 5, [f.name for f in files]
    assert _conditions_of(files) == ["45A"]


def test_45a_is_not_confused_with_450a(tmp_path):
    """A prefix match would take both, and 450A is nine times the current."""
    folder = _campaign(tmp_path / "campaign",
                       "RO2611976-01_Current_{cond}_Test_01_Karte_{card}.DAT")
    for cond in ("45A", "450A"):
        cfg = DEFAULT.replace(dat_dir=folder, leepa="2611976", condition=cond)
        assert _conditions_of(bronze.discover_files(cfg)) == [cond]


def test_all_really_does_mean_all(tmp_path):
    folder = _campaign(tmp_path / "campaign",
                       "RO2611976-01_Current_{cond}_Test_01_Karte_{card}.DAT")
    cfg = DEFAULT.replace(dat_dir=folder, leepa="2611976", condition="ALL")
    assert len(bronze.discover_files(cfg)) == 20


def test_an_unknown_convention_is_still_narrowed_by_condition(tmp_path):
    """The fallback keeps the condition instead of taking the folder."""
    folder = _campaign(tmp_path / "odd",
                       "unknown_Current_{cond}_Test_01_Karte_{card}.DAT")
    cfg = DEFAULT.replace(dat_dir=folder, leepa="2611976", condition="60A")
    files = bronze.discover_files(cfg)
    assert len(files) == 5
    assert _conditions_of(files) == ["60A"]


def test_it_refuses_rather_than_reading_every_condition(tmp_path):
    """The failure this whole module is about.

    Names that match nothing and cannot be narrowed must stop the run. Reading
    four conditions slowly and labelling the result as one of them is worse
    than not starting.
    """
    folder = tmp_path / "ambiguous"
    folder.mkdir()
    for cond in ("45A", "60A"):
        for card in range(1, 6):
            (folder / f"weird-{cond}-card{card}.DAT").write_bytes(b"x")

    cfg = DEFAULT.replace(dat_dir=folder, leepa="2611976", condition="45A")
    with pytest.raises(SystemExit) as excinfo:
        bronze.discover_files(cfg)
    message = str(excinfo.value)
    assert "45A" in message and "60A" in message
    assert "EIS_FAMOS_REGEX" in message, "it must say how to fix it"


def test_a_single_condition_folder_is_accepted_under_any_name(tmp_path):
    """Pointing --dat at one condition's files is a legitimate thing to do."""
    folder = tmp_path / "one"
    folder.mkdir()
    for card in range(1, 6):
        (folder / f"whatever-45A-{card}.DAT").write_bytes(b"x")
    cfg = DEFAULT.replace(dat_dir=folder, leepa="2611976", condition="45A")
    assert len(bronze.discover_files(cfg)) == 5


def test_an_empty_folder_says_what_it_looked_for(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    cfg = DEFAULT.replace(dat_dir=folder, leepa="2611976", condition="45A")
    with pytest.raises(SystemExit) as excinfo:
        bronze.discover_files(cfg)
    assert "Karte" in str(excinfo.value)
