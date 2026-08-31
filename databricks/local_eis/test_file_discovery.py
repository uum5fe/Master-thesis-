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


# ---------------------------------------------------------------------------
# what the log says was matched
# ---------------------------------------------------------------------------

def test_the_pattern_reported_is_the_one_that_matched(tmp_path):
    """The log used to name patterns[0] whichever pattern actually matched.

    A run over `RO2612025-01_Current_45A_Test_01_Karte_1.DAT` announced
    itself as `1 file(s) matching 'Leepa_2612025_Current_45A_...DAT'` -- a
    pattern that matches nothing in that folder. It reads as a filename
    problem, and the next hour goes on renaming files that were already
    right.
    """
    (tmp_path / "RO2612025-01_Current_45A_Test_01_Karte_1.DAT").write_bytes(b"")
    cfg = DEFAULT.replace(dat_dir=tmp_path, leepa="2612025",
                          condition="45A")

    files, how = bronze.discover_files_verbose(cfg)
    assert len(files) == 1
    assert "RO" in how or "2612025" in how
    assert "Leepa_2612025_Current_45A" not in how, (
        "reported a pattern that did not match")


def test_the_fallback_says_it_is_a_fallback(tmp_path):
    """Taking every .DAT in the folder is a different claim from matching.

    Reported as "matching <pattern>" it looks like the naming convention was
    recognised, which is exactly when a stray file from another condition
    goes unnoticed.
    """
    (tmp_path / "something_entirely_else_45A_x.DAT").write_bytes(b"")
    cfg = DEFAULT.replace(dat_dir=tmp_path, leepa="2612025",
                          condition="45A")

    _, how = bronze.discover_files_verbose(cfg)
    assert "no known naming convention" in how


def test_the_thin_wrapper_still_returns_just_the_files(tmp_path):
    """discover_files is called from stage_bronze and from the tests above."""
    (tmp_path / "RO2612025-01_Current_45A_Test_01_Karte_1.DAT").write_bytes(b"")
    cfg = DEFAULT.replace(dat_dir=tmp_path, leepa="2612025",
                          condition="45A")

    got = bronze.discover_files(cfg)
    assert isinstance(got, list) and all(isinstance(p, Path) for p in got)
