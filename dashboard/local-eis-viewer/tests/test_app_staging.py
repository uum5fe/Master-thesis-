r"""Staging recordings off a network share.

``FamosFile`` memory-maps each card and bronze walks it repeatedly, so over SMB
every page fault is a network round trip. Copying down first is the difference
between minutes and hours - and between a run that finishes and one that dies
on a network hiccup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.sources import RunRef                    # noqa: E402
from app.services import staging                       # noqa: E402


def _ref(tmp_path, size: int = 2048) -> RunRef:
    source = tmp_path / "share"
    source.mkdir(exist_ok=True)
    files = []
    for card in range(1, 4):
        path = source / f"Leepa_2611976_Current_45A_Test_01_Karte_{card}.DAT"
        path.write_bytes(b"x" * size)
        files.append(str(path))
    return RunRef(kind="famos", measurement_id="2611976", condition="45A",
                  path=str(source), layout="famos", files=tuple(files))


@pytest.mark.parametrize("path,expected", [
    (r"\\bosch.com\DfsRB\DfsDE\LOC\Fe\ILM\Charan\Lokale_EIS\Daten", True),
    ("//fileserver/share/data", True),
    (r"C:\Users\me\OneDrive - Bosch Group\Local_Eis", False),
    ("/home/me/data", False),
    (r"Z:\mapped\drive", False),          # indistinguishable from a local disk
])
def test_network_paths_are_recognised(path, expected):
    assert staging.is_network_path(path) is expected


def test_staging_copies_every_card(tmp_path):
    ref = _ref(tmp_path)
    staged = staging.stage(ref, tmp_path / "stage")

    assert len(staged.files) == len(ref.files)
    assert all(Path(f).is_file() for f in staged.files)
    assert [Path(f).stat().st_size for f in staged.files] == \
           [Path(f).stat().st_size for f in ref.files]
    # The new ref points at the copy, and remembers where it came from.
    assert staged.path != ref.path
    assert staged.detail["staged_from"] == ref.path
    assert staged.measurement_id == ref.measurement_id
    assert staged.condition == ref.condition


def test_staging_is_resumable(tmp_path):
    """An interrupted run must not start the copy from nothing."""
    ref = _ref(tmp_path)
    staging.stage(ref, tmp_path / "stage")

    copied = []
    staging.stage(ref, tmp_path / "stage",
                  lambda done, total, message="": copied.append(message))
    assert not any("copying" in m for m in copied)      # nothing re-copied


def test_a_changed_source_is_copied_again(tmp_path):
    ref = _ref(tmp_path, size=2048)
    staging.stage(ref, tmp_path / "stage")
    Path(ref.files[0]).write_bytes(b"y" * 4096)         # source grew

    copied = []
    staging.stage(ref, tmp_path / "stage",
                  lambda done, total, message="": copied.append(message))
    assert sum("copying" in m for m in copied) == 1


def test_clearing_removes_the_copy_and_nothing_else(tmp_path):
    ref = _ref(tmp_path)
    staged = staging.stage(ref, tmp_path / "stage")
    staging.clear(ref, tmp_path / "stage")

    assert not any(Path(f).exists() for f in staged.files)
    assert all(Path(f).is_file() for f in ref.files)    # originals untouched


def test_clearing_a_run_that_was_never_staged_is_harmless(tmp_path):
    staging.clear(_ref(tmp_path), tmp_path / "stage")


def test_size_is_reported_for_the_warning(tmp_path):
    ref = _ref(tmp_path, size=1024 * 1024)
    assert staging.staged_size_mb(ref) == pytest.approx(3.0, abs=0.01)


def test_size_of_unreachable_files_does_not_raise(tmp_path):
    ref = RunRef(kind="famos", measurement_id="x", condition="y",
                 files=(str(tmp_path / "gone.DAT"),))
    assert staging.staged_size_mb(ref) == 0.0
