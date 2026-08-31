"""The viewer ships its own copy of the pipeline; it has to be the same copy.

`dashboard/local-eis-viewer/local_eis/` is a mirror of `databricks/local_eis/`
minus the tests. Nothing enforces that -- it is kept by hand -- so a fix made
in one place runs from the other unfixed, and the symptom appears in a
traceback pointing at the file that was never changed.

That is not a hypothesis. A crash was fixed in `databricks/local_eis/bronze.py`
while the run that produced it executed
`dashboard/local-eis-viewer/local_eis/bronze.py`, and the two would have
disagreed silently. This test is the thing that would have said so.
"""

from __future__ import annotations

from pathlib import Path

VIEWER = Path(__file__).resolve().parents[1] / "local_eis"
SOURCE = Path(__file__).resolve().parents[3] / "databricks" / "local_eis"


def _mirrored() -> list[str]:
    return sorted(p.name for p in VIEWER.glob("*.py"))


def test_the_two_copies_of_the_pipeline_are_identical():
    """Byte-for-byte, because "nearly the same" is what causes this bug.

    A diff of one line in an accept rule changes which steps enter a
    spectrum, and reads as a measurement difference rather than a stale file.
    """
    drift = []
    for name in _mirrored():
        src, dst = SOURCE / name, VIEWER / name
        if not src.exists():
            drift.append(f"{name}: in the viewer copy but not in "
                         f"databricks/local_eis")
        elif src.read_bytes() != dst.read_bytes():
            drift.append(f"{name}: differs -- copy "
                         f"databricks/local_eis/{name} over "
                         f"dashboard/local-eis-viewer/local_eis/{name}")
    assert not drift, "the bundled pipeline has drifted:\n  " + \
                      "\n  ".join(drift)


def test_every_module_the_pipeline_needs_is_mirrored():
    """Tests are deliberately not shipped; modules are not optional.

    A module added to the pipeline and not mirrored fails at import time in
    the viewer only -- which is the hardest place to notice it, because the
    Databricks copy and the whole test suite both pass.
    """
    missing = [p.name for p in SOURCE.glob("*.py")
               if not p.name.startswith("test_")
               and not (VIEWER / p.name).exists()]
    assert not missing, (
        "these modules exist in databricks/local_eis but are not bundled "
        f"with the viewer: {', '.join(sorted(missing))}")


def test_the_test_modules_are_not_shipped():
    """They pull in pytest, which the viewer does not require to run."""
    shipped = [n for n in _mirrored() if n.startswith("test_")]
    assert not shipped, f"tests do not belong in the bundle: {shipped}"
