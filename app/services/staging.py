r"""Copying recordings off a network share before processing them.

``eis_local.FamosFile`` opens each recording with ``np.memmap``, and bronze
then walks that array repeatedly - once per frequency step, per channel. On a
local disk that is the right design: the operating system caches the pages and
nothing is read twice. Over a UNC or DFS share every page fault becomes an SMB
round trip, so the same run can take hours instead of minutes, and an ordinary
network hiccup surfaces as an I/O error partway through rather than as a
retry.

Copying the cards to local disk first turns thousands of small network reads
into one sequential bulk read per file. The copy costs a few minutes of
bandwidth once; the run then proceeds at local-disk speed.

Nothing here decides policy. It reports that a path is remote and offers to
stage it; the caller chooses.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from app.data.sources import RunRef


def is_network_path(path: str | Path) -> bool:
    r"""True for a Windows UNC path (``\\server\share\...``).

    Mapped drive letters (``Z:\...``) are indistinguishable from local disks
    from here, so they are not detected; the warning is best-effort, and being
    wrong only means the advice is not offered.
    """
    text = str(path)
    return text.startswith("\\\\") or text.startswith("//")


def default_stage_dir() -> Path:
    return Path(tempfile.gettempdir()) / "eis_stage"


def staged_size_mb(ref: RunRef) -> float:
    total = 0
    for name in ref.files:
        try:
            total += Path(name).stat().st_size
        except OSError:
            pass
    return total / (1024 * 1024)


def stage(ref: RunRef, stage_dir: str | Path | None = None,
          progress=None) -> RunRef:
    """Copy one run's card files to local disk; return a ref pointing at them.

    The copy is skipped for any file already present at the right size, so a
    re-run after an interrupted one does not start from nothing.
    """
    destination = Path(stage_dir or default_stage_dir()) / \
        ref.measurement_id / ref.condition
    destination.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for index, name in enumerate(ref.files, 1):
        source = Path(name)
        target = destination / source.name
        try:
            same = (target.is_file()
                    and target.stat().st_size == source.stat().st_size)
        except OSError:
            same = False
        if not same:
            if progress is not None:
                progress(index - 1, len(ref.files),
                         f"copying {source.name} "
                         f"({source.stat().st_size / 1e6:.0f} MB)")
            shutil.copy2(source, target)
        copied.append(str(target))
        if progress is not None:
            progress(index, len(ref.files), f"staged {source.name}")

    return replace(ref, path=str(destination), files=tuple(copied),
                   detail={**ref.detail, "staged_from": ref.path})


def clear(ref: RunRef, stage_dir: str | Path | None = None) -> None:
    """Remove one run's staged copy. Never touches the original recordings."""
    destination = Path(stage_dir or default_stage_dir()) / \
        ref.measurement_id / ref.condition
    if destination.is_dir():
        shutil.rmtree(destination, ignore_errors=True)
