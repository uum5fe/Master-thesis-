"""
eis_paths.py
============
Puts the pipeline's folders on sys.path, so every module keeps importing its
neighbours by plain name (`import config`, `import silver`, ...) wherever it
lives in the tree:

    core/       config, utils, plate geometry, plate conditions
    pipeline/   bronze -> silver -> gold, main, and their signal processing
    checks/     plausibility and the per-run diagnostics
    analysis/   ECM / DRT fits, polarisation curves
    plotting/   heat maps, Nyquist and selector viewers, figure helpers
    readers/    Gamry .DTA, Abgleich calibration, CSV sources, vendored zstd

Imported first by the runner notebook, by tests/conftest.py and by run.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUBDIRS = ("core", "pipeline", "checks", "analysis", "plotting", "readers")


def add(root: Path | str = ROOT) -> list[str]:
    """Insert root and its sub-folders at the front of sys.path (once)."""
    root = Path(root)
    added = []
    for p in [root] + [root / d for d in SUBDIRS]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
            added.append(s)
    return added


add()
