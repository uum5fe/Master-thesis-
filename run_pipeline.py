#!/usr/bin/env python3
"""Launcher for the pipeline's main script.

The main script itself lives inside the package, at
:mod:`eis.pipeline.main`, next to the bronze, silver and gold tiers it drives.
This file only exists so that the documented

    python run_pipeline.py --demo --plots

keeps working from a clone without installing anything.  It is equivalent to

    python -m eis.pipeline.main --demo --plots
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eis.pipeline.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
