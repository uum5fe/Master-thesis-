#!/usr/bin/env python3
"""
run.py -- command-line entry point for any pipeline module.

    python run.py main --reevaluate RUN_DIR --out OUT      the pipeline
    python run.py main --self-test
    python run.py frequency_response RUN_DIR --bode DIR    a per-run check
    python run.py via_resistance ABGLEICH_DIR RUN_DIR...
    python run.py segment_scale RUN_45A RUN_60A ... -o OUT
    python run.py dc_closure RUN_DIR... -o OUT

The first argument is the module name; the rest goes to its main().
"""

from __future__ import annotations

import importlib
import sys

import eis_paths  # noqa: F401  (puts core/, pipeline/, ... on sys.path)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    mod = importlib.import_module(argv[0])
    if not hasattr(mod, "main"):
        print(f"{argv[0]} has no main()")
        return 2
    return int(mod.main(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
