#!/usr/bin/env python3
"""Start the Local EIS Viewer dashboard.

    python run_dashboard.py
    python run_dashboard.py --results /path/to/results --open
    python run_dashboard.py --famos /path/to/Famos --port 8060

Run this file from anywhere; it puts the project root on the import path
itself, so `python run_dashboard.py` and `python /full/path/run_dashboard.py`
both work. It prints the URL to open before the server starts, because a
server that starts in silence looks like a server that did not start.

On Databricks the platform starts the app from `app.yaml` instead, and this
script is not used.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", metavar="DIR",
                   help="folder of finished pipeline output, laid out "
                        "<DIR>/<order id>/<condition>/{gold,silver}/ "
                        "(sets EIS_RESULTS_ROOT)")
    p.add_argument("--famos", metavar="DIR",
                   help="folder of raw FAMOS .DAT recordings "
                        "(sets EIS_FAMOS_ROOT)")
    p.add_argument("--plate-specs", metavar="DIR",
                   help="extra folder of plate generation specs "
                        "(sets EIS_PLATE_SPEC_DIR)")
    p.add_argument("--port", type=int, default=None,
                   help="port to listen on (default 8050)")
    p.add_argument("--host", default=None,
                   help="address to bind (default 0.0.0.0; the link shown is "
                        "always http://127.0.0.1:<port>)")
    p.add_argument("--open", action="store_true", dest="open_browser",
                   help="open the dashboard in your browser once it is up")
    p.add_argument("--debug", action="store_true",
                   help="Dash debug mode with hot reload")
    a = p.parse_args(argv)

    # Command-line paths win over whatever is already in the environment, and
    # are set before app.settings is imported - Settings reads os.environ once.
    for value, name in ((a.results, "EIS_RESULTS_ROOT"),
                        (a.famos, "EIS_FAMOS_ROOT"),
                        (a.plate_specs, "EIS_PLATE_SPEC_DIR")):
        if value:
            path = Path(value).expanduser().resolve()
            if not path.is_dir():
                print(f"error: {name}: not a directory: {path}", file=sys.stderr)
                return 2
            os.environ[name] = str(path)

    try:
        from app.app import serve
    except ModuleNotFoundError as exc:
        print(f"error: {exc}\n\n"
              f"Install the dependencies first:\n"
              f"    pip install -r {ROOT / 'requirements.txt'}", file=sys.stderr)
        return 1

    serve(host=a.host, port=a.port, debug=a.debug or None,
          open_browser=a.open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
