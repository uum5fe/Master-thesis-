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


def write_env_file(args, force: bool = False) -> int:
    r"""Create the .env settings file, and say exactly where it is.

    Creating a file whose name begins with a dot is awkward on Windows:
    Explorer resists it and Notepad appends .txt, which then looks correct
    because Explorer hides the extension. Writing it from here avoids all of
    that, and printing the absolute path removes any doubt about which file the
    application will read.
    """
    target = ROOT / ".env"
    if target.exists() and not force:
        print(f"{target} already exists.\n"
              f"Edit it, or pass --force to replace it with a fresh template.")
        return 0

    template = ROOT / ".env.example"
    lines = template.read_text(encoding="utf-8").splitlines() if template.is_file() else []

    # Anything given on the command line goes in uncommented, so the file is
    # usable as written rather than needing an edit before the first run.
    supplied = {
        "EIS_FAMOS_ROOT": args.famos,
        "EIS_RESULTS_ROOT": args.results,
        "EIS_PLATE_SPEC_DIR": args.plate_specs,
    }
    supplied = {k: v for k, v in supplied.items() if v}

    out, seen = [], set()
    for line in lines:
        key = line.split("=", 1)[0].strip().lstrip("# ")
        if key in supplied and not line.strip().startswith("#"):
            out.append(f"{key}={supplied[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in supplied.items():
        if key not in seen:
            out.append(f"{key}={value}")

    target.write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"\nCreated {target}\n")
    if supplied:
        print("  filled in from the command line:")
        for key, value in supplied.items():
            print(f"    {key}={value}")
    else:
        print("  Open it and set at least one of:")
        print("    EIS_FAMOS_ROOT    folder of raw .DAT recordings")
        print("    EIS_RESULTS_ROOT  folder of finished results")
    print("\n  Then:  python run_dashboard.py --check")
    return 0


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
    p.add_argument("--init", action="store_true",
                   help="create the .env settings file next to this script "
                        "and print where it is, then exit. Combine with "
                        "--results / --famos to fill it in immediately")
    p.add_argument("--force", action="store_true",
                   help="with --init, overwrite an existing .env")
    p.add_argument("--check", action="store_true",
                   help="report what the app can and cannot find, and why, "
                        "then exit without starting the server")
    a = p.parse_args(argv)

    # Writing the settings file must not require the data to be reachable: the
    # share may be offline, or the paths may be for a different machine.
    if a.init:
        return write_env_file(a, force=a.force)

    # Command-line paths win over whatever is already in the environment, and
    # are set before app.settings is imported - Settings reads os.environ once.
    for value, name in ((a.results, "EIS_RESULTS_ROOT"),
                        (a.famos, "EIS_FAMOS_ROOT"),
                        (a.plate_specs, "EIS_PLATE_SPEC_DIR")):
        if value:
            path = Path(value).expanduser()
            if not path.is_dir():
                print(f"error: {name}: not a directory: {path}", file=sys.stderr)
                return 2
            os.environ[name] = str(path)

    if a.check:
        from app.diagnose import report
        print(report())
        return 0

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
