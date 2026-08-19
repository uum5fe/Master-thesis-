#!/usr/bin/env python3
r"""Process raw FAMOS recordings into results the dashboard can read.

    python run_evaluation.py --list                 # what recordings are there?
    python run_evaluation.py --all                  # process every condition
    python run_evaluation.py --condition 45A        # just one
    python run_evaluation.py --self-test            # check the install, no data

This is the bronze/silver/gold pipeline from ``local_eis/`` - the same code
that ran on Databricks - driven from the paths in ``.env``.  Output lands
directly in ``<EIS_RESULTS_ROOT>\<order id>\<condition>``, which is where the
dashboard looks, so there is no copying step afterwards.

Paths are never given here.  Put them in ``.env`` once:

    EIS_FAMOS_ROOT=C:\Users\me\OneDrive - Bosch Group\Local_Eis\2611976_16_07
    EIS_RESULTS_ROOT=C:\Users\me\OneDrive - Bosch Group\Local_Eis\results
    EIS_CURR_CAL=C:\Users\me\OneDrive - Bosch Group\Local_Eis\cal\curr.csv
    EIS_TEMP_CAL=C:\Users\me\OneDrive - Bosch Group\Local_Eis\cal\temp.csv

Expect minutes per condition: bronze reads every sample of every card.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--famos", metavar="DIR", help="override EIS_FAMOS_ROOT")
    p.add_argument("--out", metavar="DIR", help="override EIS_RESULTS_ROOT")
    p.add_argument("--curr-cal", metavar="CSV", help="override EIS_CURR_CAL")
    p.add_argument("--temp-cal", metavar="CSV", help="override EIS_TEMP_CAL")
    p.add_argument("--leepa", metavar="ID",
                   help="process only this order id (default: all found)")
    p.add_argument("--condition", metavar="COND", action="append",
                   help="process this condition; repeatable")
    p.add_argument("--all", action="store_true",
                   help="process every condition discovered")
    p.add_argument("--list", action="store_true",
                   help="list the recordings found, then exit")
    p.add_argument("--self-test", action="store_true",
                   help="run the pipeline's synthetic checks, then exit")
    p.add_argument("--equal-areas", action="store_true",
                   help="treat every segment as A_cell/72 — a deliberate "
                        "simplification, recorded in the manifest")
    p.add_argument("--no-png", action="store_true",
                   help="skip the pipeline's own figures; the dashboard draws "
                        "its own and never reads them")
    p.add_argument("--stop-after", choices=["bronze", "silver", "gold"],
                   default="gold")
    p.add_argument("--stage-local", action="store_true",
                   help="copy the cards to local disk before processing. "
                        "Strongly advised when the recordings live on a "
                        "network share: the reader memory-maps each card and "
                        "bronze walks it repeatedly, so over SMB every page "
                        "fault is a network round trip")
    p.add_argument("--keep-staged", action="store_true",
                   help="do not delete the local copies afterwards")
    p.add_argument("--dry-run", action="store_true",
                   help="print the command that would run, and stop")
    a = p.parse_args(argv)

    for value, name in ((a.famos, "EIS_FAMOS_ROOT"), (a.out, "EIS_RESULTS_ROOT"),
                        (a.curr_cal, "EIS_CURR_CAL"), (a.temp_cal, "EIS_TEMP_CAL")):
        if value:
            os.environ[name] = str(Path(value).expanduser())
    os.environ.setdefault("EIS_ALLOW_INLINE_PIPELINE", "1")

    from app.data.sources import FamosSource, _condition_sort_key
    from app.services import runner, staging
    from app.settings import SETTINGS

    if a.self_test:
        ok, output = runner.self_test(SETTINGS)
        print(output)
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    if not SETTINGS.famos_roots:
        print("error: EIS_FAMOS_ROOT is not set. Put it in .env, or pass "
              "--famos.", file=sys.stderr)
        return 2

    refs = FamosSource(SETTINGS.famos_roots, SETTINGS.famos_glob).scan()
    if a.leepa:
        refs = [r for r in refs if r.measurement_id == a.leepa]
    refs.sort(key=lambda r: (r.measurement_id, _condition_sort_key(r.condition)))

    if not refs:
        print(f"no recordings found under {', '.join(SETTINGS.famos_roots)}")
        print("run `python run_dashboard.py --check` to see why")
        return 1

    print(f"\nfound {len(refs)} condition(s):")
    for ref in refs:
        print(f"  {ref.measurement_id} / {ref.condition:<8} "
              f"{len(ref.files)} card file(s)  ->  "
              f"{runner.output_dir(ref, SETTINGS)}")
    if a.list:
        return 0

    if a.condition:
        wanted = {c.lower() for c in a.condition}
        refs = [r for r in refs if r.condition.lower() in wanted]
        if not refs:
            print(f"\nnone of {a.condition} is among the conditions found",
                  file=sys.stderr)
            return 1
    elif not a.all:
        print("\nNothing selected. Pass --all, or --condition <name>, "
              "or --list to look first.")
        return 0

    for note in runner.warnings_for(SETTINGS):
        print(f"\nnote: {note}")

    remote = [r for r in refs if staging.is_network_path(r.path)]
    if remote and not a.stage_local:
        total = sum(staging.staged_size_mb(r) for r in remote) / 1024
        print(f"\nnote: {len(remote)} condition(s) sit on a network share "
              f"({total:.1f} GB). The reader memory-maps each card and bronze "
              f"walks it repeatedly, so over SMB this is far slower than local "
              f"disk and a network hiccup fails the run. Pass --stage-local to "
              f"copy them down first.")

    failures = 0
    for index, ref in enumerate(refs, 1):
        out = runner.output_dir(ref, SETTINGS)
        print(f"\n{'=' * 70}\n[{index}/{len(refs)}] "
              f"{ref.measurement_id} / {ref.condition}\n{'=' * 70}")

        if a.dry_run:
            argv_ = runner.build_command(ref, out, SETTINGS, a.equal_areas,
                                         a.no_png, a.stop_after)
            print("  would run, in " + str(runner.pipeline_dir(SETTINGS)) + ":")
            print("    " + " ".join(argv_))
            continue

        problems = runner.preflight(ref, SETTINGS)
        if problems:
            print("  cannot run:")
            for problem in problems:
                print(f"    - {problem}")
            failures += 1
            continue

        started = time.time()
        source = ref
        try:
            if a.stage_local:
                print(f"  staging {staging.staged_size_mb(ref):.0f} MB to "
                      f"local disk first")
                source = staging.stage(
                    ref, SETTINGS.stage_dir or None,
                    lambda done, total, message="": print(f"    {message}"))
            runner.run_famos(
                lambda done, total, message="": print(f"  {message}"),
                source, settings=SETTINGS, equal_areas=a.equal_areas,
                no_png=a.no_png, stop_after=a.stop_after)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures += 1
            continue
        finally:
            if a.stage_local and not a.keep_staged:
                staging.clear(ref, SETTINGS.stage_dir or None)
        print(f"  done in {time.time() - started:.0f} s -> {out}")

    if not a.dry_run:
        print(f"\n{len(refs) - failures}/{len(refs)} condition(s) processed.")
        print("Start the dashboard to look at them:  python run_dashboard.py --open")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
