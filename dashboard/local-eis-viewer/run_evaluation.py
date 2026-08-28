#!/usr/bin/env python3
r"""Process raw recordings -- FAMOS cards or R2-D2 CSV sweeps -- into results
the dashboard can read.

    python run_evaluation.py --list                 # what recordings are there?
    python run_evaluation.py --all                  # process everything found
    python run_evaluation.py --condition 45A        # just one
    python run_evaluation.py --source csv --all     # only the CSV sweeps
    python run_evaluation.py --self-test            # check the install, no data

This is the bronze/silver/gold pipeline from ``local_eis/`` - the same code
that ran on Databricks - driven from the paths in ``.env``.  Output lands
directly in ``<EIS_RESULTS_ROOT>\<order id>\<condition>``, which is where the
dashboard looks, so there is no copying step afterwards.

Both recording formats are found and processed the same way; the reader is
chosen per recording, so a mixed campaign needs no flags.  A FAMOS condition
is a folder of ``.DAT`` cards; a CSV condition is one sweep folder of point
files, and the campaign folder holding several of those is what goes in
``.env``.

Paths are never given here.  Put them in ``.env`` once:

    EIS_FAMOS_ROOT=C:\Users\me\OneDrive - Bosch Group\Local_Eis\2611976_16_07
    EIS_CSV_ROOT=C:\Users\me\OneDrive - Bosch Group\Local_Eis\csv_files
    EIS_RESULTS_ROOT=C:\Users\me\OneDrive - Bosch Group\Local_Eis\results
    EIS_CURR_CAL=C:\Users\me\OneDrive - Bosch Group\Local_Eis\cal\curr.csv
    EIS_TEMP_CAL=C:\Users\me\OneDrive - Bosch Group\Local_Eis\cal\temp.csv

Either root alone is enough.  EIS_CURR_CAL is needed for FAMOS, which has no
other absolute scale; the R2-D2 CSV logger applies its own coefficients before
writing, so a CSV-only setup does not need it.

Expect minutes per FAMOS condition: bronze reads every sample of every card.
CSV sweeps are far quicker -- the logger already reduced them.
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


def survey_roots(roots, patterns) -> list[str]:
    r"""What is in these folders that did NOT become a run, and why.

    FamosSource matches on the FILE NAME and silently drops everything else,
    which is right for a scanner and useless for a person: an order whose
    recording is a DASYLab .ddf, or whose cards were renamed, simply does not
    appear, with no line of output pointing at it. "It is not listing" is
    then the only available diagnosis.

    So this walks the same roots and reports the folders that contributed
    nothing, with what they actually hold. It reads names only -- no file is
    opened -- so it costs nothing over a share.
    """
    import collections

    lines: list[str] = []
    folders = 0
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        by_dir: dict[Path, collections.Counter] = {}
        unmatched: dict[Path, list[str]] = {}
        matched: set[Path] = set()
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            by_dir.setdefault(path.parent, collections.Counter())[
                path.suffix.lower() or "(none)"] += 1
            if path.suffix.lower() != ".dat":
                continue
            if any(pat.search(path.name) for pat in patterns):
                matched.add(path.parent)
            else:
                unmatched.setdefault(path.parent, []).append(path.name)

        for directory in sorted(by_dir):
            if directory in matched:
                continue
            counts = by_dir[directory]
            # Folders of results, figures or notes are not failed recordings.
            interesting = {ext: n for ext, n in counts.items()
                           if ext in (".dat", ".ddf", ".mf4", ".tdms", ".hdf5",
                                      ".h5", ".dmd", ".dxd", ".d7d", ".dsd")}
            if not interesting:
                continue
            what = ", ".join(f"{n} x {ext}" for ext, n in
                             sorted(interesting.items(), key=lambda kv: -kv[1]))
            try:
                shown = directory.relative_to(root)
            except ValueError:
                shown = directory
            folders += 1
            lines.append(f"  {shown}  --  {what}")
            if directory in unmatched:
                sample = sorted(unmatched[directory])[0]
                lines.append(f"      .DAT present but the names do not match "
                             f"any known pattern, e.g. {sample!r}")
                lines.append(f"      set EIS_FAMOS_REGEX to a pattern with "
                             f"named groups measurement_id and condition")
            elif ".ddf" in interesting:
                example = next(iter(sorted(
                    q.name for q in directory.glob("*.ddf"))), "<file>.ddf")
                lines.append(f"      DASYLab .ddf is not a format this "
                             f"pipeline reads yet -- run")
                lines.append(f"        python local_eis/ddf_source.py probe "
                             f'"{directory / example}"')
                lines.append(f"      or export ASCII from DASYLab and use "
                             f"--source csv")
            else:
                lines.append(f"      no .DAT cards here")
    return [f"__folders__:{folders}"] + lines


def normalise_order_id(text: str) -> str:
    r"""The bare order number, however it was written.

    On disk the same order appears three ways: the folder is
    ``2612025_27_08``, the card files are ``RO2612025-01_Current_...``, and
    the catalogue key the pattern extracts is ``2612025``. Someone selecting
    an order copies whichever they are looking at, and only one of the three
    used to work -- the other two returned "no recordings found under
    <roots>", which blames the path for a filter mismatch.
    """
    import re

    core = str(text).split("_")[0]
    # Drop the station suffix FIRST. Stripping non-digits before it turns
    # "RO2612025-01" into "261202501", which matches nothing -- the suffix
    # digits get welded onto the order number.
    core = re.sub(r"-\d+$", "", core)
    return re.sub(r"[^0-9]", "", core)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--famos", metavar="DIR", help="override EIS_FAMOS_ROOT")
    p.add_argument("--csv", metavar="DIR", help="override EIS_CSV_ROOT: the "
                   "CAMPAIGN folder holding one sub-folder per sweep, not a "
                   "single sweep")
    p.add_argument("--source", choices=["auto", "famos", "csv"], default="auto",
                   help="which recordings to consider (default: both)")
    p.add_argument("--out", metavar="DIR", help="override EIS_RESULTS_ROOT")
    p.add_argument("--curr-cal", metavar="CSV", help="override EIS_CURR_CAL")
    p.add_argument("--temp-cal", metavar="CSV", help="override EIS_TEMP_CAL")
    p.add_argument("--leepa", metavar="ID", action="append",
                   help="process only this order id; repeatable. Written how "
                        "you like: 2612025, RO2612025, RO2612025-01 and "
                        "ro2612025 all select the same order (default: all "
                        "found)")
    p.add_argument("--condition", metavar="COND", action="append",
                   help="process this condition; repeatable")
    p.add_argument("--all", action="store_true",
                   help="process every condition discovered")
    p.add_argument("--list", action="store_true",
                   help="list the recordings found, then exit")
    p.add_argument("--self-test", action="store_true",
                   help="run the pipeline's synthetic checks, then exit")
    p.add_argument("--plate", choices=["gen1", "gen2"], default=None,
                   help="plate generation to evaluate with (default: the "
                        "dashboard's EIS_DEFAULT_PLATE, else gen1). The areas "
                        "differ between generations, so this changes every "
                        "area-weighted result")
    p.add_argument("--gamry", metavar="DIR",
                   help="folder of whole-cell Gamry .DTA sweeps; sets "
                        "EIS_GAMRY_ROOT for this run so the whole-cell "
                        "comparison is written")
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

    for value, name in ((a.famos, "EIS_FAMOS_ROOT"), (a.csv, "EIS_CSV_ROOT"),
                        (a.out, "EIS_RESULTS_ROOT"),
                        (a.curr_cal, "EIS_CURR_CAL"), (a.temp_cal, "EIS_TEMP_CAL"),
                        (a.gamry, "EIS_GAMRY_ROOT")):
        if value:
            os.environ[name] = str(Path(value).expanduser())
    os.environ.setdefault("EIS_ALLOW_INLINE_PIPELINE", "1")

    from app.data.sources import (CsvLoggerSource, FamosSource,
                                  _condition_sort_key)
    from app.services import runner, staging
    from app.settings import SETTINGS

    if a.self_test:
        ok, output = runner.self_test(SETTINGS)
        print(output)
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    want_famos = a.source in ("auto", "famos")
    want_csv = a.source in ("auto", "csv")
    have_famos = bool(SETTINGS.famos_roots) and want_famos
    have_csv = bool(SETTINGS.csv_roots) and want_csv

    if not (have_famos or have_csv):
        from app.settings import DOTENV_LOADED
        env_path = ROOT / ".env"
        needed = {"famos": "EIS_FAMOS_ROOT", "csv": "EIS_CSV_ROOT"}.get(
            a.source, "EIS_FAMOS_ROOT or EIS_CSV_ROOT")
        print(f"error: {needed} is not set.\n", file=sys.stderr)
        if DOTENV_LOADED:
            print(f"  {DOTENV_LOADED} was read, but it does not set it.\n"
                  f"  Add whichever applies:\n"
                  f"      EIS_FAMOS_ROOT=<folder holding the .DAT cards>\n"
                  f"      EIS_CSV_ROOT=<campaign folder of R2-D2 sweep folders>",
                  file=sys.stderr)
        else:
            print(f"  There is no settings file yet. Create one:\n"
                  f"      python run_dashboard.py --init\n"
                  f"  It will be written to:\n"
                  f"      {env_path}", file=sys.stderr)
        print("\n  Or pass the folder directly, just this once:\n"
              '      python run_evaluation.py --famos "<folder>" --list\n'
              '      python run_evaluation.py --csv "<folder>" --list',
              file=sys.stderr)
        return 2

    from app.plates import registry
    PLATE_KEYS = {"gen1": "gen1_r2d2_72", "gen2": "gen2_r2d2_naboo_72"}
    geom = registry.get(PLATE_KEYS.get(a.plate) or registry.default_key())
    print(f"\nplate: {geom.name}")
    if not geom.verified:
        print("  NOTE: this layout is unverified — its areas are provisional")

    refs = []
    searched = []
    if have_famos:
        refs += FamosSource(SETTINGS.famos_roots, SETTINGS.famos_glob).scan()
        searched += list(SETTINGS.famos_roots)
    if have_csv:
        refs += CsvLoggerSource(SETTINGS.resolved_csv_roots()).scan()
        searched += [str(r) for r in SETTINGS.resolved_csv_roots()]

    discovered = list(refs)
    if a.leepa:
        wanted = {normalise_order_id(x) for x in a.leepa}
        refs = [r for r in refs
                if normalise_order_id(r.measurement_id) in wanted]
    refs.sort(key=lambda r: (r.kind, r.measurement_id,
                             _condition_sort_key(r.condition)))

    if not refs and discovered:
        # The filter emptied the list, not the disk. Saying "no recordings
        # found under <roots>" here is simply false and sends the reader to
        # check paths that are fine, so name the filter and list what it
        # could have matched.
        have = sorted({r.measurement_id for r in discovered})
        print(f"\nno recording matches order id "
              f"{', '.join(a.leepa)}", file=sys.stderr)
        print(f"  {len(discovered)} recording(s) were found; the order ids "
              f"present are: {', '.join(have)}", file=sys.stderr)
        return 1
    if not refs:
        print(f"no recordings found under {', '.join(dict.fromkeys(searched))}")
        print("run `python run_dashboard.py --check` to see why")
        return 1

    print(f"\nfound {len(refs)} condition(s):")
    width = max((len(r.condition) for r in refs), default=8)
    for ref in refs:
        what = ("card file(s)" if ref.kind == "famos" else "CSV point file(s)")
        print(f"  [{ref.kind:<6}] {ref.measurement_id} / "
              f"{ref.condition:<{width}}  {len(ref.files)} {what}  ->  "
              f"{runner.output_dir(ref, SETTINGS)}")
    if a.list:
        from app.data.sources import famos_patterns
        skipped = survey_roots(
            [r for r in SETTINGS.famos_roots] if have_famos else [],
            famos_patterns())
        count = int(skipped[0].split(":")[1]) if skipped else 0
        if count:
            print(f"\n{count} folder(s) under the FAMOS root produced no "
                  f"run:")
            for line in skipped[1:]:
                print(line)
        return 0

    if a.condition:
        wanted = {c.lower() for c in a.condition}
        refs = [r for r in refs if r.condition.lower() in wanted]
        if not refs:
            print(f"\nnone of {a.condition} is among the conditions found",
                  file=sys.stderr)
            return 1
    elif not a.all:
        # Listing is not the same as asking for hours of processing, so nothing
        # runs without being asked. Print the command to ask with, rather than
        # the flag names on their own.
        remote = any(staging.is_network_path(r.path) for r in refs)
        stage = " --stage-local" if remote else ""
        one = refs[0].condition
        print("\nNothing processed yet — that needs asking for explicitly.\n")
        print(f"  every condition:   python run_evaluation.py --all{stage}")
        print(f"  just one:          python run_evaluation.py "
              f"--condition {one}{stage}")
        if remote:
            print("\n  --stage-local copies each condition's cards to local disk "
                  "first.\n  The recordings are on a network share, where the "
                  "reader memory-maps\n  each card and bronze walks it "
                  "repeatedly, so every page fault is\n  an SMB round trip.")
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
            runner.run_pipeline(
                lambda done, total, message="": print(f"  {message}"),
                source, geom=geom, settings=SETTINGS,
                equal_areas=a.equal_areas,
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
