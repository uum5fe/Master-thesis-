#!/usr/bin/env python3
r"""
preflight.py
============
Answer, from the headers and a few seconds of signal, the three questions
that are expensive to get wrong and cheap to check:

    1. Are these cards one measurement, on one time base?
    2. Is the calibration -- current AND temperature -- actually applied?
    3. What excitation is this, and does the analysis band reach it?

    python preflight.py --dat <folder of .DAT cards> \
                        --curr-cal curr.csv --temp-cal temp.csv
    python preflight.py --dat <folder> --leepa 2612025 --json report.json

Exits 0 when nothing is wrong, 1 when something is.  Nothing here processes
a sweep, so it runs in seconds on a share -- as against the minutes per
condition bronze needs once it starts reading every sample of every card.

WHY THIS IS A SEPARATE COMMAND
------------------------------
All three checks also run inside bronze, which is the right place for them:
they gate the run.  But by the time bronze reports them the cards have been
staged and read, and the answer to "should I have started this?" arrives
after the cost of starting it.  Three of the failures this catches --
cards from two different runs, a temperature calibration whose channel
names do not match, an analysis ceiling below the excitation -- are visible
in the file headers alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bronze as B                                          # noqa: E402
import r2d2_geometry as geom                                # noqa: E402
import utils                                                # noqa: E402
from config import DEFAULT                                  # noqa: E402
from eis_local import (open_famos, PlateCalibration,          # noqa: E402
                       classify_excitation)


def _cards(files, cfg, log, max_samples=None):
    utils.section("cards", log)
    return B.inventory_channels(files, cfg, log, max_samples=max_samples)


def excitation_survey(files, cards, cfg, log, seconds: float = 20.0) -> dict:
    """What the excitation is, and whether the band reaches it.

    Only the first `seconds` of each card is read.  The excitation type and
    the position of the band relative to it do not change halfway through a
    recording, and reading the whole card here would cost exactly what this
    command exists to avoid.
    """
    utils.section("excitation", log)
    out = {}
    for fp in files:
        stem = fp.stem
        if stem not in cards:
            continue
        fam = open_famos(fp)
        n = min(fam.n_samples, int(seconds * fam.fs))
        # Bounded at the memmap, not after it: see FamosFile.channel. Slicing
        # a materialised column reads the whole recording first, which over
        # SMB is the entire cost this command exists to avoid.
        ref = fam.channel(cards[stem].ref_name, 0, n)
        f_hi = cfg.f_hi(fam.fs)
        verdict = classify_excitation(ref, fam.fs, cfg.f_min_hz, f_hi)
        head = B.excitation_above_ceiling(ref, fam.fs, f_hi)
        out[stem] = {"kind": verdict["kind"],
                     "n_dwells": verdict["n_dwells"],
                     "n_tones": verdict["n_tones"],
                     "tones_at_once": verdict["tones_at_once"],
                     "f_hi_hz": f_hi, "nyquist_hz": fam.fs / 2,
                     "above_ceiling": head}
        log.info(f"  {stem[-8:]}: {verdict['kind']} — "
                 f"{verdict['n_tones']} strong tone(s), median "
                 f"{verdict['tones_at_once']:.1f} on at once"
                 + (f", {verdict['n_dwells']} dwell(s)"
                    if verdict['n_dwells'] else "")
                 + f"; search ceiling {f_hi:.0f} Hz of "
                   f"{fam.fs / 2:.0f} Hz Nyquist")
        if head.get("significant"):
            log.warning(
                f"    {100 * head['fraction']:.0f} % of the AC energy is "
                f"ABOVE the {f_hi:.0f} Hz ceiling, strongest at "
                f"{head['peak_hz']:.0f} Hz. Those tones are not being "
                f"searched for -- raise --f-max, or leave it at 'auto' so "
                f"the ceiling follows the sample rate.")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dat", type=Path, required=True,
                   help="folder holding the .DAT cards")
    p.add_argument("--leepa", default="", help="order id, e.g. 2612025")
    p.add_argument("--condition", default="ALL")
    p.add_argument("--curr-cal", type=Path)
    p.add_argument("--temp-cal", type=Path)
    p.add_argument("--json", type=Path, help="write the full report here")
    p.add_argument("--seconds", type=float, default=20.0,
                   help="how much of each card to read for the excitation "
                        "survey (default 20 s)")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    cfg = DEFAULT.replace(dat_dir=a.dat, leepa=a.leepa, condition=a.condition,
                          curr_cal=a.curr_cal, temp_cal=a.temp_cal,
                          verbose=not a.quiet)
    log = utils.get_logger(cfg.verbose)
    utils.banner("PREFLIGHT  --  before anything is evaluated", log)

    files = B.discover_files(cfg)
    log.info(f"  {len(files)} file(s) matching {cfg.famos_pattern()!r}")
    if not files:
        log.error("  nothing to check")
        return 1

    # Everything below reads only a bounded slice of each card. The point of
    # this command is to answer, over a network share, in seconds -- and a
    # survey that pulls every card across the wire is not a survey, it is the
    # expensive half of the run without the results.
    # Samples, from the highest rate any card could plausibly run at, so the
    # bound is never tighter than `--seconds` on the fastest card. Each read
    # clamps to that card's own length anyway.
    look = int(a.seconds * 200_000)
    channels, cards = _cards(files, cfg, log, max_samples=look)

    timebase = B.timebase_report(cards, cfg, log)

    cal = PlateCalibration.load(cfg.curr_cal, cfg.temp_cal)
    rejected: dict[str, str] = {}
    utils.section("plate temperature", log)
    T_seg, sensor_T = B.plate_temperatures(files, cards, cal, cfg, log,
                                           rejected=rejected,
                                           max_samples=look)
    calrep = B.calibration_report(cal, cfg, sensor_T, T_seg, rejected, log)

    excitation = excitation_survey(files, cards, cfg, log, a.seconds)

    report = {
        "files": [str(f) for f in files],
        "leepa": a.leepa,
        "timebase": timebase.summary(),
        "calibration": calrep.summary(),
        "excitation": excitation,
    }
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=2, default=str),
                          encoding="utf-8")
        log.info(f"\n  written: {a.json}")

    problems = list(timebase.problems) + list(calrep.problems)
    ceiling = [s for s, e in excitation.items()
               if e["above_ceiling"].get("significant")]
    if ceiling:
        problems.append(
            f"excitation above the search ceiling on {len(ceiling)} card(s); "
            f"those tones will not be found")

    utils.section("verdict", log)
    if problems:
        log.error(f"  {len(problems)} problem(s) would affect this run:")
        for problem in problems:
            log.error(f"    - {problem}")
        log.info("\n  Fix these before spending the run.")
        return 1
    log.info("  time base, calibration and band all check out.")
    n_multi = sum(1 for e in excitation.values() if e["kind"] == "multisine")
    if n_multi:
        log.info(f"  {n_multi} card(s) carry a stepped multisine; the tones "
                 f"of each dwell will be fitted jointly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
