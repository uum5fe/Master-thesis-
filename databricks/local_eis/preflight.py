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
import re
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
            peak, nyquist = head["peak_hz"], head["nyquist_hz"]
            log.warning(
                f"    {100 * head['fraction']:.0f} % of the AC energy is "
                f"ABOVE the {f_hi:.0f} Hz ceiling, strongest at "
                f"{peak:.0f} Hz.")
            # TWO READINGS THAT NEED OPPOSITE RESPONSES. Saying only "raise
            # --f-max" assumes it is excitation, and on a fuel cell a tone in
            # the tens of kHz usually is not -- a load bank's switching
            # frequency lands there. Chasing it wastes the run; what it
            # needs is a notch.
            log.warning(
                f"    If those are excitation tones, raise --f-max. If they "
                f"are interference -- which is what energy in the tens of "
                f"kHz usually is on a cell -- a wider band makes the "
                f"detector chase it, and a notch is what removes it.")
            if peak > 0.85 * nyquist:
                log.warning(
                    f"    {peak:.0f} Hz is within 15 % of this card's "
                    f"Nyquist, so it may itself be an alias of something "
                    f"higher the anti-alias filter did not stop.")
    return out


def alias_report(excitation: dict, cards, log) -> list[dict]:
    """Where a fast card's high-frequency content lands on a slower one.

    WHY THIS IS WORTH CHECKING ACROSS CARDS
    ---------------------------------------
    Content above a card's Nyquist does not disappear: it folds back into
    the band, and it folds to a DIFFERENT place on every sample rate. So a
    plate recorded at two rates can carry the same physical interference at
    46 kHz on the fast cards -- where it is visible, above the analysis
    band, and obviously not electrochemistry -- and at 4 kHz on the slow
    ones, where it is indistinguishable from a real excitation tone and
    sits in the middle of the measurement.

    Neither card can see this on its own. The fast card cannot know another
    card sampled slower; the slow card cannot know the 4 kHz peak was ever
    at 46 kHz. It is only visible by comparing them, which is why it is
    checked here rather than in the per-card survey.
    """
    rates = {info["nyquist_hz"] * 2 for info in excitation.values()}
    peaks = [(stem, info["above_ceiling"]["peak_hz"])
             for stem, info in excitation.items()
             if info["above_ceiling"].get("significant")]
    if len(rates) < 2 or not peaks:
        return []

    utils.section("aliasing across the sample rates on this plate", log)
    slower = sorted(rates)[:-1]
    out = []
    for stem, peak in peaks:
        for fs in slower:
            folded = abs(peak - round(peak / fs) * fs)
            entry = {"seen_on": stem, "peak_hz": peak, "at_rate_hz": fs,
                     "folds_to_hz": folded}
            out.append(entry)
            log.warning(
                f"  {peak:.0f} Hz (strong on {stem[-8:]}) folds to "
                f"{folded:.0f} Hz on a card sampled at {fs:.0f} Hz.")
            if folded < 5000.0:
                log.error(
                    f"    That is inside the analysis band, where it is "
                    f"indistinguishable from an excitation tone. The slower "
                    f"cards will carry it as a peak that is not "
                    f"electrochemistry, and no per-card check can see that.")
                entry["in_band"] = True
    return out


def safe_ceiling(fs: float, interferers, default_hi: float) -> tuple[float, str]:
    """The highest analysis ceiling that stays below any aliased interference.

    An interferer above a card's Nyquist folds INTO the band, and there is
    nothing about the folded peak that marks it as an alias -- it is a
    perfectly ordinary tone at a perfectly ordinary frequency. It cannot be
    removed after sampling and it cannot be told from signal.

    What CAN be done is to stop the analysis below it. That does not recover
    the band above; it keeps the part below honest, which is the difference
    between a spectrum with a fabricated point in it and a shorter spectrum.
    """
    folded = []
    for peak in interferers:
        alias = abs(peak - round(peak / fs) * fs)
        if 0 < alias < default_hi:
            folded.append((alias, peak))
    if not folded:
        return default_hi, "auto"
    alias, source = min(folded)
    ceiling = 0.8 * alias
    return ceiling, (f"{source:.0f} Hz folds to {alias:.0f} Hz at this rate; "
                     f"stopping at {ceiling:.0f} Hz keeps the band below it")


def _card_tag(stem: str) -> str:
    """The shortest substring of a card's name that still identifies it."""
    match = re.search(r"(Karte[_-]?\w+)", stem, re.IGNORECASE)
    return match.group(1) if match else stem


def group_plan(cards, excitation, cfg, log, a_condition: str = "COND"
               ) -> list[dict]:
    """Which subsets of this folder can be evaluated, and how.

    A folder is not a measurement. When part of it is evaluable the useful
    answer is which part and what command runs it -- not a refusal of the
    whole folder, and not an evaluation that treats cards from two runs as
    one event.
    """
    groups = B.consistent_groups(cards)
    utils.section("what CAN be evaluated from this folder", log)

    # Only peaks seen on the FASTEST cards are true frequencies. A peak on a
    # slower card may itself already be folded, and folding a fold again
    # would put the ceiling in an arbitrary place.
    fastest = max((info["nyquist_hz"] * 2 for info in excitation.values()),
                  default=0.0)
    interferers = [info["above_ceiling"]["peak_hz"]
                   for info in excitation.values()
                   if info["above_ceiling"].get("significant")
                   and info["nyquist_hz"] * 2 >= fastest]

    total = sum(g.get("n_segments", 0) for g in groups)
    out = []
    for index, group in enumerate(groups, 1):
        fs = group["fs_hz"]
        ceiling, why = safe_ceiling(fs, interferers, cfg.f_hi(fs))
        group["f_max_hz"] = ceiling
        group["f_max_reason"] = why
        out.append(group)

        log.info(f"\n  group {index}: {group['n_cards']} card(s) at "
                 f"{fs:.0f} Hz, {group.get('n_segments', 0)} of 72 segments")
        log.info(f"    cards   : {', '.join(c[-8:] for c in group['cards'])}")
        if group.get("timed"):
            log.info(f"    together: {group['overlap_s']:.1f} s")
        else:
            log.warning("    no |NT stamps: grouped by sample rate alone, so "
                        "whether these were recorded together is unverified")
        log.info(f"    ceiling : {ceiling:.0f} Hz  ({why})")
        if group["n_cards"] == 1:
            log.warning("    one card only -- there is no second reference to "
                        "measure its timing against, so its alignment is "
                        "assumed rather than verified")

    utils.section("the commands", log)
    for index, group in enumerate(out, 1):
        cards_arg = ",".join(_card_tag(c) for c in group["cards"])
        suffix = f"{a_condition}_g{index}_{group['fs_hz'] / 1000:.0f}kHz"
        log.info(f"\n  group {index}  ->  condition {suffix}")
        log.info(f'    python main.py --dat "<folder>" --leepa <id> \\')
        # --condition FINDS the files, --label NAMES the result. They differ
        # here: this is a subset of the condition, not the condition.
        log.info(f'        --condition {a_condition} --cards {cards_arg} \\')
        log.info(f'        --label {suffix} \\')
        log.info(f'        --f-max {group["f_max_hz"]:.0f} \\')
        log.info(f'        --curr-cal "<curr.csv>" --temp-cal "<temp.csv>" \\')
        log.info(f'        --out "<results root>/<id>/{suffix}"')
    log.info("\n  Each writes its own condition folder, so the dashboard "
             "lists them side by side and the Compare tab can put two of "
             "them together.")

    if len(groups) > 1:
        log.warning(
            f"\n  These groups CANNOT be combined into one plate: they differ "
            f"in sample rate or in when they were recorded, which is what "
            f"makes them separate groups. Each is a partial plate -- "
            f"{total} of 72 segments across all of them, and fewer in any "
            f"one. Every plate-wide number (current closure, area-weighted "
            f"aggregate, the parallel R_s) is computed over the segments "
            f"present, so read those as partial too.")
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
    p.add_argument("--groups", action="store_true",
                   help="report which subsets of this folder can be evaluated "
                        "together, and print the command for each")
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

    aliases = alias_report(excitation, cards, log)
    plan = (group_plan(cards, excitation, cfg, log,
                       a.condition if a.condition != 'ALL' else 'COND')
            if a.groups else [])

    report = {
        "files": [str(f) for f in files],
        "aliasing": aliases,
        "groups": plan,
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
            f"energy above the search ceiling on {len(ceiling)} card(s). If "
            f"it is excitation, raise --f-max; if it is interference, it "
            f"needs a notch rather than a wider band")
    in_band = [a for a in aliases if a.get("in_band")]
    if in_band:
        problems.append(
            f"{in_band[0]['peak_hz']:.0f} Hz, strong on the faster cards, "
            f"folds to {in_band[0]['folds_to_hz']:.0f} Hz on the "
            f"{in_band[0]['at_rate_hz']:.0f} Hz cards -- inside the analysis "
            f"band, where it cannot be told from a real tone")

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
