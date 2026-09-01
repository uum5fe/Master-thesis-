#!/usr/bin/env python3
r"""
evaluate_per_card.py
====================
Evaluate every card of a condition, whatever its sample rate and whenever it
was armed, and put all the spectra on one plot.

    python evaluate_per_card.py --leepa 2612025 --condition 45A
    python evaluate_per_card.py --leepa 2612025 --condition 45A --stage-local
    python evaluate_per_card.py --leepa 2612025 --all

Paths come from .env, the same file the dashboard reads.

WHY THIS IS SOUND AND NOT A SHORTCUT
------------------------------------
The impedance of a segment is

    Z_k(f) = U_cell(f) / I_k(f)

and every card records its OWN copy of the cell voltage.  When the segment
current and the cell voltage it is divided by come from the same card, a bulk
timing offset of that card multiplies numerator and denominator alike and
CANCELS EXACTLY.  It is not approximately removed and it is not corrected --
it is absent from the ratio.  That is the pipeline's own accurate default
(`uc_strategy: same_card`), and it means the inter-card alignment is not
needed to compute a per-segment spectrum at all.

What the alignment IS needed for is the CONSENSUS SCHEDULE: pooling the steps
detected on several cards onto one time base so they can vote on each other,
and verifying the cancellation rather than assuming it.  Evaluating one card
at a time gives that up -- each card finds its own schedule, on its own clock,
and there is no cross-card vote -- and gives up nothing else.

So a folder whose cards cannot be evaluated TOGETHER can still be evaluated
CARD BY CARD:

  * different sample rates stop mattering, because no quantity in samples
    ever crosses between cards; each card is analysed at its own rate, with
    its own ceiling;
  * different arming times stop mattering, because no window from one card is
    ever applied to another;
  * the results are spectra in HERTZ, and hertz are comparable across cards
    however they were sampled.

WHAT IS GIVEN UP, PLAINLY
-------------------------
  * Cross-card agreement as evidence.  A step is kept on one reference
    channel rather than two, so the SNR and grid tests carry the whole
    burden of deciding what is real.
  * The verification that the skew cancelled.  It cancels by construction
    here; nothing measures that it did.
  * A common frequency grid.  Each card detects its own steps, so two cards
    may report slightly different frequencies. Fine for plotting and for
    per-segment work; it is why the plate-wide aggregate is not recomputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------

#: The keys this needs, and whether it can run without them.
ENV_KEYS = {
    "EIS_FAMOS_ROOT": "folder holding the .DAT cards",
    "EIS_RESULTS_ROOT": "where results are written",
    "EIS_CURR_CAL": "per-segment shunt calibration (curr.csv)",
    "EIS_TEMP_CAL": "temperature sensor calibration (temp.csv)",
    "EIS_GAMRY_ROOT": "whole-cell Gamry sweeps (optional)",
}
REQUIRED = ("EIS_FAMOS_ROOT", "EIS_RESULTS_ROOT", "EIS_CURR_CAL")


def load_env(path: Path | None = None) -> dict[str, str]:
    """Read a .env of KEY=VALUE lines, without needing python-dotenv.

    Windows paths are taken verbatim: no unescaping, no expansion.  A UNC
    path is full of backslashes and every one of them is part of the path,
    so anything that treats a backslash as an escape turns
    \\\\bosch.com\\DfsRB into something that does not exist -- and the error
    that produces is "no cards found", which points at the share rather than
    at the parser.
    """
    candidates = [path] if path else [
        Path.cwd() / ".env", ROOT / ".env", ROOT.parent / ".env"]
    for candidate in candidates:
        if candidate and candidate.is_file():
            values: dict[str, str] = {}
            for line in candidate.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")
            values["_source"] = str(candidate)
            return values
    searched = ", ".join(str(c) for c in candidates if c)
    raise SystemExit(f"error: no .env found. Looked in: {searched}")


# ---------------------------------------------------------------------------
# running one card
# ---------------------------------------------------------------------------

#: measurement id and condition out of a card's file name.
_NAME = re.compile(
    r"(?:Leepa[_-]?|RO)?(?P<leepa>\d{6,})(?:-\d+)?_Current_"
    r"(?P<condition>.+?)_Test_\d+_Karte_\w+\.DAT$", re.IGNORECASE)


def cards_of(root: Path, condition: str, leepa: str) -> list[Path]:
    """The .DAT files of one condition, searched RECURSIVELY.

    The root in .env is a campaign folder, and campaigns are usually kept
    with one sub-folder per condition. bronze's own discovery globs a single
    directory -- it is handed one condition's folder by its caller -- so
    searching only the root finds nothing and reports it as "no FAMOS files",
    which reads as a missing share rather than as a folder one level up.
    """
    found = []
    for path in sorted(set(list(root.rglob("*.DAT")) + list(root.rglob("*.dat")))):
        match = _NAME.search(path.name)
        if not match:
            continue
        if match.group("leepa") != leepa:
            continue
        if match.group("condition").lower() != condition.lower():
            continue
        found.append(path)
    return found


def conditions_of(root: Path, leepa: str) -> list[str]:
    """Every condition present for this order id."""
    out = set()
    for path in list(root.rglob("*.DAT")) + list(root.rglob("*.dat")):
        match = _NAME.search(path.name)
        if match and match.group("leepa") == leepa:
            out.add(match.group("condition"))
    return sorted(out)


def card_tag(path: Path) -> str:
    """A short, unique handle for one card: Karte_3, or the file stem."""
    import re
    match = re.search(r"(Karte[_-]?\w+)", path.name, re.IGNORECASE)
    return match.group(1) if match else path.stem


def card_rate(card: Path) -> float:
    """This card's sample rate, from its header alone."""
    from eis_local import open_famos
    return float(open_famos(card).fs)


def ceiling_for(fs: float, interferer: float | None,
                fixed: float | None) -> tuple[float | None, str]:
    """The analysis ceiling for ONE card, decided from ITS OWN sample rate.

    This is the per-rate branch the whole approach turns on.  An interferer
    above a card's Nyquist folds INTO its band, and it folds to a different
    place at every rate: 45996 Hz sits at 45996 Hz on a 100 kHz card, well
    above anything electrochemical, and at 4004 Hz on a 50 kHz card, in the
    middle of the measurement.  A folded peak carries no mark that it is one
    -- it is an ordinary tone at an ordinary frequency -- so it cannot be
    removed afterwards and cannot be told from signal.

    What can be done is to stop each card's analysis below ITS OWN fold.
    One ceiling for the whole plate cannot do that: the value that protects
    the 50 kHz cards throws away three quarters of the band on the 100 kHz
    ones.
    """
    if fixed is not None:
        return fixed, f"--f-max {fixed:.0f} for every card"
    if interferer is None:
        return None, "auto (0.45 * this card's rate)"
    alias = abs(interferer - round(interferer / fs) * fs)
    nyquist_limit = 0.45 * fs
    if not (0 < alias < nyquist_limit):
        return None, (f"{interferer:.0f} Hz stays at {alias:.0f} Hz here, "
                      f"outside the band; auto")
    ceiling = 0.8 * alias
    return ceiling, (f"{interferer:.0f} Hz folds to {alias:.0f} Hz at "
                     f"{fs / 1000:.0f} kHz; stopping at {ceiling:.0f} Hz")


def run_one_card(card: Path, leepa: str, condition: str,
                 out_dir: Path, env: dict, f_max: float | None,
                 plate: str | None, extra: list[str]) -> tuple[bool, str]:
    """Run the pipeline over a single card, in its own output folder."""
    tag = card_tag(card)
    # The card's OWN folder, so a campaign kept one-condition-per-directory
    # works without the root having to be the condition's directory.
    argv = [sys.executable, "main.py",
            "--dat", str(card.parent), "--leepa", leepa,
            "--condition", condition, "--cards", tag,
            "--label", f"{condition}_{tag}",
            "--out", str(out_dir / tag),
            "--curr-cal", env["EIS_CURR_CAL"],
            "--no-png", "--no-html", "--no-require-timebase"]
    if env.get("EIS_TEMP_CAL"):
        argv += ["--temp-cal", env["EIS_TEMP_CAL"]]
    if f_max is not None:
        argv += ["--f-max", str(f_max)]
    if plate:
        argv += ["--plate", plate]
    argv += extra

    child = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(argv, cwd=str(ROOT), capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            env=child)
    if result.returncode != 0:
        tail = "\n".join((result.stdout or "").splitlines()[-6:])
        return False, tail or (result.stderr or "")[-400:]
    return True, ""


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------

def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def merge(out_dir: Path, tags: list[str], target: Path) -> dict:
    """Concatenate the per-card tables into one condition folder.

    The segments of two cards are disjoint -- each card is wired to its own
    block of the plate -- so this is a concatenation and not a merge with a
    rule for collisions. Where a segment does appear twice (it should not),
    the first card wins and the duplicate is counted and reported.
    """
    (target / "silver").mkdir(parents=True, exist_ok=True)
    (target / "gold").mkdir(parents=True, exist_ok=True)

    spectra, summary, seen, dupes = [], [], set(), []
    per_card = {}
    for tag in tags:
        rows = _read(out_dir / tag / "silver" / "spectra_clean.csv")
        plate = _read(out_dir / tag / "gold" / "plate_summary.csv")
        measured = {r["segment"] for r in plate if r.get("class") == "measured"}
        per_card[tag] = {"n_points": len(rows), "n_measured": len(measured)}

        for row in rows:
            seg = row.get("segment", "")
            if seg in seen and seg not in {r["segment"] for r in spectra}:
                pass
            row["card"] = tag
            spectra.append(row)
        for row in plate:
            seg = row.get("segment", "")
            if row.get("class") != "measured":
                continue
            if seg in seen:
                dupes.append(seg)
                continue
            seen.add(seg)
            row["card"] = tag
            summary.append(row)

    # keep only the spectra of segments that survived, and drop duplicates
    keep = {r["segment"] for r in summary}
    spectra = [r for r in spectra if r["segment"] in keep]

    if spectra:
        fields = list(dict.fromkeys(k for r in spectra for k in r))
        with (target / "silver" / "spectra_clean.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(spectra)
    if summary:
        fields = list(dict.fromkeys(k for r in summary for k in r))
        with (target / "gold" / "plate_summary.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(summary,
                                    key=lambda r: int(r["segment"])
                                    if r["segment"].isdigit() else 0))

    return {"n_segments": len(summary), "n_points": len(spectra),
            "duplicates": sorted(set(dupes)), "per_card": per_card}


# ---------------------------------------------------------------------------
# the plot
# ---------------------------------------------------------------------------

def plot(target: Path, title: str) -> Path | None:
    """One Nyquist and one Bode over every segment of every card."""
    rows = _read(target / "silver" / "spectra_clean.csv")
    if not rows:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib is not installed; the CSVs are written but the "
              "plot is not. pip install matplotlib")
        return None
    import numpy as np

    by_seg: dict[str, list[tuple[float, float, float]]] = {}
    card_of: dict[str, str] = {}
    for row in rows:
        try:
            f = float(row["freq_hz"])
            re_ = float(row["z_re_mohm_cm2"])
            im = float(row["z_im_mohm_cm2"])
        except (KeyError, ValueError):
            continue
        seg = row["segment"]
        by_seg.setdefault(seg, []).append((f, re_, im))
        card_of[seg] = row.get("card", "")

    if not by_seg:
        return None
    cards = sorted(set(card_of.values()))
    # One colour per CARD, not per segment: seventy-two colours are not
    # distinguishable, and the question this plot answers is whether the
    # cards agree with each other.
    cmap = plt.get_cmap("tab10")
    colour = {c: cmap(i % 10) for i, c in enumerate(cards)}

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for seg, points in sorted(by_seg.items(),
                              key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        points.sort()
        f = np.array([p[0] for p in points])
        re_ = np.array([p[1] for p in points])
        im = np.array([p[2] for p in points])
        col = colour.get(card_of[seg], "0.5")
        ax[0].plot(re_, -im, "-", lw=0.9, alpha=0.65, color=col)
        ax[1].plot(f, np.hypot(re_, im), "-", lw=0.9, alpha=0.65, color=col)
        ax[2].plot(f, np.degrees(np.arctan2(im, re_)), "-", lw=0.9,
                   alpha=0.65, color=col)

    ax[0].set_xlabel("Z' [mΩ·cm²]"); ax[0].set_ylabel("-Z'' [mΩ·cm²]")
    ax[0].set_title("Nyquist"); ax[0].axis("equal"); ax[0].grid(alpha=.3)
    ax[1].set_xscale("log"); ax[1].set_xlabel("f [Hz]")
    ax[1].set_ylabel("|Z| [mΩ·cm²]"); ax[1].set_title("|Z|"); ax[1].grid(alpha=.3)
    ax[2].set_xscale("log"); ax[2].set_xlabel("f [Hz]")
    ax[2].set_ylabel("phase [°]"); ax[2].set_title("Phase"); ax[2].grid(alpha=.3)

    handles = [plt.Line2D([], [], color=colour[c], lw=2,
                          label=f"{c}  ({sum(1 for s in card_of if card_of[s] == c)} seg)")
               for c in cards]
    ax[0].legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle(f"{title} — {len(by_seg)} segments, evaluated card by card")
    fig.tight_layout()
    path = target / "spectra_all_cards.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def evaluate(condition: str, env: dict, a) -> int:
    dat_root = Path(env["EIS_FAMOS_ROOT"])
    results = Path(env["EIS_RESULTS_ROOT"])
    leepa = a.leepa

    cards = cards_of(dat_root, condition, leepa)
    if not cards:
        print(f"  {condition}: no cards found under {dat_root}")
        return 1

    label = f"{condition}{a.suffix}"
    target = results / leepa / label
    scratch = target / "_per_card"
    print(f"\n{'=' * 70}\n{leepa} / {condition}  ->  {label}\n{'=' * 70}")
    print(f"  {len(cards)} card(s); each is evaluated on its own clock and at "
          f"its own sample rate.")

    ok, failed = [], []
    for index, card in enumerate(cards, 1):
        tag = card_tag(card)
        print(f"  [{index}/{len(cards)}] {tag}: ", end="", flush=True)
        if a.dry_run:
            print("(dry run)")
            ok.append(tag)
            continue
        try:
            fs = card_rate(card)
        except Exception as exc:                            # noqa: BLE001
            failed.append((tag, f"could not read the header: {exc}"))
            print("FAILED (header)")
            continue
        ceiling, why_ceiling = ceiling_for(fs, a.interferer_hz, a.f_max)
        print(f"{fs / 1000:.0f} kHz, {why_ceiling} ...", end=" ", flush=True)
        good, why = run_one_card(card, leepa, condition, scratch,
                                 env, ceiling, a.plate, a.extra)
        if good:
            ok.append(tag)
            print("done")
        else:
            failed.append((tag, why))
            print("FAILED")
            for line in why.splitlines()[-4:]:
                print(f"      {line}")

    if a.dry_run:
        return 0
    if not ok:
        print("\n  no card produced results; nothing to merge")
        return 1

    stats = merge(scratch, ok, target)
    print(f"\n  merged: {stats['n_segments']} segments, "
          f"{stats['n_points']} spectrum points")
    for tag, info in stats["per_card"].items():
        print(f"    {tag:<10} {info['n_measured']:>3} segments, "
              f"{info['n_points']:>5} points")
    if stats["duplicates"]:
        print(f"    NOTE: {len(stats['duplicates'])} segment(s) appeared on "
              f"more than one card; the first card kept them: "
              f"{', '.join(stats['duplicates'][:8])}")

    image = plot(target, f"{leepa} / {label}")
    if image:
        print(f"  plot   : {image}")
    (target / "per_card_manifest.json").write_text(
        json.dumps({"condition": condition, "label": label,
                    "cards_ok": ok, "cards_failed": [t for t, _ in failed],
                    **stats}, indent=2), encoding="utf-8")
    print(f"  results: {target}")
    if failed:
        print(f"  {len(failed)} card(s) failed; the rest are in the result.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", type=Path, help="path to the .env (default: "
                                            "beside this script or the cwd)")
    p.add_argument("--leepa", required=True, help="order id, e.g. 2612025")
    p.add_argument("--condition", action="append",
                   help="condition to evaluate; repeatable")
    p.add_argument("--all", action="store_true",
                   help="every condition found for this order id")
    p.add_argument("--suffix", default="_percard",
                   help="appended to the condition to name the result "
                        "(default _percard), so it sits beside a whole-plate "
                        "run rather than overwriting it")
    p.add_argument("--f-max", type=float, default=None,
                   help="one analysis ceiling in Hz for EVERY card. Leave "
                        "unset and each card uses 0.45 * its own rate")
    p.add_argument("--interferer-hz", type=float, default=None,
                   help="a known interference frequency, e.g. 45996. Each "
                        "card's ceiling is then set below where that "
                        "frequency FOLDS at that card's own rate -- which is "
                        "a different place on a 50 kHz card than on a "
                        "100 kHz one, and is why one ceiling cannot serve "
                        "both")
    p.add_argument("--plate", choices=["gen1", "gen2"], default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after this is passed to main.py verbatim")
    a = p.parse_args(argv)

    env = load_env(a.env)
    print(f"settings: {env['_source']}")
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing:
        print("error: .env is missing " + ", ".join(missing), file=sys.stderr)
        for key in missing:
            print(f"  {key} = {ENV_KEYS[key]}", file=sys.stderr)
        return 2
    for key in ("EIS_FAMOS_ROOT", "EIS_RESULTS_ROOT"):
        print(f"  {key} = {env[key]}")

    conditions = list(a.condition or [])
    if a.all or not conditions:
        root = Path(env["EIS_FAMOS_ROOT"])
        conditions = conditions_of(root, a.leepa)
        if not conditions:
            print(f"error: no cards for order id {a.leepa} under {root}",
                  file=sys.stderr)
            return 1
        print(f"  conditions: {', '.join(conditions)}")

    worst = 0
    for condition in conditions:
        worst = max(worst, evaluate(condition, env, a))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
