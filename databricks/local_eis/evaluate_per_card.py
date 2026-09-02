#!/usr/bin/env python3
r"""
evaluate_per_card.py
====================
Evaluate every card of a condition, whatever its sample rate and whenever it
was armed, and put all the spectra on one plot.

    python evaluate_per_card.py --leepa 2612025 --condition 45A
    python evaluate_per_card.py --leepa 2612025 --all
    python evaluate_per_card.py --leepa 2612025 --all --group card

Paths come from .env, the same file the dashboard reads.

THE PROBLEM THIS SOLVES
-----------------------
A run recorded on five cards at two sample rates cannot be evaluated as one
plate.  Not because the alignment gives a poor answer on mixed rates -- it
gives NO answer: every quantity the cards exchange is a count of SAMPLES (the
lag from `bronze.estimate_card_lags`, the dwell window indices of the
consensus schedule), and a count of samples is only the same amount of time
on two cards clocked alike.  `estimate_card_lags` refuses such a selection
outright rather than returning a lag that means one thing on one card and
half of it on another.

So the plate is partitioned into sets that CAN be evaluated together, each
set is put through the ordinary pipeline, and the spectra are concatenated
at the end -- in HERTZ, which are comparable across rates however they were
sampled.

TWO WAYS TO PARTITION, AND WHY THE DEFAULT IS BY RATE
-----------------------------------------------------
`--group rate` (the default) puts every card sharing a sample rate in one
group: the 100 kHz cards together, the 50 kHz cards together.  Inside a
group nothing is given up at all.  The cards are aligned to each other by
cross-correlation exactly as before -- arming times seconds apart is the
case that alignment exists to solve, and within a group it still solves it
-- and they still pool their detected steps into one consensus schedule and
vote on each other.  What is given up is only what had to be: the vote does
not cross the rate boundary.

`--group card` puts every card in a group of its own.  Use it when a rate
group fails as a whole, or when only one card per rate is readable.  It is
sound but weaker, and the reason it is sound is worth stating:

    Z_k(f) = U_cell(f) / I_k(f)

and every card records its OWN copy of the cell voltage.  When the segment
current and the cell voltage it is divided by come from the same card, a
bulk timing offset of that card multiplies numerator and denominator alike
and CANCELS EXACTLY.  It is not approximately removed and it is not
corrected -- it is absent from the ratio.  That is the pipeline's own
default (`uc_strategy: same_card`), and it is why a single card needs no
inter-card alignment to produce a per-segment spectrum.

WHAT `--group card` GIVES UP, PLAINLY
-------------------------------------
  * Cross-card agreement as evidence.  A step is kept on one reference
    channel rather than two, so the SNR and grid tests carry the whole
    burden of deciding what is real.
  * The verification that the skew cancelled.  It cancels by construction;
    nothing measures that it did.
  * A common frequency grid.  Each card detects its own steps, so two cards
    may report slightly different frequencies. Fine for plotting and for
    per-segment work; it is why the plate-wide aggregate is not recomputed.

`--group rate` gives up the last of those three across groups only, and the
first two not at all within a group.

THE SEPARATE MATTER OF THE CEILING
----------------------------------
Grouping fixes what crosses between cards.  It does not fix aliasing, which
is a property of one card: an interferer above a card's Nyquist folds INTO
its band, and to a DIFFERENT place at every rate.  45996 Hz sits at 45996 Hz
on a 100 kHz card and at 4004 Hz on a 50 kHz card, in the middle of the
measurement, carrying no mark that it is a fold.  `--interferer-hz 45996`
sets each group's analysis ceiling below its own fold.  See `ceiling_for`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Group:
    """A set of cards that may be evaluated TOGETHER, and its handle.

    Membership is decided by the two things that make a joint evaluation
    valid or invalid, and by nothing else:

      * ONE SAMPLE RATE.  Every quantity the cards exchange -- the lag from
        `estimate_card_lags`, the dwell window indices of the consensus
        schedule -- is a count of SAMPLES.  A count of samples is only the
        same amount of time on two cards clocked alike.  bronze does not
        paper over this: `estimate_card_lags` refuses a mixed-rate selection
        outright (see the SystemExit it raises), so a mixed group does not
        produce a wrong answer, it produces no answer at all.
      * ONE DIRECTORY.  `bronze.discover_files` globs a single folder, so
        cards that a campaign filed apart cannot be discovered in one run
        whatever their rate.

    Arming time is deliberately NOT a criterion.  Cards armed seconds apart
    is the case the cross-correlation exists to solve, and within a group it
    solves it; splitting on it would throw away the alignment rather than
    use it.
    """

    tag: str
    fs: float
    cards: tuple[Path, ...]
    directory: Path

    @property
    def card_tags(self) -> list[str]:
        return [card_tag(c) for c in self.cards]


def _rate_tag(fs: float) -> str:
    """A label a person can read: 100kHz, 50kHz, 12.5kHz."""
    khz = fs / 1000.0
    return (f"{khz:.0f}kHz" if abs(khz - round(khz)) < 0.05
            else f"{khz:g}kHz".replace(".", "p"))


def group_cards(cards: list[Path], mode: str = "rate",
                on_error=None) -> tuple[list[Group], list[tuple[str, str]]]:
    """Partition the cards into sets that can be evaluated together.

    ``mode="rate"`` puts every card sharing a sample rate (and a directory)
    in one group -- the 100 kHz cards together, the 50 kHz cards together --
    so that inside each group the alignment and the consensus schedule still
    run, and the cards still vote on each other's steps.  That is what a
    per-CARD split gives up, and it is worth keeping wherever the rate
    permits it.

    ``mode="card"`` puts every card in a group of its own.  It is the
    fallback: it needs no card to agree with any other, so it survives a
    plate where only one card per rate is readable.

    A card whose header cannot be read is not guessed at -- it is returned
    in the failure list with the reason.
    """
    groups: dict[tuple[float, str], list[Path]] = {}
    failed: list[tuple[str, str]] = []
    for card in cards:
        tag = card_tag(card)
        try:
            fs = card_rate(card)
        except Exception as exc:                            # noqa: BLE001
            failed.append((tag, f"could not read the header: "
                                f"{type(exc).__name__}: {exc}"))
            if on_error is not None:
                on_error(tag, exc)
            continue
        key = ((round(fs, 3), str(card.parent)) if mode == "rate"
               else (round(fs, 3), f"{card.parent}\x00{tag}"))
        groups.setdefault(key, []).append(card)

    # Two directories at the same rate would collide on the rate label, so
    # the label is only unique-ified when it actually has to be.
    by_rate: dict[float, int] = {}
    for fs, _ in groups:
        by_rate[fs] = by_rate.get(fs, 0) + 1

    out: list[Group] = []
    seen: dict[float, int] = {}
    for (fs, _key), members in sorted(groups.items(),
                                      key=lambda kv: (-kv[0][0], kv[0][1])):
        members = sorted(members)
        if mode == "card":
            tag = card_tag(members[0])
        else:
            tag = _rate_tag(fs)
            if by_rate[fs] > 1:
                seen[fs] = seen.get(fs, 0) + 1
                tag = f"{tag}_{seen[fs]}"
        out.append(Group(tag=tag, fs=fs, cards=tuple(members),
                         directory=members[0].parent))
    return out, failed


def run_group(group: Group, leepa: str, condition: str,
              out_dir: Path, env: dict, f_max: float | None,
              plate: str | None, extra: list[str]) -> tuple[bool, str]:
    """Run the pipeline over one group, in its own output folder.

    ``--cards`` restricts discovery to this group's members, so bronze aligns
    and schedules across exactly these cards and no others; because they
    share a rate, the mixed-rate refusal never fires.
    """
    argv = [sys.executable, "main.py",
            "--dat", str(group.directory), "--leepa", leepa,
            "--condition", condition,
            "--cards", ",".join(group.card_tags),
            "--label", f"{condition}_{group.tag}",
            "--out", str(out_dir / group.tag),
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

        # Which physical card each segment was wired to. In a per-rate group
        # the run covers several cards, so the run tag ("100kHz") does not
        # answer that question and silver's own table has to. Falling back to
        # the run tag keeps the column populated for a single-card run, where
        # the two are the same thing anyway.
        wiring = {r["segment"]: r.get("card", "")
                  for r in _read(out_dir / tag / "silver"
                                 / "segments_summary.csv")}

        for row in rows:
            seg = row.get("segment", "")
            row["run"] = tag
            row["card"] = wiring.get(seg) or tag
            spectra.append(row)
        for row in plate:
            seg = row.get("segment", "")
            if row.get("class") != "measured":
                continue
            if seg in seen:
                dupes.append(seg)
                continue
            seen.add(seg)
            row["run"] = tag
            row["card"] = wiring.get(seg) or tag
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

    cell = _cell_aggregate(spectra, summary)
    if cell:
        with (target / "silver" / "cell_aggregate.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2",
                                "n_segments"])
            writer.writeheader()
            writer.writerows(cell)

    return {"n_segments": len(summary), "n_points": len(spectra),
            "duplicates": sorted(set(dupes)), "per_card": per_card,
            "n_cell_points": len(cell)}


def _cell_aggregate(spectra: list[dict], summary: list[dict]) -> list[dict]:
    """The cell spectrum the merged plate implies: what an integral
    instrument -- the Gamry -- would have seen.

        Z_cell = A_used / sum_s (A_s / Z_s)

    Segments sit in PARALLEL across one cell voltage, so admittances add and
    the cell curve is the AREA-WEIGHTED HARMONIC MEAN of the local ones, not
    the arithmetic mean.  Being harmonic it is dominated by the LOW-impedance
    segments, which is exactly why an integral measurement hides local
    faults: a flooded segment has high Z, contributes little admittance and
    barely moves the cell curve.  That is the whole reason for measuring
    locally, and it is worth seeing next to the reference sweep.

    Why this is recomputed here rather than taken from a group.  Each group
    ran on its own cards and wrote a cell aggregate over ITS segments only;
    on a five-card plate split by rate, neither covers the cell.  Summing the
    admittances of every merged segment does.

    Two things this refuses to fake.  The groups detect their own steps, so
    their frequency grids need not coincide: the grid used here is the one
    actually shared, and a frequency only one group reached is dropped rather
    than interpolated across a gap.  And ``n_segments`` is carried per point,
    because a point backed by 12 segments and one backed by 60 are not the
    same measurement and the difference must not be invisible.
    """
    import math

    areas = {r["segment"]: r.get("area_cm2", "") for r in summary}
    by_freq: dict[float, list[tuple[float, complex]]] = {}
    seen_by_seg: dict[str, set[float]] = {}
    for row in spectra:
        seg = row.get("segment", "")
        try:
            f = float(row["freq_hz"])
            z = complex(float(row["z_re_mohm_cm2"]),
                        float(row["z_im_mohm_cm2"]))
            a = float(areas.get(seg) or 0.0)
        except (TypeError, ValueError, KeyError):
            continue
        if a <= 0 or z == 0 or not math.isfinite(abs(z)):
            continue
        by_freq.setdefault(round(f, 6), []).append((a, z))
        seen_by_seg.setdefault(seg, set()).add(round(f, 6))

    if not by_freq:
        return []

    # Only frequencies that most of the plate actually reached. A point held
    # up by one segment is that segment's spectrum wearing the cell's name.
    n_seg = max(len(v) for v in by_freq.values())
    floor = max(2, int(0.5 * n_seg))

    out = []
    for f in sorted(by_freq):
        members = by_freq[f]
        if len(members) < floor:
            continue
        Y = sum(a / z for a, z in members)
        if Y == 0:
            continue
        Z = sum(a for a, _ in members) / Y
        out.append({"freq_hz": round(f, 6),
                    "z_re_mohm_cm2": round(Z.real, 5),
                    "z_im_mohm_cm2": round(Z.imag, 5),
                    "n_segments": len(members)})
    return out


# ---------------------------------------------------------------------------
# the plot
# ---------------------------------------------------------------------------

def plot(target: Path, title: str, f_min: float | None = None,
         f_max: float | None = None) -> Path | None:
    """One Nyquist and one Bode over every segment of every card.

    `f_min`/`f_max` restrict what is DRAWN, not what was computed. The CSVs
    keep every point either way, so narrowing the view is reversible and
    costs nothing -- as against narrowing the analysis, which means running
    the cards again.

    A band also makes the Nyquist honest about what it is: with a ceiling of
    a few kHz the arc is a partial arc, and drawing it against an axis that
    stops there says so, where an autoscaled full-band plot invites the eye
    to complete a semicircle that was never measured.
    """
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
    reach: dict[str, float] = {}          # highest frequency each card has
    dropped = 0
    for row in rows:
        try:
            f = float(row["freq_hz"])
            re_ = float(row["z_re_mohm_cm2"])
            im = float(row["z_im_mohm_cm2"])
        except (KeyError, ValueError):
            continue
        seg = row["segment"]
        card = row.get("card", "")
        reach[card] = max(reach.get(card, 0.0), f)
        if (f_min is not None and f < f_min) or (f_max is not None and f > f_max):
            dropped += 1
            continue
        by_seg.setdefault(seg, []).append((f, re_, im))
        card_of[seg] = card

    if not by_seg:
        band = f"{f_min or 0:.0f}-{f_max or 0:.0f} Hz"
        print(f"  no point falls inside {band}. The cards reach: "
              + ", ".join(f"{c} to {r:.0f} Hz" for c, r in sorted(reach.items())))
        return None
    if dropped:
        print(f"  plotting {sum(len(v) for v in by_seg.values())} of "
              f"{sum(len(v) for v in by_seg.values()) + dropped} points "
              f"({dropped} outside the band; the CSVs keep all of them)")
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

    # A CARD WITH NOTHING IN THE BAND IS A RESULT, NOT AN OMISSION. It means
    # that card's analysis stopped below the window -- on this plate because
    # an interferer folds into its band and its ceiling was set beneath the
    # fold -- and a legend that simply lacks it looks like a card that was
    # not run.
    absent = [c for c in sorted(reach) if c and c not in cards]
    band_text = ""
    if f_min is not None and f_max is not None:
        band_text = f"  |  {f_min:.0f}-{f_max:.0f} Hz"
    elif f_max is not None:
        band_text = f"  |  up to {f_max:.0f} Hz"
    elif f_min is not None:
        band_text = f"  |  from {f_min:.0f} Hz"
    if absent:
        note = "; ".join(f"{c} reaches only {reach[c]:.0f} Hz" for c in absent)
        fig.text(0.5, 0.005, f"not in this band: {note}", ha="center",
                 fontsize=9, color="#b03030")
        print(f"  NOT in the band: {note}")
    fig.suptitle(f"{title} — {len(by_seg)} segments, evaluated card by card"
                 + band_text)
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

    groups, failed = group_cards(cards, a.group)
    if not groups:
        print(f"  {condition}: no card header could be read; nothing to run")
        for tag, why in failed:
            print(f"    {tag}: {why}")
        return 1

    if a.group == "rate":
        print(f"  {len(cards)} card(s) in {len(groups)} rate group(s); each "
              f"group is aligned and scheduled within itself, and nothing "
              f"measured in samples crosses between groups.")
    else:
        print(f"  {len(cards)} card(s), each evaluated alone on its own "
              f"clock and at its own sample rate.")
    for g in groups:
        print(f"    {g.tag:<10} {g.fs / 1000:>5.0f} kHz  "
              f"{', '.join(g.card_tags)}")
    for tag, why in failed:
        print(f"    {tag:<10} SKIPPED: {why}")

    ok = []
    for index, group in enumerate(groups, 1):
        print(f"  [{index}/{len(groups)}] {group.tag}: ", end="", flush=True)
        if a.dry_run:
            print("(dry run)")
            ok.append(group.tag)
            continue
        ceiling, why_ceiling = ceiling_for(group.fs, a.interferer_hz, a.f_max)
        print(f"{why_ceiling} ...", end=" ", flush=True)
        good, why = run_group(group, leepa, condition, scratch,
                              env, ceiling, a.plate, a.extra)
        if good:
            ok.append(group.tag)
            print("done")
        else:
            failed.append((group.tag, why))
            print("FAILED")
            for line in why.splitlines()[-4:]:
                print(f"      {line}")

    if a.dry_run:
        return 0
    if not ok:
        print("\n  nothing produced results; nothing to merge")
        if a.group == "rate" and len(groups) < len(cards):
            print("  a whole rate group failed. Retry with --group card to "
                  "evaluate each card alone: it needs no card to agree with "
                  "any other, at the cost of the cross-card vote.")
        return 1

    stats = merge(scratch, ok, target)
    print(f"\n  merged: {stats['n_segments']} segments, "
          f"{stats['n_points']} spectrum points")
    if stats.get("n_cell_points"):
        print(f"  cell   : {stats['n_cell_points']} points "
              f"(area-weighted harmonic mean over the merged segments; "
              f"this is what the Gamry sweep is compared against)")
    for tag, info in stats["per_card"].items():
        print(f"    {tag:<10} {info['n_measured']:>3} segments, "
              f"{info['n_points']:>5} points")
    if failed:
        print(f"  {len(failed)} group(s)/card(s) produced nothing; the "
              f"segments above are what the rest measured.")
    if stats["duplicates"]:
        print(f"    NOTE: {len(stats['duplicates'])} segment(s) appeared on "
              f"more than one card; the first card kept them: "
              f"{', '.join(stats['duplicates'][:8])}")

    image = plot(target, f"{leepa} / {label}", a.plot_f_min, a.plot_f_max)
    if image:
        print(f"  plot   : {image}")
    (target / "per_card_manifest.json").write_text(
        json.dumps({"condition": condition, "label": label,
                    "group_by": a.group,
                    "groups": [{"tag": g.tag, "fs_hz": g.fs,
                                "cards": g.card_tags} for g in groups],
                    "cards_ok": ok, "cards_failed": [t for t, _ in failed],
                    **stats}, indent=2), encoding="utf-8")
    print(f"  results: {target}")
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
    p.add_argument("--plot-f-min", type=float, default=None,
                   help="lowest frequency to DRAW. The CSVs keep every point "
                        "regardless, so this is a view and not a re-run")
    p.add_argument("--plot-f-max", type=float, default=None,
                   help="highest frequency to DRAW, e.g. 5000 for a Nyquist "
                        "that stops at 5 kHz")
    p.add_argument("--interferer-hz", type=float, default=None,
                   help="a known interference frequency, e.g. 45996. Each "
                        "card's ceiling is then set below where that "
                        "frequency FOLDS at that card's own rate -- which is "
                        "a different place on a 50 kHz card than on a "
                        "100 kHz one, and is why one ceiling cannot serve "
                        "both")
    p.add_argument("--group", choices=["rate", "card"], default="rate",
                   help="how to partition the cards. 'rate' (default) "
                        "evaluates every card of one sample rate together, "
                        "so they still align to each other and vote on each "
                        "other's steps, while nothing measured in samples "
                        "crosses between rates. 'card' evaluates each card "
                        "alone: it gives up the cross-card vote and needs no "
                        "card to agree with any other")
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
