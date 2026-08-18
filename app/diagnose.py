r"""Explain why the picker is empty.

"No runs discovered" has a handful of causes and they are indistinguishable
from the dashboard: a ``.env`` that was never created, one Notepad saved as
``.env.txt``, a root pointing at the folder *above* the one with the data, a
results folder that is not laid out ``<order>/<condition>/``, or recordings
whose filenames do not match any known convention.

Each of those has a different fix, so this module looks at what is actually on
disk and names the one that applies, rather than repeating the message that
sent the reader here.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.data.loaders import detect_layout
from app.data.sources import Catalog, famos_patterns
from app.settings import DOTENV_LOADED, SETTINGS, Settings

TICK, CROSS, WARN, INFO = "[ok]", "[--]", "[!!]", "  ->"


def _env_section(out: list[str]) -> None:
    out.append("1. SETTINGS FILE")
    if DOTENV_LOADED:
        out.append(f"   {TICK} read {DOTENV_LOADED}")
    else:
        root = Path(__file__).resolve().parent.parent
        out.append(f"   {CROSS} no .env found. Looked in:")
        looked = dict.fromkeys([root / ".env", Path.cwd() / ".env"])
        for candidate in looked:
            out.append(f"        {candidate}")
        # The classic Windows trap: Notepad appends .txt and Explorer hides it.
        strays = [p for d in (root, Path.cwd()) if d.is_dir()
                  for p in d.glob(".env.*") if p.suffix != ".example"]
        if strays:
            out.append(f"   {WARN} but these exist: "
                       + ", ".join(p.name for p in strays))
            out.append(f"{INFO} rename to exactly '.env' (no .txt). In Explorer, "
                       "turn on View > File name extensions first.")
        elif (root / ".env.example").is_file():
            out.append(f"{INFO} copy .env.example to .env and put your paths in it")
        out.append(f"{INFO} not fatal if you passed --results / --famos or set "
                   "the variables in your shell")

    for name in ("EIS_FAMOS_ROOT", "EIS_RESULTS_ROOT", "EIS_CURR_CAL",
                 "EIS_ALLOW_INLINE_PIPELINE", "EIS_PLATE_SPEC_DIR"):
        value = os.environ.get(name)
        if value:
            out.append(f"   {TICK} {name} = {value}")
    out.append("")


def _famos_section(out: list[str], settings: Settings) -> None:
    out.append("2. RAW FAMOS RECORDINGS  (EIS_FAMOS_ROOT)")
    if not settings.famos_roots:
        out.append(f"   {CROSS} not set — skipping. Set EIS_FAMOS_ROOT to view "
                   ".DAT recordings.")
        out.append("")
        return

    patterns = famos_patterns()
    for raw in settings.famos_roots:
        root = Path(raw)
        out.append(f"   folder: {root}")
        if not root.exists():
            out.append(f"   {CROSS} does not exist")
            out.append(f"{INFO} check for a typo, and that OneDrive has the "
                       "folder downloaded rather than online-only")
            continue
        if not root.is_dir():
            out.append(f"   {CROSS} that is a file, not a folder")
            continue

        files = sorted(list(root.rglob("*.DAT")) + list(root.rglob("*.dat")))
        if not files:
            out.append(f"   {CROSS} no .DAT files anywhere under it")
            subdirs = [p.name for p in root.iterdir() if p.is_dir()][:8]
            if subdirs:
                out.append(f"{INFO} subfolders here: {', '.join(subdirs)}")
                out.append(f"{INFO} if the recordings are in one of those, "
                           "either point at it or leave this root — subfolders "
                           "are searched, so the files would have been found")
            continue

        out.append(f"   {TICK} {len(files)} .DAT file(s) found")
        matched, unmatched = [], []
        for path in files:
            for pattern in patterns:
                m = pattern.search(path.name)
                if m:
                    matched.append((path.name, m.group("measurement_id"),
                                    m.group("condition")))
                    break
            else:
                unmatched.append(path.name)

        if matched:
            groups = sorted({(mid, cond) for _n, mid, cond in matched})
            out.append(f"   {TICK} {len(matched)} recognised, giving "
                       f"{len(groups)} selectable run(s):")
            for mid, cond in groups[:10]:
                out.append(f"        order {mid}  condition {cond}")
        if unmatched:
            out.append(f"   {WARN} {len(unmatched)} filename(s) not recognised, "
                       f"e.g. {unmatched[0]}")
            out.append(f"{INFO} recognised conventions are:")
            out.append("        Leepa_<order>_Current_<condition>_Test_<nn>_Karte_<n>.DAT")
            out.append("        RO<order>-01_Current_<condition>_Test_<nn>_Karte_<n>.DAT")
            out.append(f"{INFO} for anything else set EIS_FAMOS_REGEX with named "
                       "groups measurement_id, condition, test, card")
    out.append("")


def _results_section(out: list[str], settings: Settings) -> None:
    out.append("3. PROCESSED RESULTS  (EIS_RESULTS_ROOT)")
    roots = [r for r in settings.resolved_results_roots()]
    if not settings.results_roots:
        out.append(f"   {CROSS} not set — skipping. Set EIS_RESULTS_ROOT to view "
                   "finished pipeline output.")
        out.append("")
        return

    for root in roots:
        if str(root) not in settings.results_roots:
            continue                       # the scratch root; not user-configured
        out.append(f"   folder: {root}")
        if not root.is_dir():
            out.append(f"   {CROSS} does not exist")
            continue

        layout = detect_layout(root)
        if layout:
            out.append(f"   {WARN} this folder IS itself one run ({layout}).")
            out.append(f"{INFO} that works, but it shows up as a single entry. "
                       "The normal layout is "
                       "<root>\\<order id>\\<condition>\\gold\\plate_summary.csv")
            continue

        found = 0
        children = sorted(p for p in root.iterdir() if p.is_dir())
        if not children:
            out.append(f"   {CROSS} empty, or contains only files")
        for child in children:
            sub = detect_layout(child)
            if sub == "eis_tables":
                out.append(f"   {TICK} {child.name}: {sub}")
                found += 1
                continue
            if sub:
                out.append(f"   {TICK} {child.name}: {sub} (no condition level)")
                found += 1
                continue
            conditions = [c for c in sorted(p for p in child.iterdir() if p.is_dir())
                          if detect_layout(c)]
            if conditions:
                out.append(f"   {TICK} {child.name}: "
                           f"{', '.join(c.name for c in conditions)}")
                found += len(conditions)
            else:
                out.append(f"   {CROSS} {child.name}: nothing recognisable inside")

        if not found:
            out.append(f"{INFO} expected, under each order id and condition, a "
                       "'gold' folder with plate_summary.csv or a 'silver' "
                       "folder with spectra_clean.csv")
            out.append(f"{INFO} to use an existing results folder, copy it so it "
                       "sits at <root>\\<order id>\\<condition>\\  — e.g. "
                       "<root>\\2611976\\45A\\gold\\plate_summary.csv")
    out.append("")


def _summary(out: list[str], settings: Settings) -> None:
    out.append("4. WHAT THE APP WILL SHOW")
    catalog = Catalog(settings).refresh()
    if not catalog.runs:
        out.append(f"   {CROSS} nothing — every dropdown will be empty")
        out.append(f"{INFO} fix whichever section above is marked {CROSS} or "
                   f"{WARN}, then start the app again")
    else:
        by_kind: dict[str, int] = {}
        for ref in catalog.runs:
            by_kind[ref.kind] = by_kind.get(ref.kind, 0) + 1
        out.append(f"   {TICK} {len(catalog.runs)} run(s): "
                   + ", ".join(f"{n} {k}" for k, n in sorted(by_kind.items())))
        for ref in catalog.runs[:12]:
            out.append(f"        [{ref.kind}] {ref.measurement_id} / "
                       f"{ref.condition}   {ref.describe()}")
    for message in catalog.messages:
        out.append(f"   note: {message}")
    out.append("")


def report(settings: Settings = SETTINGS) -> str:
    out = ["", "=" * 68, "  Local EIS Viewer — configuration check", "=" * 68, ""]
    _env_section(out)
    _famos_section(out, settings)
    _results_section(out, settings)
    _summary(out, settings)
    out.append("=" * 68)
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
