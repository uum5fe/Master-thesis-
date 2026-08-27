#!/usr/bin/env python3
"""
run_rename.py  --  edit the list below, then run this file
===========================================================

For when you would rather edit a file and press Run than type a command.
Everything you change is in the EDIT HERE block; nothing below it needs
touching.  Run it with

    python run_rename.py

and it renames every file in the list, one after another, reporting what it
did to each.  A file that fails is reported and the rest still run.

The names are given in HEADER ORDER -- the order `list` prints them, which is
the order the channels sit in the file, not sorted by name.  Check with

    python famos_rename.py list "your file.DAT"

before trusting a list you have not seen applied.
"""

from __future__ import annotations

from pathlib import Path

import famos_rename as F


# ===========================================================================
# EDIT HERE
# ===========================================================================

#: One entry per file: the path, then the names for its channels in header
#: order.  Windows paths go in r"..." so the backslashes are left alone.
#:
#: The names may be written three ways, and you can mix them between files:
#:
#:   "64-79"                 shorthand for the sixteen numbers 64..79
#:   "UC1,65-79"             a literal name, then a range
#:   ["UC1", "65", "temp1"]  a plain list, when the names have no pattern
#:   "names.csv"             a template you filled in with `template`
#:
#: Any name works -- a channel that is a reference tap rather than a segment
#: can be called UC1.  There must be exactly one name per channel.

FILES = {

    r"/root/.claude/uploads/2026_08_27_KANAL_6479RO2612025_60A.DDF_1.DAT":
        "64-79",

    # Add the next file below, in the same shape:
    #
    # r"C:\EIS_data\2026_08_27_KANAL_8095RO2612025_60A.DDF_1.DAT":
    #     "80-95",
    #
    # r"C:\EIS_data\2026_08_27_KANAL_4863RO2612025_60A.DDF_1.DAT":
    #     ["UC1", "49", "50", "51", "52", "53", "54", "55",
    #      "56", "57", "58", "59", "60", "61", "62", "63"],

}

#: Where the renamed copies go.  None puts each one beside its original.
OUT_DIR = None

#: Added before the extension: "run_1.DAT" -> "run_1_renamed.DAT".
SUFFIX = "_renamed"

#: True shows what would happen and writes nothing.  Worth doing once.
DRY_RUN = False

# ===========================================================================
# EDIT ABOVE -- nothing below needs changing
# ===========================================================================


HERE = "the FILES list in run_rename.py"


def resolve_names(spec) -> tuple[list[str], str]:
    """A list stays a list; a string is a name list, a range, or a file."""
    if isinstance(spec, (list, tuple)):
        return [str(s) for s in spec], HERE
    names, source = F.load_names(str(spec))
    # load_names says "the command line" for a spec it did not read off disk;
    # from here it came out of FILES.
    return names, source if Path(str(spec)).exists() else HERE


def out_path_for(src: Path) -> Path:
    out = F.default_out(src) if SUFFIX == "_renamed" else \
        src.with_name(src.stem + SUFFIX + src.suffix)
    return Path(OUT_DIR) / out.name if OUT_DIR else out


def run_one(raw_path: str, spec) -> bool:
    """Rename one file.  Returns whether it worked."""
    src = Path(raw_path)
    print(f"\n{'=' * 70}\n{src.name}\n{'=' * 70}")

    if not src.exists():
        print(f"  ERROR: no such file\n         {src}")
        return False

    try:
        head = F.read_header(src)
        names, source = resolve_names(spec)
        F.check_names(names, head)
    except F.FamosError as exc:
        print(f"  ERROR: {exc}")
        return False

    print(f"  {head.n_channels} channels, {head.numpy_dtype()}, "
          f"{head.n_frames:,} samples at {head.fs:,.1f} Hz "
          f"({head.duration_s:.3f} s)")
    print(f"  names from {source}")
    for ch, new in zip(head.channels, names):
        arrow = "->" if ch.name != new else "  "
        print(f"    ch {ch.index:>2}   {ch.name:<12} {arrow} {new}")

    if DRY_RUN:
        print("  dry run: nothing written")
        return True

    out = out_path_for(src)
    try:
        done = F.rename(src, out, names)
    except F.FamosError as exc:
        print(f"  ERROR: {exc}")
        return False

    same = F.data_digest(head) == F.data_digest(done)
    print(f"  written: {out}")
    print(f"  samples: {'identical to the source' if same else 'DIFFER'}")
    return same


def main() -> int:
    if not FILES:
        print("Nothing to do: the FILES list at the top is empty.")
        return 1

    print(f"{len(FILES)} file(s) to rename"
          f"{'  (DRY RUN)' if DRY_RUN else ''}")
    results = {path: run_one(path, spec) for path, spec in FILES.items()}

    ok = sum(results.values())
    print(f"\n{'=' * 70}")
    print(f"{ok} of {len(results)} file(s) done")
    for path, good in results.items():
        if not good:
            print(f"  failed: {Path(path).name}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
