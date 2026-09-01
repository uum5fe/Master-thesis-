#!/usr/bin/env python3
r"""
famos_keys.py  --  what is actually in this FAMOS file?
=======================================================

    python famos_keys.py "<card>.DAT"
    python famos_keys.py "<card>.DAT" --json keys.json
    python famos_keys.py "<card>.DAT" --max-keys 400

Walks the imc FAMOS key structure and prints it.  Nothing is guessed and
nothing is pattern-matched: every FAMOS key is

    |<KK>,<KeyVersion>,<Length>,<Length bytes of content>;

so the file says where each key ends.  That makes the whole structure of a
4 GB card readable in milliseconds -- a data block declares its size, so it
is SEEKED PAST rather than read.

WHY THIS EXISTS
---------------
Searching a header with regular expressions is what the pipeline's reader
does, and it works only for the dialect it was written against.  Against a
standard FAMOS header it silently matches the wrong fields: every key's
third field is a BYTE COUNT, so `\|CD,\d+,([\d.eE+-]+)` captures the length
of the |CD block and reports 1/21 Hz as the sample rate.  It does not fail;
it returns a plausible-looking number that is not the sample rate.

Reading the structure instead of searching it removes that whole class of
error.  Paste this output when a card will not read: it is the difference
between "the header is laid out differently" and a specific list of the keys
present, their sizes, and where the data blocks start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: What each key means, for the ones that carry the layout.  Anything not
#: listed is printed with its raw content; the point is to show what is
#: there, not to filter it down to what this file's author expected.
KEY_MEANING = {
    "CF": "file format and version",
    "CK": "key group start",
    "NO": "origin: who wrote the file",
    "CB": "group definition",
    "CG": "data field: how many components follow",
    "CD": "x-axis delta (sample interval) and its unit",
    "CZ": "z-axis",
    "CT": "free text",
    "CI": "text with index",
    "CC": "component start",
    "CP": "packing: byte offset, width, dtype, mask",
    "Cb": "buffer description: offset and length of this component's samples",
    "CR": "calibration: factor, offset, unit",
    "CN": "channel name",
    "CS": "SAMPLES -- the payload of one component",
    "Cv": "event/trigger",
    "CV": "event list",
    "ND": "display settings",
}

#: Content longer than this is previewed, not printed.
_PREVIEW = 160


class FamosStructureError(RuntimeError):
    """The bytes do not form a FAMOS key at the position given."""


def _read_field(fh, stop: bytes, limit: int = 64) -> bytes:
    """Read up to and excluding `stop`; refuse to run away."""
    out = bytearray()
    while len(out) < limit:
        ch = fh.read(1)
        if not ch:
            raise FamosStructureError("end of file inside a key field")
        if ch == stop:
            return bytes(out)
        out += ch
    raise FamosStructureError(
        f"no {stop!r} within {limit} bytes: {bytes(out)[:32]!r}")


def walk(path, max_keys: int = 2000, preview: int = _PREVIEW):
    """Yield one dict per key, seeking past the content of large ones."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as fh:
        for index in range(max_keys):
            at = fh.tell()
            if at >= size:
                return
            lead = fh.read(1)
            if lead in (b"\r", b"\n", b" ", b"\t"):
                continue                       # separators between keys
            if lead != b"|":
                raise FamosStructureError(
                    f"expected '|' at byte {at}, found {lead!r}")
            key = "??"
            try:
                key = _read_field(fh, b",", 8).decode("latin-1")
                version = int(_read_field(fh, b",", 16))
                raw_length = _read_field(fh, b",", 24)
                length = int(raw_length)
            except FamosStructureError as exc:
                raise FamosStructureError(
                    f"key |{key} at byte {at}: {exc}") from None
            except ValueError:
                # The third field is a BYTE COUNT in the standard layout. A
                # dialect that writes a value there instead -- |CD,2,0.0001,1;
                # puts the sample interval where the length belongs -- lands
                # here, and which key it was is the thing that identifies the
                # dialect.
                raise FamosStructureError(
                    f"key |{key} at byte {at} has {raw_length.decode('latin-1')!r} "
                    f"where the standard layout puts a byte count. This file "
                    f"writes values in the length field, so its keys are not "
                    f"self-delimiting and cannot be walked.") from None

            content_at = fh.tell()
            head = fh.read(min(length, preview))
            if length > preview:
                fh.seek(content_at + length)   # a 4 GB block costs nothing
            terminator = fh.read(1)

            yield {
                "index": index,
                "key": key,
                "version": version,
                "length": length,
                "offset": at,
                "content_offset": content_at,
                "meaning": KEY_MEANING.get(key, ""),
                "content": head.decode("latin-1", "replace"),
                "truncated": length > preview,
                "terminated": terminator == b";",
            }
            if terminator != b";":
                raise FamosStructureError(
                    f"key |{key} at byte {at} declares {length} bytes but byte "
                    f"{content_at + length} is {terminator!r}, not ';'. Either "
                    f"the length is not a byte count in this dialect, or the "
                    f"key structure differs.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--max-keys", type=int, default=2000)
    p.add_argument("--json", type=Path, help="write the full key list here")
    p.add_argument("--all", action="store_true",
                   help="print every key; by default the per-component keys "
                        "are summarised after the first few components")
    a = p.parse_args(argv)

    if not a.path.is_file():
        print(f"error: {a.path} is not a file", file=sys.stderr)
        return 2

    size = a.path.stat().st_size
    print(f"\n{a.path.name}   {size / 1e9:.2f} GB\n")

    keys, problem = [], ""
    try:
        for entry in walk(a.path, a.max_keys):
            keys.append(entry)
    except FamosStructureError as exc:
        problem = str(exc)

    if not keys:
        print("  No FAMOS key structure at all.")
        if problem:
            print(f"  {problem}")
        print(f'\n  Try:  python identify_file.py "{a.path}"')
        return 1

    counts: dict[str, int] = {}
    for entry in keys:
        counts[entry["key"]] = counts.get(entry["key"], 0) + 1

    shown = 0
    print(f"  {'#':>4}  {'key':<4} {'ver':>3} {'length':>12} "
          f"{'offset':>14}  content")
    print(f"  {'-' * 4}  {'-' * 4} {'-' * 3} {'-' * 12} {'-' * 14}  {'-' * 40}")
    for entry in keys:
        # After the first three components the pattern repeats; printing 27
        # identical blocks buries the two or three keys that differ.
        if not a.all and entry["key"] in ("CC", "CP", "Cb", "CR", "CN", "CS") \
                and counts[entry["key"]] > 3 and shown > 24:
            continue
        shown += 1
        text = entry["content"].replace("\r", "").replace("\n", " ")
        if entry["truncated"]:
            text += " ..."
        print(f"  {entry['index']:>4}  |{entry['key']:<3} {entry['version']:>3} "
              f"{entry['length']:>12} {entry['offset']:>14}  {text[:88]}")
    if shown < len(keys):
        print(f"  ... {len(keys) - shown} further keys not shown "
              f"(--all prints them)")

    print("\n  key counts:")
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    |{key:<3} x{n:<5} {KEY_MEANING.get(key, '')}")

    data = [e for e in keys if e["key"] == "CS"]
    if data:
        total = sum(e["length"] for e in data)
        print(f"\n  payload: {len(data)} |CS block(s), {total / 1e9:.2f} GB "
              f"total, first at byte {data[0]['content_offset']}")
        if len(data) > 1:
            print("    Several |CS blocks means each component carries its "
                  "own samples CONTIGUOUSLY -- the channels are not "
                  "interleaved in one block.")
        else:
            print("    One |CS block: the components share it, so the samples "
                  "are interleaved.")
    else:
        print("\n  No |CS block found -- no payload key in the keys walked.")

    if problem:
        print(f"\n  stopped: {problem}")

    if a.json:
        a.json.write_text(json.dumps(keys, indent=2), encoding="utf-8")
        print(f"\n  written: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
