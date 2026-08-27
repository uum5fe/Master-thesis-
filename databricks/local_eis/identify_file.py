#!/usr/bin/env python3
"""
identify_file.py  --  what IS this measurement file?
====================================================

    python identify_file.py "<path to the file>"

An extension is a claim, not evidence.  ``.DAT`` is used by imc FAMOS, by
half a dozen unrelated loggers, and by anything that felt like it; ``.ddf``
is used by at least three products that share nothing.  Before writing a
reader for a format, it is worth spending one second establishing which
format it actually is, because the first bytes of a file say so and the name
does not.

This prints the size, the first bytes as hex and text, and the closest match
among the signatures this pipeline knows or could plausibly be asked to
support.  Paste its output when asking for a new reader: it is the difference
between "it is a .ddf" and "it is a zip container whose first entry is
data.xml", which are answerable and unanswerable questions respectively.

Nothing here parses a measurement.  It reads the head of the file and
refuses to guess beyond what it can see.
"""

from __future__ import annotations

import argparse
from pathlib import Path


#: (signature, offset, name, how to read it).  Ordered most specific first;
#: several formats are zip or XML underneath and would otherwise match the
#: generic entry.
SIGNATURES: list[tuple[bytes, int, str, str]] = [
    (b"|CF,", 0, "imc FAMOS raw",
     "this pipeline's own FAMOS reader (eis_local.FamosFile) -- already "
     "supported, run with --dat"),
    (b"IMCVIEW", 0, "imc FAMOS (viewer variant)",
     "imc's own tools, or FamosFile after checking the key layout"),
    (b"\x89HDF\r\n\x1a\n", 0, "HDF5",
     "h5py. Many loggers wrap their own schema in HDF5, so the group names "
     "matter more than the container"),
    (b"MDF     ", 0, "ASAM MDF v3", "asammdf"),
    (b"UnFinMF", 0, "ASAM MDF v4, unfinalised",
     "asammdf (it finalises on open)"),
    (b"TDSm", 0, "NI TDMS", "npTDMS"),
    (b"SQLite format 3\x00", 0, "SQLite database",
     "sqlite3 -- dump the table list first; the schema is the real format"),
    (b"PK\x03\x04", 0, "zip container",
     "zipfile -- the entry names below decide what it really is"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", 0, "OLE2 compound document",
     "olefile, or pandas if it turns out to be a legacy .xls"),
    (b"<?xml", 0, "XML", "an XML parser -- read the root element name"),
    (b"DEWE", 0, "Dewesoft (DEWE-prefixed header)",
     "Dewesoft's DWDataReaderLib, or their export to a documented format"),
]

#: Extensions that are claimed by more than one product. Named so the report
#: can say WHY the bytes matter rather than just showing them.
AMBIGUOUS = {
    ".ddf": ("at least three unrelated products write .ddf: DEWETRON's older "
             "DEWE-x data files, a plain-text key/value format of the same "
             "name, and several in-house bench formats. The bytes below "
             "settle which"),
    ".dat": ("imc FAMOS, and a great many loggers that simply meant 'data'"),
    ".dsd": ("Dewesoft's own sequential data format"),
    ".dmd": ("DEWETRON Oxygen -- read with pyDmdReader"),
    ".dxd": ("Dewesoft X -- read with DWDataReaderLib"),
    ".d7d": ("Dewesoft 7 -- read with DWDataReaderLib"),
}


def looks_textual(head: bytes) -> bool:
    """True when the head is plausibly text in some 8-bit encoding."""
    if not head:
        return False
    printable = sum(1 for b in head if 32 <= b < 127 or b in (9, 10, 13))
    return printable / len(head) > 0.85


def identify(path: Path, n: int = 512) -> list[str]:
    out: list[str] = []
    if not path.exists():
        return [f"{path}: does not exist"]
    if path.is_dir():
        kids = sorted(p.name for p in path.iterdir())[:20]
        return [f"{path} is a DIRECTORY holding {len(kids)} entr(ies):",
                *(f"    {k}" for k in kids),
                "", "Point this at one of the files inside it."]

    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(n)

    out.append(f"file      : {path.name}")
    out.append(f"size      : {size:,} bytes")
    if path.suffix.lower() in AMBIGUOUS:
        out.append(f"extension : {path.suffix} -- "
                   f"{AMBIGUOUS[path.suffix.lower()]}")

    hits = [(name, how) for sig, off, name, how in SIGNATURES
            if head[off:off + len(sig)] == sig]
    if hits:
        name, how = hits[0]
        out.append(f"format    : {name}")
        out.append(f"read with : {how}")
        if len(hits) > 1:
            out.append(f"            (also matched: "
                       f"{', '.join(n for n, _ in hits[1:])})")
    elif looks_textual(head):
        text = head.decode("latin-1", errors="replace")
        first = text.splitlines()[0] if text.splitlines() else ""
        sep = max(("\t", ";", ",", "|"), key=first.count)
        out.append("format    : TEXT of some kind, not a known binary "
                   "container")
        out.append(f"            first line has {first.count(sep)} "
                   f"{sep!r} separator(s)")
        out.append("read with : csv, once the delimiter and the header rows "
                   "are known -- see csv_source.detect_dialect")
    else:
        out.append("format    : BINARY, no known signature. The hex below is "
                   "what a reader would have to be written against.")

    # A container tells you almost nothing; its contents tell you everything.
    # Listing them here saves the round trip that "list them first" would cost.
    if head[:4] == b"PK\x03\x04":
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
            out.append("")
            out.append(f"zip contains {len(names)} entr(ies):")
            for name in names[:25]:
                out.append(f"    {name}")
            if len(names) > 25:
                out.append(f"    ... and {len(names) - 25} more")
        except (zipfile.BadZipFile, OSError) as exc:
            out.append(f"            (zip listing failed: {exc})")

    out.append("")
    out.append("first bytes:")
    for i in range(0, min(len(head), 128), 16):
        chunk = head[i:i + 16]
        hexed = " ".join(f"{b:02x}" for b in chunk).ljust(47)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"  {i:04x}  {hexed}  |{text}|")

    if looks_textual(head):
        out.append("")
        out.append("first lines, as text:")
        for line in head.decode("latin-1", errors="replace").splitlines()[:8]:
            out.append(f"  {line[:150]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("-n", type=int, default=512,
                    help="how many bytes to read from the head")
    args = ap.parse_args(argv)

    for i, path in enumerate(args.paths):
        if i:
            print()
        print("\n".join(identify(path, args.n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
