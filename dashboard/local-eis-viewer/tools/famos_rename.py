#!/usr/bin/env python3
r"""
famos_rename.py  --  rename the channels in an imc FAMOS .DAT file
==================================================================

    python famos_rename.py list     "file.DAT"
    python famos_rename.py template "file.DAT" names.csv
    python famos_rename.py rename   "file.DAT" "UC2,UC1,1-14"

WHY THIS EXISTS
---------------
A DASYLab FAMOS export names its channels "0", "1", ... "15" -- the hardware
channel index, which says nothing about which plate segment each one is
wired to.  The pipeline files a channel by NAME (`UC1`/`UC2` for the cell
voltage taps, a bare number for a segment), so until the names are right the
recording cannot be evaluated even though every sample in it is good.

WHAT THE FILE ACTUALLY LOOKS LIKE
---------------------------------
imc FAMOS is a stream of keys::

    |<KK>,<key version>,<body length>,<body>;

and the length is a byte count of the body.  Three things about the DASYLab
variant break a parser written against the imc-written files, and all three
are in this one header::

    |CN,1,12,1,0,0,1,0,0,;
    |Cb,1,40,1,0,1,1,0,         0,0,         0,1,0,0,;
    |CS,1,                   0,1,<raw samples>

1.  NUMBERS ARE SPACE PADDED.  `|CS,1,` is followed by nineteen spaces and a
    zero.  A parser that reads digits immediately after the comma finds none
    and either raises or, worse, resynchronises somewhere meaningless.  Every
    integer here is therefore read as `int(field.strip())`.

2.  THE |CS LENGTH IS ZERO.  DASYLab streams the data and never goes back to
    fill the byte count in, so the declared length is a placeholder.  The
    samples run to the end of the file, and that -- not the declared number
    -- is what the sample count comes from.

3.  THE NAME IS LENGTH-PREFIXED, NOT COMMA-DELIMITED.  A |CN body is
    ``<a>,<b>,<c>,<name length>,<name>,<comment length>,<comment>,`` and the
    name is exactly `<name length>` bytes, which may contain commas.  Walking
    to the next comma works by luck on `0`..`9` and breaks on `10`..`15`.

Renaming rewrites the |CN keys and their length fields and copies the |CS
block through byte for byte, so the samples are bit-identical.  `rename`
checks that afterwards rather than asserting it.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class FamosError(Exception):
    """Anything wrong with the file or with the names given for it."""


#: imc numeric format code -> numpy dtype. The DASYLab export writes 8
#: (double); the imc-written cards write 4 (float). Reading one as the other
#: is silent and catastrophic -- a float64 stream read as float32 gives
#: alternating zeros and nonsense, which still plots.
NUMFORMAT = {
    1: np.dtype("<u1"), 2: np.dtype("<i1"),
    3: np.dtype("<u2"), 4: np.dtype("<i2"),
    5: np.dtype("<u4"), 6: np.dtype("<i4"),
    7: np.dtype("<f4"), 8: np.dtype("<f8"),
}


@dataclass
class Channel:
    index: int                 # 1-based, header order = acquisition order
    name: str
    key_start: int             # byte offset of this channel's |CN key
    key_stop: int              # one past its terminating ';'
    prefix: str                # the three fields before the name
    comment: str


@dataclass
class Header:
    path: Path
    channels: list[Channel] = field(default_factory=list)
    fs: float = float("nan")
    numformat: int = 0
    bytes_per_sample: int = 0
    data_offset: int = 0
    raw: bytes = b""

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def numpy_dtype(self) -> np.dtype:
        if self.numformat not in NUMFORMAT:
            raise FamosError(f"unsupported imc numeric format {self.numformat}")
        return NUMFORMAT[self.numformat]

    @property
    def n_frames(self) -> int:
        width = self.numpy_dtype().itemsize * max(1, self.n_channels)
        return max(0, (self.path.stat().st_size - self.data_offset) // width)

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fs if self.fs else float("nan")

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.channels]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _int_at(body: bytes, pos: int) -> tuple[int, int]:
    """One comma-terminated integer, tolerating the space padding.

    Returns (value, position after the comma). Raises rather than returning a
    wrong number: a resynchronised parser produces a header that looks fine
    and describes a different file.
    """
    end = body.find(b",", pos)
    if end < 0:
        raise FamosError(f"expected a comma after offset {pos} in {body[:60]!r}")
    text = body[pos:end].strip()
    if not text or not text.lstrip(b"+-").isdigit():
        raise FamosError(f"expected an integer at offset {pos}, got {text!r}")
    return int(text), end + 1


def _str_at(body: bytes, pos: int) -> tuple[str, int]:
    """A length-prefixed string: ``<n>,<n bytes>,``.

    The length prefix is what makes a name containing a comma readable, and
    channel names here are "10".."15", so this is not hypothetical.
    """
    n, pos = _int_at(body, pos)
    if pos + n > len(body):
        raise FamosError(f"string of {n} bytes runs past the end of the key")
    text = body[pos:pos + n].decode("latin-1")
    pos += n
    if pos < len(body) and body[pos:pos + 1] == b",":
        pos += 1
    return text, pos


def iter_keys(raw: bytes):
    """Yield (name, version, body_start, body_len, key_start) for each key.

    Stops at |CS, whose body is the raw samples and is never text.
    """
    pos = 0
    while True:
        start = raw.find(b"|", pos)
        if start < 0:
            return
        comma = raw.find(b",", start)
        if comma < 0:
            return
        name = raw[start + 1:comma].decode("latin-1")
        try:
            version, p = _int_at(raw, comma + 1)
            length, p = _int_at(raw, p)
        except FamosError:
            return
        if name == "CS":
            # The body is binary and its declared length is a placeholder.
            # Everything from here is data; the caller copies it verbatim.
            yield name, version, p, -1, start
            return
        yield name, version, p, length, start
        pos = p + length


def read_header(path) -> Header:
    path = Path(path)
    raw = path.open("rb").read(1 << 20)      # the header is a few kB at most
    head = Header(path=path, raw=raw)

    if not raw.startswith(b"|CF,"):
        raise FamosError(f"{path.name}: not an imc FAMOS file "
                         f"(expected |CF, got {raw[:8]!r})")

    index = 0
    for name, _ver, bstart, blen, kstart in iter_keys(raw):
        if name == "CS":
            head.data_offset = _skip_cs_index(raw, bstart)
            break
        body = raw[bstart:bstart + blen]
        if name == "CD":
            dt, _ = _float_at(body, 0)
            head.fs = 1.0 / dt if dt else float("nan")
        elif name == "CP":
            # <buffer>,<bytes per sample>,<numformat>,<bits>,...
            _buf, p = _int_at(body, 0)
            head.bytes_per_sample, p = _int_at(body, p)
            head.numformat, p = _int_at(body, p)
        elif name == "CN":
            index += 1
            a, p = _int_at(body, 0)
            b, p = _int_at(body, p)
            c, p = _int_at(body, p)
            prefix = f"{a},{b},{c},"
            channel_name, p = _str_at(body, p)
            comment, p = _str_at(body, p) if p < len(body) else ("", p)
            head.channels.append(Channel(
                index=index, name=channel_name, key_start=kstart,
                key_stop=bstart + blen + 1, prefix=prefix, comment=comment))

    if not head.channels:
        raise FamosError(f"{path.name}: no |CN channel keys found")
    if not head.data_offset:
        raise FamosError(f"{path.name}: no |CS data block found")
    return head


def _float_at(body: bytes, pos: int) -> tuple[float, int]:
    end = body.find(b",", pos)
    if end < 0:
        raise FamosError("expected a comma after a number")
    return float(body[pos:end].strip()), end + 1


def _skip_cs_index(raw: bytes, pos: int) -> int:
    """Past the |CS index field, to the first sample byte."""
    comma = raw.find(b",", pos)
    if comma < 0:
        raise FamosError("malformed |CS key")
    return comma + 1


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

def expand(spec: str) -> list[str]:
    """`"UC1,65-79"` -> ["UC1", "65", ..., "79"]."""
    out: list[str] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:] and all(
                s.strip().isdigit() for s in part.split("-", 1)):
            lo, hi = (int(s) for s in part.split("-", 1))
            if hi < lo:
                raise FamosError(f"range {part!r} counts backwards")
            out.extend(str(n) for n in range(lo, hi + 1))
        else:
            out.append(part)
    return out


def load_names(spec: str) -> tuple[list[str], str]:
    """A range, a comma list, or a file of one name per line."""
    path = Path(str(spec))
    if path.exists() and path.is_file():
        names = [ln.split(",")[-1].strip()
                 for ln in path.read_text(encoding="utf-8-sig").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        return names, str(path)
    return expand(spec), "the command line"


def check_names(names: list[str], head: Header) -> None:
    if len(names) != head.n_channels:
        raise FamosError(
            f"{len(names)} name(s) for {head.n_channels} channel(s). "
            f"The file's channels are, in header order: "
            f"{', '.join(head.names)}")
    seen = [n for n in names if names.count(n) > 1]
    if seen:
        raise FamosError(f"duplicate name(s): {', '.join(sorted(set(seen)))}. "
                         f"The pipeline files a channel by name, so two "
                         f"channels sharing one would silently overwrite.")
    for n in names:
        if not n or "," in n or ";" in n:
            raise FamosError(f"{n!r} cannot be a channel name: a name must be "
                             f"non-empty and free of ',' and ';', which "
                             f"delimit the key it lives in")


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _cn_key(prefix: str, name: str, comment: str) -> bytes:
    body = (f"{prefix}{len(name)},{name},{len(comment)},"
            f"{comment}{',' if comment else ''}").encode("latin-1")
    return b"|CN,1," + str(len(body)).encode() + b"," + body + b";"


def rename(src, dst, names: list[str]) -> Header:
    """Write `src` to `dst` with new channel names. Samples untouched."""
    src, dst = Path(src), Path(dst)
    head = read_header(src)
    check_names(names, head)

    raw = src.open("rb").read()
    out = bytearray()
    cursor = 0
    for channel, new in zip(head.channels, names):
        out += raw[cursor:channel.key_start]
        out += _cn_key(channel.prefix, new, channel.comment)
        cursor = channel.key_stop
    out += raw[cursor:]

    dst.write_bytes(bytes(out))
    return read_header(dst)


def data_digest(head: Header) -> str:
    """A hash of the SAMPLES, so a rename can be shown not to touch them."""
    with head.path.open("rb") as fh:
        fh.seek(head.data_offset)
        digest = hashlib.sha256()
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def default_out(src) -> Path:
    src = Path(src)
    return src.with_name(src.stem + "_renamed" + src.suffix)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def describe(head: Header) -> str:
    lines = [f"{head.path.name}",
             f"  {head.n_channels} channels, {head.numpy_dtype()}, "
             f"{head.n_frames:,} samples at {head.fs:,.1f} Hz "
             f"({head.duration_s:.3f} s)",
             f"  data starts at byte {head.data_offset}",
             "  channels, in header order:"]
    lines += [f"    {c.index:>3}   {c.name}" for c in head.channels]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="show the channels as they are now")
    p.add_argument("path", type=Path)

    p = sub.add_parser("template", help="write a csv to fill in")
    p.add_argument("path", type=Path)
    p.add_argument("out", type=Path)

    p = sub.add_parser("rename", help="write a copy with new names")
    p.add_argument("path", type=Path)
    p.add_argument("names", help='"UC2,UC1,1-14", or a csv from `template`')
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")

    a = ap.parse_args(argv)
    try:
        head = read_header(a.path)
    except FamosError as exc:
        print(f"ERROR: {exc}")
        return 2

    if a.cmd == "list":
        print(describe(head))
        return 0

    if a.cmd == "template":
        rows = ["# one row per channel, in header order. Edit the second",
                "# column; the first is what the channel is called now.",
                "old,new"]
        rows += [f"{c.name},{c.name}" for c in head.channels]
        a.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"wrote {a.out} -- fill in the `new` column, then:")
        print(f'  python famos_rename.py rename "{a.path}" "{a.out}"')
        return 0

    try:
        names, source = load_names(a.names)
        check_names(names, head)
    except FamosError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(describe(head))
    print(f"  names from {source}")
    for channel, new in zip(head.channels, names):
        arrow = "->" if channel.name != new else "  "
        print(f"    {channel.index:>3}   {channel.name:<12} {arrow} {new}")
    if a.dry_run:
        print("  dry run: nothing written")
        return 0

    out = a.out or default_out(a.path)
    try:
        done = rename(a.path, out, names)
    except FamosError as exc:
        print(f"ERROR: {exc}")
        return 2
    same = data_digest(head) == data_digest(done)
    print(f"  written: {out}")
    print(f"  samples: {'identical to the source' if same else 'DIFFER'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
