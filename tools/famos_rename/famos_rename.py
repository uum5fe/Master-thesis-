#!/usr/bin/env python3
"""
famos_rename.py  --  rename FAMOS / DASYLab channels from a list you write
==========================================================================

A standalone tool.  It imports nothing from the EIS pipeline and the pipeline
imports nothing from it: copy this one file anywhere and it works.

    python famos_rename.py list     KANAL_6479.DAT
    python famos_rename.py template KANAL_6479.DAT -o names.csv
    python famos_rename.py apply    KANAL_6479.DAT --names names.csv --out renamed.DAT
    python famos_rename.py verify   renamed.DAT --against KANAL_6479.DAT

THE PROBLEM
-----------
DASYLab names the channels it records after their position on the card, not
after the thing that is wired to that position.  A card carrying segments
64..79 is written out with the channel names

    "0", "1", "2", ... "15"

The measurement is fine -- only the labels are wrong.  But a spectrum filed
under segment 3 that belongs to segment 67 does not look wrong at all, which
is why this is worth a tool rather than a mental note.

NOTHING IS GUESSED
------------------
The new names come from you and only from you -- a file you have edited, or
a list on the command line.  This tool will not read a range out of the file
name, or assume the channels are in ascending order, or fill in a name it was
not given.  You know the wiring; it does not.

    python famos_rename.py template FILE -o names.csv   # current names
    # edit the new_name column in names.csv
    python famos_rename.py apply FILE --names names.csv --out renamed.DAT

`--names` also takes the list directly, in header order:

    --names 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79
    --names 64-79          # the same thing, written as a range

WHAT IT WRITES
--------------
A FAMOS file identical to the input except for the channel-name keys: every
other key and every single sample byte is copied through untouched, and the
input file is never modified.  The result is an ordinary FAMOS file that any
reader -- including your EIS evaluation -- opens exactly as it opened the
original, but with the right names on the channels.

`verify` re-reads the result and checks the sample bytes against the source,
so the copy is not taken on trust.

`export` is there for an evaluation that would rather have a CSV than a
FAMOS file; the renamed .DAT is the deliverable, the CSV is a convenience.

THE HEADER IS PARSED, NOT PATTERN-MATCHED
-----------------------------------------
FAMOS keys are `|XX,version,length,<length bytes of content>;` and the
strings inside them are length-prefixed, precisely so a name may contain a
comma or a semicolon.  Splitting the header on punctuation works right up
until a channel is called "U,ref" -- so this walks the keys and honours the
declared lengths.  The channel table comes from the |CP keys (byte offset,
sample width, number format) and the names from the |CN keys after them.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: FAMOS number-format codes -> (numpy type code, bytes per sample).
NUMBER_FORMATS: dict[int, tuple[str, int]] = {
    1: ("u1", 1), 2: ("i1", 1),
    3: ("u2", 2), 4: ("i2", 2),
    5: ("u4", 4), 6: ("i4", 4),
    7: ("f4", 4), 8: ("f8", 8),
    10: ("u2", 2), 11: ("u8", 6),
}

#: How much of the file may be searched for the end of the key section. These
#: headers run to a few kB; a megabyte is generous and still fails fast on a
#: file that is not FAMOS at all.
MAX_HEADER = 1 << 20

#: FAMOS headers are 8-bit text -- channel names carry umlauts.
ENCODING = "latin-1"


class FamosError(ValueError):
    """The file is not FAMOS, or is FAMOS this tool will not guess about."""


# ---------------------------------------------------------------------------
# 1. Key-level parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Key:
    """One `|XX,version,length,content;` key, located in the raw bytes."""

    name: str
    version: int
    start: int          # index of the leading '|'
    body_start: int     # first byte of the length-counted content
    body_end: int       # one past the content, i.e. the ';'
    raw: bytes = field(repr=False)

    @property
    def body(self) -> bytes:
        return self.raw[self.body_start:self.body_end]

    @property
    def end(self) -> int:
        """One past the terminating ';'."""
        return self.body_end + 1


def _int_field(body: bytes, p: int) -> tuple[int, int]:
    """Read an integer up to the next comma; return it and the next position."""
    q = body.index(b",", p)
    return int(body[p:q]), q + 1


def _str_field(body: bytes, p: int) -> tuple[str, int]:
    """Read a length-prefixed string: `<n>,<n bytes>`, then past its comma."""
    n, p = _int_field(body, p)
    s = body[p:p + n].decode(ENCODING)
    # The separator after the string is absent when it ends the key.
    return s, min(p + n + 1, len(body))


def iter_keys(raw: bytes, limit: int = MAX_HEADER):
    """Walk the key section, yielding every key up to and including |CS."""
    p = 0
    while p < min(len(raw), limit):
        if raw[p:p + 1] in (b"\r", b"\n", b" ", b"\t"):
            p += 1
            continue
        if raw[p:p + 1] != b"|":
            raise FamosError(f"expected a FAMOS key at byte {p}, "
                             f"found {raw[p:p + 8]!r}")
        name = raw[p + 1:p + 3].decode(ENCODING)
        q = raw.index(b",", p + 3) + 1          # past the name's comma
        version, q = _int_field(raw, q)
        length, q = _int_field(raw, q)
        key = Key(name, version, p, q, q + length, raw)
        if name == "CS":
            # The data key's content is the measurement itself: it is neither
            # read into `raw` nor followed by any further key.
            yield key
            return
        if key.body_end > len(raw):
            raise FamosError(f"|{name} key at {p} declares {length} bytes, "
                             f"past the end of the header")
        yield key
        p = key.end


# ---------------------------------------------------------------------------
# 2. The channel table
# ---------------------------------------------------------------------------


@dataclass
class Channel:
    """One recorded channel: where its samples sit, and what it is called."""

    index: int                  # position in the header = acquisition order
    byte_offset: int            # of its sample within the frame
    bytes_per_sample: int
    number_format: int
    bytes_to_next: int = 0      # gap to this channel's next sample; 0 = unset
    name: str = ""              # the name the file carries
    comment: str = ""
    name_key: Key | None = None  # the |CN key the name came from

    @property
    def dtype(self) -> str:
        code, _ = NUMBER_FORMATS.get(self.number_format, ("", 0))
        return code


@dataclass
class FamosHeader:
    """Everything about the recording except the samples themselves."""

    path: Path
    channels: list[Channel]
    data_offset: int            # first byte of interleaved sample data
    frame_bytes: int            # one sample of every channel
    n_frames: int               # samples per channel
    dt: float                   # seconds between samples
    little_endian: bool
    declared_bytes: int         # what |Cb says the buffer holds
    available_bytes: int        # what the file actually carries

    @property
    def fs(self) -> float:
        return 1.0 / self.dt if self.dt else float("nan")

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def duration_s(self) -> float:
        return self.n_frames * self.dt

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.channels]

    def numpy_dtype(self) -> str:
        codes = {c.dtype for c in self.channels}
        if len(codes) != 1 or not codes.pop():
            raise FamosError("the channels do not share one number format; "
                             "this tool reads a uniform frame only")
        return ("<" if self.little_endian else ">") + self.channels[0].dtype


def read_header(path) -> FamosHeader:
    """Parse the key section of a FAMOS / DASYLab file."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(min(MAX_HEADER, size))

    if not raw.startswith(b"|CF"):
        raise FamosError(f"{path.name}: does not start with a |CF key, "
                         f"so it is not a FAMOS file")

    channels: list[Channel] = []
    little_endian = True
    dt = 0.0
    data_offset = 0
    declared = 0

    for key in iter_keys(raw):
        body = key.body
        if key.name == "CF":
            # Processor code: 1 = Intel (little endian), 2 = Motorola.
            little_endian = int(body.split(b",")[0]) == 1
        elif key.name == "CD":
            dt = float(body.split(b",")[0])
        elif key.name == "CP":
            f = [int(x) for x in body.split(b",")[:8]]
            # buffer_ref, bytes_per_sample, number_format, significant_bits,
            # mask, byte_offset, direct_block_size, byte_offset_to_next
            channels.append(Channel(index=len(channels), byte_offset=f[5],
                                    bytes_per_sample=f[1], number_format=f[2],
                                    bytes_to_next=f[7]))
        elif key.name == "Cb" and channels:
            parts = body.split(b",")
            if len(parts) > 7:
                declared = max(declared, int(parts[7]))
        elif key.name == "CN" and channels:
            ch = channels[-1]
            p = 0
            for _ in range(3):                   # group index, and two zeros
                _, p = _int_field(body, p)
            ch.name, p = _str_field(body, p)
            ch.comment, p = _str_field(body, p)
            ch.name_key = key
        elif key.name == "CS":
            _, p = _int_field(body, 0)           # index of the buffer
            data_offset = key.body_start + p

    if not channels:
        raise FamosError(f"{path.name}: no |CP keys, so no channels")
    if not data_offset:
        raise FamosError(f"{path.name}: no |CS key, so no sample data")

    frame = _frame_bytes(path.name, channels)
    available = size - data_offset

    return FamosHeader(path=path, channels=channels, data_offset=data_offset,
                       frame_bytes=frame, n_frames=available // frame, dt=dt,
                       little_endian=little_endian, declared_bytes=declared,
                       available_bytes=available)


def _frame_bytes(name: str, channels: list[Channel]) -> int:
    """Bytes per interleaved frame, cross-checked three ways.

    The |CP keys state it over and over -- as the span of the byte offsets, as
    the samples packed end to end, and as the stride from one sample of a
    channel to the next.  All three must agree.  If they do not, the frame
    carries padding or a layout this tool does not model, and reading it as a
    plain interleave would put every sample after the first in the wrong
    channel: refuse loudly, because the alternative is plausible nonsense.
    """
    span = max(c.byte_offset + c.bytes_per_sample for c in channels)
    packed = sum(c.bytes_per_sample for c in channels)
    if span != packed:
        raise FamosError(f"{name}: the channel offsets span {span} bytes but "
                         f"the samples total {packed}; the frame is not a "
                         f"plain interleave and this tool will not guess it")

    # A writer that does not interleave leaves the stride at 0; only a stated
    # stride is evidence.
    strides = {c.bytes_to_next + c.bytes_per_sample for c in channels
               if c.bytes_to_next}
    if strides and strides != {span}:
        stated = ", ".join(str(x) for x in sorted(strides))
        raise FamosError(f"{name}: the |CP keys stride {stated} bytes between "
                         f"samples but the frame spans {span}; the frame is "
                         f"not a plain interleave and this tool will not "
                         f"guess it")
    return span


# ---------------------------------------------------------------------------
# 3. The names you supply
# ---------------------------------------------------------------------------

TEMPLATE_COLUMNS = ["channel", "label_in_file", "new_name"]


def write_template(head: FamosHeader, out_path) -> Path:
    """A CSV of the current channels, for you to edit the new_name column.

    new_name starts out as the name already in the file, so editing a few
    channels does not mean retyping the rest -- and so a row left alone is a
    decision to keep that name rather than an oversight.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(TEMPLATE_COLUMNS)
        for ch in head.channels:
            w.writerow([ch.index, ch.name, ch.name])
    return out_path


def read_names_file(path) -> list[str]:
    """Names from a file: either the edited template, or one name per line."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise FamosError(f"{path.name}: no names in it")

    header = [c.strip().lower() for c in lines[0].split(",")]
    if "new_name" in header:
        col = header.index("new_name")
        names = []
        for n, line in enumerate(lines[1:], start=2):
            cells = next(csv.reader([line]))
            if col >= len(cells):
                raise FamosError(f"{path.name} line {n}: no new_name column")
            names.append(cells[col].strip())
        return names

    # A plain list: one name per line, comments and blanks ignored.
    return [ln.strip() for ln in lines if not ln.lstrip().startswith("#")]


def parse_names_arg(spec: str) -> list[str]:
    """`--names` as a literal list, or as numeric ranges: `64-79`, `1,5-7`."""
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if hi < lo:
                raise FamosError(f"the range {part!r} counts backwards")
            out.extend(str(v) for v in range(lo, hi + 1))
        else:
            out.append(part)
    return out


def load_names(spec: str) -> tuple[list[str], str]:
    """Resolve `--names`: a path to a file, or the list itself."""
    if Path(spec).exists():
        return read_names_file(spec), f"the file {Path(spec).name}"
    return parse_names_arg(spec), "the command line"


def check_names(names: list[str], head: FamosHeader) -> None:
    """Every way a hand-written list goes wrong, caught before anything is
    written.

    Silence is the enemy here: a list one short would otherwise rename fifteen
    channels correctly and leave the sixteenth carrying its old name, which is
    exactly the kind of result nobody notices.
    """
    if len(names) != head.n_channels:
        raise FamosError(
            f"{len(names)} names given for {head.n_channels} channels. "
            f"Give one name per channel, in header order.")

    blank = [i for i, n in enumerate(names) if not n.strip()]
    if blank:
        raise FamosError(
            f"no name given for channel(s) {_join(blank)}. Every channel "
            f"needs one -- keep a name by writing it out.")

    seen: dict[str, int] = {}
    for i, n in enumerate(names):
        if n in seen:
            raise FamosError(
                f"channels {seen[n]} and {i} were both named {n!r}; "
                f"a duplicate name makes the two impossible to tell apart")
        seen[n] = i

    for i, n in enumerate(names):
        try:
            n.encode(ENCODING)
        except UnicodeEncodeError as exc:
            raise FamosError(
                f"the name for channel {i}, {n!r}, has a character that a "
                f"FAMOS header cannot hold (it is 8-bit {ENCODING})") from exc
        if any(c in n for c in "\r\n"):
            raise FamosError(f"the name for channel {i} has a line break in it")


def _join(xs) -> str:
    return ", ".join(str(x) for x in xs)


# ---------------------------------------------------------------------------
# 4. Writing the renamed file
# ---------------------------------------------------------------------------


def build_cn_key(group: int, name: str, comment: str) -> bytes:
    """A |CN key carrying `name`, with its length field made to match."""
    n = name.encode(ENCODING)
    c = comment.encode(ENCODING)
    body = b"%d,0,0,%d,%s,%d,%s" % (group, len(n), n, len(c), c)
    return b"|CN,1,%d,%s;" % (len(body), body)


def default_out(path) -> Path:
    """Where a renamed copy goes when you do not say: beside the original."""
    p = Path(path)
    return p.with_name(p.stem + "_renamed" + p.suffix)


def rename(path, out_path, names: list[str]) -> FamosHeader:
    """Copy the file with `names` in its |CN keys, one per channel.

    Only the name keys change.  Their length fields are rebuilt to match, so
    the keys after them stay findable; everything else -- every other key and
    every sample byte -- is copied through, and the input is not touched.
    """
    path, out_path = Path(path), Path(out_path)
    if out_path.exists() and out_path.resolve() == path.resolve():
        raise FamosError("refusing to write over the file being renamed; "
                         "--out must name a different file")

    head = read_header(path)
    check_names(names, head)

    missing = [c.index for c in head.channels if c.name_key is None]
    if missing:
        raise FamosError(f"channel(s) {_join(missing)} have no |CN key to "
                         f"rename; this file does not carry channel names")

    raw = path.open("rb").read(head.channels[-1].name_key.end)

    pieces, cursor = [], 0
    for ch, new in zip(head.channels, names):
        key = ch.name_key
        group, _ = _int_field(key.body, 0)
        pieces.append(raw[cursor:key.start])
        pieces.append(build_cn_key(group, new, ch.comment))
        cursor = key.end
    pieces.append(raw[cursor:])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as src, out_path.open("wb") as dst:
        for piece in pieces:
            dst.write(piece)
        src.seek(cursor)
        shutil.copyfileobj(src, dst, 1 << 20)

    return read_header(out_path)


# ---------------------------------------------------------------------------
# 5. Reading the samples
# ---------------------------------------------------------------------------


def read_data(head: FamosHeader, channels: list[str] | None = None,
              step: int = 1):
    """The samples as an (n_frames, n_channels) array, memory-mapped.

    `channels` selects by name, in the order given; `step` decimates.
    """
    import numpy as np

    mm = np.memmap(head.path, dtype=head.numpy_dtype(), mode="r",
                   offset=head.data_offset,
                   shape=(head.n_frames, head.n_channels))
    if channels is None:
        return mm[::step]

    missing = [c for c in channels if c not in head.names]
    if missing:
        raise FamosError(f"no channel named {_join(repr(m) for m in missing)}; "
                         f"the file has {_join(head.names)}")
    cols = [head.names.index(c) for c in channels]
    return mm[::step][:, cols]


def data_digest(head: FamosHeader) -> str:
    """SHA-256 of the whole sample region, for checking a copy."""
    import hashlib

    h = hashlib.sha256()
    with head.path.open("rb") as fh:
        fh.seek(head.data_offset)
        remaining = head.n_frames * head.frame_bytes
        while remaining > 0:
            block = fh.read(min(1 << 22, remaining))
            if not block:
                break
            remaining -= len(block)
            h.update(block)
    return h.hexdigest()


def export_csv(head: FamosHeader, out_path, channels: list[str] | None = None,
               step: int = 1, time_column: bool = True) -> int:
    """Write the samples as CSV, for an evaluation that wants one.

    The renamed FAMOS file is the deliverable; this is for a tool that would
    rather read text.  It is a big file -- one row per sample -- so `step`
    is there to thin it.
    """
    import numpy as np

    block = np.asarray(read_data(head, channels, step))
    names = channels if channels else head.names
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow((["time_s"] if time_column else []) + list(names))
        dt = head.dt * step
        for i, row in enumerate(block):
            cells = [f"{v:.9g}" for v in row]
            w.writerow(([f"{i * dt:.9g}"] if time_column else []) + cells)
    return len(block)


# ---------------------------------------------------------------------------
# 6. Command line
# ---------------------------------------------------------------------------


def describe(head: FamosHeader) -> str:
    lines = [
        f"file       : {head.path.name}",
        f"channels   : {head.n_channels}",
        f"format     : {head.numpy_dtype()}  "
        f"({head.frame_bytes} bytes per frame)",
        f"data at    : byte {head.data_offset:,}",
        f"samples    : {head.n_frames:,} per channel",
        f"rate       : {head.fs:,.1f} Hz  ({head.duration_s:.3f} s)",
    ]
    used = head.n_frames * head.frame_bytes
    if head.declared_bytes and head.declared_bytes != used:
        lines.append(f"NOTE       : |Cb declares {head.declared_bytes:,} "
                     f"bytes, {used:,} read from the file")
    if head.available_bytes - used > 1:
        lines.append(f"NOTE       : {head.available_bytes - used} bytes after "
                     f"the last whole frame were ignored")
    return "\n".join(lines)


def channel_table(head: FamosHeader, new: list[str] | None = None) -> str:
    cols = ["ch", "name in file", "byte off", "format"]
    if new:
        cols.insert(2, "new name")
    rows = []
    for i, ch in enumerate(head.channels):
        row = [str(i), ch.name, str(ch.byte_offset),
               ch.dtype or f"fmt{ch.number_format}"]
        if new:
            row.insert(2, new[i])
        rows.append(row)

    w = [max(len(r[c]) for r in [cols] + rows) for c in range(len(cols))]
    out = ["  ".join(h.rjust(x) for h, x in zip(cols, w)),
           "  ".join("-" * x for x in w)]
    out += ["  ".join(v.rjust(x) for v, x in zip(r, w)) for r in rows]
    return "\n".join(out)


def cmd_list(a) -> int:
    head = read_header(a.file)
    print(describe(head))
    print()
    print(channel_table(head))
    return 0


def cmd_template(a) -> int:
    head = read_header(a.file)
    out = write_template(head, a.out)
    print(f"written    : {out}  ({head.n_channels} channels)")
    print("Edit the new_name column, then:")
    print(f"    python famos_rename.py apply {a.file} "
          f"--names {out} --out renamed.DAT")
    return 0


def cmd_apply(a) -> int:
    names, source = load_names(a.names)
    head = read_header(a.file)
    check_names(names, head)
    out_path = a.out or default_out(a.file)

    print(describe(head))
    print(f"names from : {source}")
    print()
    print(channel_table(head, names))

    if a.dry_run:
        print("\ndry run    : nothing written")
        return 0

    out = rename(a.file, out_path, names)
    print(f"\nwritten    : {out_path}")
    print(f"names now  : {', '.join(out.names)}")
    if a.verify:
        ok = data_digest(head) == data_digest(out)
        print(f"samples    : {'identical to the source' if ok else 'DIFFER'}")
        if not ok:
            return 1
    return 0


def cmd_verify(a) -> int:
    head = read_header(a.file)
    print(describe(head))
    print()
    print(channel_table(head))
    if a.against:
        src = read_header(a.against)
        if src.n_channels != head.n_channels:
            print(f"\nchannels   : {src.n_channels} in {Path(a.against).name}, "
                  f"{head.n_channels} here -- NOT the same recording")
            return 1
        ok = data_digest(src) == data_digest(head)
        print(f"\nagainst    : {Path(a.against).name}")
        print(f"was named  : {', '.join(src.names)}")
        print(f"samples    : {'identical' if ok else 'DIFFER'}")
        return 0 if ok else 1
    return 0


def cmd_export(a) -> int:
    head = read_header(a.file)
    chans = [c.strip() for c in a.channels.split(",")] if a.channels else None
    n = export_csv(head, a.out, chans, a.step, not a.no_time)
    print(f"written    : {a.out}  ({n:,} rows)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Rename FAMOS/DASYLab channels from a list you write.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list", help="show the channels the file carries")
    ls.add_argument("file")
    ls.set_defaults(func=cmd_list)

    tp = sub.add_parser("template", help="write a CSV of names for you to edit")
    tp.add_argument("file")
    tp.add_argument("-o", "--out", default="names.csv")
    tp.set_defaults(func=cmd_template)

    ap = sub.add_parser("apply", help="write a copy carrying your names")
    ap.add_argument("file")
    ap.add_argument("--names", required=True,
                    help="an edited template, a file of one name per line, "
                         "or the names themselves: '64-79' or 'UC1,64,65,...'")
    ap.add_argument("--out", default=None,
                    help="the file to write. Defaults to the input with "
                         "'_renamed' before the extension; the input itself "
                         "is never written over.")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be renamed and write nothing")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip the sample-for-sample check of the copy")
    ap.set_defaults(func=cmd_apply)

    vf = sub.add_parser("verify", help="re-read a renamed file")
    vf.add_argument("file")
    vf.add_argument("--against", default=None,
                    help="the original, to check the samples against")
    vf.set_defaults(func=cmd_verify)

    ex = sub.add_parser("export", help="write the samples as CSV")
    ex.add_argument("file")
    ex.add_argument("--out", required=True)
    ex.add_argument("--channels", default=None,
                    help="comma-separated names, in the order you want them")
    ex.add_argument("--step", type=int, default=1,
                    help="keep every Nth sample (default 1: all of them)")
    ex.add_argument("--no-time", action="store_true",
                    help="leave out the leading time_s column")
    ex.set_defaults(func=cmd_export)

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except FamosError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
