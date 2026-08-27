#!/usr/bin/env python3
"""
famos_segments.py  --  put the real segment numbers back on FAMOS channels
==========================================================================

    python famos_segments.py map     "2026_08_27_KANAL_6479RO2612025_60A.DDF_1.DAT"
    python famos_segments.py map     FILE --segments 64-79 --stats --csv map.csv
    python famos_segments.py relabel FILE --out corrected.DAT

THE PROBLEM
-----------
DASYLab names the channels it records after their position on the card, not
after the plate segment that is wired to that position.  A card that carries
segments 64..79 is written out with the channel names

    "0", "1", "2", ... "15"

so every downstream tool that trusts the name is off by 64.  The measurement
itself is fine -- only the label is wrong -- but a spectrum attributed to
segment 3 when it belongs to segment 67 is worse than a missing spectrum,
because nothing about it looks wrong.

WHAT THIS DOES
--------------
`map` reads the FAMOS header, recovers the real channel table, and prints
which actual segment each channel belongs to.  `relabel` writes a corrected
copy of the file with the segment numbers in the channel-name keys, so the
rest of the pipeline reads the right names without knowing any of this.

WHERE THE SEGMENT NUMBERS COME FROM
-----------------------------------
Two sources, in this order:

  1. `--segments 64-79` (or a comma list).  Explicit, always wins.
  2. The file name.  These recordings carry the range in the name --
     "..._KANAL_6479RO..." is card "Kanal 64..79" -- so a four-digit run
     after KANAL is read as two two-digit bounds.

Whichever source is used, the range must contain exactly as many segments as
the file has channels.  If it does not, this refuses to map rather than
silently pairing the two lists off against each other: the whole point of
the script is that a wrong-but-plausible segment number is the failure that
costs a measurement campaign.

The pairing itself is positional and ascending -- first channel in the
header (lowest byte offset in the sample frame) to lowest segment number.
`--reverse` flips it for a card that was wired the other way round.

THE HEADER IS PARSED, NOT PATTERN-MATCHED
-----------------------------------------
FAMOS keys are `|XX,version,length,<length bytes of content>;` and the
strings inside them are length-prefixed, precisely so that a name may
contain a comma or a semicolon.  Splitting the header on punctuation
therefore works right up until a channel is named "U,ref" -- so this walks
the keys and honours the declared lengths instead.  The channel table comes
from the |CP keys (byte offset, sample width, number format) and the names
from the |CN keys that follow them.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: FAMOS number-format codes -> (numpy dtype without byte order, bytes).
NUMBER_FORMATS: dict[int, tuple[str, int]] = {
    1: ("u1", 1), 2: ("i1", 1),
    3: ("u2", 2), 4: ("i2", 2),
    5: ("u4", 4), 6: ("i4", 4),
    7: ("f4", 4), 8: ("f8", 8),
    10: ("u2", 2), 11: ("u8", 6),
}

#: How many bytes of a file may be searched for the end of the key section.
#: The header of these recordings is under 4 kB; a megabyte is a generous
#: ceiling that still fails fast on a file that is not FAMOS at all.
MAX_HEADER = 1 << 20


class FamosError(ValueError):
    """The file is not FAMOS, or is FAMOS this script will not guess about."""


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
    """Read an integer up to the next comma; return it and the position after."""
    q = body.index(b",", p)
    return int(body[p:q]), q + 1


def _str_field(body: bytes, p: int) -> tuple[str, int]:
    """Read a length-prefixed string: `<n>,<n bytes>` followed by a comma."""
    n, p = _int_field(body, p)
    s = body[p:p + n].decode("latin-1")
    # The separator after the string is absent when the string ends the key.
    return s, min(p + n + 1, len(body))


def iter_keys(raw: bytes, limit: int = MAX_HEADER):
    """Walk the key section, yielding every key until the data key or the end.

    Stops after |CS, whose content is the measurement itself rather than
    more keys.
    """
    p = 0
    while p < min(len(raw), limit):
        if raw[p:p + 1] in (b"\r", b"\n", b" ", b"\t"):
            p += 1
            continue
        if raw[p:p + 1] != b"|":
            raise FamosError(f"expected a FAMOS key at byte {p}, "
                             f"found {raw[p:p + 8]!r}")
        name = raw[p + 1:p + 3].decode("latin-1")
        q = raw.index(b",", p + 3) + 1          # past the name's comma
        version, q = _int_field(raw, q)
        length, q = _int_field(raw, q)
        key = Key(name, version, p, q, q + length, raw)
        if name == "CS":
            # The data key's content is the measurement, which is neither
            # read into `raw` nor followed by more keys.
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
    label: str = ""             # the name the file carries
    comment: str = ""
    name_key: Key | None = None  # the |CN key the label came from

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

    def numpy_dtype(self) -> str:
        codes = {c.dtype for c in self.channels}
        if len(codes) != 1 or not codes.pop():
            raise FamosError("channels do not share one number format; "
                             "this script reads a uniform frame only")
        return ("<" if self.little_endian else ">") + self.channels[0].dtype


def read_header(path) -> FamosHeader:
    """Parse the key section of a FAMOS/DASYLab .DAT file."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(min(MAX_HEADER, size))

    if not raw.startswith(b"|CF"):
        raise FamosError(f"{path.name}: does not start with a |CF key; "
                         f"this is not a FAMOS file")

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
            # The buffer key's 8th field is the buffer length in bytes.
            parts = body.split(b",")
            if len(parts) > 7:
                declared = max(declared, int(parts[7]))
        elif key.name == "CN" and channels:
            ch = channels[-1]
            p = 0
            for _ in range(3):                   # group index, and two zeros
                _, p = _int_field(body, p)
            ch.label, p = _str_field(body, p)
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
    n_frames = available // frame

    return FamosHeader(path=path, channels=channels, data_offset=data_offset,
                       frame_bytes=frame, n_frames=n_frames, dt=dt,
                       little_endian=little_endian, declared_bytes=declared,
                       available_bytes=available)


def _frame_bytes(name: str, channels: list[Channel]) -> int:
    """Bytes per interleaved frame, cross-checked three ways.

    The |CP keys state it over and over -- as the span of the byte offsets, as
    the samples packed end to end, and as the stride from one sample of a
    channel to the next.  All three must agree.  If they do not, the frame
    carries padding or a layout this script does not model, and reading it as
    a plain interleave would put every sample after the first in the wrong
    channel: loud refusal, because the alternative is plausible nonsense.
    """
    span = max(c.byte_offset + c.bytes_per_sample for c in channels)
    packed = sum(c.bytes_per_sample for c in channels)
    if span != packed:
        raise FamosError(f"{name}: channel offsets span {span} bytes but the "
                         f"samples total {packed}; the frame is not a plain "
                         f"interleave and this script will not guess it")

    # A writer that does not interleave leaves the stride at 0; only a
    # stated one is evidence.
    strides = {c.bytes_to_next + c.bytes_per_sample for c in channels
               if c.bytes_to_next}
    if strides and strides != {span}:
        stated = ", ".join(str(x) for x in sorted(strides))
        raise FamosError(f"{name}: the |CP keys stride {stated} bytes between "
                         f"samples but the frame spans {span}; the frame is "
                         f"not a plain interleave and this script will not "
                         f"guess it")
    return span


# ---------------------------------------------------------------------------
# 3. Which segment is which channel
# ---------------------------------------------------------------------------

#: "..._KANAL_6479RO..." -- the card's segment range, run together.
KANAL_RE = re.compile(r"KANAL[\s_-]*(\d+)(?:[\s_-]+(\d+))?", re.IGNORECASE)


def segments_from_filename(name: str, n_channels: int) -> list[int] | None:
    """Read the segment range out of a file name, or return None.

    Returns None rather than a guess: a range that does not match the channel
    count is not evidence, and the caller must be told to say what it wants.
    """
    m = KANAL_RE.search(name)
    if not m:
        return None

    lo_txt, hi_txt = m.group(1), m.group(2)
    if hi_txt is None:
        # One run of digits: "6479" is 64..79.  Only an even-length run can
        # split into two equal bounds, and only that reading is accepted.
        if len(lo_txt) % 2:
            return None
        half = len(lo_txt) // 2
        lo_txt, hi_txt = lo_txt[:half], lo_txt[half:]

    lo, hi = int(lo_txt), int(hi_txt)
    if hi < lo or hi - lo + 1 != n_channels:
        return None
    return list(range(lo, hi + 1))


def parse_segment_spec(spec: str) -> list[int]:
    """`64-79`, or `64,65,66`, or a mix of both."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if hi < lo:
                raise FamosError(f"segment range {part!r} counts backwards")
            out.extend(range(lo, hi + 1))
        elif part.isdigit():
            out.append(int(part))
        else:
            raise FamosError(f"cannot read {part!r} as a segment or a range")
    return out


def resolve_segments(head: FamosHeader, spec: str | None = None,
                     reverse: bool = False) -> tuple[list[int], str]:
    """The real segment number of every channel, and where it came from."""
    if spec:
        segments, source = parse_segment_spec(spec), "--segments"
    else:
        segments = segments_from_filename(head.path.name, head.n_channels)
        source = "the file name"
        if segments is None:
            raise FamosError(
                f"{head.path.name}: the file name does not carry a segment "
                f"range matching {head.n_channels} channels. Say it "
                f"explicitly, e.g. --segments 64-{63 + head.n_channels}")

    if len(segments) != head.n_channels:
        raise FamosError(
            f"{len(segments)} segments given but the file has "
            f"{head.n_channels} channels; refusing to pair them off")

    if reverse:
        segments = segments[::-1]
    return segments, source


def segment_map(path, spec: str | None = None,
                reverse: bool = False) -> dict[str, int]:
    """{channel label in the file: real segment number}, for importers."""
    head = read_header(path)
    segments, _ = resolve_segments(head, spec, reverse)
    return {ch.label: seg for ch, seg in zip(head.channels, segments)}


# ---------------------------------------------------------------------------
# 4. Reading samples, for the sanity check
# ---------------------------------------------------------------------------


def channel_stats(head: FamosHeader, max_samples: int = 200_000) -> list[dict]:
    """Per-channel min/mean/max over a stride across the whole recording.

    Not a measurement, a sanity check: a channel that is flat, railed or
    orders of magnitude away from its neighbours is a sign that the frame was
    read at the wrong alignment -- or that the channel is not a segment at all.
    """
    import numpy as np

    step = max(1, head.n_frames // max_samples)
    mm = np.memmap(head.path, dtype=head.numpy_dtype(), mode="r",
                   offset=head.data_offset,
                   shape=(head.n_frames, head.n_channels))
    block = np.asarray(mm[::step], dtype=np.float64)
    return [{"min": float(block[:, i].min()),
             "mean": float(block[:, i].mean()),
             "max": float(block[:, i].max()),
             "std": float(block[:, i].std())}
            for i in range(head.n_channels)]


# ---------------------------------------------------------------------------
# 5. Writing the corrected file
# ---------------------------------------------------------------------------


def build_cn_key(group: int, name: str, comment: str) -> bytes:
    """A |CN key carrying `name`, with its length field made to match."""
    n = name.encode("latin-1")
    c = comment.encode("latin-1")
    body = b"%d,0,0,%d,%s,%d,%s" % (group, len(n), n, len(c), c)
    return b"|CN,1,%d,%s;" % (len(body), body)


def relabel(path, out_path, spec: str | None = None, reverse: bool = False,
            prefix: str = "") -> FamosHeader:
    """Copy the file with the real segment numbers in its |CN keys.

    Only the name keys change.  Every other key and every sample byte is
    copied through untouched, so the result is the same measurement under the
    right names -- and the input is never modified.
    """
    path, out_path = Path(path), Path(out_path)
    if out_path.resolve() == path.resolve():
        raise FamosError("refusing to relabel a file onto itself; "
                         "--out must name a different file")

    head = read_header(path)
    segments, _ = resolve_segments(head, spec, reverse)

    missing = [c.index for c in head.channels if c.name_key is None]
    if missing:
        raise FamosError(f"channels {missing} have no |CN key to rewrite")

    raw = path.open("rb").read(head.channels[-1].name_key.end)

    pieces, cursor = [], 0
    for ch, seg in zip(head.channels, segments):
        key = ch.name_key
        group, _ = _int_field(key.body, 0)
        pieces.append(raw[cursor:key.start])
        pieces.append(build_cn_key(group, f"{prefix}{seg}", ch.comment))
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
# 6. Command line
# ---------------------------------------------------------------------------


def format_table(head: FamosHeader, segments: list[int],
                 stats: list[dict] | None = None) -> str:
    rows = [("ch", "label in file", "segment", "byte off", "format")]
    if stats:
        rows[0] += ("min", "mean", "max")
    body = []
    for i, (ch, seg) in enumerate(zip(head.channels, segments)):
        row = (str(i), f'"{ch.label}"', str(seg), str(ch.byte_offset),
               ch.dtype or f"fmt{ch.number_format}")
        if stats:
            s = stats[i]
            row += (f"{s['min']:.5g}", f"{s['mean']:.5g}", f"{s['max']:.5g}")
        body.append(row)

    widths = [max(len(r[c]) for r in [rows[0]] + body)
              for c in range(len(rows[0]))]
    out = ["  ".join(h.rjust(w) for h, w in zip(rows[0], widths)),
           "  ".join("-" * w for w in widths)]
    out += ["  ".join(v.rjust(w) for v, w in zip(r, widths)) for r in body]
    return "\n".join(out)


def describe(head: FamosHeader) -> str:
    lines = [
        f"file       : {head.path.name}",
        f"channels   : {head.n_channels}",
        f"format     : {head.numpy_dtype()}  "
        f"({head.frame_bytes} bytes per frame)",
        f"data at    : byte {head.data_offset:,}",
        f"samples    : {head.n_frames:,} per channel",
        f"rate       : {head.fs:,.1f} Hz  "
        f"({head.duration_s:.3f} s of recording)",
    ]
    used = head.n_frames * head.frame_bytes
    if head.declared_bytes and head.declared_bytes != used:
        lines.append(f"NOTE       : |Cb declares {head.declared_bytes:,} "
                     f"bytes, {used:,} read from the file")
    if head.available_bytes - used > 1:
        lines.append(f"NOTE       : {head.available_bytes - used} bytes after "
                     f"the last whole frame were ignored")
    return "\n".join(lines)


def cmd_map(a) -> int:
    head = read_header(a.file)
    segments, source = resolve_segments(head, a.segments, a.reverse)
    stats = channel_stats(head) if a.stats else None

    print(describe(head))
    print(f"segments   : {segments[0]}..{segments[-1]} from {source}"
          f"{', reversed' if a.reverse else ''}")
    print()
    print(format_table(head, segments, stats))

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            cols = ["channel", "label_in_file", "segment", "byte_offset",
                    "format"]
            if stats:
                cols += ["min", "mean", "max", "std"]
            w.writerow(cols)
            for i, (ch, seg) in enumerate(zip(head.channels, segments)):
                row = [i, ch.label, seg, ch.byte_offset, ch.dtype]
                if stats:
                    s = stats[i]
                    row += [s["min"], s["mean"], s["max"], s["std"]]
                w.writerow(row)
        print(f"\nwritten    : {a.csv}")
    return 0


def cmd_relabel(a) -> int:
    out = a.out or str(Path(a.file).with_suffix("")) + "_segments.DAT"
    head = relabel(a.file, out, a.segments, a.reverse, a.prefix)
    print(f"written    : {out}")
    print(f"labels now : {', '.join(c.label for c in head.channels)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Give FAMOS/DASYLab channels their real segment numbers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("file", help="the .DAT / .DDF FAMOS file")
        sp.add_argument("--segments", default=None,
                        help="the real segments, e.g. '64-79'. Read from the "
                             "file name when omitted.")
        sp.add_argument("--reverse", action="store_true",
                        help="pair the highest segment to the first channel")

    m = sub.add_parser("map", help="print the channel-to-segment table")
    common(m)
    m.add_argument("--stats", action="store_true",
                   help="also read the data and report each channel's range")
    m.add_argument("--csv", default=None, help="write the table to a CSV")
    m.set_defaults(func=cmd_map)

    r = sub.add_parser("relabel", help="write a copy with corrected names")
    common(r)
    r.add_argument("--out", default=None, help="output file")
    r.add_argument("--prefix", default="",
                   help="put before the number, e.g. 'Seg' -> 'Seg64'. "
                        "Bare digits by default, which is what the pipeline "
                        "recognises as a segment channel.")
    r.set_defaults(func=cmd_relabel)

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except FamosError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
