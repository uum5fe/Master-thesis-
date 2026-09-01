#!/usr/bin/env python3
r"""Standard-layout ("v2") FAMOS, read from the key structure.

The reader in ``eis_local.FamosFile`` handles a simplified export in which a
key's third field carries a value.  In the standard layout it carries a BYTE
COUNT, and every field the simplified reader wants sits one place further
along -- which is why that reader does not fail on a standard file so much as
answer wrongly: ``\|CD,\d+,([\d.eE+-]+)`` captures the length of the |CD
block, so a 100 kHz card reports 1/21 Hz.

So this does not search the header.  It walks it (:mod:`famos_keys`), which
is exact because every key declares its own length, and then CHECKS what it
derived against the sizes the file states.  The distinction matters more than
it sounds: every plausible-looking wrong answer in this file's history came
from a layout assumption that was never tested against the file's own
numbers.

WHAT IS DERIVED, AND WHAT IS REFUSED
------------------------------------
* **Sample interval** from |CD's first content field, and every component
  must agree.  A card whose channels disagree on dx is not a card this can
  return one ``fs`` for.
* **Names** from |CN, read by the declared name length rather than by
  splitting on commas, so a name containing a comma does not shift every
  field after it.
* **Sample format** from |CP's (Bytes, NumberFormat) pair, against an
  explicit table.  An unknown pair is REFUSED rather than defaulted: eight
  bytes is float64 or int64 and four is float32 or int32, and reading one as
  the other produces numbers, not an error.
* **Layout** from how many |CS blocks there are.  One means the components
  share it and the samples are interleaved; one each means every channel is
  contiguous.  These need completely different reads and nothing but the
  block count distinguishes them.
* **Calibration** from |CR is parsed and REPORTED but never auto-applied,
  for the same reason the v1 reader does not apply it: a factor guessed from
  a field position silently rescales every impedance, and the whole point of
  the shunt calibration is that it is the one absolute scale in the chain.

Finally the derived geometry is checked against the declared payload size.
If the product does not come out, that is a wrong assumption showing itself
as an error instead of as a spectrum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from famos_keys import FamosStructureError, walk

#: (bytes per value, |CP NumberFormat code) -> numpy dtype.
#: Only the pairs whose meaning is unambiguous. Anything else is refused,
#: because a wrong choice here does not fail -- it returns noise.
NUMBER_FORMATS: dict[tuple[int, int], str] = {
    (1, 1): "u1", (1, 2): "i1",
    (2, 3): "<u2", (2, 4): "<i2",
    (4, 5): "<u4", (4, 6): "<i4",
    (4, 7): "<f4",
    (8, 8): "<f8",
}


@dataclass
class Component:
    """One channel: where its samples are, and how wide they are."""

    name: str = ""
    dx: float = float("nan")
    n_bytes: int = 0
    number_format: int = -1
    buffer_ref: int = 0
    byte_offset: int = 0           #: |CP offset of this channel inside a row
    buffer_length: int = 0         #: |Cb length of this component's samples
    transform: int = 0             #: |CR transform flag; 0 means "not calibrated"
    factor: float = 1.0
    offset: float = 0.0
    unit: str = ""
    data_offset: int = -1          #: byte offset of this component's |CS
    data_length: int = 0           #: declared length of that |CS

    @property
    def dtype(self) -> str:
        key = (self.n_bytes, self.number_format)
        if key not in NUMBER_FORMATS:
            raise FamosStructureError(
                f"channel {self.name!r}: |CP says {self.n_bytes} bytes, "
                f"number format {self.number_format}, which is not a "
                f"combination this reader will guess at. Eight bytes is "
                f"float64 or int64 and four is float32 or int32; reading one "
                f"as the other returns numbers rather than an error, so this "
                f"refuses instead. Add the pair to NUMBER_FORMATS once the "
                f"instrument's documentation says which it is.")
        return NUMBER_FORMATS[key]

    @property
    def identity_calibration(self) -> bool:
        # THE TRANSFORM FLAG DECIDES.  DASYLab writes |CR,1,10,0,0,0,1,0 --
        # transform 0, and the factor and offset fields are then both 0 and
        # mean nothing. Reading them as a scaling reports "factor 0" on every
        # channel of a perfectly ordinary file, and applying it would zero
        # the plate.
        if self.transform == 0:
            return True
        return abs(self.factor - 1.0) < 1e-12 and abs(self.offset) < 1e-12


def _famos_string(fields: list[str], index: int) -> tuple[str, int]:
    """Read a FAMOS length-prefixed string starting at `fields[index]`.

    Returns (text, index after it).  Uses the DECLARED length rather than
    trusting the comma split, so a name containing a comma -- which the split
    would tear in half and shift every field after -- is read whole.
    """
    try:
        want = int(fields[index])
    except (ValueError, IndexError):
        return "", index + 1
    if want <= 0:
        return "", index + 1
    text = ""
    j = index + 1
    while j < len(fields) and len(text) < want:
        text = fields[j] if not text else f"{text},{fields[j]}"
        j += 1
    return text[:want] if len(text) >= want else text, j


def _parse_nt(content: str) -> "datetime | None":
    """|NT trigger time: day, month, year, hour, minute, seconds."""
    parts = [p.strip() for p in content.split(",")]
    nums = [p for p in parts if re.fullmatch(r"[\d.]+", p or "")]
    if len(nums) < 6:
        return None
    try:
        day, month, year, hour, minute = (int(float(n)) for n in nums[:5])
        seconds = float(nums[5])
        return datetime(year, month, day, hour, minute, int(seconds),
                        int(round((seconds % 1) * 1e6)))
    except (ValueError, OverflowError):
        return None


@dataclass
class FamosFileV2:
    """A standard-layout FAMOS card, with the same surface as FamosFile."""

    path: Path
    fs: float = field(init=False)
    names: list[str] = field(init=False)
    n_ch: int = field(init=False)
    n_samples: int = field(init=False)
    offset: int = field(init=False)
    start_time: "datetime | None" = field(init=False, default=None)
    interleaved: bool = field(init=False, default=True)
    components: list[Component] = field(init=False, default_factory=list)
    notes: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.path = Path(self.path)
        self._parse()

    # -- header ----------------------------------------------------------
    def _parse(self) -> None:
        comps: list[Component] = []
        current: Component | None = None
        payloads: list[tuple[int, int]] = []

        for entry in walk(self.path):
            key, content = entry["key"], entry["content"]
            fields = content.split(",")

            if key == "NT" and self.start_time is None:
                self.start_time = _parse_nt(content)
            elif key == "CC":
                current = Component()
                comps.append(current)
            elif key == "CD":
                dx = _first_float(fields)
                if current is None:
                    current = Component()
                    comps.append(current)
                current.dx = dx
            elif key == "CN" and current is not None:
                current.name = _channel_name(fields)
            elif key == "CP" and current is not None:
                # BufferReference, Bytes, NumberFormat, SignBits, ...
                nums = _ints(fields)
                if len(nums) >= 3:
                    current.buffer_ref = nums[0]
                    current.n_bytes = nums[1]
                    current.number_format = nums[2]
                # BufferReference, Bytes, NumberFormat, SignificantBits,
                # Mask, Offset, ... -- field 5 is where this channel sits
                # inside an interleaved row. Taken from the file rather than
                # assumed to be index*itemsize, because that assumption is
                # only right while the components are in order.
                if len(nums) >= 6:
                    current.byte_offset = nums[5]
            elif key == "Cb" and current is not None:
                # NumBuffers, UserInfo, BufferRef, SamplesKeyIndex,
                # OffsetInKey, BufferLengthBytes, ...
                nums = _ints(fields)
                if len(nums) >= 6:
                    current.buffer_length = nums[5]
            elif key == "CR" and current is not None:
                _apply_cr(current, fields)
            elif key == "CS":
                # |CS content is "<samples-key index>,<raw bytes>", so the
                # payload does not start at the content offset. On the
                # 150 A cards that prefix is exactly the two bytes by which
                # the declared |CS length exceeds the |Cb buffer length --
                # and two bytes of skew shifts every sample of every channel
                # by a fraction of a value, which is noise, not an error.
                skip = _index_prefix(entry["content"])
                start = entry["content_offset"] + skip
                length = max(0, entry["length"] - skip)
                if current is not None and current.data_offset < 0:
                    current.data_offset = start
                    current.data_length = length
                payloads.append((start, length))

        comps = [c for c in comps if c.name or c.n_bytes]
        if not comps:
            raise FamosStructureError(
                f"{self.path.name}: the key structure walked cleanly but "
                f"carried no channel components (|CC / |CN / |CP).")
        if not payloads:
            raise FamosStructureError(
                f"{self.path.name}: no |CS payload key anywhere in the file.")

        self.components = comps
        self.names = [c.name or str(i + 1) for i, c in enumerate(comps)]
        self.n_ch = len(comps)

        # ---- one sample interval for the card --------------------------
        rates = sorted({round(1.0 / c.dx, 6) for c in comps
                        if np.isfinite(c.dx) and c.dx > 0})
        if not rates:
            raise FamosStructureError(
                f"{self.path.name}: no usable sample interval in any |CD.")
        if len(rates) > 1:
            raise FamosStructureError(
                f"{self.path.name}: the components declare different sample "
                f"rates ({', '.join(f'{r:.0f} Hz' for r in rates)}). There is "
                f"no single fs for this card.")
        self.fs = float(rates[0])

        # ---- interleaved, or one block per channel? --------------------
        self.interleaved = len(payloads) == 1
        self.offset = payloads[0][0]
        self._verify(payloads)

    def _verify(self, payloads) -> None:
        """Check the derived geometry against the sizes the file declares."""
        widths = {c.n_bytes for c in self.components}
        formats = {c.number_format for c in self.components}
        if len(widths) > 1 or len(formats) > 1:
            raise FamosStructureError(
                f"{self.path.name}: components differ in sample format "
                f"(bytes {sorted(widths)}, formats {sorted(formats)}). This "
                f"reader returns one array shape for the card and cannot.")
        itemsize = np.dtype(self.components[0].dtype).itemsize

        if self.interleaved:
            # THE ROW LAYOUT COMES FROM |CP, NOT FROM THE COMPONENT ORDER.
            # Each component states its byte offset inside the row; assuming
            # index*itemsize is right only while the components happen to be
            # in order, and wrong silently when they are not -- every channel
            # would be read as its neighbour.
            offsets = [c.byte_offset for c in self.components]
            expected = {i * itemsize for i in range(self.n_ch)}
            if set(offsets) != expected:
                raise FamosStructureError(
                    f"{self.path.name}: the |CP byte offsets are "
                    f"{sorted(set(offsets))}, which is not one slot per "
                    f"channel in a row of {self.n_ch} x {itemsize} bytes. "
                    f"This reader cannot place the channels from that.")
            per_row = itemsize * self.n_ch

            declared = payloads[0][1]
            # |Cb states the buffer length directly; prefer it over the |CS
            # length, which also counts the samples-key index prefix.
            buffers = {c.buffer_length for c in self.components
                       if c.buffer_length > 0}
            usable = min(buffers) if len(buffers) == 1 else declared
            available = self.path.stat().st_size - self.offset
            if not (0 < usable <= available):
                usable = min(declared, available) if declared else available
            self.n_samples = usable // per_row
            if usable % per_row:
                self.notes.append(
                    f"the payload is {usable} bytes, which is not a whole "
                    f"number of {self.n_ch}-channel rows of {itemsize} bytes "
                    f"({usable % per_row} left over). Either the channel "
                    f"count or the sample width is wrong.")
        else:
            if len(payloads) != self.n_ch:
                raise FamosStructureError(
                    f"{self.path.name}: {len(payloads)} |CS blocks for "
                    f"{self.n_ch} channels. Neither one shared block nor one "
                    f"per channel, so how the samples are laid out is not "
                    f"something this can infer.")
            counts = {c.data_length // itemsize for c in self.components}
            if len(counts) > 1:
                self.notes.append(
                    f"the per-channel blocks hold different sample counts "
                    f"({sorted(counts)}); the shortest is used.")
            self.n_samples = min(counts)

        if self.n_samples <= 0:
            raise FamosStructureError(
                f"{self.path.name}: the declared payload works out to "
                f"{self.n_samples} samples.")

        scaled = [c for c in self.components if not c.identity_calibration]
        if scaled:
            self.notes.append(
                f"{len(scaled)} channel(s) carry a non-identity |CR "
                f"calibration (e.g. {scaled[0].name!r}: "
                f"factor {scaled[0].factor:g}, offset {scaled[0].offset:g}). "
                f"It is REPORTED, NOT APPLIED -- a factor read from a guessed "
                f"field position rescales every impedance silently, and the "
                f"shunt calibration is meant to be the only absolute scale in "
                f"the chain. Confirm the field order before applying it.")

    # -- data ------------------------------------------------------------
    def channel(self, name: str, start: int = 0,
                stop: int | None = None) -> np.ndarray:
        """One channel, optionally only the samples in [start, stop)."""
        if name not in self.names:
            raise KeyError(f"{name!r} not in {self.path.name}: {self.names}")
        index = self.names.index(name)
        stop = self.n_samples if stop is None else min(int(stop),
                                                       self.n_samples)
        start = max(0, int(start))
        if stop <= start:
            return np.empty(0, dtype=np.float64)

        comp = self.components[index]
        dtype = comp.dtype
        if self.interleaved:
            column = comp.byte_offset // np.dtype(dtype).itemsize
            mm = np.memmap(self.path, dtype=dtype, mode="r",
                           offset=self.offset,
                           shape=(self.n_samples, self.n_ch))
            return np.asarray(mm[start:stop, column], dtype=np.float64)
        # Contiguous: this channel has its own block, so the range is a
        # straight slice of it rather than a stride through every channel.
        mm = np.memmap(self.path, dtype=dtype, mode="r",
                       offset=comp.data_offset, shape=(self.n_samples,))
        return np.asarray(mm[start:stop], dtype=np.float64)

    def position(self, name: str) -> int:
        return self.names.index(name)

    @property
    def segment_names(self) -> list[str]:
        return [n for n in self.names if re.fullmatch(r"\d+", n)]

    @property
    def uc_names(self) -> list[str]:
        return [n for n in self.names if n.upper().startswith("UC")]

    @property
    def temp_names(self) -> list[str]:
        return [n for n in self.names if n.lower().startswith("temp")]

    def describe(self) -> str:
        layout = ("interleaved in one block" if self.interleaved
                  else "one contiguous block per channel")
        return (f"{self.path.name}: FAMOS standard layout, {self.fs:.0f} Hz, "
                f"{self.n_ch} ch, {self.n_samples} samples "
                f"({self.n_samples / self.fs:.1f} s), "
                f"{self.components[0].dtype}, {layout}")


def _first_float(fields: list[str]) -> float:
    for token in fields:
        try:
            value = float(token)
        except ValueError:
            continue
        if value > 0:
            return value
    return float("nan")


def _ints(fields: list[str]) -> list[int]:
    out = []
    for token in fields:
        try:
            out.append(int(token))
        except ValueError:
            break
    return out


def _channel_name(fields: list[str]) -> str:
    """The name out of a |CN block.

    Layout is GroupIndex, BitIndex, GroupType, NameLength, Name,
    CommentLength, Comment -- so the name is field 4 and its length field 3.
    That is checked rather than trusted: if the declared length matches, it
    is the name.

    THE FALLBACK TAKES THE LAST MATCH, NOT THE FIRST.
    Scanning for "an integer followed by a string of that length" finds
    |CN,1,0,0,3,UC2,0 at field 0 as well -- "1" followed by "0", which is
    one character long -- and returns "0". Every channel then comes back
    named "0". The name pair is the last such pair in the block, because
    what follows it is the comment length and the comment.
    """
    if len(fields) > 4:
        try:
            if int(fields[3]) == len(fields[4]):
                return fields[4]
        except ValueError:
            pass
    best = ""
    for i, token in enumerate(fields[:-1]):
        try:
            want = int(token)
        except ValueError:
            continue
        if want <= 0:
            continue
        text, _ = _famos_string(fields, i)
        if text and len(text) == want:
            best = text
    if best:
        return best
    for token in reversed(fields):
        token = token.strip()
        if token and not re.fullmatch(r"[\d.\-+eE]+", token):
            return token
    return ""


def _index_prefix(content: str) -> int:
    """Bytes of the leading "<index>," in a |CS content, or 0 if absent."""
    match = re.match(r"\d+,", content)
    return len(match.group(0)) if match else 0


def _apply_cr(comp: Component, fields: list[str]) -> None:
    """Transform flag, factor and offset out of |CR."""
    nums = []
    for token in fields:
        try:
            nums.append(float(token))
        except ValueError:
            break
    if nums:
        comp.transform = int(nums[0])
    if len(nums) >= 3:
        comp.factor, comp.offset = nums[1], nums[2]
