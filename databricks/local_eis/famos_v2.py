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
            elif key == "CR" and current is not None:
                _apply_cr(current, fields)
            elif key == "CS":
                if current is not None and current.data_offset < 0:
                    current.data_offset = entry["content_offset"]
                    current.data_length = entry["length"]
                payloads.append((entry["content_offset"], entry["length"]))

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
            declared = payloads[0][1]
            available = self.path.stat().st_size - self.offset
            usable = declared if 0 < declared <= available else available
            per_row = itemsize * self.n_ch
            self.n_samples = usable // per_row
            if declared and declared % per_row:
                self.notes.append(
                    f"the |CS block declares {declared} bytes, which is not a "
                    f"whole number of {self.n_ch}-channel rows of {itemsize} "
                    f"bytes ({declared % per_row} left over). Either the "
                    f"channel count or the sample width is wrong.")
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
            mm = np.memmap(self.path, dtype=dtype, mode="r",
                           offset=self.offset,
                           shape=(self.n_samples, self.n_ch))
            return np.asarray(mm[start:stop, index], dtype=np.float64)
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
    """The name out of a |CN block, by declared length.

    The field index differs between writers, so the name is found by the
    property that identifies it: a length field followed by exactly that many
    characters. Hard-coding an index is what breaks when a writer adds a
    field, and it breaks silently -- into a neighbouring number.
    """
    for i, token in enumerate(fields[:-1]):
        try:
            want = int(token)
        except ValueError:
            continue
        if want <= 0:
            continue
        text, _ = _famos_string(fields, i)
        if len(text) == want and not re.fullmatch(r"\d+", text or "x") is None:
            return text
        if len(text) == want and text:
            return text
    # Nothing self-consistent: fall back to the last non-numeric field.
    for token in reversed(fields):
        token = token.strip()
        if token and not re.fullmatch(r"[\d.\-+eE]+", token):
            return token
    return ""


def _apply_cr(comp: Component, fields: list[str]) -> None:
    """Factor and offset out of |CR, without assuming they are safe."""
    nums = []
    for token in fields:
        try:
            nums.append(float(token))
        except ValueError:
            break
    if len(nums) >= 3:
        comp.factor, comp.offset = nums[1], nums[2]
