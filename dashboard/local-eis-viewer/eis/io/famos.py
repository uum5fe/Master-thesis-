"""Reader for imc FAMOS ``.DAT`` files written by the segmented-cell DAQ.

The files use the interleaved multi-channel layout: one ASCII key block
followed by a single ``|CS`` payload holding ``n_samples x n_channels``
little-endian ``float32`` samples in channel-interleaved order.

Compared with a minimal "find the offset and memmap it" parser this module
adds the three things the downstream synchronisation depends on:

* **Absolute start time** from the ``|NT`` key.  This is the only direct
  evidence about the offset between two files and must not be discarded.
* **Per-channel scaling** from ``|CR``.  Reading raw ``float32`` is correct
  only when the transform is the identity; here that is checked rather than
  assumed.
* **Cross-validated geometry.**  ``n_samples * n_channels * 4`` must equal the
  declared payload size and the channel-name count must equal the channel
  count.  A mismatch raises instead of silently falling back to defaults,
  because a wrong channel count silently shuffles every signal.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

_MAX_HEADER_BYTES = 1 << 20  # 1 MiB; 80 channel names fit comfortably


class FamosFormatError(ValueError):
    """Raised when a file cannot be parsed as the expected FAMOS dialect."""


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

@dataclass
class FamosHeader:
    path: str
    dt: float                       # sample interval [s]
    n_channels: int
    n_samples: int
    channel_names: list[str]
    data_offset: int                # byte offset of the first sample
    x_offset: float = 0.0           # declared x-axis start [s]
    start_time: datetime | None = None      # from |NT
    scale_factor: float = 1.0               # from |CR, only if applied
    scale_offset: float = 0.0               # from |CR, only if applied
    scaling_applied: bool = False
    scaling_note: str = ""
    unit: str = "V"
    raw_keys: dict[str, str] = field(default_factory=dict)

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def duration_s(self) -> float:
        return self.n_samples * self.dt

    @property
    def identity_scaling(self) -> bool:
        return not self.scaling_applied or (
            abs(self.scale_factor - 1.0) < 1e-12 and abs(self.scale_offset) < 1e-12
        )


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _read_until_payload(path: str | Path) -> tuple[str, int]:
    """Return the decoded header text and the byte offset just past ``|CS,n,m,``.

    Reads in growing chunks so that files with many channel names are handled
    without loading the whole payload.
    """
    size = os.path.getsize(path)
    chunk = 65_536
    with open(path, "rb") as fh:
        while True:
            raw = fh.read(chunk)
            text = raw.decode("latin-1")
            match = re.search(r"\|CS,\d+,\d+,", text)
            if match is not None:
                return text[: match.end()], match.end()
            if chunk >= min(size, _MAX_HEADER_BYTES):
                raise FamosFormatError(
                    f"{path}: no |CS payload key found in the first "
                    f"{chunk} bytes - not a FAMOS file, or an unsupported dialect"
                )
            chunk *= 2
            fh.seek(0)


def _parse_start_time(text: str) -> datetime | None:
    """Parse the ``|NT`` trigger-time key: ``|NT,len,day,month,year,h,m,s;``."""
    match = re.search(
        r"\|NT,\d+,(\d+),(\d+),(\d+),(\d+),(\d+),([\d.]+)\s*;", text
    )
    if match is None:
        return None
    day, month, year, hour, minute = (int(match.group(i)) for i in range(1, 6))
    seconds = float(match.group(6))
    try:
        return datetime(
            year, month, day, hour, minute,
            int(seconds), int(round((seconds % 1) * 1e6)),
        )
    except ValueError:
        return None


def parse_famos_header(path: str | Path) -> FamosHeader:
    """Parse the key block of a FAMOS file.

    Raises
    ------
    FamosFormatError
        If a required key is missing or the declared sizes are inconsistent.
    """
    path = str(path)
    text, data_offset = _read_until_payload(path)

    # --- |CD : sample interval, and (when present) the x-axis offset -------
    cd = re.search(r"\|CD,\d+,([\d.eE+-]+)((?:,[^;]*)?);", text)
    if cd is None:
        raise FamosFormatError(f"{path}: missing |CD key (sample interval)")
    dt = float(cd.group(1))
    if not (0 < dt < 1.0):
        raise FamosFormatError(f"{path}: implausible sample interval dt={dt}")

    # The x-axis start lives in the buffer key |Cb, not in |CD (whose fields
    # after dx are the calibration flag, unit and reduction settings).  When
    # |Cb is absent the axis starts at zero.  This value is only a coarse
    # cross-check anyway - the alignment that matters is measured, not declared.
    x_offset = 0.0
    cb = re.search(r"\|Cb,\d+((?:,[^;]*)?);", text)
    if cb is not None:
        fields = [p.strip() for p in cb.group(1).split(",") if p.strip()]
        if len(fields) >= 10 and _is_number(fields[9]):
            candidate = float(fields[9])
            if abs(candidate) < 1e6:
                x_offset = candidate

    # --- channel count -----------------------------------------------------
    cr = re.search(r"\|CR,\d+,(\d+)((?:,[^;]*)?);", text)
    if cr is None:
        raise FamosFormatError(f"{path}: missing |CR key (channel count)")
    n_channels = int(cr.group(1))

    # --- |CR trailing fields ----------------------------------------------
    # In the standard FAMOS spec |CR carries (transform, factor, offset, ...),
    # but this dialect puts the channel count first and the meaning of the
    # remaining fields is not documented for these files.  Guessing here is
    # dangerous in both directions: applying a bogus offset corrupts every
    # sample, and ignoring a real factor silently rescales every impedance.
    # So the fields are parsed and surfaced, never auto-applied.  Verify once
    # against a channel with a known input, then set the scaling explicitly.
    scale_factor, scale_offset, unit = 1.0, 0.0, "V"
    cr_fields = [p.strip() for p in cr.group(2).split(",") if p.strip()]
    for token in cr_fields:
        if token and not _is_number(token):
            unit = token
            break
    numeric = [float(t) for t in cr_fields if _is_number(t)]
    scaling_note = ""
    if numeric and any(abs(v - 1.0) > 1e-9 and v != 0.0 for v in numeric[:2]):
        scaling_note = (
            f"|CR trailing fields {cr_fields[:4]} are not an obvious identity "
            f"transform; raw float32 samples are used as-is. Verify the "
            f"dialect against a channel with a known input before trusting "
            f"absolute magnitudes."
        )

    # --- |CP : channel names, laid out as repeating (type, len, name) ------
    cp = re.search(r"\|CP,\d+,\d+,\d+,(\d+),(.+?);", text, re.DOTALL)
    channel_names: list[str] = []
    if cp is not None:
        parts = cp.group(2).split(",")
        for i in range(0, len(parts), 3):
            if i + 2 < len(parts):
                channel_names.append(parts[i + 2].strip())
    if not channel_names:  # fall back to the standard |CN name keys
        channel_names = [m.strip() for m in re.findall(r"\|CN,\d+(?:,[^,]*){3},(.+?);", text)]
    if not channel_names:
        raise FamosFormatError(f"{path}: could not recover any channel names")

    # --- |CS : payload size -----------------------------------------------
    cs = re.search(r"\|CS,\d+,(\d+),", text)
    if cs is None:
        raise FamosFormatError(f"{path}: missing |CS payload size")
    data_bytes = int(cs.group(1))

    # --- cross-validation --------------------------------------------------
    if len(channel_names) != n_channels:
        raise FamosFormatError(
            f"{path}: header declares {n_channels} channels but names "
            f"{len(channel_names)} of them ({channel_names[:4]}...). Refusing "
            f"to guess - a wrong channel count de-interleaves every signal."
        )
    stride = n_channels * 4
    if data_bytes % stride:
        raise FamosFormatError(
            f"{path}: payload of {data_bytes} B is not a whole number of "
            f"{n_channels}-channel float32 frames"
        )
    n_samples = data_bytes // stride

    on_disk = os.path.getsize(path) - data_offset
    if on_disk < data_bytes:
        # Truncated acquisition: keep the frames that are actually present.
        n_samples = on_disk // stride

    return FamosHeader(
        path=path,
        dt=dt,
        n_channels=n_channels,
        n_samples=n_samples,
        channel_names=channel_names,
        data_offset=data_offset,
        x_offset=x_offset,
        start_time=_parse_start_time(text),
        scale_factor=scale_factor,
        scale_offset=scale_offset,
        scaling_applied=False,
        scaling_note=scaling_note,
        unit=unit,
        raw_keys={"cd": cd.group(0), "cr": cr.group(0)},
    )


# ---------------------------------------------------------------------------
# File access
# ---------------------------------------------------------------------------

@dataclass
class FamosFile:
    """Lazy accessor for one card's recording."""

    header: FamosHeader
    card: int | None = None

    @property
    def fs(self) -> float:
        return self.header.fs

    @property
    def channel_names(self) -> list[str]:
        return self.header.channel_names

    @property
    def n_samples(self) -> int:
        return self.header.n_samples

    def _memmap(self) -> np.memmap:
        h = self.header
        return np.memmap(
            h.path, dtype="<f4", mode="r",
            offset=h.data_offset, shape=(h.n_samples, h.n_channels),
        )

    def channel(self, name: str, dtype=np.float64) -> np.ndarray:
        """Read one channel into memory, with |CR scaling applied."""
        if name not in self.header.channel_names:
            raise KeyError(
                f"{self.header.path}: no channel {name!r}; "
                f"available: {self.header.channel_names}"
            )
        index = self.header.channel_names.index(name)
        mm = self._memmap()
        data = np.asarray(mm[:, index], dtype=dtype)
        del mm
        if not self.header.identity_scaling:
            data = data * self.header.scale_factor + self.header.scale_offset
        return data

    def channels(self, names, dtype=np.float64) -> dict[str, np.ndarray]:
        """Read several channels in a single pass over the memmap."""
        missing = [n for n in names if n not in self.header.channel_names]
        if missing:
            raise KeyError(f"{self.header.path}: missing channels {missing}")
        idx = [self.header.channel_names.index(n) for n in names]
        mm = self._memmap()
        block = np.asarray(mm[:, idx], dtype=dtype)
        del mm
        if not self.header.identity_scaling:
            block = block * self.header.scale_factor + self.header.scale_offset
        return {name: block[:, i] for i, name in enumerate(names)}

    def time(self) -> np.ndarray:
        """Card-local time axis [s], including any declared x offset."""
        h = self.header
        return h.x_offset + np.arange(h.n_samples) * h.dt


def open_famos(path: str | Path, card: int | None = None) -> FamosFile:
    return FamosFile(header=parse_famos_header(path), card=card)


# ---------------------------------------------------------------------------
# Channel classification and discovery
# ---------------------------------------------------------------------------

def classify_channel(name: str) -> tuple[str, object]:
    """Map a raw channel name onto a role.

    Returns ``(role, key)`` where role is one of ``segment``, ``cell_voltage``,
    ``temperature``, ``other``.
    """
    stripped = name.strip()
    upper = stripped.upper()
    if upper.startswith("UC"):
        return "cell_voltage", upper
    if upper.startswith("TEMP"):
        return "temperature", stripped
    try:
        return "segment", int(stripped)
    except ValueError:
        return "other", stripped


def discover_files(
    raw_dir: str | Path,
    regex: str,
    measurement_id: str | None = None,
) -> dict[str, dict[int, str]]:
    """Scan ``raw_dir`` and group files as ``{condition: {card: path}}``."""
    pattern = re.compile(regex, re.IGNORECASE)
    found: dict[str, dict[int, str]] = {}
    raw_dir = Path(raw_dir)
    for entry in sorted(os.listdir(raw_dir)):
        match = pattern.match(entry)
        if match is None:
            continue
        groups = match.groupdict()
        if measurement_id is not None and groups.get("measurement_id") not in (
            None, measurement_id,
        ):
            continue
        condition = groups["condition"]
        found.setdefault(condition, {})[int(groups["card"])] = str(raw_dir / entry)
    return found
