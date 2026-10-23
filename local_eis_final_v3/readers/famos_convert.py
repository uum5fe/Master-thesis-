#!/usr/bin/env python3
"""
famos_convert.py
================
Rewrite a FAMOS-v2 card recording (DASYLab 14: one |CN block per channel,
float64, the buffer column given by each channel's |CP byte offset) as a
FAMOS-v1 .DAT in DASYLab's own v1 layout (famos_write.py: every name in one
|CP, float32, interleaved in slot order) -- the layout datago imports
correctly. Written front to back in one pass (no memory-mapped writes).

WHY
---
Datago's import of a v2 card ignores the |CP byte offset and returns column
0 -- UC2 -- under every channel name (RO2612085 450 A Karte_3; see
runner/datago.py). The pipeline catches that and falls back to the original
.DAT, but the datago rows themselves stay wrong. Converting the card to v1
BEFORE the upload gives datago a file it reads correctly.

WHAT IS KEPT
------------
  * every channel and its name;
  * the column order = the multiplexer slot order: the pipeline corrects
    the slot-to-slot sampling delay from it (`position`), so the v1 file
    lists the channels in the v2 file's buffer order, not its |CN order;
  * the sample rate, and the trigger time (|NT) if the header has one;
  * the samples, as float32 (24-bit mantissa: ~6e-8 relative, far below a
    16-bit ADC step). The |CS byte count is written correctly.

The file is read through the pipeline's own reader (eis_local.FamosFile),
converted in chunks (a 16-channel card of a full sweep is several GB as
float64), read back with the v1 reader and compared channel by channel.
Keep the original v2 file: the converted one is a copy, not the recording.

    python famos_convert.py CARD_v2.DAT [-o OUT.DAT]      (default: OUT next
                                                           to it, _v1.DAT)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHUNK_ROWS = 1_000_000


def v1_header(names, fs: float, n_samples: int, start=None) -> bytes:
    """DASYLab's own v1 header (famos_write.dasylab_v1_header): the layout
    datago imports correctly, not merely one this pipeline can read."""
    from famos_write import dasylab_v1_header
    return dasylab_v1_header(names, fs, n_samples, start)


def convert(src, dst=None, chunk_rows: int = CHUNK_ROWS, check: bool = True,
            log=print) -> Path:
    """FAMOS v2 (or v1) -> FAMOS v1; returns the written path."""
    import eis_local
    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.stem + "_v1" + src.suffix)
    if dst.resolve() == src.resolve():
        raise ValueError("refusing to overwrite the original recording")
    fam = eis_local.FamosFile(src)
    names = sorted(fam.names, key=fam.position)          # slot order
    start = eis_local.famos_start_time(src)
    n = int(fam.n_samples)
    mm = np.memmap(src, dtype=getattr(fam, "_dtype", "<f4"), mode="r",
                   offset=fam.offset, shape=(n, fam.n_ch))
    cols = [fam.position(nm) for nm in names]
    with open(dst, "wb") as fh:
        fh.write(v1_header(names, fam.fs, n, start))
        for i in range(0, n, chunk_rows):
            fh.write(np.ascontiguousarray(mm[i:i + chunk_rows, cols],
                                          dtype="<f4").tobytes())
        from famos_write import TRAILER
        fh.write(TRAILER)
    del mm
    if log:
        log(f"  {src.name}: {type(fam).__name__}, {fam.n_ch} channels, "
            f"{n} samples at {fam.fs:g} Hz"
            + (f", start {start:%d.%m.%Y %H:%M:%S}" if start else "")
            + f" -> {dst.name} (FAMOS v1)")
    if check:
        verify(src, dst, log=log)
    return dst


def verify(src, dst, n_probe: int = 8, probe_len: int = 20000, log=print) -> None:
    """Read both back with the pipeline's readers and compare every channel
    on probes spread over the record; raises if anything differs."""
    import eis_local
    a, b = eis_local.FamosFile(src), eis_local.FamosFile(dst)
    if type(b).__name__ != "FamosV1":
        raise ValueError(f"{Path(dst).name} reads as {type(b).__name__}, not v1")
    problems = []
    if set(a.names) != set(b.names):
        problems.append(f"names differ: {sorted(set(a.names) ^ set(b.names))}")
    if a.n_samples != b.n_samples:
        problems.append(f"samples {a.n_samples} vs {b.n_samples}")
    if abs(a.fs - b.fs) > 1e-9 * a.fs:
        problems.append(f"fs {a.fs} vs {b.fs}")
    order_a = sorted(a.names, key=a.position)
    order_b = sorted(b.names, key=b.position)
    if order_a != order_b:
        problems.append("slot order changed")
    starts = np.linspace(0, max(a.n_samples - probe_len, 0), n_probe).astype(int)
    worst = 0.0
    for nm in a.names:
        if nm not in b.names:
            continue
        x, y = a.channel(nm), b.channel(nm)
        for s0 in starts:
            u, v = x[s0:s0 + probe_len], y[s0:s0 + probe_len]
            scale = max(float(np.max(np.abs(u))), 1e-30)
            worst = max(worst, float(np.max(np.abs(u - v))) / scale)
    if worst > 1e-6:
        problems.append(f"samples differ by up to {worst:.1e} (relative)")
    if problems:
        raise ValueError(f"{Path(dst).name}: " + "; ".join(problems))
    if log:
        log(f"  verified: {len(a.names)} channels, slot order, fs and "
            f"{n_probe} probes per channel identical (largest relative "
            f"difference {worst:.1e}, float32 rounding)")


def main(argv=None) -> int:
    import eis_paths  # noqa: F401  -- the pipeline's folders on sys.path
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, nargs="+", help="FAMOS v2 .DAT file(s)")
    ap.add_argument("-o", "--out", type=Path,
                    help="output file (one input) or folder (several)")
    a = ap.parse_args(argv)
    for s in a.src:
        dst = None
        if a.out:
            dst = a.out / s.name if (len(a.src) > 1 or a.out.is_dir()) else a.out
        convert(s, dst)
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
