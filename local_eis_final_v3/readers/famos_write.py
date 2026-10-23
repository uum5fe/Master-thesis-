"""
famos_write.py
==============
Write a FAMOS card in the exact layout DASYLab's v1 export uses -- the
layout datago imports correctly.

WHY. Datago reads the older DASYLab FAMOS files (one |CP listing every
channel name in column order) correctly, and the DASYLab 14 files (one |CN
block per channel, the column given by a |CP byte offset) wrongly: it puts
column 0 (UC2) under every name (RO2612085 / RO2612086 / RO2612087 Karte_3).
Converting a DASYLab 14 card to the old layout lets datago import all its
segments. The converted file must look like a real DASYLab v1 file, not
merely be readable by this pipeline: a header that says |CF,2 (FAMOS
version 2) but carries no |CN / |CG / |Cb blocks is neither format.

The layout, byte for byte as on RO2612090 Karte_2 (DASYLab, v1):

    |CF,1;|CT,1;|KS;|CE,1;
    |CD,1,4.000000e-005,1,0,1,1,s;                    sample interval
    |CR,1,16,0,2,-1E+030,1E+030,0,0,0,0,1,1,V, ...;   one block per channel
    |NO,1,0;|NL,1,DASYLab;
    |NT,1,13,21.10.2026,132,09:58:24.00;              start (PC local time)
    |CP,3,10,64,16,7,32,UC2     ,7,32,22      , ...;  names, padded to 8
    |CJ,1,0541065234;                                 data bytes + 18
    |CS,1,0270532608,                                 samples * channels * 2
    <float32 samples, channel-interleaved>
    ;|CA,1,0000000000;

(no separators between the keys; the |CS count is half the real data, as
DASYLab writes it -- eis_local.FamosV1 reads the file size instead.)

    python readers/famos_write.py IN.DAT OUT.DAT
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

TRAILER = b";|CA,1,0000000000;"
_CR_BLOCK = "0,2,-1E+030,1E+030,0,0,0,0,1,1,V"


def _sci3(x: float) -> str:
    """4e-05 as DASYLab writes it: 4.000000e-005 (three exponent digits)."""
    m, e = f"{x:.6e}".split("e")
    sign = "-" if e.startswith("-") else "+"
    return f"{m}e{sign}{abs(int(e)):03d}"


def dasylab_v1_header(names, fs: float, n_samples: int,
                      start: datetime | None = None) -> bytes:
    n = len(names)
    start = start or datetime(1970, 1, 1)
    data_bytes = n_samples * n * 4
    hdr = ("|CF,1;|CT,1;|KS;|CE,1;"
           f"|CD,1,{_sci3(1.0 / fs)},1,0,1,1,s;"
           f"|CR,1,{n}," + ",".join([_CR_BLOCK] * n) + ";"
           "|NO,1,0;|NL,1,DASYLab;"
           f"|NT,1,13,{start:%d.%m.%Y},132,{start:%H:%M:%S}.00;"
           f"|CP,3,10,64,{n}," + ",".join(f"7,32,{str(nm):<8}" for nm in names)
           + ";"
           f"|CJ,1,{data_bytes + len(TRAILER):010d};"
           f"|CS,1,{n_samples * n * 2:010d},")
    return hdr.encode("latin-1")


def write_dasylab_v1(path, names, channel, n_samples: int, fs: float,
                     start: datetime | None = None) -> Path:
    """Write `names` (in this column order) to `path`. `channel(name)` returns
    that channel's samples; one channel is held in memory at a time."""
    path = Path(path)
    hdr = dasylab_v1_header(names, fs, n_samples, start)
    n = len(names)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        fh.write(hdr)
        fh.truncate(len(hdr) + n_samples * n * 4)
    mm = np.memmap(tmp, dtype="<f4", mode="r+", offset=len(hdr),
                   shape=(n_samples, n))
    for j, nm in enumerate(names):
        x = np.asarray(channel(nm), dtype=np.float64)[:n_samples]
        if x.size < n_samples:
            raise ValueError(f"{nm}: {x.size} samples, {n_samples} expected")
        mm[:, j] = x.astype("<f4")
    mm.flush()
    del mm
    with open(tmp, "ab") as fh:
        fh.write(TRAILER)
    tmp.replace(path)
    return path


def convert(src, dst) -> Path:
    """Any FAMOS card this pipeline reads (v1 or DASYLab 14 / v2) -> the
    DASYLab v1 layout, same channels in the same column order, same start
    time. float32 samples (both DASYLab exports store float32)."""
    from eis_local import FamosFile, famos_start_time
    fam = FamosFile(src)
    names = sorted(fam.names, key=fam.position)
    return write_dasylab_v1(dst, names, fam.channel, fam.n_samples, fam.fs,
                            famos_start_time(src))


def convert_folder(src_dir, dst_dir, pattern="*.DAT", only_v2=True,
                   say=print) -> list[Path]:
    """Convert every DASYLab 14 (FAMOS v2) card under `src_dir` to the v1
    layout in `dst_dir`, keeping the file names. Cards already in the v1
    layout are skipped (only_v2) -- datago imports those correctly."""
    from eis_local import FamosFile, FamosV2
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for src in sorted(Path(src_dir).glob(pattern)):
        try:
            fam = FamosFile(src)
        except Exception as exc:                            # noqa: BLE001
            say(f"  {src.name}: not read ({exc})")
            continue
        if only_v2 and not isinstance(fam, FamosV2):
            continue
        out = convert(src, dst_dir / src.name)
        say(f"  {src.name}: DASYLab 14 -> v1 layout, {fam.n_ch} channels, "
            f"{fam.n_samples:,} samples -> {out}")
        done.append(out)
    return done


if __name__ == "__main__":                                  # pragma: no cover
    here = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(here))
    import eis_paths  # noqa: F401
    if len(sys.argv) != 3:
        raise SystemExit("usage: famos_write.py IN.DAT OUT.DAT")
    print(convert(sys.argv[1], sys.argv[2]))
