#!/usr/bin/env python3
"""
convert_karte_v2_to_v1.py
=========================
Convert ONE FAMOS-v2 card recording (DASYLab 14, e.g. Karte_3) into a
FAMOS-v1 .DAT, saved next to the original under a new name. Standalone:
needs only Python 3 and numpy, nothing from the pipeline.

HOW TO USE
----------
1. Put the full path of the Karte_3 file in KARTE_PATH below
   (or pass it on the command line: python convert_karte_v2_to_v1.py <path>).
2. Run the file (Databricks notebook cell, %run, or python).
3. The converted card is written into the SAME folder as

       <original name>_v1.DAT
       e.g.  RO2612085-01_Current_450A_Test_01_Karte_3.DAT
          -> RO2612085-01_Current_450A_Test_01_Karte_3_v1.DAT

   The original is never changed or overwritten.

IMPORTANT: if that folder is the one the pipeline reads its FAMOS cards
from, it now holds TWO Karte_3 files of the same test. Upload the _v1 file
to datago (or move one of the two elsewhere) before running the pipeline
on that folder.

THE FILE WRITTEN
----------------
Exactly the layout DASYLab's own v1 export uses (readers/famos_write.py;
byte for byte as on RO2612090 Karte_2), because datago imports that layout
correctly -- a merely "readable" v1 header is not enough:

    |CF,1;|CT,1;|KS;|CE,1;|CD,1,4.000000e-005,1,0,1,1,s;|CR,1,16,...;
    |NO,1,0;|NL,1,DASYLab;|NT,1,13,dd.mm.yyyy,132,hh:mm:ss.00;
    |CP,3,10,64,16,7,32,UC2     ,7,32,22      ,...;|CJ,...;|CS,1,<half>,
    <float32 samples, channel-interleaved>;|CA,1,0000000000;

(|CS states half the data bytes, as DASYLab writes it; the pipeline reads
the file size, so it prints the usual "|CS declares N bytes, 2N present".)
The file is written front to back in one pass -- no memory-mapped writes
and no rename, which a Databricks Volume may not allow.

WHAT IS KEPT
------------
every channel and its name; the column order = the multiplexer slot order
(taken from each channel's |CP byte offset, as the pipeline does -- the
pipeline corrects the slot-to-slot sampling delay from it); the sample rate;
the trigger time (|NT); the samples, as float32 (exact for a float32 v2
file, ~6e-8 relative for float64). The result is read back and compared
channel by channel before the script reports success.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# >>> put the path of the Karte_3 file here <<<
# ---------------------------------------------------------------------------
KARTE_PATH = r"/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Famos/RO2612085-01_Current_450A_Test_01_Karte_3.DAT"

SUFFIX = "_v1"                 # new name: <original stem><SUFFIX>.DAT
CHUNK_ROWS = 1_000_000         # rows converted at a time (memory)
HEADER_BYTES = 512_000         # v2 headers can run to hundreds of kB


# ---------------------------------------------------------------------------
# reading FAMOS v2
# ---------------------------------------------------------------------------
class V2Card:
    """The parts of a FAMOS-v2 file needed to rewrite it (same rules as the
    pipeline's eis_local.FamosV2)."""

    def __init__(self, path):
        self.path = Path(path)
        raw = self.path.open("rb").read(HEADER_BYTES)
        hdr = raw.decode("latin-1", errors="replace")
        if "|CN," not in hdr:
            raise ValueError(f"{self.path.name}: no |CN blocks -- not a FAMOS-v2 "
                             "file (a v1 card needs no conversion)")

        # names: one |CN per channel, the name is field 6
        names = []
        for m in re.finditer(r"\|CN,([^;]+);", hdr):
            f = m.group(1).split(",")
            names.append((f[6] if len(f) > 6 else f[-2]).strip())
        self.names = names
        self.n_ch = len(names)

        # sample interval: one |CD per channel, all equal
        dts = [float(m.group(1)) for m in
               re.finditer(r"\|CD,\d+,\d+,([\d.eE+-]+)", hdr)]
        if not dts:
            raise ValueError(f"{self.path.name}: no |CD sample interval")
        if max(dts) / min(dts) > 1.000001:
            raise ValueError(f"{self.path.name}: channels declare different "
                             "sample intervals -- not one interleaved block")
        self.fs = 1.0 / dts[0]

        # sample width: declared in |CP
        cp = re.search(r"\|CP,\d+,\d+,\d+,(\d+)", hdr)
        self.bytes_per_val = int(cp.group(1)) if cp else 8
        if self.bytes_per_val not in (4, 8):
            raise ValueError(f"{self.path.name}: {self.bytes_per_val} bytes "
                             "per value -- only 4 and 8 are supported")
        self.dtype = "<f8" if self.bytes_per_val == 8 else "<f4"
        row = self.bytes_per_val * self.n_ch

        # column of each channel: its own |CP byte offset (field 7)
        cps = [(m.start(), m.group(1).split(","))
               for m in re.finditer(r"\|CP,([^;]+);", hdr)]
        cols = []
        for m in re.finditer(r"\|CN,([^;]+);", hdr):
            prev = [f for p, f in cps if p < m.start()]
            try:
                f = prev[-1]
                nbytes, off = int(f[3]), int(f[7])
            except (IndexError, ValueError):
                cols = []
                break
            if nbytes != self.bytes_per_val or off % nbytes:
                cols = []
                break
            cols.append(off // nbytes)
        if len(cols) == self.n_ch and sorted(cols) == list(range(self.n_ch)):
            self.column, self.column_basis = cols, "|CP byte offset"
        else:
            self.column, self.column_basis = list(range(self.n_ch)), "|CN order"

        # where the samples start: the offset |CS's byte count confirms
        size = self.path.stat().st_size
        cs = raw.rfind(b"|CS,")
        if cs < 0:
            raise ValueError(f"{self.path.name}: no |CS data block")
        m_cs = re.match(rb"\|CS,(\d+),(\d+),", raw[cs:])
        declared = int(m_cs.group(2)) if m_cs else None
        cands = []
        if m_cs:
            cands.append(cs + m_cs.end())
        off = cs + 4
        for _ in range(3):
            nxt = raw.find(b",", off)
            if nxt < 0:
                break
            off = nxt + 1
        else:
            cands.append(off)
        chosen = next((c for c in cands if size - c == declared), None)
        if chosen is None:
            chosen = next((c for c in cands if size - c >= row), None)
        if chosen is None:
            raise ValueError(f"{self.path.name}: cannot locate the samples")
        self.offset = chosen
        self.n_samples = (size - chosen) // row
        self.declared = declared
        self.start = _start_time(hdr)

    def data(self) -> np.memmap:
        return np.memmap(self.path, dtype=self.dtype, mode="r",
                         offset=self.offset, shape=(self.n_samples, self.n_ch))


def _start_time(hdr: str):
    """|NT trigger time, v2 (d,m,y,h,m,s) or v1 (dd.mm.yyyy ... hh:mm:ss)."""
    m = re.search(r"\|NT,\d+,\d+,\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{4}),"
                  r"\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2})(?:\.\d+)?;", hdr)
    if not m:
        m = re.search(r"\|NT,\d+,\d+,\s*(\d{1,2})\.(\d{1,2})\.(\d{4}),[^,;]*,"
                      r"\s*(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?;", hdr)
    if not m:
        return None
    d, mo, y, h, mi, s = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# writing FAMOS v1 in DASYLab's own layout (= readers/famos_write.py)
# ---------------------------------------------------------------------------
TRAILER = b";|CA,1,0000000000;"
_CR_BLOCK = "0,2,-1E+030,1E+030,0,0,0,0,1,1,V"


def _sci3(x: float) -> str:
    """4e-05 as DASYLab writes it: 4.000000e-005 (three exponent digits)."""
    m, e = f"{x:.6e}".split("e")
    sign = "-" if e.startswith("-") else "+"
    return f"{m}e{sign}{abs(int(e)):03d}"


def _v1_header(names, fs, n_samples, start) -> bytes:
    n = len(names)
    start = start or datetime(1970, 1, 1)
    data_bytes = n_samples * n * 4
    hdr = ("|CF,1;|CT,1;|KS;|CE,1;"
           f"|CD,1,{_sci3(1.0 / fs)},1,0,1,1,s;"
           f"|CR,1,{n}," + ",".join([_CR_BLOCK] * n) + ";"
           "|NO,1,0;|NL,1,DASYLab;"
           f"|NT,1,13,{start:%d.%m.%Y},132,{start:%H:%M:%S}.00;"
           # 64 = 4 bytes x 16 channels on every DASYLab card seen
           f"|CP,3,10,64,{n}," + ",".join(f"7,32,{str(nm):<8}" for nm in names)
           + ";"
           f"|CJ,1,{data_bytes + len(TRAILER):010d};"
           f"|CS,1,{n_samples * n * 2:010d},")
    return hdr.encode("latin-1")


def _read_v1(path):
    """(names, fs, offset, n_samples) of a v1 file written here: the samples
    run from the |CS field to the trailer."""
    raw = Path(path).open("rb").read(65536).decode("latin-1", errors="replace")
    fs = 1.0 / float(re.search(r"\|CD,\d+,([\d.eE+-]+)", raw).group(1))
    names = [nm.strip() for nm in
             re.findall(r"7,32,([^,;]*)", re.search(r"\|CP,([^;]*);", raw).group(1))]
    off = re.search(r"\|CS,(\d+),(\d+),", raw).end()
    n = (Path(path).stat().st_size - off - len(TRAILER)) // (4 * len(names))
    return names, fs, off, n


def convert(src, suffix: str = SUFFIX) -> Path:
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = src.with_name(src.stem + suffix + src.suffix)
    if dst.resolve() == src.resolve():
        raise ValueError("the new name equals the original -- refusing to overwrite")

    card = V2Card(src)
    order = sorted(range(card.n_ch), key=lambda i: card.column[i])  # slot order
    names = [card.names[i] for i in order]
    cols = [card.column[i] for i in order]
    print(f"reading  {src}")
    print(f"  FAMOS v2: {card.n_ch} channels, {card.n_samples:,} samples at "
          f"{card.fs:g} Hz, {card.bytes_per_val * 8}-bit floats, columns from "
          f"{card.column_basis}"
          + (f", start {card.start:%d.%m.%Y %H:%M:%S}" if card.start else ""))
    print(f"  channels in slot order: {', '.join(names)}")

    mm = card.data()
    tmp = dst                  # written in place (Volumes may not allow a rename)
    with open(tmp, "wb") as fh:
        fh.write(_v1_header(names, card.fs, card.n_samples, card.start))
        for i in range(0, card.n_samples, CHUNK_ROWS):
            fh.write(np.ascontiguousarray(mm[i:i + CHUNK_ROWS, cols],
                                          dtype="<f4").tobytes())
        fh.write(TRAILER)

    # ---- read back and compare every channel ------------------------------
    v_names, v_fs, v_off, v_n = _read_v1(tmp)
    problems = []
    if v_names != names:
        problems.append("names / order differ")
    if v_n != card.n_samples:
        problems.append(f"{v_n} samples instead of {card.n_samples}")
    if abs(v_fs - card.fs) > 1e-9 * card.fs:
        problems.append(f"sample rate {v_fs} instead of {card.fs}")
    out = np.memmap(tmp, dtype="<f4", mode="r", offset=v_off,
                    shape=(v_n, len(v_names)))
    worst = 0.0
    starts = np.linspace(0, max(card.n_samples - 20000, 0), 8).astype(int)
    for j, c in enumerate(cols):
        for s0 in starts:
            a = np.asarray(mm[s0:s0 + 20000, c], dtype=np.float64)
            b = np.asarray(out[s0:s0 + 20000, j], dtype=np.float64)
            scale = max(float(np.max(np.abs(a))), 1e-30)
            worst = max(worst, float(np.max(np.abs(a - b))) / scale)
    del out, mm
    if worst > 1e-6:
        problems.append(f"samples differ by up to {worst:.1e}")
    if problems:
        tmp.unlink(missing_ok=True)
        raise ValueError("conversion check failed: " + "; ".join(problems))
    print(f"written  {dst}")
    print(f"  FAMOS v1 (DASYLab layout), checked: every channel identical on 8 probes "
          f"(largest relative difference {worst:.1e}, float32 rounding)")
    print("  NOTE: this folder now holds the original AND the _v1 card -- "
          "upload the _v1 file, or move one of them before the pipeline "
          "reads this folder.")
    return dst


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a.lower().endswith(".dat")]
    convert(args[0] if args else KARTE_PATH)
