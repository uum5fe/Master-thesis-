#!/usr/bin/env python3
"""
ddf_source.py  --  reading DASYLab .DDF measurement files
=========================================================

    python ddf_source.py probe "C:\\EIS_data\\2612025_27_08\\file.ddf"
    python ddf_source.py read  "C:\\EIS_data\\2612025_27_08\\file.ddf" --out out.csv

WHY A PROBE AND NOT JUST A READER
---------------------------------
DASYLab's DDF is proprietary and its byte layout is not published.  What IS
publicly established about it:

  * it is a binary container of BLOCKS, with a file header and a per-block
    header that together preserve the timing, so the data reads back exactly
    as it was stored;
  * the classic layout stores samples as float32, with a block size of at
    most 32768;
  * later versions extended it to double precision and larger blocks;
  * the header carries the channel count and the sampling rate.

That is enough to search for the layout but not enough to assume one, and a
reader that assumes the wrong offset does not fail loudly -- it returns
plausible-looking numbers that are actually a misaligned view of the file,
which is the single worst outcome for a measurement pipeline.  So `probe`
establishes the layout from the file itself and reports its evidence, and
`read` only runs once a layout is known.

HOW THE PROBE DECIDES
---------------------
Real measurement data is not random bytes.  Interpreted at the correct
offset, stride and dtype it is finite, bounded, and SMOOTH in time -- each
sample close to the one before it, because a physical signal sampled fast
enough cannot jump arbitrarily.  Interpreted at the wrong alignment it is
none of those: exponents come out absurd, NaNs appear, and successive values
are uncorrelated.  The probe scores every candidate layout on exactly that
and reports the ranked list rather than a single answer, so a close call is
visible instead of hidden.

IF THE PROBE CANNOT DECIDE
--------------------------
Two routes that need no reverse engineering at all, in order of preference:

  1. DASYLab writes ASCII directly.  Open the worksheet, put a "Write Data"
     module in ASCII mode on the same signals, re-run or replay the
     recording.  The result drops straight into this pipeline's existing CSV
     path (`--source csv`), which already handles a tab-separated logger
     export.  This is the supported route and it cannot be wrong about the
     byte layout, because DASYLab does the interpreting.

  2. NI's DataPlugin for DASYLab reads DDF into DIAdem, which exports TDMS,
     which `npTDMS` reads in Python.  More moving parts, but it keeps the
     full precision and the channel metadata.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


#: Sizes worth trying for the fixed header. DASYLab's own tools describe a
#: file header followed by block headers; these are the round numbers such
#: headers actually take, plus a fine sweep the search fills in around any
#: promising one.
COMMON_HEADER_SIZES = (0, 128, 256, 512, 1024, 2048, 4096, 8192)

DTYPES = {
    "float32-le": np.dtype("<f4"),
    "float32-be": np.dtype(">f4"),
    "float64-le": np.dtype("<f8"),
    "int16-le": np.dtype("<i2"),
}


@dataclass
class Layout:
    """One candidate interpretation of the file, and how well it holds up."""

    offset: int
    n_channels: int
    dtype_name: str
    score: float = 0.0
    finite_fraction: float = 0.0
    smoothness: float = float("inf")
    span: tuple[float, float] = (float("nan"), float("nan"))
    n_samples: int = 0
    #: What, other than looking smooth, agrees with this channel count.
    corroborated: str = ""

    def describe(self) -> str:
        why = f"  [{self.corroborated}]" if self.corroborated else ""
        return (f"offset {self.offset:>6}  {self.n_channels:>3} ch  "
                f"{self.dtype_name:<11}  score {self.score:6.3f}  "
                f"roughness {self.smoothness:8.3g}  "
                f"range [{self.span[0]:.4g}, {self.span[1]:.4g}]  "
                f"{self.n_samples} samples/ch{why}")


@dataclass
class Probe:
    path: Path
    size: int = 0
    ascii_header: str = ""
    binary_starts: int = 0
    numbers_in_header: dict = field(default_factory=dict)
    candidates: list[Layout] = field(default_factory=list)
    stride_measured: int = 0
    stride_strength: float = 0.0

    def report(self) -> str:
        lines = [f"file          : {self.path.name}",
                 f"size          : {self.size:,} bytes"]
        if self.ascii_header:
            lines.append(f"ascii header  : {len(self.ascii_header)} bytes of "
                         f"readable text before the binary starts at "
                         f"{self.binary_starts}")
            lines.append("")
            lines.append("--- header text ---")
            for line in self.ascii_header.splitlines()[:40]:
                lines.append(f"  {line[:160]}")
            lines.append("--- end header ---")
        else:
            lines.append("ascii header  : none found -- the file is binary "
                         "from byte 0")
        if self.numbers_in_header:
            lines.append("")
            lines.append("numbers named in the header (these usually ARE the "
                         "answer):")
            for k, v in self.numbers_in_header.items():
                lines.append(f"    {k} = {v}")

        lines.append("")
        if not self.candidates:
            lines.append("No layout scored well enough to propose. Send the "
                         "header text above; failing that, use one of the two "
                         "export routes in this module's docstring.")
            return "\n".join(lines)

        if self.stride_measured:
            lines.append(f"interleave period measured by autocorrelation: "
                         f"{self.stride_measured} "
                         f"({self.stride_strength:.1f} sigma above the rest "
                         f"of the curve)")
            lines.append("")
        lines.append("candidate layouts, best first:")
        for layout in self.candidates[:8]:
            lines.append(f"  {layout.describe()}")

        best = self.candidates[0]
        runner = self.candidates[1] if len(self.candidates) > 1 else None
        margin = best.score - runner.score if runner else float("inf")
        lines.append("")

        # The verdict leans on CORROBORATION rather than on the score alone.
        # The score is a heuristic and its weights are arguable; "the header
        # says 16 channels and the interleave period measures 16" is not. So
        # a corroborated leader is reported as the answer, and an
        # uncorroborated one never is, whatever it scored.
        if best.score < 0.55:
            lines.append(f"VERDICT: nothing scored convincingly (best "
                         f"{best.score:.3f}). Either the data is compressed, "
                         f"or the layout is not a plain interleaved array. "
                         f"Use the export route.")
        elif not best.corroborated:
            lines.append(f"VERDICT: the best-scoring layout is not "
                         f"corroborated by the header or by the measured "
                         f"interleave period, so it rests on smoothness "
                         f"alone -- which cannot tell N similar channels read "
                         f"as 1 from the truth. Use the export route.")
        else:
            lines.append(f"VERDICT: {best.n_channels} channels, "
                         f"{best.dtype_name}, data from byte {best.offset}")
            lines.append(f"  corroborated by : {best.corroborated}")
            lines.append(f"  value range     : {best.span[0]:.4g} .. "
                         f"{best.span[1]:.4g}  <-- CHECK THIS. It should look "
                         f"like the physical quantity you recorded (a cell "
                         f"voltage near 0.7 V, a shunt voltage in mV). If it "
                         f"does not, the layout is wrong however well it "
                         f"scored.")
            if runner is not None and margin < 0.05:
                lines.append(f"  close second    : {runner.n_channels} ch "
                             f"{runner.dtype_name} at {runner.offset}, range "
                             f"{runner.span[0]:.4g} .. {runner.span[1]:.4g} "
                             f"(margin {margin:.3f}) -- the ranges are what "
                             f"separate these, not the scores")
            lines.append("")
            lines.append(f"  read it with:  python ddf_source.py read "
                         f'"{self.path}" --offset {best.offset} '
                         f"--channels {best.n_channels} "
                         f"--dtype {best.dtype_name} --out out.tsv")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# finding the header
# ---------------------------------------------------------------------------

def split_header(raw: bytes, max_scan: int = 65536) -> tuple[str, int]:
    """The readable preamble, and where the binary body starts.

    DASYLab writes a text header before the blocks. Rather than assume its
    length, walk until the text stops looking like text: a run of bytes that
    are not printable and not whitespace is the body beginning.
    """
    limit = min(len(raw), max_scan)
    run = 0
    for i in range(limit):
        b = raw[i]
        printable = 32 <= b < 127 or b in (9, 10, 13)
        if printable:
            run = 0
            continue
        run += 1
        if run >= 8:                     # eight non-text bytes in a row
            end = i - run + 1
            return raw[:end].decode("latin-1", errors="replace"), end
    return raw[:limit].decode("latin-1", errors="replace"), limit


_KEYS = ("channel", "kanal", "kanäle", "kanaele", "rate", "abtast", "sample",
         "block", "size", "count", "anzahl", "freq", "delta", "interval")


def numbers_from_header(header: str) -> dict:
    """Key/value pairs whose key mentions something layout-relevant.

    DASYLab's header names the channel count and the sample rate outright.
    When it does, that beats anything the byte search can infer, so it is
    pulled out and shown first.
    """
    import re

    found: dict[str, str] = {}
    for line in header.splitlines():
        low = line.lower()
        if not any(k in low for k in _KEYS):
            continue
        m = re.search(r"([A-Za-zÄÖÜäöüß _.\-]{2,40})[\t:=;]+\s*"
                      r"([-+]?\d[\d ,.eE+\-]*)", line)
        if m:
            found[m.group(1).strip()] = m.group(2).strip()
    return found


# ---------------------------------------------------------------------------
# scoring a candidate layout
# ---------------------------------------------------------------------------

def score_layout(raw: bytes, offset: int, n_channels: int,
                 dtype_name: str, want: int = 4096) -> Layout | None:
    """How much does this interpretation look like a physical measurement?

    Three independent things have to hold at once, and noise passes none of
    them: the values are finite, they sit in a physically plausible range,
    and successive samples of one channel are CLOSE to each other. The last
    is the discriminating one -- a misaligned float array is white noise in
    the exponent, so its step-to-step change is the same size as its overall
    spread, while any real sampled signal changes far less between adjacent
    samples than it does over the record.
    """
    dtype = DTYPES[dtype_name]
    itemsize = dtype.itemsize
    avail = (len(raw) - offset) // (itemsize * n_channels)
    if avail < 64:
        return None
    take = min(avail, want)
    block = np.frombuffer(raw, dtype=dtype, count=take * n_channels,
                          offset=offset).reshape(take, n_channels)
    block = block.astype(np.float64, copy=False)

    # Reading arbitrary bytes as floats legitimately produces inf and nan --
    # that is the signal that the alignment is wrong, not a fault to warn
    # about. numpy's warnings here would be one line per wrong candidate.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _score_block(block, offset, n_channels, dtype_name, take)


def _score_block(block, offset, n_channels, dtype_name, take):
    finite = np.isfinite(block)
    finite_fraction = float(finite.mean())
    if finite_fraction < 0.99:
        return Layout(offset, n_channels, dtype_name, 0.0, finite_fraction,
                      float("inf"), (float("nan"), float("nan")), take)

    # PADDING IS NOT DATA. A header padded to a block boundary is a run of
    # exact zeros, and a run of exact zeros is maximally smooth -- so without
    # this the search settles a few hundred bytes short of the real offset and
    # reports a beautiful score for reading the padding. A real measurement
    # channel is essentially never exactly 0.0, so the fraction of exact zeros
    # separates the two cleanly. Visible in the candidate list as a range that
    # starts at 0 where the true offset's range starts at the signal level.
    zero_fraction = float(np.mean(block == 0.0))
    if zero_fraction > 0.5:
        return Layout(offset, n_channels, dtype_name, 0.0, finite_fraction,
                      float("inf"), (0.0, 0.0), take)

    lo, hi = float(np.min(block)), float(np.max(block))
    # A measurement channel carries volts, amps, bar or degrees, so its full
    # scale sits between roughly a millivolt and a few thousand. Grading this
    # rather than accepting anything under 1e12 is what separates the true
    # dtype from a wrong one that happens to be smooth: reading float32 data
    # as float64 pairs up adjacent samples into a number around 7e-4, which
    # is beautifully smooth and is not a reading of anything.
    magnitude = max(abs(lo), abs(hi))
    if magnitude == 0.0:
        plausible = 0.0
    else:
        decades = abs(np.log10(magnitude))
        # 1.0 within 1e-3 .. 1e3, falling away outside, 0 by 1e-9 / 1e9.
        plausible = float(np.clip((9.0 - decades) / 6.0, 0.0, 1.0))

    # Roughness: mean |x[n] - x[n-1]| against the channel's own spread.
    # ~0 for a smooth signal, ~1.4 for white noise.
    spread = np.std(block, axis=0)
    ok = spread > 0
    if not ok.any():
        return Layout(offset, n_channels, dtype_name, 0.0, finite_fraction,
                      float("inf"), (lo, hi), take)
    steps = np.mean(np.abs(np.diff(block[:, ok], axis=0)), axis=0)
    roughness = float(np.median(steps / spread[ok]))

    smooth_score = float(np.clip(1.0 - roughness / 1.4, 0.0, 1.0))
    score = (0.5 * smooth_score + 0.3 * plausible + 0.2 * finite_fraction
             - zero_fraction)
    return Layout(offset, n_channels, dtype_name, score, finite_fraction,
                  roughness, (lo, hi), take)


def stride_from_autocorrelation(raw: bytes, offset: int,
                                dtype_name: str = "float32-le",
                                max_stride: int = 256) -> tuple[int, float]:
    """The interleave period, measured rather than searched for.

    An interleaved multi-channel array is PERIODIC with period n_channels:
    sample k and sample k+n_channels are the same physical channel one tick
    apart, so they are highly correlated, while k and k+1 are different
    channels and are not. Autocorrelating the flat stream therefore puts a
    peak exactly at the channel count.

    This replaces guessing. Scoring candidate channel counts by smoothness
    cannot work on its own, because reading N similar channels as 1 channel
    also looks smooth when the channels resemble each other -- and on a
    segmented fuel cell every channel resembles every other. Periodicity does
    not have that failure: it is a property of the interleaving itself, not
    of how alike the channels happen to be.

    Returns (stride, strength) with strength the peak's height above the
    surrounding correlation, so a flat autocorrelation reports weakly.
    """
    dtype = DTYPES[dtype_name]
    n = min((len(raw) - offset) // dtype.itemsize, 1 << 16)
    if n < 4 * max_stride:
        return 0, 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.frombuffer(raw, dtype=dtype, count=n, offset=offset)
        x = x.astype(np.float64)
    if not np.all(np.isfinite(x)):
        return 0, 0.0
    x = x - x.mean()
    if x.std() == 0:
        return 0, 0.0

    spectrum = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:max_stride + 1]
    if ac[0] <= 0:
        return 0, 0.0
    ac = ac / ac[0]

    body = ac[1:]
    best = int(np.argmax(body)) + 1
    others = np.delete(body, best - 1)
    strength = float((body[best - 1] - np.median(others))
                     / (np.std(others) or 1.0))
    return best, strength


def first_nonzero_after(raw: bytes, start: int, limit: int = 1 << 20) -> int:
    """The byte index where the padding ends.

    A header padded to a block boundary reads as a run of zero floats, which
    score as perfectly smooth and drag the search to an offset short of where
    the data really starts. Scanned BYTE by byte rather than in dtype steps:
    the text length is not a multiple of the sample size, so stepping by 4
    from it lands one byte out and every candidate built on it is misaligned.
    """
    end = min(len(raw), start + limit)
    for i in range(start, end):
        if raw[i]:
            return i
    return start


def probe(path, channels: tuple[int, ...] | None = None,
          max_bytes: int = 4 << 20) -> Probe:
    """Work out how this file is laid out, and show the evidence."""
    path = Path(path)
    raw = path.read_bytes()[:max_bytes]
    out = Probe(path=path, size=path.stat().st_size)

    header, start = split_header(raw)
    out.ascii_header = header
    out.binary_starts = start
    out.numbers_in_header = numbers_from_header(header)

    # WHERE THE DATA STARTS. The text ends at `start`, but a header padded to
    # a block boundary leaves a run of zero floats after it, and zeros are
    # perfectly smooth, so they pull the search to an offset short of the real
    # one. Skipping the padding removes that whole family of near misses.
    data_start = first_nonzero_after(raw, start)
    offsets = sorted({0, start, data_start,
                      *(start + k for k in COMMON_HEADER_SIZES),
                      *COMMON_HEADER_SIZES})
    offsets = [o for o in offsets if 0 <= o < len(raw) - 1024]

    # HOW MANY CHANNELS. Two sources independent of smoothness, both better
    # than a scan: what the header says, and what the interleave period
    # measures. Smoothness alone CANNOT settle this -- reading N similar
    # channels as 1 also looks smooth, and on a segmented cell every channel
    # resembles every other -- so a candidate that only scores well loses to
    # one that is corroborated.
    named = []
    for key, value in out.numbers_in_header.items():
        if "chan" in key.lower() or "kan" in key.lower():
            try:
                named.append(int(float(value.replace(",", "."))))
            except ValueError:
                pass
    named = [c for c in named if 1 <= c <= 512]

    stride, strength = stride_from_autocorrelation(raw, data_start)
    out.stride_measured = stride
    out.stride_strength = strength
    measured = [stride] if stride and strength >= 4.0 else []

    tries = channels or tuple(dict.fromkeys(
        named + measured + [72, 80, 16, 8, 4, 2, 1] + list(range(1, 97))))

    results: list[Layout] = []
    for offset in offsets:
        for n_channels in tries:
            for dtype_name in DTYPES:
                got = score_layout(raw, offset, n_channels, dtype_name)
                if got is None or got.score <= 0:
                    continue
                if n_channels in named:
                    got.score += 0.35
                    got.corroborated = "header"
                if measured and n_channels == measured[0]:
                    got.score += 0.35
                    got.corroborated = ("header+autocorrelation"
                                        if got.corroborated
                                        else "autocorrelation")
                # FRAME ALIGNMENT. The data begins at data_start, so a correct
                # read has to be a whole number of frames from it. An offset
                # 16 bytes along reads the same samples with the channels
                # ROTATED -- every value real, every label wrong -- and scores
                # identically, because rotating channels that resemble each
                # other changes nothing a smoothness test can see. Nothing but
                # the alignment distinguishes them, so the alignment has to be
                # part of the score.
                frame = n_channels * DTYPES[dtype_name].itemsize
                if (got.offset - data_start) % frame == 0:
                    got.score += 0.20
                    got.corroborated = (f"{got.corroborated}+aligned"
                                        if got.corroborated else "aligned")
                results.append(got)
    # COLLAPSE EQUIVALENT OFFSETS. Two offsets that differ by a whole number
    # of frames (n_channels * itemsize) are the SAME layout read from a
    # different starting sample -- not two rival answers. Left uncollapsed
    # they fill the leaderboard with near-identical scores and trip the
    # "too close to call" guard on what is actually one confident result.
    # The representative kept is the first frame boundary at or after the end
    # of the padding, which starts on real data and loses none of it.
    best_of: dict[tuple[int, str, int], Layout] = {}
    for got in results:
        frame = got.n_channels * DTYPES[got.dtype_name].itemsize
        key = (got.n_channels, got.dtype_name, got.offset % frame)
        keep = best_of.get(key)
        if keep is None or got.score > keep.score:
            best_of[key] = got
    for key, got in best_of.items():
        frame = got.n_channels * DTYPES[got.dtype_name].itemsize
        if got.offset < data_start:
            bump = -(-(data_start - got.offset) // frame) * frame
            moved = score_layout(raw, got.offset + bump, got.n_channels,
                                 got.dtype_name)
            if moved is not None:
                moved.score = got.score
                moved.corroborated = got.corroborated
                best_of[key] = moved

    results = sorted(best_of.values(), key=lambda L: -L.score)
    out.candidates = results
    return out


# ---------------------------------------------------------------------------
# reading, once the layout is known
# ---------------------------------------------------------------------------

def read_ddf(path, offset: int, n_channels: int,
             dtype_name: str = "float32-le") -> np.ndarray:
    """The samples as (n_samples, n_channels). No block handling yet.

    This is deliberately the simplest thing that can work: one contiguous
    interleaved array after a fixed offset. If the file turns out to carry a
    per-block header between blocks -- which DASYLab's documentation says it
    can -- this will read those header bytes as samples and the probe's
    roughness score will have said so, because periodic garbage every N
    samples is not smooth. Do not extend this to strip block headers on a
    guess; get a file whose layout is confirmed first.
    """
    dtype = DTYPES[dtype_name]
    raw = Path(path).read_bytes()
    count = (len(raw) - offset) // (dtype.itemsize * n_channels) * n_channels
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
    return data.reshape(-1, n_channels).astype(np.float64)


def to_csv(path, out_path, offset: int, n_channels: int,
           dtype_name: str = "float32-le", names: list[str] | None = None
           ) -> Path:
    """Write the samples as a tab-separated file the CSV path can read."""
    data = read_ddf(path, offset, n_channels, dtype_name)
    names = names or [f"s{i + 1}" for i in range(n_channels)]
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(names) + "\n")
        for row in data:
            fh.write("\t".join(f"{v:.6g}" for v in row) + "\n")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="work out the layout")
    p_probe.add_argument("path", type=Path)
    p_probe.add_argument("--channels", type=int, nargs="*",
                         help="only try these channel counts")

    p_read = sub.add_parser("read", help="read with a known layout")
    p_read.add_argument("path", type=Path)
    p_read.add_argument("--offset", type=int, required=True)
    p_read.add_argument("--channels", type=int, required=True)
    p_read.add_argument("--dtype", default="float32-le", choices=list(DTYPES))
    p_read.add_argument("--out", type=Path,
                        help="write a tab-separated file instead of a summary")

    a = ap.parse_args(argv)
    if a.cmd == "probe":
        print(probe(a.path, tuple(a.channels) if a.channels else None).report())
        return 0

    data = read_ddf(a.path, a.offset, a.channels, a.dtype)
    if a.out:
        print(f"wrote {to_csv(a.path, a.out, a.offset, a.channels, a.dtype)}")
    else:
        print(f"{data.shape[0]} samples x {data.shape[1]} channels")
        print(f"range {np.min(data):.6g} .. {np.max(data):.6g}")
        print(data[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
