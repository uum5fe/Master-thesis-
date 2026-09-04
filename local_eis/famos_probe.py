#!/usr/bin/env python3
r"""
famos_probe.py  --  why won't this .DAT open, and what IS its layout?
=====================================================================

    python famos_probe.py "\\server\share\RO2612025-01_Current_45A_Test_01_Karte_1.DAT"
    python famos_probe.py "...\Karte_1.DAT" --keys        # every key, verbatim
    python famos_probe.py "...\*.DAT"                     # a whole card set

`eis_local.FamosFile` raises ``incomplete FAMOS header`` when any one of four
keys is missing from the first 8192 bytes:

    |CD   the sample interval        -> fs
    |CR   (its second field is read as the channel count)
    |CP   the channel names, as ``7,32,<name>`` inside ONE field
    |CS   the data block

That message says which reader gave up. It does not say why, and the four
causes need completely different answers:

  a. THE HEADER IS LONGER THAN 8192 BYTES. A dialect that writes one |CN
     block per channel pushes |CS far down the file, and on a 60-channel card
     it lands well past the window the reader looks in. The keys are all
     there; the reader never sees them. This is a two-line fix.
  b. THE KEYS ARE THERE IN A DIFFERENT SHAPE. Names carried one-per-|CN
     rather than packed into a single |CP; the channel count somewhere other
     than |CR's second field. A reader change, not a big one.
  c. THE SAMPLES ARE NOT float32. `FamosFile.channel` hardcodes ``<f4`` and a
     stride of 4*n_ch. Read a float64 file that way and it does NOT fail --
     it returns a misaligned view of real numbers, which is the worst
     possible outcome for a measurement pipeline.
  d. IT IS NOT A FAMOS FILE AT ALL. `identify_file.py` settles that in one
     second; run it first if |CF is missing below.

This probe answers which. It reads the WHOLE header, not a window, reports
every key it finds with the byte offset it was found at, and cross-checks the
sample layout -- while saying plainly when that cross-check cannot decide,
which on a small signal riding a large DC level it often cannot. The sample
format is DECLARED in |CP; ``--keys`` prints it verbatim.

It reads. It never writes, and it never guesses past what it can show you.
Paste its output and the reader can be written against the file rather than
against a description of it.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

#: How much of the file to search for header keys. The data block starts
#: wherever |CS does; a header running past this is itself the finding.
HEAD_BYTES = 4 * 1024 * 1024

#: What the current reader looks at, and the cause (a) above turns on.
READER_WINDOW = 8192

KEY_MEANING = {
    "CF": "file format / version",
    "CK": "key block",
    "NO": "origin (software that wrote it)",
    "CB": "group",
    "CT": "text",
    "CI": "component index",
    "CG": "component group",
    "CD": "sample interval dx  ->  fs = 1/dx",
    "NT": "timestamp",
    "CC": "channel component",
    "CP": "packing: components, bits, dtype, offsets",
    "Cb": "buffer description",
    "CR": "calibration: factor, offset, unit",
    "ND": "display",
    "CN": "channel name block",
    "CS": "the sample data itself",
}


# ===========================================================================
# 1. The header
# ===========================================================================


def read_head(path: Path, n: int = HEAD_BYTES) -> bytes:
    with path.open("rb") as fh:
        return fh.read(n)


def scan_keys(raw: bytes) -> list[dict]:
    """Every ``|XX,`` in the file head, with where it was found.

    Deliberately a scan and not a strict parse of FAMOS's length framing.
    The framing is what a reader will eventually rely on, but a file that
    does not open is exactly the file whose framing might be the problem, so
    the probe must not depend on it to tell you what is there.
    """
    text = raw.decode("latin-1", errors="replace")
    out = []
    for m in re.finditer(r"\|([A-Za-z]{2}),", text):
        key = m.group(1)
        end = text.find(";", m.end())
        body = text[m.end():end if end != -1 else m.end() + 200]
        out.append({
            "key": key,
            "offset": m.start(),
            "len": (end - m.end()) if end != -1 else -1,
            "body": body,
            "terminated": end != -1,
        })
    return out


def framing_holds(raw: bytes) -> tuple[bool, str]:
    """Does ``|XX,<len>,<len bytes>;`` actually describe this file?

    FAMOS frames every key with the byte count of its data. Where that holds,
    a reader can walk the header exactly. Where it does not -- and the
    synthetic fixture in this repository does not, because it was written to
    satisfy a regex -- a reader has to scan. Worth knowing which, before
    writing one.
    """
    text = raw.decode("latin-1", errors="replace")
    pos = text.find("|")
    if pos < 0:
        return False, "no | at all"
    checked = 0
    while pos >= 0 and checked < 12:
        m = re.match(r"\|([A-Za-z]{2}),(\d+),", text[pos:])
        if not m:
            return False, f"key at byte {pos} is not |XX,<digits>,"
        data_start = pos + m.end()
        declared = int(m.group(2))
        end = data_start + declared
        if end >= len(text):
            return False, f"{m.group(1)} at byte {pos} declares {declared} bytes, past the head"
        if text[end] != ";":
            return (False, f"{m.group(1)} at byte {pos} declares {declared} bytes "
                           f"but byte {end} is {text[end]!r}, not ';'")
        checked += 1
        pos = text.find("|", end)
    return True, f"holds for the first {checked} keys"


# ===========================================================================
# 2. What the current reader would make of it
# ===========================================================================


def reader_verdict(raw: bytes) -> dict:
    """Exactly the four regexes `eis_local.FamosFile` runs, and where each
    would have matched had the window been big enough."""
    window = raw[:READER_WINDOW].decode("latin-1", errors="replace")
    whole = raw.decode("latin-1", errors="replace")
    probes = {
        "|CD (dx)": r"\|CD,\d+,([\d.eE+-]+)",
        "|CR (n_ch)": r"\|CR,\d+,(\d+)",
        "|CP (names)": r"\|CP,([^;]*);",
        "|CS (data)": r"\|CS,(\d+),(\d+),",
    }
    out = {}
    for label, pattern in probes.items():
        in_window = re.search(pattern, window)
        anywhere = re.search(pattern, whole)
        out[label] = {
            "in_reader_window": bool(in_window),
            "found_at": anywhere.start() if anywhere else None,
            "value": (anywhere.group(0)[:70] if anywhere else None),
        }
    return out


def channel_names(raw: bytes) -> dict:
    """Names by both conventions, without preferring either."""
    text = raw.decode("latin-1", errors="replace")
    packed = []
    m_cp = re.search(r"\|CP,([^;]*);", text)
    if m_cp:
        packed = [n.strip() for n in re.findall(r"7,32,([^,;]*)", m_cp.group(1))]
    per_cn = [n.strip() for n in
              re.findall(r"\|CN,[^,]*,[^,]*,[^,]*,[^,]*,\d+,([^,;]*)", text)]
    if not per_cn:
        # a looser sweep: the last comma-separated field of each |CN
        per_cn = []
        for m in re.finditer(r"\|CN,([^;]*);", text):
            parts = [p.strip() for p in m.group(1).split(",")]
            if parts:
                per_cn.append(parts[-1])
    return {"packed_in_one_CP": packed, "one_per_CN": [n for n in per_cn if n]}


# ===========================================================================
# 3. The samples: float32 or float64?
# ===========================================================================


def _smoothness(x: np.ndarray) -> float:
    """Ratio of step size to spread. Small means smooth means aligned.

    A physical signal sampled fast enough cannot jump arbitrarily, so
    successive samples are close. Read the same bytes at the wrong dtype or
    the wrong stride and successive values are uncorrelated, which puts this
    ratio near sqrt(2) -- the value for white noise.
    """
    x = x[np.isfinite(x)]
    if x.size < 64:
        return float("inf")
    s = float(np.std(x))
    if s == 0:
        return float("inf")
    return float(np.mean(np.abs(np.diff(x))) / s)


def _exponent_spread(x: np.ndarray) -> float:
    """How many decades the magnitudes are spread over.

    THE DISCRIMINATOR THAT SMOOTHNESS ALONE IS NOT.
    Reading float32 data as float64 pairs each two adjacent float32s into one
    float64, and the exponent of the result is whatever those bytes happen to
    encode -- so the values sprawl over tens of decades, with wild ones next
    to denormal ones. Measured on this repository's own fixture: the correct
    <f4 reading spreads over 0.5 decades, the <f8 misreading over 40.

    Smoothness does not catch it, because a misread of a SMOOTH signal is
    still locally correlated -- neighbouring garbage shares most of its bytes
    with the garbage before it. That is why the first version of this
    function ranked <f8 above <f4 on a file that is unambiguously <f4.
    """
    x = x[np.isfinite(x)]
    x = x[x != 0]
    if x.size < 64:
        return float("inf")
    return float(np.std(np.log10(np.abs(x))))


def score_dtype(path: Path, offset: int, n_ch: int, n_probe: int = 8192) -> list[dict]:
    """Rank the plausible sample layouts, on evidence rather than on a guess.

    This is the check that stops cause (c) from being silent. It reports the
    whole ranking, not a single answer, so a close call stays visible.
    """
    size = path.stat().st_size - offset
    rows = []
    for name, dt in (("<f4 (float32)", "<f4"), ("<f8 (float64)", "<f8"),
                     ("<i2 (int16)", "<i2"), ("<i4 (int32)", "<i4")):
        itemsize = np.dtype(dt).itemsize
        stride = itemsize * n_ch
        if n_ch <= 0 or size < stride * 128:
            continue
        n_rows = min(size // stride, n_probe)
        try:
            mm = np.memmap(path, dtype=dt, mode="r", offset=offset,
                           shape=(int(n_rows), n_ch))
            block = np.asarray(mm[:, :min(n_ch, 8)], dtype=np.float64)
        except Exception as exc:                        # pragma: no cover
            rows.append({"dtype": name, "error": str(exc)})
            continue
        finite = float(np.mean(np.isfinite(block)))
        with np.errstate(invalid="ignore", divide="ignore"):
            smooth = float(np.median([_smoothness(block[:, c])
                                      for c in range(block.shape[1])]))
            spread = float(np.median([_exponent_spread(block[:, c])
                                      for c in range(block.shape[1])]))
            amax = float(np.nanmax(np.abs(block))) if finite > 0 else float("nan")
        rows.append({
            "dtype": name, "stride_bytes": stride,
            "samples_per_channel": int(size // stride),
            "finite_frac": finite, "max_abs": amax, "smoothness": smooth,
            "decades": spread,
        })
    # Finite first, then a sane exponent range, then smoothness. The order
    # matters: a misread is locally smooth but never sits inside a couple of
    # decades, so testing smoothness first ranks the misread top.
    rows.sort(key=lambda r: (-(r.get("finite_frac", 0) > 0.999),
                             r.get("decades", float("inf")) > 3.0,
                             r.get("smoothness", float("inf"))))
    return rows


# ===========================================================================
# 4. Report
# ===========================================================================


def probe(path: Path, show_keys: bool = False) -> dict:
    raw = read_head(path)
    size = path.stat().st_size
    print("=" * 78)
    print(f"  {path.name}")
    print(f"  {size:,} bytes")
    print("=" * 78)

    if not raw.startswith(b"|CF,"):
        print("\n  DOES NOT START WITH |CF, so it is not a raw imc FAMOS file.")
        print(f"  first 32 bytes: {raw[:32]!r}")
        print("  -> run identify_file.py on it; the reader question is moot")
        print("     until the container is known.")
        return {"famos": False}

    keys = scan_keys(raw)
    counts: dict[str, int] = {}
    for k in keys:
        counts[k["key"]] = counts.get(k["key"], 0) + 1
    print(f"\n  KEYS FOUND ({len(keys)} in the first {len(raw):,} bytes)")
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        first = next(k for k in keys if k["key"] == key)
        print(f"    |{key} x{n:<5d} first at byte {first['offset']:>9,}   "
              f"{KEY_MEANING.get(key, '')}")

    ok, why = framing_holds(raw)
    print(f"\n  LENGTH FRAMING: {'holds' if ok else 'DOES NOT HOLD'} - {why}")

    print(f"\n  WHAT eis_local.FamosFile SEES (it reads only the first "
          f"{READER_WINDOW:,} bytes)")
    verdict = reader_verdict(raw)
    missing_window, missing_entirely = [], []
    for label, r in verdict.items():
        if r["found_at"] is None:
            missing_entirely.append(label)
            print(f"    {label:<14s} NOT PRESENT ANYWHERE in the head")
        elif not r["in_reader_window"]:
            missing_window.append(label)
            print(f"    {label:<14s} present at byte {r['found_at']:,} -- "
                  f"BEYOND the {READER_WINDOW:,}-byte window")
        else:
            print(f"    {label:<14s} ok at byte {r['found_at']:,}   "
                  f"{r['value']}")

    if missing_window and not missing_entirely:
        print("\n    -> CAUSE (a): the header is longer than the window the")
        print("       reader looks in. Every key it needs is present. Nothing")
        print("       about this file is unusual except its length.")
    elif missing_entirely:
        print("\n    -> CAUSE (b): a key the reader requires is genuinely not")
        print("       in the shape it expects. The key list above says what")
        print("       this file carries instead.")

    names = channel_names(raw)
    print("\n  CHANNEL NAMES")
    print(f"    packed into one |CP : {len(names['packed_in_one_CP'])}"
          + (f"   {', '.join(names['packed_in_one_CP'][:8])}"
             + (" ..." if len(names['packed_in_one_CP']) > 8 else "")
             if names["packed_in_one_CP"] else ""))
    print(f"    one per |CN         : {len(names['one_per_CN'])}"
          + (f"   {', '.join(names['one_per_CN'][:8])}"
             + (" ..." if len(names['one_per_CN']) > 8 else "")
             if names["one_per_CN"] else ""))

    text = raw.decode("latin-1", errors="replace")
    m_cd = re.search(r"\|CD,\d+,([\d.eE+-]+)", text)
    if m_cd:
        dx = float(m_cd.group(1))
        print(f"\n  SAMPLE INTERVAL  dx = {dx:g} s  ->  fs = {1 / dx:,.0f} Hz")

    m_cs = re.search(r"\|CS,(\d+),(\d+),", text)
    if m_cs:
        offset = m_cs.end()
        declared = int(m_cs.group(2))
        available = size - offset
        print(f"\n  DATA BLOCK")
        print(f"    starts at byte {offset:,}")
        print(f"    |CS declares {declared:,} bytes, {available:,} present"
              + ("   (the reader ignores the declared value and uses the "
                 "file size)" if abs(available - declared) > 64 else ""))

        n_ch = max(len(names["one_per_CN"]), len(names["packed_in_one_CP"]))
        m_cr = re.search(r"\|CR,\d+,(\d+)", text)
        n_cr = int(m_cr.group(1)) if m_cr else 0
        print(f"    channel count: {n_ch} from the name blocks, "
              f"{n_cr} from |CR's second field"
              + ("   -- THESE DISAGREE" if n_ch and n_cr and n_ch != n_cr
                 else ""))
        n_use = n_ch or n_cr
        if n_use:
            print(f"\n    SAMPLE LAYOUT, scored on {n_use} interleaved channels")
            print("    (smoothness near 1.4 is white noise = wrong alignment;")
            print("     a real signal sits far below it)")
            print(f"      {'dtype':<16s} {'stride':>7s} {'samples/ch':>12s} "
                  f"{'finite':>8s} {'max|x|':>12s} {'decades':>8s} "
                  f"{'smooth':>8s}")
            ranked = score_dtype(path, offset, n_use)
            for r in ranked:
                if "error" in r:
                    print(f"      {r['dtype']:<16s} {r['error']}")
                    continue
                print(f"      {r['dtype']:<16s} {r['stride_bytes']:>7d} "
                      f"{r['samples_per_channel']:>12,} "
                      f"{r['finite_frac']:>8.3f} {r['max_abs']:>12.4g} "
                      f"{r['decades']:>8.2f} {r['smoothness']:>8.3f}")

            # WHETHER THIS RANKING DECIDES ANYTHING IS ITSELF A FINDING.
            # A float32 record read as float64 pairs two adjacent samples
            # into one double. Where the signal is a small excursion on a
            # large DC level -- which is exactly what a segment shunt voltage
            # is -- every such pair has nearly the same bit pattern, so the
            # misread is ALSO narrow-banded and ALSO locally smooth. Both
            # candidates then tile the declared byte count exactly, and
            # nothing in the samples separates them.
            #
            # Saying "float64, probably" there would be worse than saying
            # nothing: it is the misalignment that returns plausible numbers,
            # and a pipeline that acts on it produces a spectrum rather than
            # an error.
            scored = [r for r in ranked if "error" not in r]
            decided = ""
            if len(scored) >= 2:
                a, b = scored[0], scored[1]
                clear = (abs(a["max_abs"]) > 1e-4
                         and (b["decades"] > a["decades"] + 1.0
                              or b["smoothness"] > 3 * a["smoothness"]
                              or abs(b["max_abs"]) < 1e-6))
                decided = a["dtype"] if clear else ""
            print()
            if decided:
                print(f"    -> the evidence separates them: {decided}")
            else:
                print("    -> AMBIGUOUS on the samples alone, and that is the")
                print("       honest answer rather than a coin toss. A float32")
                print("       record read as float64 pairs two adjacent samples,")
                print("       and where the signal is a small excursion on a big")
                print("       DC level -- a segment shunt voltage exactly -- the")
                print("       misread is just as narrow and just as smooth.")
                print("       FAMOS declares the sample format in |CP. Run again")
                print("       with --keys and read it there; that is evidence,")
                print("       this is only a cross-check.")
            print("\n    decades = powers of ten the magnitudes span; smooth =")
            print("    mean |step| over standard deviation, ~1.4 for white noise.")
            print("\n    eis_local.FamosFile hardcodes <f4 and a stride of")
            print("    4*n_ch. On a file that is not <f4 it does not fail -- it")
            print("    returns a misaligned view of real numbers, which is why")
            print("    this has to be settled before the reader is pointed at")
            print("    the campaign.")

    if show_keys:
        print("\n  EVERY KEY, VERBATIM")
        for k in keys:
            body = k["body"]
            if len(body) > 160:
                body = body[:160] + f" ... (+{len(k['body']) - 160} chars)"
            print(f"    @{k['offset']:>9,}  |{k['key']}  {body}")

    print()
    return {"famos": True, "keys": counts, "names": names, "verdict": verdict}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="one or more .DAT files; wildcards are expanded here "
                         "too, for shells that do not")
    ap.add_argument("--keys", action="store_true",
                    help="print every key verbatim - this is what to paste "
                         "when asking for a reader")
    a = ap.parse_args(argv)

    files: list[Path] = []
    for pattern in a.paths:
        hits = [Path(p) for p in glob.glob(pattern)]
        files.extend(hits or [Path(pattern)])
    if not files:
        print("famos_probe: nothing to probe")
        return 2
    for path in files:
        if not path.is_file():
            print(f"famos_probe: not a file: {path}")
            continue
        probe(path, show_keys=a.keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
