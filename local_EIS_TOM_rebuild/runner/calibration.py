"""
calibration.py
==============
The plate calibration as Delta tables -- one VERSION per upload.

What a plate needs (core/site_paths.py lists it): curr.csv (72 rows c0;c1),
temp.csv (4 rows c0;c1), chain_gain.csv (the measuring-chain response) and
the Abgleich delivery it was built from (coefficients/, bode/, Step files).
Instead of reading them from EIS_config/<plate>/ on the Volume, publish them
once and every run takes them from here:

    <prefix>calibration_versions  one row per (plate, version): valid_from,
                                  source folder, note, checksums, published
    <prefix>calibration_files     every file of the version, byte for byte
                                  (path, chunk, n_chunks, n_bytes, sha, content)
    <prefix>calibration           the coefficients, readable with SQL:
                                  kind 'curr' (index = segment) / 'temp'
                                  (index = sensor), c0, c1
    <prefix>chain_gain            the chain response, readable with SQL:
                                  segment, freq_hz, gain_real, gain_imag

A recalibration is a NEW version; nothing is overwritten, so a run keeps the
calibration it was evaluated with and can be reproduced. Which version a run
uses: settings.CALIBRATION_VERSION when it pins one (per order, else per
plate), otherwise the newest version whose valid_from is on or before the
measurement date (the newest one when the date is unknown). The first version
of a plate is valid from 2000-01-01, i.e. for every earlier measurement.

The pipeline itself still reads FILES: `materialize` writes the version's
files, byte for byte, into a scratch folder and returns the same dictionary
core/site_paths.resolve() does, so bronze / silver / gold, the command line,
--reevaluate and the tests are unchanged. Because the bytes are identical, a
run's settings digest (which hashes chain_gain.csv) does not change when its
calibration moves from the Volume into the tables.

Every function takes `spark` explicitly.

    import runner
    runner.calibration.publish(spark, "gen1", "/Volumes/.../EIS_config/gen1")
    runner.calibration.versions(spark)
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings as S
from . import tables

VERSIONS = "calibration_versions"
FILES = "calibration_files"
COEFFS = "calibration"
GAIN = "chain_gain"
PARTITION = "plate"
#: valid_from of a plate's first version: every earlier measurement uses it
FIRST_VALID_FROM = "2000-01-01"
REQUIRED = ("curr.csv", "temp.csv")
#: written into a materialized folder, so the next run reuses it
MARK = ".calibration_version"


def table_name(suffix: str) -> str:
    return tables.table_name(suffix)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path) -> str:
    """Checksum of a calibration file ('' when there is none)."""
    try:
        return _sha(Path(path).read_bytes()) if path else ""
    except OSError:
        return ""


def _plate(plate: str) -> str:
    import r2d2_geometry as geom
    return geom.plate_key(plate)


# ---------------------------------------------------------------------------
# reading a calibration folder
# ---------------------------------------------------------------------------
def _pairs(path: Path) -> list[tuple[float, float]]:
    from eis_local import _read_pairs
    return _read_pairs(path)


def _files_of(folder: Path, include_raw: bool) -> list[Path]:
    """curr.csv, temp.csv, chain_gain.csv and the Abgleich delivery; the
    bode sweeps' *_Raw.DTA waveforms (not used by the pipeline, large) only
    with include_raw."""
    out = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.name == MARK or p.name.startswith("."):
            continue
        rel = p.relative_to(folder)
        top = rel.parts[0]
        if len(rel.parts) == 1 and p.name not in ("curr.csv", "temp.csv",
                                                  "chain_gain.csv", "README.md"):
            continue
        if len(rel.parts) > 1 and top != "Abgleich":
            continue
        if not include_raw and re.search(r"_raw\.dta$", p.name, re.I):
            continue
        out.append(p)
    return out


def check_folder(folder, plate: str) -> list[str]:
    """What is wrong with a calibration folder ([] when it can be published)."""
    import r2d2_geometry as geom
    folder = Path(folder)
    bad = [f"{n} missing" for n in REQUIRED if not (folder / n).is_file()]
    if bad:
        return bad
    n_seg = len(geom.plate(plate).segments)
    if len(_pairs(folder / "curr.csv")) != n_seg:
        bad.append(f"curr.csv has {len(_pairs(folder / 'curr.csv'))} rows, "
                   f"the {plate} plate has {n_seg} segments")
    if len(_pairs(folder / "temp.csv")) != 4:
        bad.append(f"temp.csv has {len(_pairs(folder / 'temp.csv'))} rows, "
                   f"expected 4 sensors")
    return bad


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------
def _append(spark, pdf: pd.DataFrame, suffix: str) -> None:
    if not len(pdf):
        return
    name = table_name(suffix)
    sdf = spark.createDataFrame(pdf)
    if not tables.exists(spark, name):
        (sdf.write.format("delta").mode("overwrite")
         .partitionBy(PARTITION).saveAsTable(name))
    else:
        (sdf.write.format("delta").mode("append")
         .option("mergeSchema", "true").saveAsTable(name))


def versions(spark, plate: str | None = None) -> pd.DataFrame:
    """Every published calibration version (of one plate), oldest first."""
    df = tables.read(spark, VERSIONS,
                     f"plate = {tables._sql_str(_plate(plate))}" if plate else None)
    if not len(df):
        return df
    df["version"] = df["version"].astype(int)
    return df.sort_values(["plate", "version"]).reset_index(drop=True)


def _rows(folder: Path, plate: str, version: int, files: list[Path]):
    key = {"plate": plate, "version": version}
    frows, shas = [], {}
    for p in files:
        data = p.read_bytes()
        rel = p.relative_to(folder).as_posix()
        shas[rel] = _sha(data)
        n = max(1, -(-len(data) // tables.CHUNK_BYTES))
        for i in range(n):
            frows.append({**key, "path": rel, "chunk": i, "n_chunks": n,
                          "n_bytes": len(data), "sha": shas[rel],
                          "content": bytes(data[i * tables.CHUNK_BYTES:
                                                (i + 1) * tables.CHUNK_BYTES])})
    crows = []
    for kind, name in (("curr", "curr.csv"), ("temp", "temp.csv")):
        for i, (c0, c1) in enumerate(_pairs(folder / name), start=1):
            crows.append({**key, "kind": kind, "index": i,
                          "c0": float(c0), "c1": float(c1)})
    gain = pd.DataFrame()
    if (folder / "chain_gain.csv").is_file():
        import utils
        g = utils.load_gain(folder / "chain_gain.csv")
        gain = pd.DataFrame([{**key, "segment": str(seg), "freq_hz": float(f),
                              "gain_real": float(z.real), "gain_imag": float(z.imag)}
                             for seg, (fs, zs) in g.items() for f, z in zip(fs, zs)])
    return pd.DataFrame(frows), pd.DataFrame(crows), gain, shas


def publish(spark, plate: str, folder, valid_from: str | None = None,
            note: str = "", include_raw: bool = False, say=print) -> int:
    """Publish the calibration folder of `plate` (curr.csv, temp.csv,
    chain_gain.csv, Abgleich/) as a NEW version and return its number. A
    folder identical to the newest version is not published again (that
    version is returned). Refused, with nothing written, when curr.csv or
    temp.csv is missing or has the wrong number of rows."""
    plate = _plate(plate)
    folder = Path(folder)
    bad = check_folder(folder, plate)
    if bad:
        raise ValueError(f"calibration {plate} in {folder} not published: "
                         + "; ".join(bad))
    files = _files_of(folder, include_raw)
    old = versions(spark, plate)
    shas_now = {p.relative_to(folder).as_posix(): _sha(p.read_bytes()) for p in files}
    if len(old):
        last = old.iloc[-1]
        prev = read_shas(spark, plate, int(last["version"]))
        if prev == shas_now:
            say(f"  calibration {plate}: unchanged -- version {int(last['version'])} "
                f"kept")
            return int(last["version"])
    version = int(old["version"].max()) + 1 if len(old) else 1
    if valid_from is None:
        valid_from = FIRST_VALID_FROM if version == 1 else date.today().isoformat()
    f_df, c_df, g_df, shas = _rows(folder, plate, version, files)
    # the version row last: a version listed there has all its rows
    _append(spark, f_df, FILES)
    _append(spark, c_df, COEFFS)
    _append(spark, g_df, GAIN)
    _append(spark, pd.DataFrame([{
        "plate": plate, "version": version, "valid_from": str(valid_from),
        "published": datetime.now().isoformat(timespec="seconds"),
        "source": str(folder), "note": str(note or ""),
        "curr_sha": shas.get("curr.csv", ""), "temp_sha": shas.get("temp.csv", ""),
        "gain_sha": shas.get("chain_gain.csv", ""),
        "has_abgleich": any(k.startswith("Abgleich/") for k in shas),
        "n_files": len(shas)}]), VERSIONS)
    say(f"  calibration {plate}: version {version} published (valid from "
        f"{valid_from}; {len(shas)} files, {len(c_df)} coefficients, "
        f"{len(g_df)} chain-gain points) in {S.TABLE_SCHEMA}")
    return version


def read_shas(spark, plate: str, version: int) -> dict:
    """{path: sha} of one version's files."""
    df = tables.read(spark, FILES, f"plate = {tables._sql_str(plate)} AND "
                                   f"version = {tables._sql_str(str(version))}")
    if not len(df):
        return {}
    return dict(zip(df["path"].astype(str), df["sha"].astype(str)))


# ---------------------------------------------------------------------------
# which version a run uses
# ---------------------------------------------------------------------------
def _as_date(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return pd.Timestamp(str(v)).date()
    except Exception:                                       # noqa: BLE001
        return None


def select(spark, plate: str, when=None, order: str | None = None,
           pins: dict | None = None):
    """The version row (a dict) a run of `plate` measured at `when` uses, or
    None when the plate has none. `pins` (settings.CALIBRATION_VERSION):
    {order: version} wins over {plate: version}."""
    plate = _plate(plate)
    v = versions(spark, plate)
    if not len(v):
        return None
    pins = S.CALIBRATION_VERSION if pins is None else pins
    pin = pins.get(str(order)) if order is not None else None
    if pin is None:
        pin = pins.get(plate)
    if pin is not None:
        hit = v[v["version"] == int(pin)]
        if not len(hit):
            raise ValueError(f"calibration {plate} version {pin} (pinned in "
                             f"CALIBRATION_VERSION) does not exist; versions: "
                             f"{sorted(v['version'].tolist())}")
        return hit.iloc[-1].to_dict()
    day = _as_date(when)
    if day is not None:
        ok = v[[(_as_date(x) or date.min) <= day for x in v["valid_from"]]]
        return (ok if len(ok) else v.iloc[:1]).iloc[-1].to_dict()
    return v.iloc[-1].to_dict()


def materialize(spark, plate: str, version: int, dest, say=print) -> dict:
    """Write one version's files into dest/<plate>/v<version>/ (reused when
    already there) and return what core/site_paths.resolve() returns, plus
    'source' = 'tables' and 'calibration_version'."""
    plate = _plate(plate)
    out_dir = Path(dest) / plate / f"v{int(version)}"
    mark = out_dir / MARK
    want = read_shas(spark, plate, version)
    if not want:
        raise ValueError(f"calibration {plate} version {version}: no files in "
                         f"{table_name(FILES)}")
    have = ({p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*")
             if p.is_file() and p.name != MARK} if out_dir.is_dir() else set())
    if not (mark.is_file() and mark.read_text().strip() == str(version)
            and have == set(want)
            and all(file_sha(out_dir / k) == s for k, s in want.items())):
        df = tables.read(spark, FILES, f"plate = {tables._sql_str(plate)} AND "
                                       f"version = {tables._sql_str(str(version))}")
        if out_dir.exists():
            import shutil
            shutil.rmtree(out_dir)
        for rel, rows in df.groupby("path", sort=True):
            p = out_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(tables._join(rows))
        mark.write_text(str(version))
        say(f"  calibration {plate} v{version}: {len(want)} files from "
            f"{table_name(FILES)}")
    res = {"plate": plate, "root": out_dir.parent, "target_dir": out_dir,
           "found_in": {}, "missing": [], "searched": [f"{table_name(FILES)} "
                                                       f"{plate} v{version}"],
           "source": "tables", "calibration_version": int(version)}
    for kind, name in (("curr", "curr.csv"), ("temp", "temp.csv"),
                       ("gain", "chain_gain.csv")):
        p = out_dir / name
        res[kind] = p if p.is_file() else None
        if res[kind] is not None:
            res["found_in"][kind] = str(out_dir)
        elif kind != "gain":
            res["missing"].append(kind)
    res["abgleich"] = (out_dir / "Abgleich" if (out_dir / "Abgleich" / "bode").is_dir()
                       else None)
    return res
