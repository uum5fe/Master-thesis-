"""
tables.py
=========
The silver / gold layer as Delta tables -- one row set per run.

Every finished run is a folder EIS_Results/<order>[_gen2][_rot180]/
<condition>_Test<nn>[_<mode>]/. Publishing copies its result files into
tables in settings.TABLE_SCHEMA, so any number of runs can be read with one
query (e.g. R_ohmic against current for every order):

    <prefix>runs                 one row per run: settings digest, runner,
                                 sources, Gamry sweep, segments, folder
    <prefix>segment_meta         bronze/segment_meta.csv   (card, K, T, j_dc)
    <prefix>bronze_spectra       bronze/raw_spectra.csv
    <prefix>silver_spectra       silver/spectra_clean.csv
    <prefix>silver_reconstructed silver/spectra_reconstructed.csv
    <prefix>silver_segments      silver/segments_summary.csv
    <prefix>gold_segments        gold/plate_summary.csv
    <prefix>gold_plate           gold/plate_statistics.csv
    <prefix>gold_cell            silver/cell_aggregate.csv
    <prefix>gamry_reference      gamry_reference_full.csv
    <prefix>gamry_comparison     gamry_comparison.csv
    <prefix>checks               plausibility.csv
    <prefix>ecm                  ecm/ecm_parameters.csv
    <prefix>run_files            EVERY file of the run, byte for byte
                                 (path, chunk, n_chunks, n_bytes, mtime,
                                 content) -- CSV, JSON, PNG, HTML

Every row carries the run key -- `run_key` (= the folder's path under
EIS_Results, e.g. '2612090/450A_Test01'), `order_id`, `plate`,
`orientation`, `condition`, `current_a`, `test_id`, `eval_mode` -- and the
tables are partitioned by `order_id`.

Publishing a run REPLACES that run's rows in every table (replaceWhere on
run_key), exactly as a rerun replaces its folder: the tables never hold two
versions of one run. The runs row is written last, so a run listed in
<prefix>runs has all its tables.

Because <prefix>run_files holds the whole folder, a published run can be
written back into a folder (`restore`) and shown or re-evaluated from there
without the Volume: 02 Results (RESULTS_SOURCE = 'tables'), 01 Process in
Run mode cache (CACHE_FROM_TABLES) and 09 Maintenance (`rebuild`) do that.

ONE TYPE PER COLUMN. Delta cannot merge bigint with double or double with
string, and pandas guesses a CSV column's type from its values (L_nH = 12,
15, 9 reads as integers in one run, 12.4 as decimals in the next). So
`tidy` writes every number as double (only the counts of the runs row stay
bigint), True/False as boolean, everything else as string, and leaves out a
column that is empty in a run. Before a run is written, its columns are
checked against the existing tables: a column the table holds as text is
written as text; any other clash is refused before any of the run's rows
are touched, with the DROP TABLE that fixes it (09 Maintenance, "Rebuild").

Every function takes `spark` explicitly.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings as S

#: table suffix -> result file inside the run folder
FILES = {
    "segment_meta": "bronze/segment_meta.csv",
    "bronze_spectra": "bronze/raw_spectra.csv",
    "silver_spectra": "silver/spectra_clean.csv",
    "silver_reconstructed": "silver/spectra_reconstructed.csv",
    "silver_segments": "silver/segments_summary.csv",
    "gold_segments": "gold/plate_summary.csv",
    "gold_plate": "gold/plate_statistics.csv",
    "gold_cell": "silver/cell_aggregate.csv",
    "gamry_reference": "gamry_reference_full.csv",
    "gamry_comparison": "gamry_comparison.csv",
    "checks": "plausibility.csv",
    "ecm": "ecm/ecm_parameters.csv",
}
#: the table holding every file of a run
RUN_FILES = "run_files"
#: every table this module writes, the runs table first
ALL_TABLES = ("runs",) + tuple(FILES) + (RUN_FILES,)
KEY_COLUMNS = ("run_key", "order_id", "plate", "orientation", "condition",
               "current_a", "test_id", "eval_mode")
PARTITION = "order_id"
#: a file larger than this is stored in several rows (Spark and Delta handle
#: many moderate binary values far better than one huge one)
CHUNK_BYTES = 4 * 1024 * 1024
#: written into a folder restored from the tables (and never published)
RESTORED_MARK = ".restored_from_tables"


def table_name(suffix: str) -> str:
    return f"{S.TABLE_SCHEMA}.{S.TABLE_PREFIX}{suffix}"


# ---------------------------------------------------------------------------
# the run key, from the folder
# ---------------------------------------------------------------------------
_ORDER_DIR = re.compile(r"^(?P<order>[^_]+)(?P<tags>(?:_[A-Za-z0-9]+)*)$")
_COND_DIR = re.compile(r"^(?P<cond>[^_]+)(?:_Test(?P<test>\d+))?(?:_(?P<mode>[a-z]+))?$")


def run_key(run_dir) -> dict:
    """The key columns of the run saved in `run_dir`
    (.../EIS_Results/2612091_gen2_rot180/45A_Test01_strict)."""
    run_dir = Path(run_dir)
    order_dir, cond_dir = run_dir.parent.name, run_dir.name
    m = _ORDER_DIR.match(order_dir)
    tags = (m.group("tags") if m else "").split("_")
    c = _COND_DIR.match(cond_dir)
    cond = c.group("cond") if c else cond_dir
    cur = re.match(r"([0-9]+(?:[.,][0-9]+)?)\s*A$", cond)
    return {
        "run_key": f"{order_dir}/{cond_dir}",
        "order_id": m.group("order") if m else order_dir,
        "plate": "gen2" if "gen2" in tags else "gen1",
        "orientation": "rot180" if "rot180" in tags else "standard",
        "condition": cond,
        "current_a": float(cur.group(1).replace(",", ".")) if cur else np.nan,
        "test_id": (c.group("test") or "") if c else "",
        "eval_mode": (c.group("mode") or "default") if c else "default",
    }


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:                                       # noqa: BLE001
        return {}


def run_record(run_dir) -> dict:
    """One row of the runs table, from the folder's identity and manifest."""
    run_dir = Path(run_dir)
    ident = _json(run_dir / "run_identity.json")
    man = _json(run_dir / "run_manifest.json")
    cfg = _json(run_dir / "config_used.json")
    gold = (man.get("stages") or {}).get("gold") or {}
    bronze = (man.get("stages") or {}).get("bronze") or {}
    gam = (man.get("stages") or {}).get("gamry") or []
    gam = gam[0] if isinstance(gam, list) and gam else {}
    verdicts = [r.get("verdict") for r in
                ((man.get("stages") or {}).get("plausibility") or [])
                if isinstance(r, dict)]
    src = ident.get("famos_source")
    return {
        **run_key(run_dir),
        "digest": ident.get("digest", ""),
        "runner": ident.get("runner", ""),
        "settings_json": json.dumps(ident.get("settings", {}), sort_keys=True),
        "famos_source": json.dumps(src) if isinstance(src, dict) else str(src or ""),
        "cards": int(ident.get("cards") or len(bronze.get("cards") or []) or 0),
        "gamry_sweep": str(gam.get("reference") or ""),
        "gamry_version": str(cfg.get("gamry_version") or ""),
        "n_measured": int(gold.get("n_measured") or 0),
        "n_inferred": int(gold.get("n_inferred") or 0),
        "n_substituted": int(gold.get("n_substituted") or 0),
        "plausibility_fail": int(sum(v == "fail" for v in verdicts)),
        "plausibility_warn": int(sum(v == "warn" for v in verdicts)),
        "elapsed_s": float(ident.get("elapsed_s") or man.get("elapsed_s") or 0.0),
        "written": str(ident.get("written") or ""),
        "results_dir": str(run_dir),
        "published": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# folder -> frames
# ---------------------------------------------------------------------------
def _clean_name(c: str) -> str:
    c = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
    return c if c and not c[0].isdigit() else f"c_{c}"


_TRUE = {"true", "True", "TRUE"}
_FALSE = {"false", "False", "FALSE"}


def _missing(v) -> bool:
    try:
        return v is None or bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _as_bool(s: pd.Series):
    """The column as Python True / False / None, or None when it is not a
    boolean column (every non-empty value True/False, as bool or as text)."""
    vals = [v for v in s.tolist() if not _missing(v)]
    if not vals:
        return None
    if not all(isinstance(v, (bool, np.bool_)) or v in _TRUE or v in _FALSE
               for v in vals):
        return None
    return pd.Series([None if _missing(v) else
                      (bool(v) if isinstance(v, (bool, np.bool_)) else v in _TRUE)
                      for v in s.tolist()], index=s.index, dtype=object)


def _text(s: pd.Series) -> pd.Series:
    """Text with None for gaps, as plain Python objects (pandas 3's own str
    type would turn None into NaN)."""
    return pd.Series([None if _missing(v) else str(v) for v in s.tolist()],
                     index=s.index, dtype=object)


def tidy(df: pd.DataFrame, key: dict, integers: bool = False) -> pd.DataFrame:
    """Key columns first, column names Delta accepts, one type per column:
    every number double (`integers=True` keeps whole-number columns as
    integers -- the counts of the runs row), True/False boolean, everything
    else text; a column with no value at all is left out (Delta fills it
    with null), so it cannot fix a type the next run contradicts."""
    df = df.copy()
    df.columns = [_clean_name(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    out = {}
    for c in df.columns:
        s = df[c]
        if s.isna().all():
            continue
        if str(s.dtype) == "bool":
            out[c] = s.astype(bool)
            continue
        numeric = pd.api.types.is_numeric_dtype(s.dtype)
        b = None if numeric else _as_bool(s)
        if b is not None:
            out[c] = b
        elif numeric:
            if integers and pd.api.types.is_integer_dtype(s.dtype):
                out[c] = s.astype("int64")
            else:
                out[c] = s.astype("float64")
        else:
            out[c] = _text(s)
    df = pd.DataFrame(out, index=df.index)
    for k in reversed(KEY_COLUMNS):
        if k in df.columns:
            df = df.drop(columns=k)
        df.insert(0, k, key[k])
    return df


def frames(run_dir) -> dict:
    """{table suffix: DataFrame} of one saved run (missing files skipped).
    The files themselves are in `file_rows`."""
    run_dir = Path(run_dir)
    key = run_key(run_dir)
    out = {"runs": tidy(pd.DataFrame([run_record(run_dir)]), key, integers=True)}
    for suffix, rel in FILES.items():
        p = run_dir / rel
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:                                   # noqa: BLE001
            continue
        if len(df):
            out[suffix] = tidy(df, key)
    return out


def file_rows(run_dir) -> pd.DataFrame:
    """Every file of the run folder, byte for byte, one row per chunk."""
    run_dir = Path(run_dir)
    key = run_key(run_dir)
    rows = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file() or p.name == RESTORED_MARK:
            continue
        data = p.read_bytes()
        n = max(1, -(-len(data) // CHUNK_BYTES))
        rel = p.relative_to(run_dir).as_posix()
        mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        for i in range(n):
            rows.append({**key, "path": rel, "chunk": i, "n_chunks": n,
                         "n_bytes": len(data), "mtime": mtime,
                         "content": bytes(data[i * CHUNK_BYTES:(i + 1) * CHUNK_BYTES])})
    cols = list(KEY_COLUMNS) + ["path", "chunk", "n_chunks", "n_bytes", "mtime",
                                "content"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# writing and reading
# ---------------------------------------------------------------------------
def exists(spark, name: str) -> bool:
    try:
        return bool(spark.catalog.tableExists(name))
    except Exception:                                       # noqa: BLE001
        return False


def _sql_str(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


_INTEGRAL = {"tinyint", "smallint", "int", "bigint"}
_FRACTIONAL = {"float", "double", "decimal"}


class TypeClash(ValueError):
    """A run's column cannot take the type the table already has."""


def _table_types(spark, name: str) -> dict:
    """{column: Spark type} of an existing table ({} when it cannot be read)."""
    try:
        fields = spark.read.table(name).schema.fields
    except Exception:                                       # noqa: BLE001
        return {}
    return {f.name: f.dataType.simpleString() for f in fields}


def _align(spark, pdf: pd.DataFrame, name: str) -> pd.DataFrame:
    """The run's rows in the types the table already has. A column the table
    holds as text is written as text; a whole-number column the table holds
    as bigint (tables written before every number became double) stays
    integral; any other clash raises TypeClash before anything is written."""
    have = _table_types(spark, name) if exists(spark, name) else {}
    if not have:
        return pdf
    pdf = pdf.copy()
    for c in pdf.columns:
        t = have.get(c)
        if t is None or c in KEY_COLUMNS:
            continue
        s = pdf[c]
        base = t.split("(")[0]
        if base == "string":
            pdf[c] = _text(s)
            continue
        if base in _FRACTIONAL and pd.api.types.is_numeric_dtype(s.dtype) \
                and str(s.dtype) != "bool":
            pdf[c] = s.astype("float64")
            continue
        if base in _INTEGRAL and pd.api.types.is_numeric_dtype(s.dtype) \
                and str(s.dtype) != "bool":
            v = s.dropna()
            if (v == v.round()).all():
                # plain ints, None for gaps (pandas' nullable Int64 is not
                # something every Spark conversion path accepts)
                pdf[c] = (s.astype("int64") if not s.isna().any() else
                          s.map(lambda x: None if _missing(x) else int(x))
                          .astype(object))
                continue
        if base == "boolean" and not pd.api.types.is_numeric_dtype(s.dtype) \
                and _as_bool(s) is not None:
            continue
        if base == "boolean" and str(s.dtype) == "bool":
            continue
        if base == "binary":
            continue
        raise TypeClash(
            f"{name}.{c}: the table holds {t}, this run has "
            f"{s.dtype if pd.api.types.is_numeric_dtype(s.dtype) else 'text'}. "
            f"Delta cannot merge the "
            f"two. Drop the table once and publish again (09 Maintenance, "
            f"'Rebuild all tables', or: DROP TABLE {name})")
    return pdf


def _write(spark, pdf: pd.DataFrame, name: str, key: str) -> None:
    sdf = spark.createDataFrame(pdf)
    where = f"run_key = {_sql_str(key)}"
    if not exists(spark, name):
        (sdf.write.format("delta").mode("overwrite")
         .partitionBy(PARTITION).saveAsTable(name))
        return
    try:
        (sdf.write.format("delta").mode("overwrite")
         .option("replaceWhere", where).option("mergeSchema", "true")
         .saveAsTable(name))
    except Exception:                                       # noqa: BLE001
        # older runtimes: replaceWhere only on partition columns
        spark.sql(f"DELETE FROM {name} WHERE {where}")
        (sdf.write.format("delta").mode("append")
         .option("mergeSchema", "true").saveAsTable(name))


def publish(spark, run_dir, say=print) -> dict:
    """Write one saved run into every table, every file into run_files; its
    earlier rows are replaced. Returns {table suffix: rows written}.
    A type clash with an existing table is refused before anything is
    written, so the run's old rows stay intact."""
    fr = frames(run_dir)
    fr[RUN_FILES] = file_rows(run_dir)
    key = fr["runs"]["run_key"].iloc[0]
    aligned = {s: _align(spark, df, table_name(s)) for s, df in fr.items()}
    done = {}
    # results and files first, the runs row last: a run listed in
    # <prefix>runs has all its tables written
    for suffix in [s for s in aligned if s != "runs"] + ["runs"]:
        if not len(aligned[suffix]):
            continue
        _write(spark, aligned[suffix], table_name(suffix), key)
        done[suffix] = len(aligned[suffix])
    n_files = int(fr[RUN_FILES]["path"].nunique()) if len(fr[RUN_FILES]) else 0
    say(f"    tables: {key} -> {len(done)} tables "
        f"({sum(v for s, v in done.items() if s != RUN_FILES):,} rows, "
        f"{n_files} files) in {S.TABLE_SCHEMA}")
    return done


def delete_run(spark, key: str, say=print) -> list:
    """Remove one run (its run_key) from every table. No undo."""
    gone = []
    for suffix in ALL_TABLES:
        name = table_name(suffix)
        if exists(spark, name):
            spark.sql(f"DELETE FROM {name} WHERE run_key = {_sql_str(key)}")
            gone.append(name)
    say(f"  {key}: removed from {len(gone)} tables")
    return gone


def read(spark, suffix: str, where: str | None = None) -> pd.DataFrame:
    """A table as pandas ('' filter = all rows); empty when it does not exist."""
    name = table_name(suffix)
    if not exists(spark, name):
        return pd.DataFrame()
    df = spark.read.table(name)
    if where:
        df = df.where(where)
    return df.toPandas()


def runs(spark) -> pd.DataFrame:
    """The runs table, newest first."""
    df = read(spark, "runs")
    if len(df) and "published" in df:
        df = df.sort_values("published", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# the files of a run, back out of run_files
# ---------------------------------------------------------------------------
def _join(rows: pd.DataFrame) -> bytes:
    rows = rows.sort_values("chunk") if "chunk" in rows else rows
    return b"".join(bytes(c) for c in rows["content"].tolist() if c is not None)


def file_bytes(spark, key: str, path: str) -> bytes | None:
    """One file of a published run (its path inside the run folder), or None."""
    df = read(spark, RUN_FILES, f"run_key = {_sql_str(key)} AND path = {_sql_str(path)}")
    return _join(df) if len(df) else None


def restore(spark, key: str, dest, published: str | None = None,
            say=print) -> Path | None:
    """Write every file of the published run `key` into the folder `dest`
    (replacing what is there) and return it; None when run_files does not
    hold the run. With `published` (the runs table's time stamp) a folder
    already restored from that publication is reused as it is."""
    dest = Path(dest)
    mark = dest / RESTORED_MARK
    if published is not None and mark.is_file():
        try:
            if mark.read_text().strip() == str(published):
                return dest
        except OSError:
            pass
    df = read(spark, RUN_FILES, f"run_key = {_sql_str(key)}")
    if not len(df):
        return None
    if dest.exists():
        shutil.rmtree(dest)
    for rel, rows in df.groupby("path", sort=True):
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_join(rows))
    mark.write_text(str(published or ""))
    say(f"    {key}: restored from {table_name(RUN_FILES)} "
        f"({df['path'].nunique()} files) into {dest}")
    return dest


# ---------------------------------------------------------------------------
# maintenance: drop everything, publish everything again
# ---------------------------------------------------------------------------
def _usable(d: Path) -> bool:
    return any((d / s / "spectra_clean.csv").is_file() for s in ("silver", "csv"))


def run_folders(root) -> dict:
    """{run_key: folder} of every saved run under the results root."""
    root = Path(root)
    out = {}
    try:
        orders = sorted(d for d in root.iterdir() if d.is_dir())
    except OSError:
        return out
    for o in orders:
        try:
            conds = sorted(d for d in o.iterdir() if d.is_dir())
        except OSError:
            continue
        for d in conds:
            if _usable(d):
                out[run_key(d)["run_key"]] = d
    return out


def drop_all(spark, say=print) -> list:
    """DROP every local_eis table. No undo."""
    gone = []
    for suffix in ALL_TABLES:
        name = table_name(suffix)
        if exists(spark, name):
            spark.sql(f"DROP TABLE {name}")
            gone.append(name)
    say(f"  dropped {len(gone)} tables")
    return gone


def republish_all(spark, root, say=print) -> dict:
    """Publish every saved run folder under `root`. {run_key: 'ok' | error}."""
    out = {}
    for key, d in sorted(run_folders(root).items()):
        try:
            publish(spark, d, say=say)
            out[key] = "ok"
        except Exception as exc:                            # noqa: BLE001
            out[key] = f"{type(exc).__name__}: {exc}"
    return out


def rebuild(spark, root=None, scratch=None, say=print) -> dict:
    """Drop every table and publish every run again -- from its folder under
    `root` (default settings.RESULTS_ROOT) or, when that folder is gone, from
    its copy in run_files, read back into `scratch` BEFORE anything is
    dropped. A published run that is in neither place stops the rebuild with
    nothing dropped: {run_key: 'no folder, no run_files copy'}. Otherwise
    {run_key: 'ok' | error}. Nothing is evaluated, folders are not touched."""
    root = Path(root) if root is not None else Path(S.RESULTS_ROOT)
    scratch = (Path(scratch) if scratch is not None
               else Path(tempfile.gettempdir()) / "eis_rebuild")
    folders = run_folders(root)
    published = runs(spark)
    stamps = (dict(zip(published["run_key"].astype(str),
                       published.get("published", pd.Series(dtype=str)).astype(str)))
              if len(published) else {})
    lost = {}
    for key in sorted(set(stamps) - set(folders)):
        d = restore(spark, key, scratch / key, published=stamps[key], say=say)
        if d is not None and _usable(d):
            folders[key] = d
        else:
            lost[key] = "no folder, no run_files copy"
    if lost:
        for key, why in lost.items():
            say(f"  {key}: {why} -- nothing dropped")
        return lost
    drop_all(spark, say=say)
    out = {}
    for key, d in sorted(folders.items()):
        try:
            publish(spark, d, say=say)
            out[key] = "ok"
        except Exception as exc:                            # noqa: BLE001
            out[key] = f"{type(exc).__name__}: {exc}"
    return out
