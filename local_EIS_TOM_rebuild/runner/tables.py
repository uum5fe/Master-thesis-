"""
tables.py
=========
The silver / gold layer as Delta tables -- one row set per run.

Every finished run is a folder EIS_Results/<order>[_gen2][_rot180]/
<condition>_Test<nn>[_<mode>]/ (unchanged). Publishing copies its result
files into tables in settings.TABLE_SCHEMA, so any number of runs can be
read with one query (e.g. R_ohmic against current for every order):

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

Every row carries the run key -- `run_key` (= the folder's path under
EIS_Results, e.g. '2612090/450A_Test01'), `order_id`, `plate`,
`orientation`, `condition`, `current_a`, `test_id`, `eval_mode` -- and the
tables are partitioned by `order_id`.

Publishing a run REPLACES that run's rows in every table (replaceWhere on
run_key), exactly as a rerun replaces its folder: the tables never hold two
versions of one run. Nothing here computes anything; the folders stay the
source, the tables are their queryable copy.

Every function takes `spark` explicitly.
"""

from __future__ import annotations

import json
import re
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
KEY_COLUMNS = ("run_key", "order_id", "plate", "orientation", "condition",
               "current_a", "test_id", "eval_mode")
PARTITION = "order_id"


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


def tidy(df: pd.DataFrame, key: dict) -> pd.DataFrame:
    """Key columns first, column names Delta accepts, one type per column
    (numbers stay numbers, everything else becomes text)."""
    df = df.copy()
    df.columns = [_clean_name(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith(("string", "category")):
            df[c] = df[c].map(lambda v: None if (v is None or (isinstance(v, float)
                                                               and np.isnan(v)))
                              else str(v)).astype(object)
        elif str(df[c].dtype) == "bool":
            df[c] = df[c].astype(bool)
    for k in reversed(KEY_COLUMNS):
        if k in df.columns:
            df = df.drop(columns=k)
        df.insert(0, k, key[k])
    return df


def frames(run_dir) -> dict:
    """{table suffix: DataFrame} of one saved run (missing files skipped)."""
    run_dir = Path(run_dir)
    key = run_key(run_dir)
    out = {"runs": tidy(pd.DataFrame([run_record(run_dir)]), key)}
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
_FRACTIONAL = {"float", "double"}


def _as_text(s: pd.Series) -> pd.Series:
    return s.map(lambda v: None if pd.isna(v) else str(v)).astype(object)


def _align(spark, pdf: pd.DataFrame, name: str) -> pd.DataFrame:
    """Give every column the type it already has in the table.

    pandas guesses a column's type per file (L_nH: whole numbers -> bigint in
    one run, double or text in the next) and Delta cannot merge two types
    (DELTA_FAILED_TO_MERGE_FIELDS). A column that cannot take the table's
    type without losing values is widened in the table first (bigint ->
    double, number -> string)."""
    if not exists(spark, name):
        return pdf
    have = {f.name: f.dataType.simpleString()
            for f in spark.table(name).schema.fields}
    pdf = pdf.copy()
    widen = {}
    for c in pdf.columns:
        t = have.get(c)
        if t is None or c in KEY_COLUMNS:
            continue
        s = pdf[c]
        if t == "string":
            if s.dtype != object:
                pdf[c] = _as_text(s)
            continue
        if t not in _INTEGRAL and t not in _FRACTIONAL:
            continue
        v = pd.to_numeric(s, errors="coerce")
        if (s.notna() & v.isna()).any():          # text that is no number
            widen[c] = "string"
            pdf[c] = _as_text(s)
        elif t in _FRACTIONAL:
            pdf[c] = v.astype("float64")
        elif (v.dropna() == v.dropna().round()).all():
            pdf[c] = v.astype("Int64")
        else:                                     # fractions in a bigint column
            widen[c] = "double"
            pdf[c] = v.astype("float64")
    if widen:
        old = spark.table(name)
        for c, t in widen.items():
            old = old.withColumn(c, old[c].cast(t))
        (old.write.format("delta").mode("overwrite")
         .option("overwriteSchema", "true")
         .partitionBy(PARTITION).saveAsTable(name))
    return pdf


def _write(spark, pdf: pd.DataFrame, name: str, key: str) -> None:
    pdf = _align(spark, pdf, name)
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
    """Write one saved run into every table; its earlier rows are replaced.
    Returns {table: rows written}."""
    fr = frames(run_dir)
    key = fr["runs"]["run_key"].iloc[0]
    done = {}
    # results first, the runs row last: a run listed in <prefix>runs has
    # all its tables written
    for suffix in [s for s in fr if s != "runs"] + ["runs"]:
        name = table_name(suffix)
        _write(spark, fr[suffix], name, key)
        done[suffix] = len(fr[suffix])
    say(f"    tables: {key} -> {len(done)} tables "
        f"({sum(done.values()):,} rows) in {S.TABLE_SCHEMA}")
    return done


def delete_run(spark, key: str, say=print) -> list:
    """Remove one run (its run_key) from every table. No undo."""
    gone = []
    for suffix in ["runs"] + list(FILES):
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
