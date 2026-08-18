"""Where measurements come from, and how they are found.

Three kinds of source, deliberately behind one interface so the view layer
asks "give me the conditions for this order id" without caring which one is
selected in the dropdown:

``results``
    A directory tree of finished pipeline output.  Normally a Databricks
    Volumes path; the fast path, because nothing has to be recomputed.

``famos``
    Raw FAMOS ``.DAT`` recordings, discovered by filename.  Selecting one of
    these does not produce a spectrum on its own - it has to go through the
    pipeline first (see :mod:`app.services.runner`).

``datago``
    The Unity Catalog metadata tables, queried over a SQL warehouse.  Used to
    list which order ids exist and what was measured, so the picker is not
    limited to whatever happens to be on the file system.

Discovery is filename- and directory-driven, never hard-coded to one person's
home directory: every root comes from :mod:`app.settings`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.data.loaders import detect_layout, load
from app.data.model import RunData
from app.settings import SETTINGS, Settings

#: Patterns tried, in order, against a FAMOS filename.  Both the campaign
#: convention (``Leepa_<id>_...``) and the order convention (``RO<id>-01_...``)
#: appear in the recordings, and a site can add a third through the
#: ``EIS_FAMOS_REGEX`` environment variable without touching this file.
_FAMOS_PATTERNS = [
    r"Leepa_(?P<measurement_id>[^_]+)_Current_(?P<condition>.+?)"
    r"_Test_(?P<test>\d+)_Karte_(?P<card>\w+)\.DAT$",
    r"RO(?P<measurement_id>[^_-]+)(?:-\d+)?_Current_(?P<condition>.+?)"
    r"_Test_(?P<test>\d+)_Karte_(?P<card>\w+)\.DAT$",
]


def famos_patterns() -> list[re.Pattern]:
    extra = os.environ.get("EIS_FAMOS_REGEX", "").strip()
    pats = ([extra] if extra else []) + _FAMOS_PATTERNS
    return [re.compile(p, re.IGNORECASE) for p in pats]


@dataclass(frozen=True)
class RunRef:
    """One selectable thing: an order id at one operating condition."""

    kind: str                      # results | famos | datago
    measurement_id: str
    condition: str
    path: str = ""
    layout: str = ""               # gold_silver | eis_tables | generic_csv | famos
    files: tuple[str, ...] = ()
    modified: float = 0.0
    detail: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.measurement_id}:{self.condition}"

    @property
    def modified_text(self) -> str:
        if not self.modified:
            return ""
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")

    def describe(self) -> str:
        if self.kind == "famos":
            return f"{len(self.files)} card file(s)"
        return self.layout or self.kind


# ---------------------------------------------------------------------------
# results on disk
# ---------------------------------------------------------------------------

class ResultsSource:
    """Finished pipeline output under one or more roots.

    Layout expected::

        <root>/<measurement_id>/<condition>/{gold,silver}/*.csv
        <root>/<measurement_id>/{impedance,segments}.parquet

    A root that is itself a single results directory also works, so pointing
    the app straight at one run is a valid thing to do while debugging.
    """

    kind = "results"

    def __init__(self, roots) -> None:
        self.roots = [Path(r) for r in roots]

    def scan(self) -> list[RunRef]:
        out: list[RunRef] = []
        for root in self.roots:
            if not root.is_dir():
                continue
            out.extend(self._scan_root(root))
        return out

    def _scan_root(self, root: Path) -> list[RunRef]:
        found: list[RunRef] = []

        layout = detect_layout(root)
        if layout:
            found.append(self._ref(root, root.name, "", layout))
            return found

        for mid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            layout = detect_layout(mid_dir)
            if layout == "eis_tables":
                # One table set can hold several conditions; enumerate them.
                for condition in _conditions_in_tables(mid_dir):
                    found.append(self._ref(mid_dir, mid_dir.name, condition, layout))
                continue
            if layout:
                found.append(self._ref(mid_dir, mid_dir.name, "", layout))
                continue
            for cond_dir in sorted(p for p in mid_dir.iterdir() if p.is_dir()):
                sub = detect_layout(cond_dir)
                if sub:
                    found.append(self._ref(cond_dir, mid_dir.name, cond_dir.name, sub))
        return found

    @staticmethod
    def _ref(path: Path, measurement_id: str, condition: str, layout: str) -> RunRef:
        return RunRef(kind="results", measurement_id=measurement_id,
                      condition=condition or "(all)", path=str(path),
                      layout=layout, modified=path.stat().st_mtime)

    def load(self, ref: RunRef, plate_key: str) -> RunData:
        condition = "" if ref.condition == "(all)" else ref.condition
        return load(ref.path, ref.measurement_id, condition, plate_key,
                    layout=ref.layout or None)


def _conditions_in_tables(directory: Path) -> list[str]:
    """Distinct conditions inside an ``eis``-pipeline table set."""
    import pandas as pd
    for name in ("segments.parquet", "segments.csv",
                 "impedance.parquet", "impedance.csv"):
        path = directory / name
        if not path.is_file():
            continue
        try:
            frame = (pd.read_parquet(path, columns=["condition"])
                     if path.suffix == ".parquet"
                     else pd.read_csv(path, usecols=["condition"]))
        except Exception:
            continue
        return sorted({str(c) for c in frame["condition"].dropna().unique()})
    return [""]


# ---------------------------------------------------------------------------
# raw FAMOS recordings
# ---------------------------------------------------------------------------

class FamosSource:
    """Raw ``.DAT`` recordings, grouped into (order id, condition) by name."""

    kind = "famos"

    def __init__(self, roots, glob: str = "Leepa_*_Current_*_Test_*_Karte_*.DAT") -> None:
        self.roots = [Path(r) for r in roots]
        self.glob = glob

    def scan(self) -> list[RunRef]:
        groups: dict[tuple[str, str], list[Path]] = {}
        unmatched = 0
        patterns = famos_patterns()
        for root in self.roots:
            if not root.is_dir():
                continue
            # rglob so a campaign kept in per-condition subfolders still works
            for path in sorted(root.rglob("*.DAT")) + sorted(root.rglob("*.dat")):
                for pattern in patterns:
                    m = pattern.search(path.name)
                    if m:
                        key = (m.group("measurement_id"), m.group("condition"))
                        groups.setdefault(key, []).append(path)
                        break
                else:
                    unmatched += 1

        out = []
        for (mid, cond), paths in sorted(groups.items()):
            out.append(RunRef(
                kind="famos", measurement_id=mid, condition=cond,
                path=str(paths[0].parent), layout="famos",
                files=tuple(str(p) for p in sorted(paths)),
                modified=max(p.stat().st_mtime for p in paths),
                detail={"n_cards": len(paths), "unmatched_in_root": unmatched},
            ))
        return out


# ---------------------------------------------------------------------------
# datago (Unity Catalog over a SQL warehouse)
# ---------------------------------------------------------------------------

class DatagoSource:
    """Order ids and conditions from the datago metadata tables.

    Works two ways: through ``databricks-sql-connector`` against a SQL
    warehouse (how a Databricks App reaches data), or through an ambient
    ``SparkSession`` when the module happens to be imported inside a notebook.
    Neither available means the source simply reports itself unavailable.
    """

    kind = "datago"

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self.error = ""

    def available(self) -> bool:
        return bool(self.settings.datago_metadata_table) and (
            self.settings.datago_configured() or _spark() is not None)

    def scan(self) -> list[RunRef]:
        s = self.settings
        if not self.available():
            # Not configured at all is a deployment choice, not a problem worth
            # reporting on every scan; half-configured is worth reporting.
            self.error = (
                "datago not configured: set EIS_DATAGO_METADATA_TABLE and "
                "DATABRICKS_WAREHOUSE_ID" if s.datago_metadata_table else "")
            return []
        where = ""
        select_cond = "NULL AS condition"
        join = ""
        if s.datago_properties_table:
            join = (f"JOIN {s.datago_properties_table} gp "
                    f"ON m.measurement_id = gp.measurement_id")
            where = f"WHERE gp.measurement_type = '{s.datago_measurement_type}'"
        sql = f"""
            SELECT DISTINCT m.orderId AS order_id, {select_cond}
            FROM {s.datago_metadata_table} m
            {join}
            {where}
            ORDER BY order_id
        """
        try:
            rows = _query(sql, s)
        except Exception as exc:
            self.error = f"datago query failed: {exc}"
            return []
        out = []
        for row in rows:
            mid = str(row[0] or "").replace("RO", "")
            if not mid:
                continue
            out.append(RunRef(kind="datago", measurement_id=mid,
                              condition=str(row[1] or "(all)"),
                              layout="datago",
                              detail={"order_id": str(row[0])}))
        return out


def _spark():
    try:
        from pyspark.sql import SparkSession  # type: ignore
        return SparkSession.getActiveSession()
    except Exception:
        return None


def _query(sql: str, settings: Settings = SETTINGS) -> list[tuple]:
    """Run SQL through a warehouse connection, or an ambient Spark session."""
    if settings.datago_configured():
        from databricks import sql as dbsql          # type: ignore
        token = (os.environ.get("DATABRICKS_TOKEN")
                 or os.environ.get("DATABRICKS_CLIENT_SECRET", ""))
        with dbsql.connect(
            server_hostname=settings.databricks_host.replace("https://", ""),
            http_path=f"/sql/1.0/warehouses/{settings.warehouse_id}",
            access_token=token,
        ) as conn, conn.cursor() as cur:
            cur.execute(sql)
            return [tuple(r) for r in cur.fetchall()]

    spark = _spark()
    if spark is not None:
        return [tuple(r) for r in spark.sql(sql).collect()]
    raise RuntimeError("no SQL warehouse and no Spark session available")


# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------

class Catalog:
    """The union of every configured source, refreshed on demand."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self.results = ResultsSource(settings.resolved_results_roots())
        self.famos = FamosSource(settings.famos_roots, settings.famos_glob)
        self.datago = DatagoSource(settings)
        self._runs: list[RunRef] = []
        self.messages: list[str] = []
        self.scanned_at: float = 0.0

    def refresh(self, kinds: tuple[str, ...] = ("results", "famos", "datago")):
        runs: list[RunRef] = []
        self.messages = []
        for kind in kinds:
            source = {"results": self.results, "famos": self.famos,
                      "datago": self.datago}[kind]
            try:
                found = source.scan()
            except Exception as exc:
                self.messages.append(f"{kind}: {exc}")
                continue
            if kind == "datago" and self.datago.error:
                self.messages.append(self.datago.error)
            runs.extend(found)
        self._runs = runs
        self.scanned_at = datetime.now().timestamp()
        return self

    @property
    def runs(self) -> list[RunRef]:
        if not self._runs:
            self.refresh()
        return self._runs

    def kinds_present(self) -> list[str]:
        return sorted({r.kind for r in self.runs})

    def measurements(self, kind: str = "") -> list[str]:
        ids = {r.measurement_id for r in self.runs if not kind or r.kind == kind}
        return sorted(ids)

    def conditions(self, measurement_id: str, kind: str = "") -> list[str]:
        conds = {r.condition for r in self.runs
                 if r.measurement_id == measurement_id and (not kind or r.kind == kind)}
        return sorted(conds, key=_condition_sort_key)

    def find(self, measurement_id: str, condition: str, kind: str = "") -> RunRef | None:
        for ref in self.runs:
            if (ref.measurement_id == measurement_id and ref.condition == condition
                    and (not kind or ref.kind == kind)):
                return ref
        return None

    def load(self, ref: RunRef, plate_key: str) -> RunData:
        if ref.kind == "results":
            return self.results.load(ref, plate_key)
        run = RunData(measurement_id=ref.measurement_id, condition=ref.condition,
                      plate_key=plate_key, source=ref.path, loader=ref.kind)
        run.warnings.append(
            "Raw recordings hold no spectra yet - run the pipeline on this "
            "selection first, then it appears under the results source.")
        return run


def _condition_sort_key(condition: str):
    """Sort ``45A`` before ``150A`` before ``450A``, not lexicographically."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)", str(condition))
    return (0, float(m.group(1))) if m else (1, str(condition))
