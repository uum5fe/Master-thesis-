"""
results.py
==========
The read-only side: the saved runs and the tables, never an evaluation.

`Results` is the Session of the "02 Local EIS - Results" notebook. Its
selection lists what has been SAVED -- published runs (the runs table) and
saved folders (EIS_Results/<order>[_gen2][_rot180]/<condition>_Test<nn>/)
-- and `run()` refuses.

With settings.RESULTS_SOURCE = 'tables' (default) a selected run is read
from the Delta tables: its files come back from <prefix>run_files into
scratch/from_tables/<run_key>/ and every plot of the process notebook --
Nyquist, median Nyquist, heat maps, frequency response, MF4, ECM -- is drawn
from there, unchanged; the Volume folder is not needed. A run that was never
published is read from its folder. With 'folders' only the folders are
used. Across orders it reads the tables: the runs list and the plate
parameters against current.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from . import settings as S
from . import tables
from .session import Session, condition_key


class Results(Session):
    """Saved runs and tables only. Evaluate in "01 Local EIS - Process"."""

    # ------------------------------------------------------------------
    # what is saved
    # ------------------------------------------------------------------
    def _from_tables(self) -> bool:
        return S.RESULTS_SOURCE == "tables" and self.spark is not None

    def _cond_dirs(self, order_dir: Path) -> list[Path]:
        """Saved run folders of this evaluation mode in one order folder."""
        out = []
        try:
            for d in sorted(order_dir.iterdir()):
                if (d.is_dir() and self._usable(d)
                        and tables.run_key(d)["eval_mode"] == S.EVALUATION_MODE):
                    out.append(d)
        except OSError:
            pass
        return out

    def _table_keys(self) -> dict:
        """{order folder name: [run_key of this evaluation mode]} of the
        runs table ({} with RESULTS_SOURCE = 'folders' or without it)."""
        if not self._from_tables():
            return {}
        out: dict = {}
        for k in self._published():
            key = tables.run_key(self.results_root / k)
            if key["eval_mode"] == S.EVALUATION_MODE:
                out.setdefault(k.split("/")[0], []).append(key)
        return out

    def _saved_keys(self, name: str) -> list[dict]:
        """Run keys of one order: published and/or saved in a folder."""
        keys = {k["run_key"]: k for k in self._table_keys().get(name, [])}
        for d in self._cond_dirs(self.results_root / name):
            keys.setdefault(tables.run_key(d)["run_key"], tables.run_key(d))
        return [keys[k] for k in sorted(keys)]

    def result_sets(self) -> list[str]:
        """Order folders with saved runs ('2612090', '2612091_gen2', ...):
        published ones and those under the results root."""
        try:
            dirs = [d for d in self.results_root.iterdir() if d.is_dir()]
        except OSError:
            dirs = []
        names = {d.name for d in dirs if self._cond_dirs(d)}
        names |= set(self._table_keys())
        return sorted(names)

    def _config_used(self, keys) -> dict:
        for k in keys:
            d = self.results_root / k["run_key"]
            try:
                return json.loads((d / "config_used.json").read_text())
            except Exception:                               # noqa: BLE001
                pass
            if self._from_tables() and k["run_key"] in self._published():
                try:
                    b = tables.file_bytes(self.spark, k["run_key"],
                                          "config_used.json")
                    if b:
                        return json.loads(b.decode())
                except Exception:                           # noqa: BLE001
                    pass
        return {}

    def run_dirs(self) -> dict:
        """{condition: folder} of the selection: restored from the tables
        (RESULTS_SOURCE = 'tables'), else the saved folder."""
        out = {}
        for cond in self.sel.get("conditions", []):
            d = None
            if self._from_tables():
                d = self.restore_run(cond)
            if d is None and self._usable(self.saved_dir(cond)):
                d = self.saved_dir(cond)
            if d is not None:
                out[cond] = Path(d)
        if not out:
            self.say("  no saved results for the selected conditions")
        return out

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------
    def widgets(self):
        import r2d2_geometry as geom
        import site_paths
        self._published(refresh=True)
        sets = self.result_sets() or ["(no saved results)"]
        self._make("dropdown", "result_set", sets[-1], sets,
                   "Order (saved results)")
        name = self._w("result_set", sets[-1])
        if name not in sets:
            name = sets[-1]
        order_dir = self.results_root / name
        keys = self._saved_keys(name)
        cond_tests: dict[str, set] = {}
        for k in keys:
            cond_tests.setdefault(k["condition"], set()).add(k["test_id"])
        cond_tests = {c: sorted(t) for c, t in sorted(
            cond_tests.items(), key=lambda kv: condition_key(kv[0]))}
        conditions = list(cond_tests) or ["-"]
        tests = sorted({t for ts in cond_tests.values() for t in ts}) or ["01"]
        self._make("multiselect", "conditions", "ALL", ["ALL"] + conditions,
                   "Conditions")
        self._make("dropdown", "test_id", tests[0], tests, "FAMOS test")
        raw = [x.strip() for x in str(self._w("conditions", "ALL")).split(",")
               if x.strip()]
        selected = (conditions if not raw or any(x.upper() == "ALL" for x in raw)
                    else sorted(dict.fromkeys(raw), key=condition_key))
        test_id = str(self._w("test_id", tests[0]) or "01").strip().zfill(2)

        k0 = keys[0] if keys else tables.run_key(order_dir / "x")
        plate, orient = k0["plate"], k0["orientation"]
        cfg = self._config_used(keys)
        geom.use_plate(plate, orient)
        files = site_paths.resolve(plate, root=self.config_root,
                                   package_dir=self.package_dir)
        order = k0["order_id"]
        csv = order.startswith("csv")
        tag = name[len(order):]
        self.sel = dict(
            leepa=order, source="csv" if csv else "famos", plate=plate,
            plate_orientation=orient, plate_tag=tag, conditions=selected,
            all_conditions=conditions, tests_by_condition=cond_tests,
            test_id=test_id, exclude=frozenset(), substitute=frozenset(),
            fill_gaps=False, run_mode="cache", plate_files=files,
            csv_folders={}, csv_campaign="",
            gain_file=str(files.get("gain") or ""),
            gamry_version=str(cfg.get("gamry_version") or ""),
            gamry_dir=str(cfg.get("gamry_dir") or ""),
            bench_log=str(cfg.get("bench_log") or ""))
        src = ("Delta tables (" + tables.table_name("run_files") + ")"
               if self._from_tables() else str(order_dir))
        self.say(f"  Saved results  {name}   from {src}\n"
                 f"  Conditions     {', '.join(selected)}   Test_{test_id}\n"
                 f"  (read-only: nothing is evaluated here)")
        return self.sel

    def run(self):
        raise RuntimeError("02 Local EIS - Results is read-only: evaluate in "
                           "01 Local EIS - Process (or the 03 Auto job)")

    def publish(self):
        raise RuntimeError("02 Local EIS - Results is read-only")

    # ------------------------------------------------------------------
    # across orders: the tables
    # ------------------------------------------------------------------
    def _plate_stats(self) -> pd.DataFrame:
        """gold_plate rows of every run (the table, else the saved folders)."""
        if self.spark is not None:
            df = tables.read(self.spark, "gold_plate")
            if len(df):
                return df
        rows = []
        for name in self.result_sets():
            for d in self._cond_dirs(self.results_root / name):
                p = d / "gold" / "plate_statistics.csv"
                if p.is_file():
                    rows.append(tables.tidy(pd.read_csv(p), tables.run_key(d)))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def show_runs(self) -> pd.DataFrame:
        """Every published run, newest first (the runs table)."""
        if self.spark is None:
            self.say("  no Spark session -- the runs table cannot be read")
            return pd.DataFrame()
        df = tables.runs(self.spark)
        cols = ["run_key", "condition", "test_id", "n_measured",
                "gamry_sweep", "plausibility_fail", "plausibility_warn",
                "famos_source", "runner", "written", "published"]
        df = df[[c for c in cols if c in df]]
        if self._display is not None and len(df):
            self._display(df)
        else:
            self.say(df.to_string(index=False) if len(df) else
                     "  no runs published yet")
        return df

    def show_trends(self):
        """Plate median of every gold parameter against current: one curve
        per order and test, all published runs (Parameter drop-down)."""
        import plotly.graph_objects as go
        df = self._plate_stats()
        if not len(df):
            self.say("  no plate statistics published yet")
            return None
        df = df[pd.to_numeric(df["current_a"], errors="coerce").notna()]
        figs = {}
        for par, g in df.groupby("parameter", sort=False):
            unit = str(g["unit"].iloc[0]) if "unit" in g else ""
            fig = go.Figure()
            for (order, plate, test), h in g.groupby(
                    ["order_id", "plate", "test_id"], sort=True):
                h = h.sort_values("current_a")
                fig.add_trace(go.Scatter(
                    x=h["current_a"], y=h["median"], mode="lines+markers",
                    name=f"{order} {plate} Test{test}",
                    error_y=dict(type="data", array=h["sd"], visible=True,
                                 thickness=1) if "sd" in h else None,
                    customdata=h[["run_key"]].to_numpy(),
                    hovertemplate="%{customdata[0]}<br>%{x:g} A: "
                                  "%{y:.4g}<extra></extra>"))
            fig.update_layout(
                title=f"{par} — plate median (± sd over tiles) against current",
                xaxis_title="current [A]", yaxis_title=f"{par} [{unit}]",
                height=S.PANEL_HEIGHT, width=S.PANEL_WIDTH)
            figs[(par,)] = fig
        import viewers
        html = viewers.selector_html(figs, ["Parameter"],
                                     title="Plate parameters against current")
        p = self.results_root / "_overview" / "trends.html"
        self.html(html)
        try:
            viewers.save_page(html, p, title="Plate parameters against current")
            self.say(f"  saved: {p}")
        except OSError as exc:
            self.say(f"  not saved ({exc})")
        return p
