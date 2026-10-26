"""The spp_eval-style layer: Delta tables per run, the read-only Results
session and the incremental job -- against an in-memory stand-in for Spark
(tests/test_tables_delta.py runs the same table code on a real local Delta
Lake when pyspark + delta-spark are installed)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import runner
from runner import auto, tables
from runner import settings as S

from test_runner_session import _fake_result, _session, volume  # noqa: F401


# ---------------------------------------------------------------------------
# an in-memory Spark: enough of the API tables.py uses
# ---------------------------------------------------------------------------
class _DF:
    def __init__(self, spark, pdf):
        self.spark, self.pdf = spark, pdf.reset_index(drop=True)

    @property
    def write(self):
        return _Writer(self)

    def where(self, expr):
        return _DF(self.spark, _filter(self.pdf, expr, keep=True))

    def toPandas(self):
        return self.pdf.copy()

    def collect(self):
        return [dict(r) for _, r in self.pdf.iterrows()]


def _filter(pdf, expr, keep):
    hit = pd.Series(True, index=pdf.index)
    for part in re.split(r"\s+AND\s+", expr):
        m = re.fullmatch(r"\s*(\w+)\s*=\s*'((?:[^']|'')*)'\s*", part)
        assert m, expr
        hit &= pdf[m.group(1)].astype(str) == m.group(2).replace("''", "'")
    return pdf[hit] if keep else pdf[~hit]


class _Writer:
    def __init__(self, df):
        self.df, self.opts, self._mode = df, {}, "error"

    def format(self, f):
        assert f == "delta"
        return self

    def mode(self, m):
        self._mode = m
        return self

    def option(self, k, v):
        self.opts[k] = v
        return self

    def partitionBy(self, *cols):
        self.opts["partition"] = cols
        return self

    def saveAsTable(self, name):
        t = self.df.spark.tables
        new = self.df.pdf
        if self._mode == "overwrite" and "replaceWhere" in self.opts and name in t:
            rest = _filter(t[name], self.opts["replaceWhere"], keep=False)
            t[name] = pd.concat([rest, new], ignore_index=True)
        elif self._mode == "append" and name in t:
            t[name] = pd.concat([t[name], new], ignore_index=True)
        else:
            t[name] = new.copy()
        self.df.spark.log.append((name, self._mode, dict(self.opts)))


class _Catalog:
    def __init__(self, spark):
        self.spark = spark

    def tableExists(self, name):
        return name in self.spark.tables

    def clearCache(self):
        pass


class _Reader:
    def __init__(self, spark):
        self.spark = spark

    def table(self, name):
        return _DF(self.spark, self.spark.tables[name])


class FakeSpark:
    def __init__(self):
        self.tables, self.log = {}, []
        self.catalog, self.read = _Catalog(self), _Reader(self)

    def sql_drop(self, q):
        self.tables.pop(q.split()[2], None)

    def createDataFrame(self, pdf):
        return _DF(self, pdf)

    def sql(self, q):
        if q.startswith("DROP TABLE"):
            return self.sql_drop(q)
        m = re.match(r"DELETE FROM (\S+) WHERE (.+)$", q.strip())
        if m:
            self.tables[m.group(1)] = _filter(self.tables[m.group(1)],
                                              m.group(2), keep=False)
            return _DF(self, pd.DataFrame())
        return _DF(self, pd.DataFrame())      # datago queries: nothing there


# ---------------------------------------------------------------------------
def _run(root, order_dir, cond_dir, digest="d1", rohm=50.0):
    d = Path(root) / order_dir / cond_dir
    cond = cond_dir.split("_")[0]
    _fake_result(d, cond=cond)
    (d / "gold").mkdir()
    pd.DataFrame({"parameter": ["R_ohmic", "R_pol"], "unit": ["mΩ·cm²"] * 2,
                  "median": [rohm, 3 * rohm], "sd": [5.0, 9.0]}).to_csv(
        d / "gold" / "plate_statistics.csv", index=False)
    (d / "run_identity.json").write_text(json.dumps(
        {"digest": digest, "runner": S.RUNNER_VERSION, "settings": {"a": 1},
         "famos_source": {"Karte_2": "datago (rows)"}, "cards": 2,
         "written": "2026-10-23T10:00:00"}))
    (d / "run_manifest.json").write_text(json.dumps(
        {"stages": {"gold": {"n_measured": 3},
                    "gamry": [{"reference": "V26_112_HFR_101_CurrVal_450.dta"}],
                    "plausibility": [{"verdict": "pass"}, {"verdict": "fail"}]}}))
    return d


def test_run_key_from_the_folder():
    k = tables.run_key(Path("/x/EIS_Results/2612091_gen2_rot180/45A_Test01_strict"))
    assert k == {"run_key": "2612091_gen2_rot180/45A_Test01_strict",
                 "order_id": "2612091", "plate": "gen2", "orientation": "rot180",
                 "condition": "45A", "current_a": 45.0, "test_id": "01",
                 "eval_mode": "strict"}
    k = tables.run_key(Path("/x/2612090/7.5A_Test02"))
    assert (k["plate"], k["orientation"], k["current_a"], k["eval_mode"]) == (
        "gen1", "standard", 7.5, "default")


def test_frames_carry_the_key_and_tidy_names(tmp_path):
    d = _run(tmp_path, "2612090", "450A_Test01")
    fr = tables.frames(d)
    assert {"runs", "silver_spectra", "gold_cell", "gold_plate",
            "silver_segments", "segment_meta", "gamry_reference",
            "gamry_comparison"} <= set(fr)
    r = fr["runs"].iloc[0]
    assert (r.run_key, r.digest, r.n_measured, r.gamry_sweep,
            r.plausibility_fail) == ("2612090/450A_Test01", "d1", 3,
                                     "V26_112_HFR_101_CurrVal_450.dta", 1)
    for df in fr.values():
        assert list(df.columns[:len(tables.KEY_COLUMNS)]) == list(tables.KEY_COLUMNS)
        assert all(re.fullmatch(r"[A-Za-z_][0-9A-Za-z_]*", c) for c in df.columns)
    # the file's own 'condition' column gives way to the key's
    assert (fr["gamry_reference"]["condition"] == "450A").all()


def test_publish_replaces_a_run_and_keeps_the_others(tmp_path):
    sp = FakeSpark()
    a = _run(tmp_path, "2612090", "450A_Test01", rohm=50)
    b = _run(tmp_path, "2612090", "60A_Test01", rohm=40)
    tables.publish(sp, a, say=lambda *x: None)
    tables.publish(sp, b, say=lambda *x: None)
    spec = tables.table_name("silver_spectra")
    n_a = (sp.tables[spec]["run_key"] == "2612090/450A_Test01").sum()
    _run(tmp_path / "again", "2612090", "450A_Test01", digest="d2", rohm=55)
    tables.publish(sp, tmp_path / "again" / "2612090" / "450A_Test01",
                   say=lambda *x: None)
    runs = tables.runs(sp)
    assert sorted(runs["run_key"]) == ["2612090/450A_Test01", "2612090/60A_Test01"]
    assert runs.set_index("run_key").loc["2612090/450A_Test01", "digest"] == "d2"
    assert (sp.tables[spec]["run_key"] == "2612090/450A_Test01").sum() == n_a
    gp = tables.read(sp, "gold_plate", "parameter = 'R_ohmic'")
    assert sorted(gp["median"]) == [40.0, 55.0]
    # first write creates a partitioned table, later ones replace by key
    first = [o for n, m, o in sp.log if n == spec][0]
    assert first.get("partition") == ("order_id",)
    later = [o for n, m, o in sp.log if n == spec][-1]
    assert later["replaceWhere"] == "run_key = '2612090/450A_Test01'"
    assert tables.table_name("runs").startswith(S.TABLE_SCHEMA + "." + S.TABLE_PREFIX)


def test_delete_run_removes_it_everywhere(tmp_path):
    sp = FakeSpark()
    for c in ("450A_Test01", "60A_Test01"):
        tables.publish(sp, _run(tmp_path, "2612090", c), say=lambda *x: None)
    tables.delete_run(sp, "2612090/450A_Test01", say=lambda *x: None)
    for name, df in sp.tables.items():
        assert "2612090/450A_Test01" not in set(df["run_key"]), name


# ---------------------------------------------------------------------------
# the read-only Results session
# ---------------------------------------------------------------------------
def _results(tmp_path, spark=None, **w):
    from test_runner_session import _DB
    shown = []
    R = runner.Results({"dbutils": _DB(w), "spark": spark,
                        "displayHTML": shown.append},
                       famos_root=tmp_path / "Famos", gamry_root=tmp_path / "Gamry",
                       results_root=tmp_path / "EIS_Results",
                       config_root=Path(__file__).resolve().parents[1] / "EIS_config",
                       scratch=tmp_path / "scratch")
    return R, shown


def test_results_lists_saved_runs_and_never_evaluates(tmp_path):
    root = tmp_path / "EIS_Results"
    _run(root, "2612090", "450A_Test01")
    _run(root, "2612090", "60A_Test01")
    _run(root, "2612091_gen2", "45A_Test02")
    R, shown = _results(tmp_path, result_set="2612091_gen2")
    assert R.result_sets() == ["2612090", "2612091_gen2"]
    sel = R.widgets()
    assert (sel["leepa"], sel["plate"], sel["conditions"], sel["test_id"]) == (
        "2612091", "gen2", ["45A"], "02")
    assert R.run_dirs() == {"45A": root / "2612091_gen2" / "45A_Test02"}
    p = R.show_nyquist()
    assert p.is_file() and len(shown) == 1
    with pytest.raises(RuntimeError):
        R.run()
    with pytest.raises(RuntimeError):
        R.publish()


def test_trends_across_orders_from_the_tables(tmp_path):
    sp = FakeSpark()
    root = tmp_path / "EIS_Results"
    for order, cond, r in (("2612090", "450A_Test01", 50), ("2612090", "60A_Test01", 40),
                           ("2612091", "450A_Test01", 70)):
        tables.publish(sp, _run(root, order, cond, rohm=r), say=lambda *x: None)
    R, shown = _results(tmp_path, spark=sp)
    p = R.show_trends()
    assert p.is_file() and len(shown) == 1
    import viewers
    P = viewers.unpack_payload(shown[0])
    assert set(P["data"]) == {"R_ohmic", "R_pol"}
    runs = R.show_runs()
    assert len(runs) == 3


# ---------------------------------------------------------------------------
# the incremental job
# ---------------------------------------------------------------------------
def test_plan_takes_new_and_changed_runs_only(volume, monkeypatch):
    monkeypatch.setattr(S, "DATAGO_FIRST", False)
    sp = FakeSpark()
    R, _ = _session(volume)
    R.spark = sp
    R.dbutils = None
    # what is "published": 60A Test01 with the current digest, 60A Test02
    # with another one
    sel = auto._select(R, "2612086", test_id="01")
    cur = R.identity(R._config("60A", volume / "x"))["digest"]
    rows = pd.DataFrame([
        {**tables.run_key(R.order_dir() / "60A_Test01"), "digest": cur,
         "published": "2026-10-23T10:00:00"},
        {**tables.run_key(R.order_dir() / "60A_Test02"), "digest": "old",
         "published": "2026-10-23T10:00:00"}])
    sp.tables[tables.table_name("runs")] = rows
    todo = auto.plan(R, orders=["2612086"], say=lambda *x: None)
    got = {t["run_key"]: t["reason"] for t in todo}
    assert got == {"2612086/450A_Test01": "new",
                   "2612086/60A_Test02": "settings changed",
                   "2612086/45A_Test02": "new"}
    monkeypatch.setattr(S, "AUTO_RERUN_CHANGED", False)
    todo = auto.plan(R, orders=["2612086"], say=lambda *x: None)
    assert "2612086/60A_Test02" not in {t["run_key"] for t in todo}
    assert auto.plan(R, orders=["9999999"], say=lambda *x: None) == []


def test_plate_comes_from_the_settings_or_the_file_names(volume, monkeypatch):
    R, _ = _session(volume)
    assert auto.plate_of(R, "2612086") == "gen1"
    (volume / "Gamry" / "x_RO2612086-01_V26_109_lokale_EIS_6_Boxen_Naboo_1.mf4"
     ).write_bytes(b"")
    assert auto.plate_of(R, "2612086") == "gen2"
    monkeypatch.setattr(S, "AUTO_PLATE", {"2612086": "gen1"})
    assert auto.plate_of(R, "2612086") == "gen1"


def test_one_type_per_column_whatever_the_run():
    """Delta cannot merge bigint with double or double with string: the
    same column must come out with the same type from every run."""
    key = tables.run_key(Path("/x/2612090/45A_Test01"))
    a = tables.tidy(pd.DataFrame({"L_nH": [12, 15], "fault": [np.nan, np.nan],
                                  "drt_clean": [True, False],
                                  "segment": ["S01", "S02"]}), key)
    b = tables.tidy(pd.DataFrame({"L_nH": [12.4, np.nan], "fault": ["x", None],
                                  "drt_clean": ["True", np.nan],
                                  "segment": ["S01", "S02"]}), key)
    assert str(a["L_nH"].dtype) == str(b["L_nH"].dtype) == "float64"
    assert "fault" not in a and b["fault"].tolist() == ["x", None]
    assert a["drt_clean"].tolist() == [True, False]
    assert b["drt_clean"].tolist() == [True, None]
    # the runs row keeps its counts as integers
    r = tables.tidy(pd.DataFrame([{"cards": 3, "elapsed_s": 1.5}]), key,
                    integers=True)
    assert str(r["cards"].dtype) == "int64"


def test_rebuild_all_tables_from_the_folders(tmp_path):
    sp = FakeSpark()
    root = tmp_path / "EIS_Results"
    for c in ("450A_Test01", "60A_Test01"):
        tables.publish(sp, _run(root, "2612090", c), say=lambda *x: None)
    assert len(tables.drop_all(sp, say=lambda *x: None)) >= 8 and not sp.tables
    got = tables.republish_all(sp, root, say=lambda *x: None)
    assert got == {"2612090/450A_Test01": "ok", "2612090/60A_Test01": "ok"}
    assert sorted(tables.runs(sp)["run_key"]) == list(got)



# ---------------------------------------------------------------------------
# reading runs back from the tables (no Volume folder)
# ---------------------------------------------------------------------------
def test_run_files_restore_the_folder_byte_for_byte(tmp_path):
    sp = FakeSpark()
    d = _run(tmp_path / "EIS_Results", "2612090", "450A_Test01")
    (d / "gold" / "nyquist.png").write_bytes(bytes(range(256)) * 10)
    tables.publish(sp, d, say=lambda *x: None)
    files = sp.tables[tables.table_name("run_files")]
    assert "gold/nyquist.png" in set(files["path"])
    out = tables.restore(sp, "2612090/450A_Test01", tmp_path / "back",
                         published="t1", say=lambda *x: None)
    src = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    got = sorted(p.relative_to(out).as_posix() for p in out.rglob("*")
                 if p.is_file() and p.name != ".restored_from_tables")
    assert got == src
    for rel in src:
        assert (out / rel).read_bytes() == (d / rel).read_bytes(), rel
    assert tables.file_bytes(sp, "2612090/450A_Test01", "run_identity.json") == \
        (d / "run_identity.json").read_bytes()
    assert tables.restore(sp, "nope/450A_Test01", tmp_path / "x",
                          say=lambda *x: None) is None
    tables.delete_run(sp, "2612090/450A_Test01", say=lambda *x: None)
    assert not len(sp.tables[tables.table_name("run_files")])


def test_results_read_from_the_tables_without_the_folders(tmp_path):
    import shutil
    sp = FakeSpark()
    root = tmp_path / "EIS_Results"
    for c in ("450A_Test01", "60A_Test01"):
        tables.publish(sp, _run(root, "2612090", c), say=lambda *x: None)
    shutil.rmtree(root)                                # the Volume is gone
    R, shown = _results(tmp_path, spark=sp, result_set="2612090")
    assert R.result_sets() == ["2612090"]
    sel = R.widgets()
    assert sel["conditions"] == ["60A", "450A"]
    dirs = R.run_dirs()
    assert set(dirs) == {"60A", "450A"}
    assert all("from_tables" in str(p) for p in dirs.values())
    assert R.show_nyquist() is not None and len(shown) == 1
    # folders only: nothing to show
    import runner.settings as RS
    old = RS.RESULTS_SOURCE
    try:
        RS.RESULTS_SOURCE = "folders"
        R2, _ = _results(tmp_path, spark=sp)
        assert R2.result_sets() == []
    finally:
        RS.RESULTS_SOURCE = old


def test_cache_run_restores_a_published_run_whose_folder_is_gone(volume, monkeypatch):
    import shutil
    monkeypatch.setattr(S, "DATAGO_FIRST", False)
    sp = FakeSpark()
    R, _ = _session(volume, leepa_id="2612086", conditions="60A", test_id="02")
    R.widgets()
    R.spark = sp
    saved = R.saved_dir("60A")
    _fake_result(saved)
    tables.publish(sp, saved, say=lambda *x: None)
    shutil.rmtree(saved)                              # the Volume folder is gone
    assert R.run() == {"60A": saved} and R._usable(saved)
    # without the tables it would be evaluated (and fail on the fake files)
    monkeypatch.setattr(S, "CACHE_FROM_TABLES", False)
    shutil.rmtree(saved)
    assert "60A" not in R.run()


def test_rebuild_reads_back_runs_whose_folder_is_gone_and_never_loses_one(tmp_path):
    import shutil
    sp = FakeSpark()
    root = tmp_path / "EIS_Results"
    for c in ("450A_Test01", "60A_Test01"):
        tables.publish(sp, _run(root, "2612090", c), say=lambda *x: None)
    shutil.rmtree(root / "2612090" / "60A_Test01")    # folder gone, archived
    got = tables.rebuild(sp, root, tmp_path / "s", say=lambda *x: None)
    assert got == {"2612090/450A_Test01": "ok", "2612090/60A_Test01": "ok"}
    assert sorted(tables.runs(sp)["run_key"]) == list(got)
    # a published run with neither folder nor archive: nothing is dropped
    t = tables.table_name("run_files")
    sp.tables[t] = sp.tables[t][sp.tables[t]["run_key"] != "2612090/60A_Test01"]
    n = {k: len(v) for k, v in sp.tables.items()}
    got = tables.rebuild(sp, root, tmp_path / "s2", say=lambda *x: None)
    assert got == {"2612090/60A_Test01": "no folder, no run_files copy"}
    assert {k: len(v) for k, v in sp.tables.items()} == n


# ---------------------------------------------------------------------------
# SAVE_FOLDERS = False: the tables are the only copy of a run
# ---------------------------------------------------------------------------
def _fake_pipeline(monkeypatch):
    """bronze.discover_files and main.run_pipeline replaced: the 'run' writes
    a small finished result into cfg.out_dir."""
    import bronze
    import main as pipeline_main
    monkeypatch.setattr(bronze, "discover_files", lambda cfg: [Path("a"), Path("b")])

    def run_pipeline(cfg, stop_after="gold"):
        _fake_result(Path(cfg.out_dir), cond=cfg.condition)
        return {"stages": {"gold": {"n_measured": 3}}}
    monkeypatch.setattr(pipeline_main, "run_pipeline", run_pipeline)


def test_tables_only_run_writes_nothing_under_the_results_root(volume, monkeypatch):
    monkeypatch.setattr(S, "DATAGO_FIRST", False)
    monkeypatch.setattr(S, "SAVE_FOLDERS", False)
    _fake_pipeline(monkeypatch)
    sp = FakeSpark()
    R, _ = _session(volume, leepa_id="2612086", conditions="60A", test_id="02",
                    run_mode="rerun")
    R.widgets()
    R.spark = sp
    d = R.run()["60A"]
    assert volume / "scratch" in d.parents and (d / "run_identity.json").is_file()
    R.publish()
    key = "2612086/60A_Test02"
    assert key in set(tables.runs(sp)["run_key"])
    assert "run_identity.json" in set(sp.tables[tables.table_name("run_files")]["path"])
    assert volume / "scratch" in R.plots_dir().parents
    assert not (volume / "EIS_Results").exists()          # nothing on the Volume

    # a new session (cluster restarted, scratch gone): cache reads the tables
    import shutil
    shutil.rmtree(volume / "scratch")
    R2, _ = _session(volume, leepa_id="2612086", conditions="60A", test_id="02")
    R2.widgets()
    R2.spark = sp
    got = R2.run()["60A"]
    assert "from_tables" in str(got) and R2._usable(got)
    assert set(R2.run_dirs()) == {"60A"}
    assert not (volume / "EIS_Results").exists()


def test_tables_only_falls_back_to_folders_without_spark(volume, monkeypatch):
    monkeypatch.setattr(S, "SAVE_FOLDERS", False)
    R, _ = _session(volume)
    assert R.spark is None and R._save_folders()        # nowhere else to keep it
    monkeypatch.setattr(S, "SAVE_FOLDERS", True)
    R.spark = FakeSpark()
    assert R._save_folders()
