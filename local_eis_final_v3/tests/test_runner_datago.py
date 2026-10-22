"""runner.datago against a stand-in for Spark: every sweep of a current is
written with its start time, and the pipeline then picks by time."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from runner import datago
import gamry_compare as GC


class _Q:
    def __init__(self, df):
        self.df = df

    def toPandas(self):
        return self.df.copy()

    def collect(self):
        return [dict(r) for _, r in self.df.iterrows()]


class _Spark:
    """Answers the three queries runner.datago sends."""

    def __init__(self, files, vals):
        self.files, self.vals = files, vals

    def sql(self, q):
        if "metafiles_delta" in q:
            return _Q(self.files)
        if "values_delta" in q:
            ids = set(x.strip("'") for x in
                      q.split("file_id IN (")[1].split(")")[0].split(", "))
            return _Q(self.vals[self.vals.file_id.isin(ids)])
        return _Q(pd.DataFrame({"orderId": ["RO2612086"]}))


def _sweep_vals(fid, n=20, f0=1e4):
    f = f0 * 10 ** (-0.1 * np.arange(n))
    rows = []
    for ch, v in (("freq", f), ("zreal", 1e-3 + 0 * f), ("zimag", -2e-4 + 0 * f),
                  ("pt", np.arange(1, n + 1)), ("vdc", 0.8 + 0 * f)):
        rows += [{"file_id": fid, "channel": ch, "t": float(i), "v": float(x),
                  "_fn": "a", "_ri": i} for i, x in enumerate(v)]
    return rows


def _spark():
    files = pd.DataFrame([
        {"file_id": "g1", "raw_file_name": "RO2612086-01_V26_109_HFR_103_CurrVal_60.mf4",
         "measurement_type": "GALVEIS", "t_begin": pd.Timestamp("2026-10-20 06:39:43"),
         "t_end": pd.Timestamp("2026-10-20 06:42:08"), "starttime": "20.10.2026 08:39:41",
         "iacreq": 7, "freqinit": 30000, "freqfinal": 0.3, "ptsperdec": 10},
        {"file_id": "g2", "raw_file_name": "RO2612086-01_V26_109_HFR_111_CurrVal_60.mf4",
         "measurement_type": "GALVEIS", "t_begin": pd.Timestamp("2026-10-20 07:34:03"),
         "t_end": pd.Timestamp("2026-10-20 07:36:17"), "starttime": "20.10.2026 09:34:01",
         "iacreq": 7, "freqinit": 30000, "freqfinal": 0.3, "ptsperdec": 10},
        {"file_id": "b1", "raw_file_name": "20261020_0809_RO2612086-01_V26_109_lokale_EIS.mf4",
         "measurement_type": None, "t_begin": pd.Timestamp("2026-10-20 06:09:00"),
         "t_end": pd.Timestamp("2026-10-20 07:57:00"), "starttime": None,
         "iacreq": None, "freqinit": None, "freqfinal": None, "ptsperdec": None},
    ])
    vals = pd.DataFrame(_sweep_vals("g1") + _sweep_vals("g2") + [
        {"file_id": "b1", "channel": "I_S", "t": float(i), "v": float(i % 7),
         "_fn": "b", "_ri": i} for i in range(0, 6000, 10)])
    return _Spark(files, vals)


def test_both_sweeps_of_a_current_are_written_with_their_start_times(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(datago, "CACHE", tmp_path / "dg")
    datago._FILES.clear()
    d = datago.gamry_dir(_spark(), "2612086", say=lambda *a: None)
    names = sorted(p.name for p in d.glob("*.dta"))
    assert names == ["RO2612086-01_V26_109_HFR_103_CurrVal_60.dta",
                     "RO2612086-01_V26_109_HFR_111_CurrVal_60.dta"]
    sw = GC.find_cell_sweeps(d, order_id="2612086", version="V26_109")
    starts = sorted(s.started for s in sw)
    assert starts == [datetime(2026, 10, 20, 8, 39, 41),
                      datetime(2026, 10, 20, 9, 34, 1)]
    rec = (datetime(2026, 10, 20, 9, 32, 29), datetime(2026, 10, 20, 9, 36, 56))
    assert "HFR_111" in GC.select_sweep(sw, when=rec).path.name


def test_the_bench_log_is_read_whole(monkeypatch):
    datago._FILES.clear()
    t0, vals, name = datago.bench_channels(_spark(), "2612086")
    assert "lokale_EIS" in name and t0 == pd.Timestamp("2026-10-20 06:09:00")
    assert vals["I_S"][0][-1] == 5990.0


def test_every_current_of_the_order_gets_its_own_datago_sweep(tmp_path,
                                                             monkeypatch):
    """RO2612091 (V26_113): seven sweeps in datago, stamps 2 h off the FAMOS
    recordings. Each current gets the sweep of its own current from the
    datago folder, by file -- none from the shared Volume folder."""
    monkeypatch.setattr(datago, "CACHE", tmp_path / "dg")
    datago._FILES.clear()
    cur = {101: 611, 102: 450, 103: 300, 104: 150, 105: 60, 106: 45, 107: 30}
    files, vals = [], []
    for n, c in cur.items():
        t0 = pd.Timestamp("2026-10-21 09:00:00") + pd.Timedelta(minutes=10 * (n - 101))
        files.append({"file_id": f"g{n}",
                      "raw_file_name": f"RO2612091-01_V26_113_HFR_{n}_CurrVal_{c}.mf4",
                      "measurement_type": "GALVEIS", "t_begin": t0,
                      "t_end": t0 + pd.Timedelta(minutes=3), "starttime": None,
                      "iacreq": 7, "freqinit": 30000, "freqfinal": 0.2,
                      "ptsperdec": 10})
        vals += _sweep_vals(f"g{n}")
    said = []
    d = datago.gamry_dir(_Spark(pd.DataFrame(files), pd.DataFrame(vals)),
                         "2612091", say=said.append)
    assert len(list(d.glob("*.dta"))) == 7
    assert "7 from datago" in said[-1] and "30A, 45A, 60A" in said[-1]
    found = GC.find_cell_sweeps(d, order_id="2612091", version="V26_113")
    for n, c in cur.items():
        hits = [s for s in found if s.condition == f"{c}A"]
        rec = (datetime(2026, 10, 21, 15, 0), datetime(2026, 10, 21, 15, 4))
        sw = GC.select_sweep(hits, when=rec, test_id="01",
                             order_id="2612091", version="V26_113")
        assert sw is not None and f"HFR_{n}_CurrVal_{c}" in sw.path.name


def test_the_runner_says_when_the_sweeps_do_not_come_from_datago(tmp_path,
                                                                monkeypatch):
    import runner
    from runner import settings as S
    monkeypatch.setattr(S, "GAMRY_FROM_DATAGO", False)
    said = []
    R = runner.Session({"spark": object()}, gamry_root=tmp_path / "Gamry",
                       famos_root=tmp_path, results_root=tmp_path / "r",
                       config_root=tmp_path / "c", scratch=tmp_path / "s")
    R.say = said.append
    R.gamry_dir("2612091")
    assert "NOT from datago" in said[-1] and "GAMRY_FROM_DATAGO" in said[-1]
