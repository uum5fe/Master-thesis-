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
