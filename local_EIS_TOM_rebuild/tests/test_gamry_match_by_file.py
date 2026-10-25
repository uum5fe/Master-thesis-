"""RO2612091 45 A (2026-10-21): the Gamry sweep was there and was not used.

The datago .dta of every sweep carried a start time 1.9 h away from the
FAMOS stamps (a time-zone difference between the PCs), so the time rule
refused the only 45 A sweep of the build and the run had no Gamry reference,
no card phase calibration and no Gamry trace in the plots. With
gamry_match = "file" the sweep is identified by its file name -- order,
build, current, test -- and the stamps never reject it; a file that does not
name the cell still needs matching stamps (the v34.2 wrong-cell case)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import gamry_compare as GC

REC = (datetime(2026, 10, 21, 15, 30, 10), datetime(2026, 10, 21, 15, 34, 0))
TZ = timedelta(hours=-2)              # Gamry stamp as UTC, FAMOS as local


def _sweep(name, start, cur=45.0, dur=200.0):
    return GC.CellSweep(path=Path(name), freq=np.array([1.0]),
                        Z_ohm=np.array([1 + 0j]), current_a=cur,
                        started=start, meta={"duration_s": dur})


def _ramp():
    t0 = datetime(2026, 10, 21, 15, 30, 40) + TZ
    names = [(101, 611), (102, 450), (103, 300), (104, 150), (105, 60),
             (106, 45), (107, 30)]
    return {c: _sweep(f"RO2612091-01_V26_113_HFR_{n}_CurrVal_{c}.dta",
                      t0 + timedelta(minutes=8 * (n - 106)), cur=c)
            for n, c in names}


class _Log(logging.Logger):
    def __init__(self):
        super().__init__("t")
        self.lines = []

    def _log(self, level, msg, *a, **k):
        self.lines.append(str(msg))


def test_the_only_sweep_of_the_build_is_used_whatever_its_stamp_says():
    sw = _ramp()[45]
    log = _Log()
    got = GC.select_sweep([sw], when=REC, test_id="01", log=log,
                          order_id="2612091", version="V26_113")
    assert got is sw
    text = " ".join(log.lines)
    assert "chosen by file" in text and "time-zone" in text
    # the old rule refused it
    assert GC.select_sweep([sw], when=REC, test_id="01", match="time",
                           order_id="2612091", version="V26_113") is None


def test_an_up_down_ramp_is_told_apart_by_the_test_number():
    up = _sweep("RO2612091-01_V26_113_HFR_103_CurrVal_45.dta",
                datetime(2026, 10, 21, 13, 0) + TZ)
    down = _sweep("RO2612091-01_V26_113_HFR_111_CurrVal_45.dta",
                  datetime(2026, 10, 21, 15, 31) + TZ)
    kw = dict(when=REC, order_id="2612091", version="V26_113")
    assert GC.select_sweep([down, up], test_id="01", **kw) is up
    assert GC.select_sweep([down, up], test_id="02", **kw) is down
    # "file" never looks at the stamps, even when one overlaps the recording;
    # "time" does
    down_ok = _sweep(down.path.name, datetime(2026, 10, 21, 15, 31))
    assert GC.select_sweep([down_ok, up], test_id="01", **kw) is up
    assert GC.select_sweep([down_ok, up], test_id="01", match="time",
                           **kw) is down_ok


def test_a_file_that_does_not_name_the_cell_still_needs_its_stamps():
    silent = _sweep("HFR_106_CurrVal_45.dta", datetime(2026, 7, 3, 9, 0))
    assert GC.select_sweep([silent], when=REC, order_id="2612091",
                           version="V26_113") is None
    other = _sweep("RO2611976-01_V26_088_HFR_104_CurrVal_45.dta",
                   datetime(2026, 10, 20, 9, 0))
    assert GC.select_sweep([other], when=REC, order_id="2612091",
                           version="V26_113") is None


def test_the_run_compares_against_the_file_matched_sweep(tmp_path):
    """gamry_compare.run end to end: a 45 A local result, the build's sweeps
    with stamps 2 h off -> the 45 A sweep is the reference."""
    g = tmp_path / "Gamry"
    g.mkdir()
    f = np.geomspace(0.2, 9000, 30)
    for n, cur in ((105, 60), (106, 45)):
        z = (0.00018 + 0.0006 / (1 + 1j * 2 * np.pi * f * 2e-3))
        start = datetime(2026, 10, 21, 13, 31, 0) + timedelta(minutes=n - 106)
        lines = ["EXPLAIN", "TAG\tGALVEIS",
                 f"STARTTIME\tLABEL\t{start:%d.%m.%Y %H:%M:%S}\tStart Time",
                 "ZCURVE\tTABLE", "\tPt\tTime\tFreq\tZreal\tZimag\tIdc\tVdc",
                 "\t#\ts\tHz\tohm\tohm\tA\tV"]
        lines += [f"\t{i}\t{i * 3}\t{ff:.6g}\t{zz.real:.6g}\t{zz.imag:.6g}"
                  f"\t{cur}\t0.8" for i, (ff, zz) in enumerate(zip(f, z))]
        (g / f"RO2612091-01_V26_113_HFR_{n}_CurrVal_{cur}.dta").write_text(
            "\n".join(lines) + "\n", encoding="latin-1")
    run = tmp_path / "45A_Test01"
    (run / "silver").mkdir(parents=True)
    za = 1000 * 304.92 * (0.00018 + 0.0006 / (1 + 1j * 2 * np.pi * f * 2e-3))
    pd.DataFrame({"freq_hz": f, "z_re_mohm_cm2": za.real,
                  "z_im_mohm_cm2": za.imag}).to_csv(
        run / "silver" / "cell_aggregate.csv", index=False)
    comps = GC.run(run, g, 304.92, out_dir=run, only="45A",
                   order_id="2612091", version="V26_113", when=REC,
                   test_id="01")
    assert comps and "HFR_106" in comps[0].sweep_name
    assert comps[0].Z_ref_full.size == 30          # the Gamry trace to plot
    assert (run / "gamry_reference_full.csv").is_file()
