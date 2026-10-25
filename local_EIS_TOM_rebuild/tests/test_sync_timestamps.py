"""Header time stamps: used to pick the sweep and the test, never to align.

RO2612086 60 A Test_02: the cards' |NT stamps are 46 s apart, their records
2.0 s apart (UC2 cross-correlation); the Gamry PC is 12 s ahead of one card
and 56 s ahead of the other. Both 60 A sweeps of the up/down ramp (08:39 and
09:34) share one frequency ladder, so only the clock can tell them apart.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from eis_local import famos_start_time
import gamry_compare as GC


def _write(tmp_path, name, header):
    p = tmp_path / name
    p.write_bytes(header.encode("latin-1") + b"\x00" * 2048)
    return p


def test_v1_and_v2_stamps_are_read(tmp_path):
    v1 = _write(tmp_path, "v1.DAT", "|CF,1;|NT,1,13,20.10.2026,132,09:33:15.00;|CP,3;")
    v2 = _write(tmp_path, "v2.DAT", "|CF,2,1,1;|NT,1,22,19,10,2026,12,35,20.00;|CC,1;")
    assert famos_start_time(v1) == datetime(2026, 10, 20, 9, 33, 15)
    assert famos_start_time(v2) == datetime(2026, 10, 19, 12, 35, 20)
    assert famos_start_time(_write(tmp_path, "none.DAT", "|CF,1;")) is None


def _sweep(name, start, dur=136.0):
    return GC.CellSweep(path=Path(name), freq=np.array([1.0]),
                        Z_ohm=np.array([1 + 0j]), current_a=60.0,
                        started=start, meta={"duration_s": dur})


UP = _sweep("V26_109_HFR_103_CurrVal_60.dta", datetime(2026, 10, 20, 8, 39, 41), 147)
DOWN = _sweep("V26_109_HFR_111_CurrVal_60.dta", datetime(2026, 10, 20, 9, 34, 1))


def test_the_sweep_that_overlaps_the_recording_wins_in_any_order():
    rec = (datetime(2026, 10, 20, 9, 32, 29), datetime(2026, 10, 20, 9, 36, 56))
    assert GC.select_sweep([UP, DOWN], when=rec) is DOWN
    assert GC.select_sweep([DOWN, UP], when=rec) is DOWN
    # a recording made at the first 60 A point, with a PC clock a minute off
    rec1 = (datetime(2026, 10, 20, 8, 39, 0), datetime(2026, 10, 20, 8, 42, 30))
    assert GC.select_sweep([DOWN, UP], when=rec1) is UP


def test_without_stamps_the_test_number_decides():
    assert GC.select_sweep([DOWN, UP], test_id="01") is UP
    assert GC.select_sweep([UP, DOWN], test_id="02") is DOWN


def test_a_single_sweep_is_returned_as_is():
    assert GC.select_sweep([UP]) is UP
    assert GC.select_sweep([]) is None
