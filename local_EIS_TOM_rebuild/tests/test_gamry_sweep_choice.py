"""Which Gamry sweep is a run's reference -- RO2612090 450 A (2026-10-21).

The shared Gamry folder held 450 A sweeps of six builds; one had no
STARTTIME. The old rule needed a time on EVERY candidate, fell back to the
test number over all builds and took V26_087 from 2026-07-03 (another cell):
the sync, the card phase calibration and the whole-cell comparison then ran
against it."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

import gamry_compare as GC
import mdf4_min

DATA = Path(__file__).parent / "data"
MF4 = DATA / "RO2612090-01_V26_112_HFR_101_CurrVal_450.mf4"
REC = (datetime(2026, 10, 21, 9, 57, 31), datetime(2026, 10, 21, 10, 4, 2))


def _sweep(name, start, dur=140.0):
    return GC.CellSweep(path=Path(name), freq=np.array([1.0]),
                        Z_ohm=np.array([1 + 0j]), current_a=450.0,
                        started=start, meta={"duration_s": dur})


RIGHT = _sweep("V26_112_HFR_101_CurrVal_450.dta", datetime(2026, 10, 21, 10, 1, 6))
JULY = _sweep("V26_087_HFR_104_CurrVal_450.dta", datetime(2026, 7, 3, 9, 14, 23))
OTHER = _sweep("V26_109_HFR_106_CurrVal_450.dta", datetime(2026, 10, 20, 9, 0, 14))
NO_TIME = _sweep("V26_088_HFR_104_CurrVal_450.dta", None)


def test_a_sweep_without_start_time_no_longer_disables_the_time_choice():
    assert GC.select_sweep([JULY, NO_TIME, OTHER, RIGHT], when=REC,
                           test_id="01") is RIGHT


def test_no_sweep_near_the_recording_gives_none_not_another_cells():
    assert GC.select_sweep([JULY, NO_TIME, OTHER], when=REC,
                           test_id="01") is None
    assert GC.select_sweep([JULY], when=REC) is None      # single, far away


def test_without_time_several_builds_give_none():
    assert GC.select_sweep([JULY, OTHER, RIGHT], test_id="01") is None
    up = _sweep("V26_112_HFR_101_CurrVal_450.dta", datetime(2026, 10, 21, 10, 1))
    down = _sweep("V26_112_HFR_108_CurrVal_450.dta", datetime(2026, 10, 21, 11, 1))
    assert GC.select_sweep([down, up], test_id="02") is down   # one build


def test_mdf4_sweep_is_read():
    ch, meta = mdf4_min.read(MF4)
    assert meta["STARTTIME"] == "21.10.2026 10:01:06"
    assert len(ch["freq"]) == 53 and abs(ch["freq"][0] - 30046.88) < 0.01
    assert 1e-4 < ch["zreal"][0] < 1e-3                    # ohm


def test_mf4_sweeps_are_found_and_never_taken_as_the_bench_log(tmp_path):
    shutil.copy(MF4, tmp_path / MF4.name)
    found = GC.find_cell_sweeps(tmp_path, order_id="2612090",
                                version=None)
    assert [s.condition for s in found] == ["450A"]
    sw = found[0]
    assert sw.started == datetime(2026, 10, 21, 10, 1, 6)
    assert sw.path.is_file() and sw.path.stem == MF4.stem
    assert len(sw.freq) == 53
    assert GC.find_bench_log(tmp_path, order_id="2612090") is None


def test_the_run_compares_against_the_sweep_of_its_own_time(tmp_path):
    g = tmp_path / "Gamry"
    g.mkdir()
    shutil.copy(MF4, g / MF4.name)
    july = GC.mf4_sweep_as_dta(MF4, tmp_path / "c")
    text = july.read_text(encoding="latin-1").replace(
        "21.10.2026 10:01:06", "03.07.2026 09:14:23")
    (g / "V26_087_HFR_104_CurrVal_450.dta").write_text(text, encoding="latin-1")
    found = GC.find_cell_sweeps(g)
    hits = [s for s in found if s.condition == "450A"]
    assert len(hits) == 2
    assert GC.select_sweep(hits, when=REC).path.stem == MF4.stem


def test_one_spurious_high_snr_point_does_not_invert_a_segment():
    """RO2612090 450 A: a drift 'step' at 0.150 Hz with 150 dB SNR outvoted
    the real low-frequency points and inverted 30 segments."""
    import bronze
    from config import DEFAULT
    f = np.logspace(np.log10(0.15), 4, 47)
    Z = 0.15 - 0.02j * np.ones_like(f)          # a healthy segment, ohm*cm2
    snr = np.full_like(f, 15.0)
    Z[0], snr[0] = -0.6, 150.0                   # the drift "step"
    assert np.all(bronze._fix_polarity(Z, f, snr, DEFAULT).real[1:] > 0)
    assert np.all(bronze._fix_polarity(-Z, f, snr, DEFAULT).real[1:] > 0)
