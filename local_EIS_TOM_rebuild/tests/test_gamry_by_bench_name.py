"""The bench .mf4 name ties an order to its build, and the build to its
Gamry sweeps -- in the datago folder and on the Volume alike.

    20261021_1243_1295_70758_RO2612091-01_V26_113_lokale_EIS_6_Boxen_Naboo_1.mf4
                             ^^^^^^^^^ order  ^^^^^^^ build             ^^^^^ plate
    V26_113_HFR_101_CurrVal_611.dta        build, sweep number, current

The Gamry file does not name the order; the bench file does, and says the
build. So the run of RO2612091 at 611 A takes V26_113_HFR_101_CurrVal_611 and
not the 611 A sweep of another build lying in the same folder."""

from __future__ import annotations

from datetime import datetime

import numpy as np

import gamry_compare as GC
import runner

BENCH = "20261021_1243_1295_70758_RO2612091-01_V26_113_lokale_EIS_6_Boxen_Naboo_1.mf4"


def _dta(path, cur, start):
    f = np.geomspace(0.2, 9000, 20)
    z = 0.00018 + 0.0006 / (1 + 1j * 2 * np.pi * f * 2e-3)
    lines = ["EXPLAIN", "TAG\tGALVEIS",
             f"STARTTIME\tLABEL\t{start:%d.%m.%Y %H:%M:%S}\tStart Time",
             "ZCURVE\tTABLE", "\tPt\tTime\tFreq\tZreal\tZimag\tIdc\tVdc",
             "\t#\ts\tHz\tohm\tohm\tA\tV"]
    lines += [f"\t{i}\t{i * 3}\t{ff:.6g}\t{zz.real:.6g}\t{zz.imag:.6g}\t{cur}\t0.6"
              for i, (ff, zz) in enumerate(zip(f, z))]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")


def test_the_bench_name_gives_the_build_and_the_build_the_sweep(tmp_path):
    g = tmp_path / "Gamry"
    g.mkdir()
    (g / BENCH).write_bytes(b"")
    _dta(g / "V26_113_HFR_101_CurrVal_611.dta", 611, datetime(2026, 10, 21, 11, 0))
    _dta(g / "V26_112_HFR_101_CurrVal_611.dta", 611, datetime(2026, 10, 21, 13, 0))
    _dta(g / "V26_113_HFR_105_CurrVal_60.dta", 60, datetime(2026, 10, 21, 11, 40))
    R = runner.Session({"spark": None}, famos_root=tmp_path, gamry_root=g,
                       results_root=tmp_path / "r", config_root=tmp_path / "c",
                       scratch=tmp_path / "s")
    R.say = lambda *a: None
    version = R.plate_version("2612091")
    assert version == "V26_113"                          # from the bench name
    assert GC.find_bench_log(g, order_id="2612091", version=version).name == BENCH
    found = GC.find_cell_sweeps(R.gamry_dir("2612091"), order_id="2612091",
                                version=version)
    rec = (datetime(2026, 10, 21, 13, 5), datetime(2026, 10, 21, 13, 9))
    for cur, want in (("611A", "V26_113_HFR_101_CurrVal_611"),
                      ("60A", "V26_113_HFR_105_CurrVal_60")):
        hits = [s for s in found if s.condition == cur]
        sw = GC.select_sweep(hits, when=rec, test_id="01", order_id="2612091",
                             version=version)
        assert sw is not None and sw.path.stem == want, cur
    # no sweep of the build for 300 A: none, never another build's
    assert not [s for s in found if s.condition == "300A"]


def test_the_bench_name_flags_the_wrong_plate(tmp_path):
    g = tmp_path / "Gamry"
    g.mkdir()
    (g / BENCH).write_bytes(b"")
    R = runner.Session({"spark": None}, famos_root=tmp_path, gamry_root=g,
                       results_root=tmp_path / "r", config_root=tmp_path / "c",
                       scratch=tmp_path / "s")
    assert "gen2" in R.plate_hint("2612091", "gen1")    # Naboo, gen1 chosen
    assert R.plate_hint("2612091", "gen2") == ""
