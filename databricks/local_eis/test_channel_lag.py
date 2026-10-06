"""Per-channel lag: the diagnostic finds it, the gain file it writes undoes it,
and the FAMOS path actually applies a gain file.

The 2612030 HFR map had neighbours 20 mOhm*cm2 apart because some segment
chains lag the cell voltage by up to ~100 us (first-order, H = 1/(1+jw tau))
and silver reads R_ohmic as Re Z at the top of the band. These tests build
that situation from a known plate and check each step of the fix.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

import bronze                                               # noqa: E402
import diagnose_channel_lag as dcl                          # noqa: E402
import utils                                                # noqa: E402
from config import Config                                   # noqa: E402

F = np.geomspace(0.2, 1200.0, 40)
TAU = {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0,
       "15": 90e-6, "19": 105e-6, "23": 110e-6, "34": -35e-6}


def _true_z(f, R=60.0, Rct=150.0, Q=2e-3, n=0.9):
    """A plate where every segment really has R_s = 60 mOhm*cm2."""
    w = 2 * np.pi * f
    return R + 1.0 / (1.0 / Rct + Q * (1j * w) ** n)


def _run(tmp_path, label="60A"):
    run = tmp_path / f"snr_{label}"
    (run / "bronze").mkdir(parents=True)
    (run / "gold").mkdir()
    rows, ps = [], []
    for s, tau in TAU.items():
        z = _true_z(F) / (1 + 1j * 2 * np.pi * F * tau)       # chain lag
        rows += [dict(segment=int(s), freq_hz=f, z_re_ohm_cm2=zz.real / 1e3,
                      z_im_ohm_cm2=zz.imag / 1e3) for f, zz in zip(F, z)]
        # what silver reports: Re Z at the top of the band
        ps.append(dict(segment=int(s), **{"class": "measured"},
                       R_ohmic=float(z[-1].real)))
    pd.DataFrame(rows).to_csv(run / "bronze" / "raw_spectra.csv", index=False)
    pd.DataFrame(ps).to_csv(run / "gold" / "plate_summary.csv", index=False)
    (run / "config_used.json").write_text(json.dumps(
        {"leepa": "2612030", "condition": label}))
    return run


def test_the_lag_is_recovered_per_segment(tmp_path):
    df = dcl.analyse_run(_run(tmp_path))
    for s, tau in TAU.items():
        assert df.loc[s, "tau_us"] == pytest.approx(1e6 * tau, abs=3.0), s


def test_removing_the_lag_makes_neighbours_agree(tmp_path):
    df = dcl.analyse_run(_run(tmp_path))
    # top-of-band Re Z, as silver reads it: the lagging channels read low
    assert df.loc["19", "R_ohmic"] < df.loc["1", "R_ohmic"] - 10
    fixed = df.Rs_fit_lag_removed
    assert fixed.max() - fixed.min() < 2.0
    assert fixed.median() == pytest.approx(60.0, abs=2.0)


def test_the_gain_file_round_trips_through_the_pipeline_reader(tmp_path):
    rows = dcl.gain_rows({"19": 105e-6, "1": 0.0})
    p = tmp_path / "g.csv"
    p.write_text("segment,freq_hz,gain_real,gain_imag\n"
                 + "".join(f"{s},{f},{gr},{gi}\n" for s, f, gr, gi in rows))
    gain = utils.load_gain(p)
    g = utils.gain_at(gain, "19", F)
    assert np.allclose(g, 1 / (1 + 1j * 2 * np.pi * F * 105e-6), rtol=2e-3)
    measured = _true_z(F) / (1 + 1j * 2 * np.pi * F * 105e-6)
    assert np.allclose(measured / g, _true_z(F), rtol=2e-3)


def test_main_writes_tables_and_gain(tmp_path, capsys):
    out = tmp_path / "out"
    assert dcl.main([str(_run(tmp_path, "45A")), str(_run(tmp_path, "60A")),
                     "-o", str(out), "--write-gain", str(out / "g.csv")]) == 0
    T = pd.read_csv(out / "channel_lag_all_runs.csv", index_col=0)
    assert T.shape == (len(TAU), 2)
    assert "repeatability" in capsys.readouterr().out
    assert set(utils.load_gain(out / "g.csv")) == set(TAU)


def test_bronze_hands_the_gain_file_to_every_card(tmp_path, monkeypatch):
    """cfg.gain_file used to be accepted and never read on the FAMOS path."""
    rows = dcl.gain_rows({"19": 105e-6})
    gfile = tmp_path / "g.csv"
    gfile.write_text("segment,freq_hz,gain_real,gain_imag\n"
                     + "".join(f"{s},{f},{gr},{gi}\n" for s, f, gr, gi in rows))
    fp = tmp_path / "Karte_1.DAT"
    fp.write_bytes(b"x")
    seen = {}

    def _process_card(fp_, cal, schedule, grid, cfg, log, **kw):
        seen.update(kw)
        return {}

    cal = SimpleNamespace(has_current_cal=True, seg_c0={}, temp_c0={})
    monkeypatch.setattr(bronze, "discover_files", lambda cfg: [fp])
    monkeypatch.setattr(bronze, "inventory_channels",
                        lambda files, cfg, log=None: ({}, {fp.stem: None}))
    monkeypatch.setattr(bronze, "PlateCalibration",
                        SimpleNamespace(load=lambda *a: cal))
    monkeypatch.setattr(bronze, "estimate_card_lags", lambda *a, **k: {})
    monkeypatch.setattr(bronze, "plate_temperatures", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(bronze, "consensus_schedule",
                        lambda *a, **k: ([], {}))
    monkeypatch.setattr(bronze, "process_card", _process_card)
    cfg = Config(dat_dir=tmp_path, out_dir=tmp_path, verbose=False,
                 gain_file=gfile)
    bronze.run(cfg)
    assert "19" in seen["gain"]
    seen.clear()
    bronze.run(Config(dat_dir=tmp_path, out_dir=tmp_path, verbose=False))
    assert seen["gain"] == {}
