"""The standard pipeline (std_pipeline.py), end to end on synthetic raw
FAMOS data: cards armed at different times, a descending stepped sweep, a
Gamry .DTA, a multiplexer delay per channel, a known circuit per segment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import make_synth_sweep as synth          # noqa: E402
import std_pipeline as sp                 # noqa: E402


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("std")
    g = synth.main(d / "raw")
    cfg = sp.StdConfig(leepa=g["leepa"], condition=g["cond"], dat_dir=g["dir"],
                       gamry_dta=next(Path(g["dir"]).glob("*.DTA")),
                       curr_cal=g["dir"] / "curr.csv", temp_cal=g["dir"] / "temp.csv",
                       out_dir=d / "out", f_min_hz=0.5, f_max_hz=2400.0)
    return sp.run(cfg, log=lambda *a: None), g


def test_every_step_is_located_on_every_card(result):
    r, g = result
    s = r["schedule"]
    assert s.card.nunique() == 3
    n_band = int(((g["freq"] >= 0.5) & (g["freq"] <= 2400)).sum())
    assert (s.groupby("card").fit_start.count() == n_band).all()
    # each window lies inside its step: true step order is the sweep order
    for _c, d in s.groupby("card"):
        assert np.all(np.diff(d.fit_start.values) > 0)


def test_the_measured_impedance_is_the_true_one(result):
    r, g = result
    spec = r["spectra"]
    errs = []
    for _, row in spec[spec.kept].iterrows():
        zt = synth.true_z(g["truth"][row.segment], row.freq_hz)
        zm = (row.z_re_mohm_cm2 + 1j * row.z_im_mohm_cm2) / 1000
        errs.append(abs(zm / zt - 1))
    assert np.median(errs) < 0.005 and np.max(errs) < 0.03
    assert spec.kept.mean() > 0.95


def test_the_standard_parameters_come_back(result):
    r, g = result
    P = r["parameters"]
    P = P[P["class"] == "measured"]
    assert len(P) == 18
    for _, row in P.iterrows():
        Rs, Rct, Rmt = (1000 * v for v in g["truth"][str(int(row.segment))])
        assert row.R_ohmic == pytest.approx(Rs, rel=0.08)
        assert row.R_pol == pytest.approx(Rct + Rmt, rel=0.06)
        assert row.R_ct == pytest.approx(Rct, rel=0.2)
        assert row.R_mt == pytest.approx(Rmt, rel=0.2)
        assert row.j_dc == pytest.approx(1.4, rel=0.01)
        assert row.T_degC == pytest.approx(60.0, abs=0.5)


def test_outputs_are_written(result):
    r, _g = result
    out = Path(r["out_dir"])
    for f in ("spectra.csv", "spectra_all.csv", "schedule.csv", "parameters.csv",
              "aggregate.csv", "gamry_comparison.csv", "plate_statistics.csv",
              "manifest.json", "nyquist.png", "gamry_vs_aggregate.png",
              "plate_R_ohmic_interp.png", "plate_R_mt_interp.png"):
        assert (out / f).exists(), f
    assert not list(out.glob("map_*"))        # only interpolated maps
    ag = pd.read_csv(out / "aggregate.csv")
    assert (ag.n_segments == 18).all()


def test_without_the_adc_delay_correction_the_phase_is_off(tmp_path):
    """The multiplexer delay is the one timing correction applied, and it
    is a known hardware number: switching it off leaves a phase error that
    grows with frequency."""
    g = synth.main(tmp_path / "raw", n_cards=1)
    base = dict(leepa=g["leepa"], condition=g["cond"], dat_dir=g["dir"],
                gamry_dta=next(Path(g["dir"]).glob("*.DTA")),
                curr_cal=g["dir"] / "curr.csv", temp_cal=g["dir"] / "temp.csv",
                f_min_hz=0.5, f_max_hz=2400.0, write_png=False)
    on = sp.run(sp.StdConfig(out_dir=tmp_path / "on", **base), log=lambda *a: None)
    off = sp.run(sp.StdConfig(out_dir=tmp_path / "off", adc_skew=False, **base),
                 log=lambda *a: None)

    def phase_err(r):
        s = r["spectra"]
        s = s[(s.freq_hz > 500) & (s.segment == "6")]
        z = (s.z_re_mohm_cm2 + 1j * s.z_im_mohm_cm2).values / 1000
        zt = synth.true_z(g["truth"]["6"], s.freq_hz.values)
        return float(np.max(np.abs(np.degrees(np.angle(z / zt)))))
    assert phase_err(on) < 0.5
    assert phase_err(off) > 1.0
