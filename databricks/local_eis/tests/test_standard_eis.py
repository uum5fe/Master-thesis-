"""The standard evaluation (standard_eis.py, config.param_method = "ecm"):
R_ohmic from the high-frequency intercept, R_ct and R_mt from the two arcs
of R_ohmic + jwL + (R_ct||CPE) + (R_mt||CPE), R_pol = R_ct + R_mt. These
tests feed spectra with known values and check they come back."""

from __future__ import annotations

import numpy as np
import pytest

import gold
import silver
import standard_eis as se
from bronze import BronzeSpectrum
from config import DEFAULT
from silver import SkewModel

R_OHM, R_CT, R_MT = 0.050, 0.060, 0.090          # Ohm*cm2
TAU_CT, TAU_MT = 1 / (2 * np.pi * 150.0), 1 / (2 * np.pi * 5.0)
N_CT, N_MT = 0.85, 0.95
F = np.geomspace(0.15, 2390.0, 52)


def _z(f, loop=False):
    w = 2 * np.pi * f
    z = (R_OHM + 1j * w * 2e-7 + se.zarc(w, R_CT, TAU_CT, N_CT)
         + se.zarc(w, R_MT, TAU_MT, N_MT))
    if loop:                               # low-frequency inductive loop
        z = z - 0.03 / (1 + 1j * w * 1.0) + 0.03 / (1 + 1j * w * 0.05)
        z = z + np.where(f < 0.6, 1j * 0.004, 0)
    return z


def test_known_circuit_comes_back():
    r = se.fit_spectrum(F, _z(F))
    assert r["ok"]
    assert r["R_ohmic"] == pytest.approx(R_OHM, rel=0.06)
    assert r["R_ct"] == pytest.approx(R_CT, rel=0.10)
    assert r["R_mt"] == pytest.approx(R_MT, rel=0.10)
    assert r["R_pol"] == pytest.approx(R_CT + R_MT, rel=0.05)
    assert 100 < r["f_ct_hz"] < 220 and 3 < r["f_mt_hz"] < 8
    assert r["rms_pct"] < 2.0 and r["verdict"] == "good"
    assert r["R_ohmic_method"] in ("extrapolated", "crossing")


def test_noise_does_not_move_the_parameters_much():
    rng = np.random.default_rng(3)
    vals = []
    for _ in range(5):
        z = _z(F) * (1 + 0.02 * (rng.standard_normal(F.size)
                                 + 1j * rng.standard_normal(F.size)) / np.sqrt(2))
        r = se.fit_spectrum(F, z)
        vals.append((r["R_ohmic"], r["R_ct"], r["R_mt"], r["R_pol"]))
    v = np.array(vals)
    assert np.all(np.abs(v[:, 3] / (R_CT + R_MT) - 1) < 0.08)
    assert np.all(np.abs(v[:, 0] / R_OHM - 1) < 0.10)


def test_the_low_frequency_loop_is_left_out_of_the_fit():
    """At high current Im Z > 0 below ~0.5 Hz; the circuit has no element
    for it, so those points are not fitted (R_lf still reports them)."""
    z = _z(F, loop=True)
    assert np.any(z.imag[F < 0.6] > 0)
    r = se.fit_spectrum(F, z)
    assert r["ok"] and r["n_lf_loop_points"] > 0
    assert r["rms_pct"] < 3.0


def test_r_ohmic_is_the_crossing_when_the_band_reaches_it():
    f = np.geomspace(0.15, 30000.0, 60)
    w = 2 * np.pi * f
    z = R_OHM + 1j * w * 2e-8 + se.zarc(w, R_CT, TAU_CT, N_CT) + se.zarc(w, R_MT, TAU_MT, N_MT)
    val, how = se.hf_intercept(f, z)
    assert how == "crossing"
    assert val == pytest.approx(R_OHM, rel=0.05)


def _bronze(f, z):
    n = f.size
    snr = np.full(n, 30.0)
    return BronzeSpectrum(
        segment="5", card="Karte_1", freq=f, Z_raw=z, snr_ref_db=snr,
        snr_seg_db=snr, snr_comb_db=snr, thd=np.zeros(n), drift=np.zeros(n),
        n_per_step=np.maximum((25000 * np.maximum(8 / f, 0.05)).astype(int), 256),
        on_grid=np.ones(n, bool), channel_slot=3, ref_slot=0, n_ch_on_card=16,
        fs=25000.0, K=1.0, K_imputed=False, T_degC=60.0, u_dc=0.77,
        ref_name="UC2")


def test_the_pipeline_reports_the_circuit_values():
    """silver keeps them on the segment; gold maps them (no DRT split)."""
    skew = SkewModel(card="Karte_1", basis="structural", dt0=0.0, k_slot=0.0,
                     k_nominal=0.0, cost_gain=0.0, n_segments=1,
                     applied=False, note="")
    res = silver.process_segment(_bronze(F, _z(F)), skew, DEFAULT)
    assert res is not None
    assert res.R_ohmic_how in ("extrapolated", "crossing")
    assert res.R_ct == pytest.approx(R_CT, rel=0.10)
    assert res.R_mt == pytest.approx(R_MT, rel=0.10)
    assert res.R_pol == pytest.approx(R_CT + R_MT, rel=0.05)
    assert res.gamma.size == 0                      # no DRT was computed
    split = gold.split_processes(res, DEFAULT)
    assert split["R_ct"] == res.R_ct and split["R_mt"] == res.R_mt
    assert silver.r_ohmic_method(DEFAULT) == "ecm"
