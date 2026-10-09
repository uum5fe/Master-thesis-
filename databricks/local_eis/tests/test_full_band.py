"""Full band: evaluate every step the card can resolve, up to just below its
Nyquist frequency, with R_ohmic read where the measured spectrum crosses the
real axis -- the Gamry and bench-tool definition."""

from __future__ import annotations

import numpy as np
import pytest

import silver
from bronze import BronzeSpectrum
from config import DEFAULT
from silver import SkewModel

FS = 25_000.0
R_TRUE = 45e-3
L_TRUE = 2.3e-7          # ohm*s*cm2: the Gamry's whole-cell value on 2612030


def _skew():
    return SkewModel(card="Karte_1", basis="structural", dt0=0.0, k_slot=0.0,
                     k_nominal=0.0, cost_gain=0.0, n_segments=1,
                     applied=False, note="")


def _z(f):
    w = 2 * np.pi * f
    return (R_TRUE + 20e-3 / (1 + (1j * w * 1.5e-4) ** 0.9)
            + 60e-3 / (1 + (1j * w * 3e-2) ** 0.9) + 1j * w * L_TRUE)


def _sweep():
    """The 2612030 Gamry sweep, 30 kHz .. 0.3 Hz, 10 points per decade."""
    return 30046.88 / 10 ** (np.arange(0, 51) / 10.0)


def _spectrum(f, Z) -> BronzeSpectrum:
    n = f.size
    snr = np.full(n, 30.0)
    return BronzeSpectrum(
        segment="1", card="Karte_1", freq=f, Z_raw=Z, snr_ref_db=snr,
        snr_seg_db=snr, snr_comb_db=snr, thd=np.zeros(n), drift=np.zeros(n),
        n_per_step=np.full(n, 20000), on_grid=np.ones(n, bool), channel_slot=3,
        ref_slot=0, n_ch_on_card=16, fs=FS, K=1.0, K_imputed=False,
        T_degC=60.0, u_dc=0.77, ref_name="UC2")


def _true_intercept():
    f = np.geomspace(100, 12000, 20000)
    z = _z(f)
    i = np.flatnonzero((z.imag[:-1] < 0) & (z.imag[1:] >= 0))[0]
    return float(z.real[i])


def test_the_band_reaches_nyquist_with_full_band_and_stops_coherent_without():
    f = _sweep()
    sp = _spectrum(f, _z(f))
    kept = f[silver.gate_points(sp, DEFAULT)["keep"]]
    assert kept.max() <= 0.16 * FS                       # 4 kHz
    full = DEFAULT.replace(full_band=True)
    kept = f[silver.gate_points(sp, full)["keep"]]
    assert kept.max() == pytest.approx(11953.1, rel=1e-3)  # below fs/2
    assert not (kept > 0.5 * FS).any()                   # 15..30 kHz alias
    # the blind search keeps its ceiling; a Gamry-known step goes higher
    assert full.f_hi(FS) == pytest.approx(0.45 * FS)
    assert full.f_known_hi(FS) == pytest.approx(0.49 * FS)
    assert DEFAULT.f_known_hi(FS) == pytest.approx(0.45 * FS)


def test_intercept_is_the_real_axis_crossing_and_ignores_the_lf_loop():
    f = _sweep()
    Z = _z(f)
    Z[f < 0.5] += 1j * 5e-3              # a low-frequency inductive loop
    R, sd = silver.r_ohmic_intercept(f, Z, np.full(f.size, 0.01))
    assert R == pytest.approx(_true_intercept(), rel=0.01)
    assert np.isfinite(sd) and sd > 0
    # still capacitive at the top: no crossing, no number
    cap = f <= 1000
    assert np.isnan(silver.r_ohmic_intercept(f[cap], Z[cap])[0])


def test_full_band_r_ohmic_is_the_intercept_and_the_top_band_mean_is_not():
    """With the inductive branch inside the band the top-band mean reads the
    branch, not the cell; the intercept does not care how far the band
    goes."""
    f = _sweep()
    sp = _spectrum(f, _z(f))
    full = DEFAULT.replace(full_band=True)
    res = silver.process_segment(sp, _skew(), full)
    assert res is not None
    assert res.R_ohmic == pytest.approx(_true_intercept(), rel=0.02)
    assert res.hf_closure == 1.0
    assert silver.r_ohmic_method(full) == "intercept"
    assert silver.r_ohmic_method(DEFAULT) == "topband"
    assert silver.r_ohmic_method(DEFAULT.replace(r_ohmic_method="intercept")
                                 ) == "intercept"


def test_the_channel_lag_fit_stays_below_the_coherent_limit():
    f = _sweep()
    sp = _spectrum(f, _z(f))
    full = DEFAULT.replace(full_band=True)
    items = silver.channel_lag_items({"1": sp}, {}, full)
    fq, _Z, use = items["1"]
    assert fq[use].max() <= 0.16 * FS
