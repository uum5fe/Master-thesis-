"""A distribution of relaxation times cannot be negative.

gamma is a sum of RC elements of a passive network, so every element is a
resistance and every resistance is positive. The posterior in `bayesian_drt`
is an unconstrained Gaussian, which does not know that, and on noisy data it
does not merely wobble: it describes the low-frequency arc with excess weight
at mid tau and cancels it with negative weight at slow tau.

`gold.split_processes` then took max(sum, 0), so that negative slow bucket
became a hard 0.0 -- and 0.0 reads as a measurement. The plate reported "no
mass transport" on a cell whose Nyquist plainly shows a transport arc, while
R_ct was inflated by the resistance that had gone missing.
"""

from __future__ import annotations

import numpy as np
import pytest

import gold
import silver
from bronze import BronzeSpectrum
from config import DEFAULT
from silver import SkewModel


FS = 10_000.0
R_OHM_TRUE, R_CT_TRUE, R_MT_TRUE = 60e-3, 40e-3, 30e-3
TAU_CT, TAU_MT = 2e-3, 0.2


def _skew():
    return SkewModel(card="Karte_1", basis="structural", dt0=0.0, k_slot=0.0,
                     k_nominal=0.0, cost_gain=0.0, n_segments=1,
                     applied=False, note="")


def _spectrum(noise: float = 0.0, seed: int = 1, n: int = 40,
              r_mt: float = R_MT_TRUE) -> BronzeSpectrum:
    """A cell with a charge-transfer arc AND a slow transport arc.

    The declared SNR is derived from the noise actually injected, so the
    weights the model is given are the truth. Telling the fit the data are
    40 dB clean while handing it 3 % scatter makes it chase noise, and any
    conclusion drawn from that is about the fixture, not the estimator.
    """
    f = np.logspace(np.log10(0.2), np.log10(3000), n)
    w = 2 * np.pi * f
    Z = (R_OHM_TRUE
         + R_CT_TRUE / (1 + (1j * w * TAU_CT) ** 0.9)
         + 1j * w * 2.6e-7)
    if r_mt:
        Z = Z + r_mt / (1 + (1j * w * TAU_MT) ** 0.9)
    n_per = np.maximum((FS * np.maximum(8.0 / f, 0.05)).astype(int), 256)
    if noise:
        rng = np.random.default_rng(seed)
        Z = Z * (1 + noise / np.sqrt(2)
                 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))
        snr = 10 * np.log10(1.0 / (n_per * noise ** 2))
    else:
        snr = np.full(n, 40.0)
    return BronzeSpectrum(
        segment="1", card="Karte_1", freq=f, Z_raw=Z, snr_ref_db=snr,
        snr_seg_db=snr, snr_comb_db=snr, thd=np.zeros(n), drift=np.zeros(n),
        n_per_step=n_per, on_grid=np.ones(n, bool), channel_slot=3, ref_slot=0,
        n_ch_on_card=16, fs=FS, K=1.0, K_imputed=False, T_degC=60.0,
        u_dc=0.77, ref_name="UC2")


def _run(noise=0.0, seed=1, nonneg=True, r_mt=R_MT_TRUE):
    cfg = DEFAULT.replace(verbose=False, drt_nonneg=nonneg)
    res = silver.process_segment(_spectrum(noise, seed, r_mt=r_mt), _skew(),
                                 cfg, None)
    assert res is not None, "fixture was rejected by the gates"
    return res, gold.split_processes(res, cfg)


# ---------------------------------------------------------------------------
# the constraint itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("noise", [0.0, 0.01, 0.03, 0.08])
def test_gamma_is_never_negative(noise) -> None:
    res, _ = _run(noise=noise)
    assert res.gamma.min() >= -1e-12, f"min gamma {res.gamma.min():.4g}"


def test_the_unconstrained_fit_really_does_go_negative() -> None:
    """The control. Without this the test above proves nothing."""
    res, _ = _run(noise=0.0, nonneg=False)
    assert res.gamma.min() < 0


# ---------------------------------------------------------------------------
# what it fixes
# ---------------------------------------------------------------------------


def test_a_real_transport_arc_is_recovered_not_zeroed() -> None:
    """The reported failure: R_mt = 0 on every segment of a plate with an arc."""
    _res, split = _run(noise=0.0)
    assert split["R_mt"] == pytest.approx(R_MT_TRUE, rel=0.20)
    assert split["R_ct"] == pytest.approx(R_CT_TRUE, rel=0.20)


def test_the_old_behaviour_loses_half_the_transport_arc() -> None:
    """Documents what the numbers in the thesis would have been."""
    _res, split = _run(noise=0.0, nonneg=False)
    assert split["R_mt"] < 0.6 * R_MT_TRUE


@pytest.mark.parametrize("noise", [0.01, 0.03, 0.08])
def test_r_pol_is_less_biased_with_the_constraint(noise) -> None:
    """The unconstrained fit loses polarisation resistance to cancellation.

    -37 % at 8 % noise, against +3.5 % constrained.
    """
    true = R_CT_TRUE + R_MT_TRUE
    off = np.mean([_run(noise=noise, seed=s, nonneg=False)[0].R_pol
                   for s in range(4)])
    on = np.mean([_run(noise=noise, seed=s, nonneg=True)[0].R_pol
                  for s in range(4)])
    assert abs(on - true) < abs(off - true)


@pytest.mark.parametrize("noise", [0.01, 0.03, 0.08])
def test_the_constraint_does_not_cost_fit_quality(noise) -> None:
    """It must not buy the split at the expense of the curve.

    Z_model is what the cell aggregate is built from, so a constraint that
    improved the parameters by degrading the fit would be a bad trade.
    """
    def err(nonneg):
        out = []
        for seed in range(4):
            res, _ = _run(noise=noise, seed=seed, nonneg=nonneg)
            f = res.freq
            w = 2 * np.pi * f
            truth = (R_OHM_TRUE + R_CT_TRUE / (1 + (1j * w * TAU_CT) ** 0.9)
                     + R_MT_TRUE / (1 + (1j * w * TAU_MT) ** 0.9)
                     + 1j * w * 2.6e-7)
            out.append(np.median(np.abs(res.Z_model - truth) / np.abs(truth)))
        return float(np.mean(out))

    assert err(True) < 1.5 * err(False) + 0.005


def test_r_ohmic_is_unaffected_by_the_constraint() -> None:
    """R_ohmic comes from the top of the band, not from the DRT.

    Worth pinning: the plate map everyone looks at must not move because of
    a change to the relaxation-time model.
    """
    on, _ = _run(noise=0.01, nonneg=True)
    off, _ = _run(noise=0.01, nonneg=False)
    assert on.R_ohmic == pytest.approx(off.R_ohmic, rel=1e-9)


# ---------------------------------------------------------------------------
# a genuinely absent process
# ---------------------------------------------------------------------------


def test_no_transport_arc_still_reads_small() -> None:
    """The constraint must not invent a process that is not there."""
    _res, split = _run(noise=0.0, r_mt=0.0)
    assert split["R_mt"] < 0.3 * R_CT_TRUE


def test_a_negative_bucket_is_unavailable_not_zero() -> None:
    """With the constraint off, a negative sum is NaN, never a hard 0.0.

    0.0 is a claim -- "this segment has no mass transport". NaN is the truth:
    the fit did not resolve it.
    """
    class _FakeSpectrum:
        gamma = np.array([-1.0, -1.0, 1.0])
        tau_grid = np.array([1e-5, 1e-1, 1e-3])
        R_ohmic, R_pol = 0.06, 0.07

    out = gold.split_processes(_FakeSpectrum(), DEFAULT)
    assert np.isnan(out["R_mt"])
    assert out["R_mt"] != 0.0
