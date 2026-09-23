"""How should the high-frequency intercept be measured?

"HFR is where the spectrum crosses the real axis" is the textbook definition,
and taking it literally is the worst thing you can do with a band-limited
measurement. When the arc does not close inside the band there is no crossing
to find, and a search for one latches onto the inductive tail instead -- and
still returns a number, which is what makes it dangerous on a plate map where
nobody inspects 72 Nyquists.

These tests put four estimators on spectra whose R_ohm is known and let the
numbers decide.
"""

from __future__ import annotations

import numpy as np
import pytest

import silver
from bronze import BronzeSpectrum
from config import DEFAULT
from silver import SkewModel


FS = 10_000.0
R_TRUE = 60e-3
L_TRUE = 2.6e-7


def _skew():
    return SkewModel(card="Karte_1", basis="structural", dt0=0.0, k_slot=0.0,
                     k_nominal=0.0, cost_gain=0.0, n_segments=1,
                     applied=False, note="")


def _spectrum(f_hi: float, noise: float = 0.02, seed: int = 0,
              extra=None, n: int = 40) -> BronzeSpectrum:
    f = np.logspace(np.log10(0.2), np.log10(f_hi), n)
    w = 2 * np.pi * f
    Z = R_TRUE + 40e-3 / (1 + (1j * w * 2e-3) ** 0.9) + 1j * w * L_TRUE
    if extra:
        Z = Z + extra[0] / (1 + (1j * w * extra[1]) ** 0.9)
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


def _zero_crossing(freq, Z) -> float:
    """The literal reading: interpolate Im Z to zero, from the top down."""
    o = np.argsort(freq)
    f, z = np.asarray(freq)[o], np.asarray(Z)[o]
    im = np.imag(z)
    for i in range(len(f) - 1, 0, -1):
        if im[i] * im[i - 1] < 0:
            t = im[i] / (im[i] - im[i - 1])
            return float(np.real(z[i]) * (1 - t) + np.real(z[i - 1]) * t)
    return float("nan")


def _run(f_hi, seeds=8, extra=None):
    out = {"topband": [], "xint": [], "crossing": [], "at_fmax": []}
    for seed in range(seeds):
        res = silver.process_segment(_spectrum(f_hi, seed=seed, extra=extra),
                                     _skew(), DEFAULT.replace(verbose=False),
                                     None)
        if res is None:
            continue
        out["topband"].append(res.R_ohmic)
        out["xint"].append(res.R_ohmic_xint)
        out["at_fmax"].append(float(np.real(res.Z_corr[np.argmax(res.freq)])))
        out["crossing"].append(_zero_crossing(res.freq, res.Z_corr))
    return {k: np.array(v, float) for k, v in out.items()}


def _bias(v):
    v = v[np.isfinite(v)]
    return float(np.mean(v) - R_TRUE) if v.size else float("nan")


# ---------------------------------------------------------------------------
# when the arc closes inside the band, everything works
# ---------------------------------------------------------------------------


def test_with_a_closed_arc_every_estimator_agrees() -> None:
    r = _run(f_hi=30_000)
    for name in ("topband", "xint", "crossing", "at_fmax"):
        assert abs(_bias(r[name])) < 3e-3, f"{name}: {1000*_bias(r[name]):+.2f}"


# ---------------------------------------------------------------------------
# when it does not, the crossing fails and keeps a straight face
# ---------------------------------------------------------------------------


def test_the_zero_crossing_breaks_when_the_arc_stays_open() -> None:
    """The finding. +40 mOhm*cm2 on a true 60 -- a 67 % error, returned as a
    number with no warning attached."""
    r = _run(f_hi=1000)
    assert _bias(r["crossing"]) > 20e-3


def test_the_axis_fit_survives_the_open_arc() -> None:
    """Same idea as the crossing -- read the intercept at Im Z = 0 -- but
    extrapolated to the axis instead of requiring the data to reach it."""
    r = _run(f_hi=1000)
    assert abs(_bias(r["xint"])) < 3e-3


def test_the_shipped_estimator_degrades_gracefully() -> None:
    """The top-band mean is biased when the arc is open, but by a millohm,
    not by forty."""
    r = _run(f_hi=1000)
    assert 0 < _bias(r["topband"]) < 5e-3


def test_with_a_slow_arc_the_crossing_finds_nothing_at_all() -> None:
    r = _run(f_hi=1000, extra=(30e-3, 0.2))
    assert not np.isfinite(r["crossing"]).any() or _bias(r["crossing"]) > 20e-3
    assert abs(_bias(r["xint"])) < 5e-3


# ---------------------------------------------------------------------------
# the two shipped numbers, together
# ---------------------------------------------------------------------------


def test_the_pair_brackets_the_truth_when_the_arc_is_closed() -> None:
    r = _run(f_hi=30_000)
    lo = min(np.nanmean(r["topband"]), np.nanmean(r["xint"]))
    hi = max(np.nanmean(r["topband"]), np.nanmean(r["xint"]))
    assert lo - 1e-3 <= R_TRUE <= hi + 1e-3


def test_disagreement_between_them_tracks_arc_closure() -> None:
    """The pair is a diagnostic, not just a second opinion: they separate
    when the band stops short of closing the arc."""
    closed = _run(f_hi=30_000)
    open_ = _run(f_hi=1000)
    gap_closed = abs(np.nanmean(closed["topband"]) - np.nanmean(closed["xint"]))
    gap_open = abs(np.nanmean(open_["topband"]) - np.nanmean(open_["xint"]))
    assert gap_open > 2 * gap_closed


def test_the_cross_check_is_reported_per_segment() -> None:
    res = silver.process_segment(_spectrum(3000, seed=1), _skew(),
                                 DEFAULT.replace(verbose=False), None)
    assert np.isfinite(res.R_ohmic_xint)
    assert res.R_ohmic_xint != res.R_ohmic       # genuinely a second estimate
