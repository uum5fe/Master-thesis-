"""Card-level tone estimation (tone_estimation.py) and its wiring into bronze.

The estimator's own accuracy checks live in tone_estimation._selftest; these
tests pin the parts the pipeline depends on: the self-test passes, phasors at
the card frequency match the known truth, and `card_ml` is really the default
the evaluation runs with -- also from the command line.
"""
import numpy as np

import config
import tone_estimation


FS = 10_000.0


def test_selftest_passes():
    assert tone_estimation._selftest() == 0


def test_card_frequency_recovers_the_shared_tone_from_an_offset_prior():
    rng = np.random.default_rng(3)
    f, n = 1234.567, int(0.3 * FS)
    t = np.arange(n) / FS
    Y = np.array([a * np.cos(2 * np.pi * f * t + p) + 0.02 * rng.normal(size=n)
                  for a, p in zip(rng.uniform(0.1, 1, 8),
                                  rng.uniform(-np.pi, np.pi, 8))])
    est = tone_estimation.card_frequency(Y, FS, f_prior=f * 1.0009)
    assert not est.at_bound
    assert est.n_channels == 8
    assert abs(est.freq - f) < 10 * est.crlb_hz + 1e-3


def test_phasors_at_returns_the_true_ratio():
    f, n = 317.0, int(0.5 * FS)
    t = np.arange(n) / FS
    z = 0.7 * np.exp(-0.4j)
    ref = np.cos(2 * np.pi * f * t) + 0.5
    seg = abs(z) * np.cos(2 * np.pi * f * t + np.angle(z)) + 0.1 * t
    A, _, _ = tone_estimation.phasors_at(np.vstack([ref, seg]), FS, f)
    assert abs(A[1] / A[0] - z) < 1e-9


def test_stationarity_test_is_skipped_on_too_short_windows():
    d, p = tone_estimation.stationarity_test(np.ones(20), FS, 10.0)
    assert np.isnan(d) and np.isnan(p)


def test_card_ml_is_the_default_everywhere():
    assert config.DEFAULT.phasor_method == "card_ml"
    # an argparse default must not silently put joint7 back
    assert config.Config.from_cli([]).phasor_method == "card_ml"
    cfg = config.Config.from_cli(["--phasor", "joint7", "--drift-alpha", "0.01"])
    assert cfg.phasor_method == "joint7" and cfg.drift_alpha == 0.01
