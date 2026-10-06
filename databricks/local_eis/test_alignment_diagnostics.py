"""The alignment band, the clock-drift diagnostic and the closure check.

Each of these answers a question the single global lag cannot: was the band
the lag was measured in recorded, did the two cards keep the same sample RATE
across the record, and do all the card offsets belong to one consistent
timing solution.

The synthetics here are built so the right answer is known in advance --
a shift that was put there deliberately, a clock error of a stated ppm --
because a diagnostic verified only against field data whose truth nobody
knows is not verified.
"""

from __future__ import annotations

import numpy as np
import pytest

import bronze as B
import make_synth_famos as M
from bronze import CardInfo
from config import DEFAULT
from eis_local import FamosFile


FS = 10_000.0


def sweep(n: int, seed: int = 0, noise: float = 0.0,
          n_steps: int = 24) -> np.ndarray:
    """A stepped-sine record like the reference channel carries.

    The dwells are spread over the WHOLE record, unlike the fixture in
    test_card_alignment.py, which fills only its first seconds. The blockwise
    diagnostic below re-estimates the lag inside each fifth of the record, so
    a record that is silent after the first block has nothing for four of the
    five blocks to measure and the ppm slope is then fitted to noise.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    t = np.arange(n) / FS
    span = n // n_steps                 # one dwell + one gap per step
    dwell = int(0.7 * span)
    for k, freq in enumerate(np.geomspace(1.0, 250.0, n_steps)):
        a = k * span
        b = min(n, a + dwell)
        if a >= n:
            break
        x[a:b] = np.sin(2 * np.pi * freq * t[a:b])
    if noise:
        x = x + noise * rng.standard_normal(n)
    return x - x.mean()


# ---------------------------------------------------------------------------
# sign and magnitude: the convention everything else is built on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [+12_345, -7_777, 0])
def test_a_known_shift_comes_back_with_the_right_sign(shift: int) -> None:
    """`lag` is how far this card's record is DELAYED behind the reference.

    Every other piece of arithmetic in the alignment depends on this: the
    schedule subtracts the lag to reach the common base, process_card adds it
    to get back, and the closure check below predicts lag(a) - lag(b) from
    it. A sign error here would be invisible in the correlation scores and
    catastrophic downstream, so it is pinned.
    """
    n = int(40 * FS)
    clean = sweep(n)
    ref = clean + 0.2 * np.random.default_rng(1).standard_normal(n)
    card = np.roll(clean, shift) + 0.2 * np.random.default_rng(2).standard_normal(n)

    lag, corr, prom = B._best_lag(card, ref, max_lag=int(5 * FS))
    assert lag == pytest.approx(shift, abs=2)
    assert abs(corr) > 0.1


# ---------------------------------------------------------------------------
# 2. the alignment band is configurable, and the lag does not depend on it
# ---------------------------------------------------------------------------


def _write_card(path, x, fs=FS):
    """A FAMOS card carrying `x` on UC2 plus two segment channels."""
    n = len(x)
    t = np.arange(n) / fs
    data = np.column_stack([0.78 + 0.01 * x,
                            0.13 + 0.002 * x,
                            0.13 + 0.002 * np.roll(x, 5)])
    return M.write_v2(path, ["UC2", "1", "2"], data, fs)


def _cards_for(paths):
    out = {}
    for p in paths:
        fam = FamosFile(p)
        out[p.stem] = CardInfo(path=p, stem=p.stem, fs=fam.fs, n_ch=fam.n_ch,
                               n_samples=fam.n_samples,
                               duration_s=fam.n_samples / fam.fs,
                               ref_name="UC2", ref_slot=0, n_segments=2)
    return out


@pytest.mark.parametrize("f_hi", [100.0, 300.0, 1000.0, 3000.0])
def test_the_lag_is_stable_across_alignment_bands(tmp_path, f_hi) -> None:
    """The band is a knob on the ESTIMATOR, not on the answer.

    If the recovered lag moved with the band, the band would be a free
    parameter fitted to taste. It does not, over a 30x range of upper limits,
    which is what makes it safe to expose.
    """
    n = int(40 * FS)
    clean = sweep(n)
    shift = int(2.5 * FS)
    a = _write_card(tmp_path / "Karte_1.DAT",
                    clean + 0.2 * np.random.default_rng(1).standard_normal(n))
    b = _write_card(tmp_path / "Karte_2.DAT",
                    np.roll(clean, shift)
                    + 0.2 * np.random.default_rng(2).standard_normal(n))
    cards = _cards_for([a, b])
    cfg = DEFAULT.replace(align_f_hi_hz=f_hi)

    ta = B._alignment_trace("Karte_1", cards, cfg)
    tb = B._alignment_trace("Karte_2", cards, cfg)
    lag, _corr, _prom = B._best_lag(tb, ta, int(12 * FS),
                                    B._guard_samples(cfg, FS))
    assert lag == pytest.approx(shift, abs=int(0.002 * FS))


def test_a_band_above_nyquist_is_clamped_not_zeroed(tmp_path) -> None:
    """A band chosen for a fast card must not blank a slow one.

    Without the clamp, `X[(fr < f_lo) | (fr > f_hi)] = 0` on a card whose
    Nyquist sits below f_lo zeroes the entire trace, and an all-zero
    correlation is a silent refusal rather than a stated one.
    """
    n = int(20 * FS)
    p = _write_card(tmp_path / "Karte_1.DAT", sweep(n))
    cards = _cards_for([p])
    cfg = DEFAULT.replace(align_f_hi_hz=500_000.0)     # far above 0.45*fs
    tr = B._alignment_trace("Karte_1", cards, cfg)
    assert np.any(np.abs(tr) > 0)


def test_an_inverted_band_is_refused_rather_than_guessed(tmp_path) -> None:
    n = int(10 * FS)
    p = _write_card(tmp_path / "Karte_1.DAT", sweep(n))
    cards = _cards_for([p])
    cfg = DEFAULT.replace(align_f_lo_hz=400.0, align_f_hi_hz=100.0)
    with pytest.raises(ValueError, match="invalid alignment band"):
        B._alignment_trace("Karte_1", cards, cfg)


# ---------------------------------------------------------------------------
# 3. clock drift
# ---------------------------------------------------------------------------


def _with_clock_error(x: np.ndarray, ppm: float) -> np.ndarray:
    """The same record as seen by a card whose clock runs `ppm` fast."""
    n = len(x)
    idx = np.arange(n) * (1.0 + ppm * 1e-6)
    return np.interp(idx, np.arange(n), x, left=0.0, right=0.0)


def test_a_clean_pair_reports_no_clock_mismatch() -> None:
    """The null: a constant offset and no drift must read as ~0 ppm."""
    n = int(60 * FS)
    clean = sweep(n)
    ref = clean + 0.1 * np.random.default_rng(1).standard_normal(n)
    shift = int(1.5 * FS)
    card = np.roll(clean, shift) + 0.1 * np.random.default_rng(2).standard_normal(n)

    d = B._blockwise_lag_diagnostic(card, ref, shift, FS, DEFAULT)
    assert d["ok"]
    assert abs(d["clock_mismatch_ppm"]) < DEFAULT.align_max_clock_ppm
    assert d["within_limit"]


@pytest.mark.parametrize("ppm", [50.0, -50.0, 200.0])
def test_a_deliberate_clock_error_is_detected(ppm: float) -> None:
    """A stated ppm comes back with the right sign and the right order.

    50 ppm over this 60 s record is 3 ms -- an eighth of a 25 ms dwell, and
    invisible to a single global lag, which simply averages it away.
    """
    n = int(60 * FS)
    clean = sweep(n)
    ref = clean + 0.05 * np.random.default_rng(1).standard_normal(n)
    drifted = _with_clock_error(clean, ppm)
    drifted = drifted + 0.05 * np.random.default_rng(2).standard_normal(n)

    lag, _corr, _prom = B._best_lag(drifted, ref, int(12 * FS))
    d = B._blockwise_lag_diagnostic(drifted, ref, lag, FS, DEFAULT)

    assert d["ok"]
    # a clock running fast makes the record arrive EARLIER as time goes on,
    # so the lag slides negative: sign is the opposite of the clock error
    assert np.sign(d["clock_mismatch_ppm"]) == -np.sign(ppm)
    assert abs(d["clock_mismatch_ppm"]) == pytest.approx(abs(ppm), rel=0.35)
    assert not d["within_limit"]


def test_the_diagnostic_compares_the_same_physical_interval() -> None:
    """The bug this was written to avoid.

    Slicing both records at [a:b] and searching +-50 ms asks for an 8 s
    offset inside a 50 ms window, so every block fails and the diagnostic
    reports drift that is really just the global lag. Blocks must be taken
    at ref[a:b] against x[a+lag : b+lag].
    """
    n = int(60 * FS)
    clean = sweep(n)
    shift = int(8.0 * FS)            # far larger than the +-50 ms search
    ref = clean + 0.1 * np.random.default_rng(1).standard_normal(n)
    card = np.roll(clean, shift) + 0.1 * np.random.default_rng(2).standard_normal(n)

    d = B._blockwise_lag_diagnostic(card, ref, shift, FS, DEFAULT)
    assert d["ok"]
    # residuals are about the lag we already have, not about finding it again
    assert all(abs(r["residual_samples"]) < 0.005 * FS for r in d["blocks"])
    assert abs(d["clock_mismatch_ppm"]) < DEFAULT.align_max_clock_ppm


def test_the_drift_diagnostic_can_be_switched_off() -> None:
    n = int(20 * FS)
    x = sweep(n)
    d = B._blockwise_lag_diagnostic(x, x, 0, FS, DEFAULT.replace(align_drift_blocks=0))
    assert d["ok"] is False


# ---------------------------------------------------------------------------
# 4. closure
# ---------------------------------------------------------------------------


def _closure_fixture(lag_b: int, lag_c: int, claimed_c: int | None = None):
    n = int(40 * FS)
    clean = sweep(n)
    rng = np.random.default_rng(7)
    traces = {
        "A": clean + 0.1 * rng.standard_normal(n),
        "B": np.roll(clean, lag_b) + 0.1 * rng.standard_normal(n),
        "C": np.roll(clean, lag_c) + 0.1 * rng.standard_normal(n),
    }
    cards = {s: CardInfo(path=None, stem=s, fs=FS, n_ch=3, n_samples=n,
                         duration_s=n / FS, ref_name="UC2", ref_slot=0,
                         n_segments=2) for s in traces}
    lags = {
        "A": {"lag": 0, "applied": True},
        "B": {"lag": lag_b, "applied": True},
        "C": {"lag": claimed_c if claimed_c is not None else lag_c,
              "applied": True},
    }
    return list(traces), traces, lags, cards


def test_consistent_lags_close_to_zero() -> None:
    """Three cards whose offsets are mutually consistent must close."""
    stems, traces, lags, cards = _closure_fixture(+15_000, -9_000)
    out = B._pairwise_closure(stems, traces, lags, cards, DEFAULT,
                              B._guard_samples(DEFAULT, FS))
    assert out["n_pairs"] == 3
    assert out["max_abs_error_samples"] <= 2
    assert out["consistent"]


def test_one_wrong_lag_breaks_the_closure_and_names_the_pairs() -> None:
    """The point of the check: a wrong lag that correlated well is still wrong.

    Card C really sits at -9000 but its accepted lag says +30000. Nothing in
    C's own correlation score can reveal that; the closure against A and B
    can, and does.
    """
    stems, traces, lags, cards = _closure_fixture(+15_000, -9_000,
                                                  claimed_c=+30_000)
    out = B._pairwise_closure(stems, traces, lags, cards, DEFAULT,
                              B._guard_samples(DEFAULT, FS))
    assert not out["consistent"]
    bad = [r for r in out["pairs"] if abs(r["closure_error_samples"]) > 500]
    assert {tuple(sorted((r["card_a"], r["card_b"]))) for r in bad} == {
        ("A", "C"), ("B", "C")}


def test_refused_cards_are_left_out_of_the_closure() -> None:
    """An unapplied lag is not a claim, so it cannot fail a consistency test."""
    stems, traces, lags, cards = _closure_fixture(+15_000, -9_000,
                                                  claimed_c=+30_000)
    lags["C"]["applied"] = False
    out = B._pairwise_closure(stems, traces, lags, cards, DEFAULT,
                              B._guard_samples(DEFAULT, FS))
    assert out["n_pairs"] == 1
    assert out["consistent"]


# ---------------------------------------------------------------------------
# mixed sample rates
# ---------------------------------------------------------------------------


def test_cards_at_a_different_rate_are_refused_not_correlated(tmp_path) -> None:
    """A lag in samples is meaningless between two sample rates.

    `_best_lag` walks two arrays index by index. If one card sampled at
    50 kHz and the other at 100 kHz, index n is a different instant on each,
    and the offset it returns is neither seconds nor samples of anything. The
    campaign mixes rates, so this has to end in a stated refusal rather than
    a number.
    """
    n = int(20 * FS)
    clean = sweep(n)
    a = _write_card(tmp_path / "Karte_1.DAT", clean)
    b = _write_card(tmp_path / "Karte_2.DAT", clean)
    c = _write_card(tmp_path / "Karte_3.DAT", clean)
    cards = _cards_for([a, b, c])
    cards["Karte_3"].fs = 2 * FS                 # the odd card out

    lags = B.estimate_card_lags([a, b, c], cards, DEFAULT)

    assert lags["Karte_3"]["applied"] is False
    assert "sample rate" in lags["Karte_3"]["refused_reason"]
    # and the majority is still aligned to each other
    assert lags["Karte_1"]["applied"] and lags["Karte_2"]["applied"]


def test_the_anchor_is_taken_from_the_largest_rate_group(tmp_path) -> None:
    """Refusing the outlier, not the majority.

    Picking the anchor from all cards at once can land it on the single card
    whose rate differs, which then refuses every other card on the plate --
    the rare case made to reject the common one.
    """
    n = int(20 * FS)
    clean = sweep(n)
    paths = [_write_card(tmp_path / f"Karte_{i}.DAT", clean) for i in (1, 2, 3)]
    cards = _cards_for(paths)
    cards["Karte_1"].fs = 4 * FS                 # first card is the outlier

    lags = B.estimate_card_lags(paths, cards, DEFAULT)

    assert lags["Karte_1"]["applied"] is False
    assert sum(bool(v.get("applied")) for v in lags.values()) == 2
