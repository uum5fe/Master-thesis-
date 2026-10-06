"""A weak alignment peak that another card reproduces is a confirmed one.

Regression cover for RO2612030 / 150 A, where the shared UC2 reference on
cards 1 and 2 was contaminated and the two cards scored |r| = 0.083 at
prominence 7.6-7.7 against the anchor -- under the gate, so both were
refused and left on their own clock.  They had nevertheless returned
+215634 and +215687 samples: 53 samples, 2.1 ms, apart on an 8.63 s offset.
That agreement is the evidence the old gate had no way to use.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "databricks" / "local_eis"))

import bronze  # noqa: E402


class _Cfg:
    align_cards = True
    align_min_corr = 0.05
    align_min_prominence = 15.0
    align_agree_tol_s = 0.02
    align_corroborate_min_prominence = 5.0


class _Card:
    fs = 25_000.0


def _decide(cand: dict[str, tuple[int, float, float]]) -> dict[str, bool]:
    """The acceptance rule of estimate_card_lags, over measured candidates."""
    cfg, cards = _Cfg(), {s: _Card() for s in cand}
    out = {}
    for stem, (lag, corr, prom) in cand.items():
        partners = bronze._agreeing_cards(stem, lag, cand, cards, cfg)
        rescued = bool(partners and prom < cfg.align_min_prominence
                       and prom >= cfg.align_corroborate_min_prominence)
        out[stem] = bool(abs(corr) >= cfg.align_min_corr
                         and (prom >= cfg.align_min_prominence or rescued))
    return out


#: Verbatim from the RO2612030 150 A bronze log.
FIELD_150A = {
    "Karte_1": (+215_634, 0.083, 7.7),
    "Karte_2": (+215_687, 0.083, 7.6),
    "Karte_3": (-71_113, 0.995, 25.7),
    "Karte_5": (+440, 0.995, 25.7),
}


def test_the_150A_pair_is_rescued_by_mutual_agreement() -> None:
    assert _decide(FIELD_150A) == {
        "Karte_1": True, "Karte_2": True, "Karte_3": True, "Karte_5": True,
    }


def test_strong_cards_are_unaffected() -> None:
    """Removing the weak pair must not change the strong cards' verdicts."""
    strong = {k: v for k, v in FIELD_150A.items() if k in ("Karte_3", "Karte_5")}
    assert all(_decide(strong).values())


def test_a_lone_weak_peak_is_still_refused() -> None:
    """Corroboration is the only way in; nothing vouches for one card."""
    cand = dict(FIELD_150A)
    del cand["Karte_2"]
    assert _decide(cand)["Karte_1"] is False


def test_two_pieces_of_garbage_cannot_vouch_for_each_other() -> None:
    """Below align_corroborate_min_prominence, agreement proves nothing.

    4.1 is the worst genuinely-bad alignment measured in this campaign
    (RO2612025); it must not be rescuable no matter what agrees with it.
    """
    cand = {"A": (+10_000, 0.06, 4.1), "B": (+10_010, 0.06, 3.9),
            "Karte_3": (-71_113, 0.995, 25.7)}
    assert _decide(cand) == {"A": False, "B": False, "Karte_3": True}


def test_agreement_must_be_within_the_tolerance() -> None:
    """40 ms apart is not agreement: 0.02 s is the window."""
    cand = {"A": (+215_634, 0.083, 7.7), "B": (+216_634, 0.083, 7.6)}
    assert _decide(cand) == {"A": False, "B": False}


@pytest.mark.parametrize("delta_samples, expect", [(0, True), (53, True),
                                                   (500, True), (501, False)])
def test_tolerance_edge(delta_samples: int, expect: bool) -> None:
    """align_agree_tol_s = 0.02 s at 25 kHz is exactly 500 samples."""
    cand = {"A": (+215_634, 0.083, 7.7),
            "B": (+215_634 + delta_samples, 0.083, 7.6)}
    assert _decide(cand)["A"] is expect


def test_absolute_floor_still_applies_to_a_rescued_card() -> None:
    """Corroboration relaxes prominence, never the |r| floor."""
    cand = {"A": (+215_634, 0.01, 7.7), "B": (+215_687, 0.083, 7.6)}
    assert _decide(cand)["A"] is False


def test_guard_band_is_expressed_in_seconds() -> None:
    """The peak is ~1/f_lo wide, so the guard cannot be a sample count."""
    import numpy as np

    mag = np.full(600_001, 1.0)
    mag += 0.01 * np.random.default_rng(0).standard_normal(mag.size)
    k = 300_000
    # a peak 2 s wide at 25 kHz: 50000 samples of shoulder either side
    shoulder = np.exp(-0.5 * (np.arange(-50_000, 50_001) / 20_000.0) ** 2)
    mag[k - 50_000:k + 50_001] += 40.0 * shoulder

    narrow = bronze._prominence(mag, k, guard=5_000)
    wide = bronze._prominence(mag, k, guard=50_000)
    # counting the peak's own shoulders as background understates the score
    assert wide > narrow