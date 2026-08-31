"""Schedule detection when only one card was recorded.

The consensus rule asks how many cards saw a step. With one card the answer
is always 1, so the rule can never pass -- every step then depends on the
geometric grid alone, and a refused grid fit leaves nothing at all. That is
not hypothetical: a converted DASYLab recording of a single card produced 44
candidate steps and zero accepted ones, and the failure surfaced as

    IndexError: list index out of range

from the summary line reading kept[0] to print the frequency range, which
discarded the grid warning that would have explained it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import bronze as B
from bronze import Step
from config import Config

FS = 100_000.0


class _FakeFamos:
    """Stands in for the reader; these tests are about the accept rule."""
    fs = FS

    def __init__(self, path):
        pass

    def channel(self, name):
        return np.zeros(16)


def _steps(freqs, snr) -> list[Step]:
    return [Step(freq=float(f), start=i * 1000, stop=(i + 1) * 1000, amp=1.0,
                 snr_db=float(s), thd=0.01, stationarity=0.01)
            for i, (f, s) in enumerate(zip(freqs, snr))]


def _consensus(monkeypatch, steps, n_cards=1, cfg=None):
    monkeypatch.setattr(B, "FamosFile", _FakeFamos)
    monkeypatch.setattr(B, "detect_schedule", lambda *a, **k: steps)
    cards = {f"card{i}": type("C", (), {"ref_name": "UC2"})()
             for i in range(1, n_cards + 1)}
    files = [Path(f"card{i}.DAT") for i in range(1, n_cards + 1)]
    return B.consensus_schedule(files, cards, cfg or Config())


#: The real recording's shape: a sweep the detector found, plus the spurious
#: steps between the true ones that blind detection always produces.
_REAL = np.geomspace(0.239, 3714.8, 26)
_SPURIOUS = np.sort(np.random.default_rng(0).uniform(0.3, 3000.0, 18))


def _mixed():
    freqs = np.sort(np.concatenate([_REAL, _SPURIOUS]))
    strong = set(_REAL.tolist())
    return _steps(freqs, [20.0 if f in strong else 2.0 for f in freqs])


def test_one_card_recovers_its_sweep_instead_of_crashing(monkeypatch):
    """The reported failure, end to end.

    Card agreement is unavailable, and the grid fit is refused on this mix,
    so before the fix both admission routes were shut and the summary line
    crashed on the empty list.
    """
    kept, grid = _consensus(monkeypatch, _mixed())

    assert grid.get("solo_card") is True
    assert len(kept) == len(_REAL), "the confident steps are the sweep"
    # Admission came from SNR, not from corroboration that does not exist.
    assert min(s.snr_db for s in kept) >= Config().min_snr_db


def test_the_spurious_steps_are_still_rejected(monkeypatch):
    """The fallback must not become "accept everything on the one card".

    Substituting a gate that passes all 44 candidates would turn a crash into
    a spectrum built partly from noise, which is worse: it does not announce
    itself.
    """
    kept, _ = _consensus(monkeypatch, _mixed())
    freqs = np.array([s.freq for s in kept])
    for junk in _SPURIOUS:
        assert not np.any(np.isclose(freqs, junk, rtol=1e-9)), (
            f"a 2 dB candidate at {junk:.3f} Hz was accepted")


def test_a_weak_grid_cannot_certify_the_steps_it_was_fitted_on(monkeypatch):
    """Fitting the grid through the junk makes all the junk "on grid".

    With fewer than four confident steps the fit falls back to every
    candidate, and the result was then used as evidence for those same
    candidates. The flag that marks it (`weak_basis`) was set and never read,
    so a perfectly geometric run of 1 dB detections certified itself.
    """
    flat = _steps(np.geomspace(0.239, 3714.8, 44), [1.0] * 44)
    with pytest.raises(SystemExit) as exc:
        _consensus(monkeypatch, flat)
    assert "not usable as evidence" in str(exc.value)


def test_the_refusal_says_which_of_the_two_routes_failed(monkeypatch):
    """An empty schedule stops the run, so the message is the whole output.

    It has to name the card count, the best SNR seen and the grid verdict --
    those three decide which of the two admission routes was shut, and the
    person reading it cannot get them any other way.
    """
    flat = _steps(np.geomspace(0.239, 3714.8, 44), [1.0] * 44)
    with pytest.raises(SystemExit) as exc:
        _consensus(monkeypatch, flat)
    text = str(exc.value)
    assert "cards      : 1" in text
    assert "1.0 dB" in text
    assert "44 cluster(s)" in text


def test_two_cards_still_decide_by_agreement(monkeypatch):
    """The fallback is for the case where agreement cannot be asked for.

    Letting it run whenever it would admit more steps would quietly replace
    corroboration with SNR on every multi-card recording in the campaign.
    """
    both = _steps(np.geomspace(0.239, 3714.8, 44), [20.0] * 44)
    kept, grid = _consensus(monkeypatch, both, n_cards=2)
    assert grid.get("solo_card") is None
    assert len(kept) == 44


def test_a_third_card_is_still_solo_when_two_are_needed(monkeypatch):
    """`min_ref_channels` is the threshold, not the number two."""
    cfg = dataclasses.replace(Config(), min_ref_channels=4)
    steps = _steps(np.geomspace(0.239, 3714.8, 44), [20.0] * 44)
    _, grid = _consensus(monkeypatch, steps, n_cards=3, cfg=cfg)
    assert grid.get("solo_card") is True
