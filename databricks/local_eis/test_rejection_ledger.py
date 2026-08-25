"""Why did a point, or a whole segment, not make it into the result?

Nine gates run in sequence in silver, and until now only their totals were
kept. These tests plant one defect at a time and assert that the ledger names
the gate that is actually responsible -- because a wrong reason is worse than
no reason: it sends you to fix the wrong thing.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import silver
from bronze import BronzeSpectrum
from config import DEFAULT
from silver import SkewModel


FS = 10_000.0


def _spectrum(n=40, f_hi=4000.0, snr_db=None, **kw) -> BronzeSpectrum:
    f = np.logspace(np.log10(0.2), np.log10(f_hi), n)
    w = 2 * np.pi * f
    Z = 60e-3 + 300e-3 / (1 + (1j * w * 2e-3) ** 0.85) + 1j * w * 2.6e-7
    snr = np.full(n, 40.0) if snr_db is None else np.asarray(snr_db, float)
    # Dwell length as a real stepped sweep does it: long enough for several
    # cycles, so the low-frequency points are not starved by construction.
    # A fixed dwell would kill 0.2 Hz on the cycles gate and make every test
    # here about the wrong thing.
    n_per = np.maximum((FS * np.maximum(8.0 / f, 0.05)).astype(int), 256)
    d = dict(segment="1", card="Karte_1", freq=f, Z_raw=Z,
             snr_ref_db=snr, snr_seg_db=snr, snr_comb_db=snr,
             thd=np.zeros(n), drift=np.zeros(n),
             n_per_step=n_per, on_grid=np.ones(n, bool),
             channel_slot=3, ref_slot=0, n_ch_on_card=16, fs=FS,
             K=1.0, K_imputed=False, T_degC=60.0, u_dc=0.77, ref_name="UC1")
    d.update(kw)
    return BronzeSpectrum(**d)


def _skew():
    return SkewModel(card="Karte_1", basis="structural", dt0=0.0, k_slot=0.0,
                     k_nominal=0.0, cost_gain=0.0, n_segments=1,
                     applied=False, note="")


def _run(sp, cfg=None):
    cfg = cfg or DEFAULT.replace(f_min_hz=0.15, f_max_hz=4500.0)
    ledger: list[dict] = []
    res = silver.process_segment(sp, _skew(), cfg, None, ledger=ledger)
    return res, ledger


def _reasons(ledger):
    return Counter(r["reason"] for r in ledger if not r["kept"])


# ---------------------------------------------------------------------------


def test_every_recorded_point_appears_exactly_once(_=None):
    """The ledger is a census, not a sample: one row per point, always."""
    sp = _spectrum(n=40)
    res, ledger = _run(sp)
    assert res is not None
    assert len(ledger) == 40
    assert len({r["freq_hz"] for r in ledger}) == 40
    assert sum(r["kept"] for r in ledger) + sum(not r["kept"] for r in ledger) == 40


def test_a_clean_spectrum_loses_nothing():
    res, ledger = _run(_spectrum())
    assert res is not None
    assert all(r["kept"] for r in ledger), _reasons(ledger)


def test_collapsing_snr_is_named_as_the_bandwidth_limit():
    """The real shape: strong at the bottom, gone at the top.

    This is the case the user meets -- a spectrum that simply stops partway up
    -- and the ledger has to say SNR rather than leave it to be guessed.
    """
    f = np.logspace(np.log10(0.2), np.log10(4000), 40)
    snr = np.clip(60 - 22 * np.log10(f / 0.2), -20, 60)
    res, ledger = _run(_spectrum(snr_db=snr))
    assert res is not None

    kept = [r for r in ledger if r["kept"]]
    dropped = [r for r in ledger if not r["kept"]]
    assert kept and dropped
    # everything kept is below everything dropped: a clean bandwidth edge
    assert max(r["freq_hz"] for r in kept) < min(r["freq_hz"] for r in dropped)
    assert set(_reasons(ledger)) == {"snr"}

    run = silver.SilverRun(spectra={}, skew={}, dc_closure={},
                           cell_freq=np.zeros(0), Z_cell=np.zeros(0),
                           point_ledger=ledger)
    reach = run.reach()[0]
    assert reach["blocked_by"] == "snr"
    assert reach["n_kept"] == len(kept)
    assert reach["f_max_hz"] == pytest.approx(max(r["freq_hz"] for r in kept))


def test_too_few_cycles_is_not_blamed_on_snr():
    """A one-sample-per-decade dwell at 0.2 Hz has no cycles in it.

    Both gates would fire on such a point; the ledger must attribute it to the
    one that actually came first, or the fix aims at the wrong knob.
    """
    n = 40
    short = np.full(n, 200)          # 20 ms of record per point, at every f
    res, ledger = _run(_spectrum(n_per_step=short))
    low = [r for r in ledger if r["freq_hz"] < 10 and not r["kept"]]
    assert low, "the low-frequency points should not have survived"
    assert all(r["reason"] == "cycles" for r in low), _reasons(ledger)


def test_an_implausible_magnitude_is_named():
    sp = _spectrum()
    Z = np.asarray(sp.Z_raw, complex).copy()
    Z[5] = 5.0 + 0j                  # 5000 mOhm.cm2, far outside the window
    res, ledger = _run(_spectrum(Z_raw=Z))
    assert _reasons(ledger)["magnitude"] == 1


def test_a_segment_with_nothing_left_is_recorded_not_silently_dropped():
    """This is the "why is segment 33 missing" case.

    It used to return None with no trace anywhere in the outputs. The segment
    still has no spectrum -- that is correct -- but now it has a reason.
    """
    res, ledger = _run(_spectrum(segment="33", snr_db=np.full(40, -5.0)))
    assert res is None
    assert len(ledger) == 40
    assert all(not r["kept"] for r in ledger)
    verdict = ledger[0]["segment_verdict"]
    assert verdict.startswith("dropped:")
    assert "min_points_per_spectrum" in verdict
    assert set(_reasons(ledger)) == {"snr"}


def test_a_segment_that_was_never_wired_says_so_instead():
    """No ADC channel is a different fact from measured-and-rejected.

    Both end as "no spectrum", and telling them apart is the difference
    between a wiring job and a measurement problem.
    """
    run = silver.SilverRun(spectra={}, skew={}, dc_closure={},
                           cell_freq=np.zeros(0), Z_cell=np.zeros(0),
                           point_ledger=[], unwired=["33", "57"])
    rows = {r["segment"]: r for r in run.reach()}
    assert set(rows) == {"33", "57"}
    for r in rows.values():
        assert r["blocked_by"] == "no_channel"
        assert r["n_points"] == 0
        assert "never recorded" in r["explanation"]


def test_the_reason_vocabulary_is_complete():
    """Every gate name the code can emit must have an explanation."""
    seen = set()
    for snr in (np.full(40, 40.0), np.full(40, -5.0),
                np.clip(60 - 22 * np.log10(
                    np.logspace(np.log10(0.2), np.log10(4000), 40) / 0.2),
                    -20, 60)):
        _res, ledger = _run(_spectrum(snr_db=snr))
        seen |= {r["reason"] for r in ledger if not r["kept"]}
    assert seen, "no gate fired at all, so this proves nothing"
    assert seen <= set(silver.REJECT_REASONS), seen - set(silver.REJECT_REASONS)
