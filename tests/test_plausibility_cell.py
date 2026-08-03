"""The notebook plausibility cell, exercised against known faults.

Builds data shaped exactly like ``TONE_ONLY_ALL`` and asserts that each
injected defect is caught, that clean data is not, and that the arithmetic of
the parallel sum is right.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "plausibility_cell",
    Path(__file__).resolve().parents[1] / "notebooks" / "plausibility_check_cell.py",
)
cell = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cell)          # driver is skipped: no notebook globals

SEG_AREA = 4.235
CELL_AREA = 304.92                      # exactly 72 segments
TONES = np.array([2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
                  987, 1597, 2584, 3571.0])


def spectrum(rs_ohm=0.0140, rct=0.018, tau=2.5e-3, n=0.88, L=8e-9, f=TONES):
    w = 2 * np.pi * np.asarray(f, float)
    return rs_ohm + 1j * w * L + rct / (1.0 + (1j * w) ** n * (tau**n / rct) * rct)


def make_results(n_segments=24, segments_per_card=6, gradient=0.30, seed=0):
    """A healthy plate with a smooth Rs gradient, in TONE_ONLY_ALL shape."""
    rng = np.random.default_rng(seed)
    out = {}
    for seg in range(1, n_segments + 1):
        u = (seg - 1) / max(n_segments - 1, 1)
        rs = 0.0140 * (1.0 + gradient * u)
        Z = spectrum(rs_ohm=rs)
        Z = Z * (1 + 0.001 * rng.standard_normal(len(Z)))
        out[str(seg)] = {
            'f': TONES.copy(), 'Z': Z,
            'coh': np.full(len(TONES), 0.995),
            'card': (seg - 1) // segments_per_card + 1,
            'n_pts': len(TONES), 'kk_pct': 0.4, 'flipped': False,
        }
    return out


def coords_for(results, per_row=6):
    """A plain grid so the neighbour and detrend paths are exercised."""
    coords = {}
    for i, seg in enumerate(sorted(int(s) for s in results)):
        r, c = divmod(i, per_row)
        coords[seg] = (c * 2.0, r * 2.0, 0.9, 0.9)
    return coords


def evaluate(results, rejected=None, limits=None):
    limits = limits or cell.Limits()
    coords = coords_for(results)
    seg_to_card = {int(s): d['card'] for s, d in results.items()}
    return cell.run_checks(
        results, '150A', limits, rejected=rejected, seg_area_cm2=SEG_AREA,
        seg_to_card=seg_to_card, seg_coords=coords,
    )


# ---------------------------------------------------------------------------
# Clean data
# ---------------------------------------------------------------------------

def test_healthy_plate_with_a_gradient_passes() -> None:
    """A real Rs gradient is physics, not a card fault."""
    checks, summary = evaluate(make_results())
    bad = {s: c.flags for s, c in checks.items() if c.verdict != cell.OK}
    assert not bad, bad
    assert summary['plate_verdict'] == cell.OK
    assert summary['card_bias_detrended'] is True


def test_rs_is_reported_as_a_bound_when_the_arc_stays_open() -> None:
    """Below a few kHz the arc never crosses Im = 0; say so rather than guess."""
    checks, summary = evaluate(make_results())
    assert summary['n_rs_measured'] == 0
    assert any('upper bound' in f for f in checks[1].flags)


# ---------------------------------------------------------------------------
# Injected faults
# ---------------------------------------------------------------------------

def test_sign_flipped_segment_is_caught() -> None:
    results = make_results()
    results['5']['Z'] = np.conj(results['5']['Z'])        # arc into +Im
    checks, _ = evaluate(results)
    assert checks[5].verdict == cell.FAIL
    assert any('capacitive arc' in f or "-Z''" in f for f in checks[5].flags)


def test_rs_outlier_is_caught() -> None:
    results = make_results()
    results['7']['Z'] = results['7']['Z'] + 0.030          # +30 mOhm on Re
    checks, _ = evaluate(results)
    assert checks[7].verdict == cell.FAIL


def test_sparse_spectrum_is_caught() -> None:
    results = make_results()
    results['9']['f'] = TONES[-3:].copy()
    results['9']['Z'] = spectrum(f=TONES[-3:])
    results['9']['coh'] = np.full(3, 0.99)
    results['9']['n_pts'] = 3
    checks, _ = evaluate(results)
    assert checks[9].verdict == cell.FAIL
    assert any('frequency points' in f for f in checks[9].flags)


def test_low_coherence_and_kk_failure_are_caught() -> None:
    results = make_results()
    results['11']['coh'] = np.full(len(TONES), 0.55)
    results['12']['kk_pct'] = 22.0
    results['13']['flipped'] = True
    checks, _ = evaluate(results)
    assert checks[11].verdict == cell.FAIL
    assert checks[12].verdict == cell.FAIL
    assert checks[13].verdict == cell.WARN
    assert any('majority vote' in f for f in checks[13].flags)


def test_segments_rejected_upstream_appear_as_failures() -> None:
    """Without this, a condition that lost segments still reads 'all passed'."""
    results = make_results()
    dropped = results.pop('4')
    checks, summary = evaluate(results, rejected={4: 'quality_gate: 2 valid pts'})
    assert 4 in checks
    assert checks[4].verdict == cell.FAIL
    assert any('rejected upstream' in f for f in checks[4].flags)
    assert summary['n_records'] > summary['n_segments']


def test_known_fault_segment_is_checked_not_hidden() -> None:
    results = make_results(n_segments=36, segments_per_card=6)
    checks, _ = evaluate(results)
    assert 33 in checks, "a known-fault segment must still be reported"
    assert checks[33].verdict != cell.OK


def test_card_offset_is_caught_even_though_a_gradient_is_not() -> None:
    """A step at the card boundary is a fault; a smooth gradient is not."""
    results = make_results()
    for seg, d in results.items():
        if d['card'] == 3:
            d['Z'] = d['Z'] + 0.008                        # 8 mOhm step on one card
    checks, summary = evaluate(results)
    assert summary['plate_verdict'] in (cell.WARN, cell.FAIL)
    assert any('card 3' in msg for _, msg in summary['plate_flags'])
    # ...and it must not be promoted to a FAIL on every segment of that card
    card3 = [c for c in checks.values() if c.card == 3]
    assert all(c.verdict != cell.FAIL for c in card3) or any(
        'MAD' in f for c in card3 for f in c.flags
    )


# ---------------------------------------------------------------------------
# Parallel sum arithmetic
# ---------------------------------------------------------------------------

def test_parallel_sum_recovers_the_segment_asr() -> None:
    """72 identical segments in parallel must give back the single-segment ASR."""
    rs = 0.0140
    results = {
        str(s): {'f': TONES.copy(), 'Z': spectrum(rs_ohm=rs), 'card': 1,
                 'coh': np.full(len(TONES), 0.99), 'n_pts': len(TONES)}
        for s in range(1, 73)
    }
    par = cell.parallel_impedance(results, SEG_AREA, CELL_AREA)
    assert par is not None
    assert par['n_segs'] == 72
    assert par['area_scale'] == pytest.approx(1.0, abs=1e-9)
    # Z_cell * A_cell must equal the per-segment ASR.
    asr_cell = par['Rs_cell_mOhm'] * CELL_AREA
    asr_seg = min(spectrum(rs_ohm=rs).real) * 1000 * SEG_AREA
    assert asr_cell == pytest.approx(asr_seg, rel=1e-6)


def test_partial_coverage_scales_by_area_not_by_luck() -> None:
    """With 60 of 72 measured, the scaling must be 60/72."""
    rs = 0.0140
    results = {
        str(s): {'f': TONES.copy(), 'Z': spectrum(rs_ohm=rs), 'card': 1,
                 'coh': np.full(len(TONES), 0.99), 'n_pts': len(TONES)}
        for s in range(1, 61)
    }
    par = cell.parallel_impedance(results, SEG_AREA, CELL_AREA)
    assert par['area_scale'] == pytest.approx(60 / 72, rel=1e-9)
    asr_cell = par['Rs_cell_mOhm'] * CELL_AREA
    asr_seg = min(spectrum(rs_ohm=rs).real) * 1000 * SEG_AREA
    assert asr_cell == pytest.approx(asr_seg, rel=1e-6)


def test_a_segment_missing_a_frequency_is_not_treated_as_an_open_circuit() -> None:
    """The nansum trap: a gap must shrink the grid, never zero a conductance."""
    rs = 0.0140
    results = {
        str(s): {'f': TONES.copy(), 'Z': spectrum(rs_ohm=rs), 'card': 1,
                 'coh': np.full(len(TONES), 0.99), 'n_pts': len(TONES)}
        for s in range(1, 11)
    }
    # one segment is missing the two lowest tones
    results['3']['f'] = TONES[2:].copy()
    results['3']['Z'] = spectrum(rs_ohm=rs, f=TONES[2:])

    par = cell.parallel_impedance(results, SEG_AREA, CELL_AREA)
    assert par['n_freq_common'] == len(TONES) - 2, "grid must shrink to the overlap"
    assert par['n_freq_dropped'] == 2
    # every retained frequency still has all 10 segments contributing
    expected = spectrum(rs_ohm=rs, f=par['f']) / 10.0
    assert np.allclose(par['Z_par'], expected, rtol=1e-9)


def test_parallel_sum_survives_a_faulty_admittance() -> None:
    results = {
        str(s): {'f': TONES.copy(), 'Z': spectrum(), 'card': 1,
                 'coh': np.full(len(TONES), 0.99), 'n_pts': len(TONES)}
        for s in range(1, 11)
    }
    par = cell.parallel_impedance(results, SEG_AREA, CELL_AREA, exclude={3, 4})
    assert par['n_segs'] == 8
    assert np.all(np.isfinite(par['Z_cell']))


def test_gamry_comparison_uses_the_plate_band() -> None:
    """Gamry must not get a less biased intercept from frequencies the plate lacks."""
    f_gamry = np.logspace(0, 4.3, 60)                # up to 20 kHz
    Z = spectrum(f=f_gamry)
    gamry = {'freq': f_gamry, 'zreal': Z.real, 'zimag': Z.imag, 'n_pts': len(f_gamry)}

    wide, _ = cell.gamry_rs_mohm(gamry, 1.0, 20000.0)
    matched, _ = cell.gamry_rs_mohm(gamry, 1.0, 20000.0, f_ref_hz=TONES.max())
    assert matched > wide, "restricting to the plate band must not lower the bias"
    assert abs(matched - min(spectrum(f=TONES).real) * 1000) < 0.5


# ---------------------------------------------------------------------------
# Plate-level diagnostics driven by the real 45 A data
# ---------------------------------------------------------------------------

def test_an_independent_split_is_not_flagged() -> None:
    """Rs and Rct varying on their own is physics; it must pass."""
    rng = np.random.default_rng(3)
    results = make_results(n_segments=40, segments_per_card=10)
    for i, (s, d) in enumerate(sorted(results.items(), key=lambda kv: int(kv[0]))):
        rs = 0.0140 * (1 + 0.15 * rng.standard_normal())
        rct = 0.018 * (1 + 0.15 * rng.standard_normal())
        d['Z'] = spectrum(rs_ohm=rs, rct=rct)
    _, summary = evaluate(results)
    assert summary['split_independence'] > 0.6, summary['split_independence']


def test_a_wandering_intercept_is_caught_at_plate_level() -> None:
    """The 45 A signature: the total is stable, only the split moves."""
    rng = np.random.default_rng(4)
    results = make_results(n_segments=40, segments_per_card=10)
    for s, d in results.items():
        total = 0.0320
        frac = 0.20 + 0.55 * rng.random()          # split point wanders
        d['Z'] = spectrum(rs_ohm=total * frac, rct=total * (1 - frac))
    _, summary = evaluate(results)
    assert summary['split_independence'] < 0.35
    assert any('not independent' in msg for _, msg in summary['plate_flags'])
    assert summary['plate_verdict'] == cell.FAIL


def test_a_collapsed_card_mapping_is_reported() -> None:
    """All segments on 'card 0' silently disables the card-bias check."""
    results = make_results()
    for d in results.values():
        d['card'] = 0
    _, summary = evaluate(results)
    assert any('card-bias' in msg for _, msg in summary['plate_flags'])
    assert any('SEG_TO_CARD' in msg for _, msg in summary['plate_flags'])
