"""The cell aggregate, written as an admittance sum, and the gaps in it.

    1 / Z_cell(f) = SUM_v 1 / Z_v(f) = SUM_v A_v / z_v(f)

z_v is the area-specific impedance of segment v [ohm.cm2] and A_v its area,
so A_v / z_v is that segment's admittance and the sum is the cell's. Turning
the sum back into an area-specific number needs an area, and there are two
candidates that differ by exactly the part of the plate that returned nothing:
the area actually covered, and the whole plate. Both are computed, because
choosing one silently is how an aggregate over 96 % of a plate gets compared
against a whole-cell instrument as if it were the same quantity.

The gap is closed by rebuilding the missing segments from the ones that touch
them, which is what `fill_missing_from_neighbours` and `substitute_segments`
do -- the second for a segment whose own measurement is present but not
trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

import neighbours
import r2d2_geometry as geom
import silver
from config import DEFAULT
from silver import SilverSpectrum


N_F = 30


def _spectrum(seg: str, r_ohm: float = 0.060, r_pol: float = 0.30):
    f = np.geomspace(0.2, 1000.0, N_F)
    w = 2 * np.pi * f
    z = r_ohm + r_pol / (1 + 1j * w * 2e-3)
    return SilverSpectrum(
        segment=seg, card="Karte_1", freq=f, Z_corr=z, Z_model=z,
        Z_model_sd=np.full(N_F, 1e-3), sigma_rel=np.full(N_F, 0.02),
        R_ohmic=r_ohm, R_ohmic_sd=1e-3, R_ohmic_drt=r_ohm, hf_arc_open=0.0,
        hf_closure=1.0, R_pol=r_pol, L=2e-7, dt_applied=0.0, tau_peak=2e-3,
        gamma=np.zeros(4), tau_grid=np.geomspace(1e-4, 1e-1, 4),
        kk_res_max=0.01, kk_mu=0.6, zhit_dev=0.01, zhit_drift=0.0,
        hf_phase_slope=0.0, T_degC=60.0, K=1.0, K_imputed=False, u_dc=0.77,
        area_cm2=float(geom.SEGMENTS[seg].area_cm2), n_used=N_F, n_dropped=0,
        snr_med_db=30.0, thd_med=0.01, tier="A")


@pytest.fixture()
def full_plate():
    rng = np.random.default_rng(0)
    return {s: _spectrum(s, 0.060 + 0.004 * rng.standard_normal())
            for s in sorted(geom.SEGMENTS, key=int)}


# ---------------------------------------------------------------------------
# the formula
# ---------------------------------------------------------------------------


def test_the_admittances_add(full_plate) -> None:
    """1/Z_cell = sum A_v / z_v, checked against the sum done by hand."""
    agg = silver.cell_aggregate_full(full_plate)
    i = len(agg["freq"]) // 2
    f = agg["freq"][i]

    Y = 0j
    for sp in full_plate.values():
        z = np.interp(f, sp.freq, sp.Z_model.real) + 1j * np.interp(
            f, sp.freq, sp.Z_model.imag)
        Y += sp.area_cm2 / z
    assert agg["Y"][i] == pytest.approx(Y, rel=1e-6)
    assert agg["Z_ohm"][i] == pytest.approx(1.0 / Y, rel=1e-6)


def test_the_two_normalisations_differ_by_the_coverage(full_plate) -> None:
    """z_full / z_used is A_cell / A_used, exactly. Nothing else."""
    part = {k: v for k, v in full_plate.items() if k not in ("33", "65", "66")}
    agg = silver.cell_aggregate_full(part)
    ratio = agg["z_asr_full"] / agg["z_asr_used"]
    assert np.allclose(ratio.real, agg["area_cell"] / agg["area_used"])
    assert np.all(agg["area_coverage"] < 1.0)


def test_a_complete_plate_makes_them_identical(full_plate) -> None:
    agg = silver.cell_aggregate_full(full_plate)
    assert np.allclose(agg["area_coverage"], 1.0)
    assert np.allclose(agg["z_asr_used"], agg["z_asr_full"])


def test_identical_segments_give_back_their_own_asr(full_plate) -> None:
    """The sanity check the formula has to pass: 72 identical segments in
    parallel have the ASR of ONE of them, at every frequency, whatever their
    areas. The area cancels exactly, which is the whole point of working in
    area-specific units."""
    same = {s: _spectrum(s, 0.060) for s in sorted(geom.SEGMENTS, key=int)}
    agg = silver.cell_aggregate_full(same)
    one = same["19"]
    z_one = np.interp(agg["freq"], one.freq, one.Z_model.real) + 1j * np.interp(
        agg["freq"], one.freq, one.Z_model.imag)
    assert np.allclose(agg["z_asr_used"], z_one, rtol=1e-6)
    # and the absolute impedance is that ASR over the whole plate area
    assert np.allclose(agg["Z_ohm"] * agg["area_cell"], z_one, rtol=1e-6)


def test_the_low_impedance_segments_dominate(full_plate) -> None:
    """Harmonic, not arithmetic: this is why an integral measurement hides a
    local fault."""
    flooded = dict(full_plate)
    flooded["19"] = _spectrum("19", 0.060, r_pol=3.0)     # 10x the R_pol
    a = silver.cell_aggregate_full(full_plate)["z_asr_used"]
    b = silver.cell_aggregate_full(flooded)["z_asr_used"]
    shift = float(np.nanmax(np.abs(b - a) / np.abs(a)))
    assert shift < 0.05, "a 10x local fault should barely move the cell curve"


# ---------------------------------------------------------------------------
# rebuilding a segment from the ring that touches it
# ---------------------------------------------------------------------------


def test_the_donors_are_the_segments_that_touch_it() -> None:
    donors, hops = neighbours.donor_ring("33", ["61", "62", "67", "68", "19"])
    assert set(donors) == {"61", "62", "67", "68"}
    assert hops == 1


def test_the_search_widens_only_when_the_ring_is_unmeasured() -> None:
    """Segment 33's own case: 67 and 68 are missing too, so it rests on 61
    and 62 -- still one hop, because those two ARE in its ring."""
    donors, hops = neighbours.donor_ring("33", ["61", "62"])
    assert donors == ["61", "62"] and hops == 1


def test_a_segment_with_nothing_within_reach_is_refused() -> None:
    donors, hops = neighbours.donor_ring("33", ["19", "20"], max_hops=2)
    assert donors == [] and hops == 0


def test_a_rebuilt_segment_looks_like_its_neighbours(full_plate) -> None:
    part = {k: v for k, v in full_plate.items() if k != "33"}
    f, z, info = neighbours.fill_spectrum("33", part,
                                          {s: sp.area_cm2 for s, sp in part.items()})
    assert info["ok"]
    ring = [part[d] for d in info["donors"]]
    lo = min(np.min(np.real(r.Z_model)) for r in ring)
    hi = max(np.max(np.real(r.Z_model)) for r in ring)
    assert lo - 1e-9 <= np.min(np.real(z)) and np.max(np.real(z)) <= hi + 1e-9


def test_filling_the_gaps_restores_full_coverage(full_plate) -> None:
    """The point of the exercise: the aggregate covers the whole plate again,
    so the two normalisations agree and the Gamry comparison is like for
    like."""
    missing = ("33", "65", "66", "67", "68")
    part = {k: v for k, v in full_plate.items() if k not in missing}
    cfg = DEFAULT.replace(verbose=False, fill_missing_from_neighbours=True)

    filled, info = silver.build_filled_spectra(part, cfg)
    assert set(filled) == set(missing)
    agg = silver.cell_aggregate_full(part, filled)

    assert np.allclose(agg["area_coverage"], 1.0, atol=1e-6)
    truth = silver.cell_aggregate_full(full_plate)["z_asr_used"]
    err = float(np.nanmax(np.abs(agg["z_asr_used"] - truth) / np.abs(truth)))
    assert err < 0.02, f"rebuilt aggregate is {100*err:.1f} % off the truth"


def test_substitution_discards_the_segments_own_measurement(full_plate) -> None:
    """Segment 33 is present and measured, and must STILL be rebuilt."""
    odd = dict(full_plate)
    odd["33"] = _spectrum("33", r_ohm=0.500)        # absurd, and to be ignored
    cfg = DEFAULT.replace(verbose=False,
                          substitute_segments=frozenset({"33"}))
    # bronze drops it, so silver never sees it; emulate that here
    without = {k: v for k, v in odd.items() if k != "33"}
    filled, _info = silver.build_filled_spectra(without, cfg)

    assert "33" in filled
    assert filled["33"].R_ohmic == pytest.approx(0.060, abs=0.01)
    assert filled["33"].R_ohmic < 0.1, "the discarded 0.5 leaked back in"


def test_an_excluded_segment_is_never_rebuilt(full_plate) -> None:
    """Exclusion means the segment plays no part. Inventing a value for it
    is the opposite of that, so exclusion wins over filling."""
    part = {k: v for k, v in full_plate.items() if k != "33"}
    cfg = DEFAULT.replace(verbose=False, fill_missing_from_neighbours=True,
                          exclude_segments=frozenset({"33"}))
    filled, _ = silver.build_filled_spectra(part, cfg)
    assert "33" not in filled


def test_nothing_is_rebuilt_by_default(full_plate) -> None:
    """A default run reconstructs nothing: the feature has to be asked for."""
    part = {k: v for k, v in full_plate.items() if k != "33"}
    filled, info = silver.build_filled_spectra(part, DEFAULT.replace(verbose=False))
    assert filled == {} and info == {}


def test_a_reconstruction_is_not_a_silver_spectrum(full_plate) -> None:
    """It must not be possible to hand one to code expecting a measurement.

    A FilledSpectrum carries no tier, no KK residual and no DRT, because none
    of those were measured.
    """
    part = {k: v for k, v in full_plate.items() if k != "33"}
    cfg = DEFAULT.replace(verbose=False, substitute_segments=frozenset({"33"}))
    filled, _ = silver.build_filled_spectra(part, cfg)
    sp = filled["33"]
    assert not isinstance(sp, SilverSpectrum)
    assert not hasattr(sp, "tier")
    assert sp.donors and sp.card == "reconstructed"
