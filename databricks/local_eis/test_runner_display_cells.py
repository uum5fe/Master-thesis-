"""What the display cells draw, and what they leave out.

Two faults, both of the same kind: a reconstruction that reached one consumer
and not the others.

  * gold filled only R_ohmic for a rebuilt segment, because that one can be
    read straight off the rebuilt curve. R_ct and R_mt come from the DRT,
    which is not linear in Z, so there was nothing to read them from -- and
    the same run produced an HFR map with 72 segments and charge-transfer and
    mass-transport maps with 67.

  * the Nyquist cells read spectra_clean.csv, which is measurements only, so
    the rebuilt segments were missing from the segment traces while the cell
    aggregate drawn on top of them already included those segments.

The cells cannot be imported (dbutils, Volume paths, displayHTML), so they are
executed here against a temporary results tree, exactly as the widgets-cell
test does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gold
import r2d2_geometry as geom
import silver
from config import DEFAULT
from silver import SilverRun, SilverSpectrum


RUNNER = Path(__file__).with_name("Local EIS Pipeline Runner.py")
MISSING = ("33", "65", "66", "67", "68")
N_F = 24


def _spectrum(seg: str, r_ohm: float, r_ct: float = 0.040,
              r_mt: float = 0.030) -> SilverSpectrum:
    f = np.geomspace(0.2, 1000.0, N_F)
    w = 2 * np.pi * f
    z = r_ohm + (r_ct + r_mt) / (1 + 1j * w * 2e-3)
    tau = np.geomspace(1e-5, 1.0, 20)
    fast = tau < DEFAULT.tau_split_kinetic_s
    gamma = np.where(fast, r_ct / max(fast.sum(), 1),
                     r_mt / max((~fast).sum(), 1))
    return SilverSpectrum(
        segment=seg, card="Karte_1", freq=f, Z_corr=z, Z_model=z,
        Z_model_sd=np.full(N_F, 1e-3), sigma_rel=np.full(N_F, 0.02),
        R_ohmic=r_ohm, R_ohmic_sd=1e-3, R_ohmic_drt=r_ohm, hf_arc_open=0.0,
        hf_closure=1.0, R_pol=r_ct + r_mt, L=2e-7, dt_applied=0.0,
        tau_peak=2e-3, gamma=gamma, tau_grid=tau, kk_res_max=0.01, kk_mu=0.6,
        zhit_dev=0.01, zhit_drift=0.0, hf_phase_slope=0.0, T_degC=60.0, K=1.0,
        K_imputed=False, u_dc=0.77,
        area_cm2=float(geom.SEGMENTS[seg].area_cm2), n_used=N_F, n_dropped=0,
        snr_med_db=30.0, thd_med=0.01, tier="A")


@pytest.fixture()
def run_with_gaps():
    """A plate missing five segments, with those five rebuilt."""
    rng = np.random.default_rng(0)
    spec = {s: _spectrum(s, 0.060 + 0.004 * rng.standard_normal())
            for s in sorted(geom.SEGMENTS, key=int) if s not in MISSING}
    cfg = DEFAULT.replace(verbose=False, infer_missing_segments=False,
                          substitute_segments=frozenset({"33"}),
                          fill_missing_from_neighbours=True)
    filled, info = silver.build_filled_spectra(spec, cfg)
    agg = silver.cell_aggregate_full(spec, filled)
    sr = SilverRun(spectra=spec, skew={}, dc_closure={}, cell_freq=agg["freq"],
                   Z_cell=agg["z_asr_used"], cell_n_seg=agg["n_seg"],
                   filled=filled, fill_info=info, aggregate=agg)
    return sr, cfg


# ---------------------------------------------------------------------------
# the maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", ["R_ohmic", "R_ct", "R_mt", "R_pol"])
def test_every_map_covers_every_segment(run_with_gaps, param) -> None:
    """Not just the HFR one. The same run must not produce three plates."""
    sr, cfg = run_with_gaps
    g = gold.run(sr, cfg, log=None)
    blank = [s for s, r in g.records.items()
             if not np.isfinite(r.values.get(param, np.nan))]
    assert blank == [], f"{param} has no value on {blank}"


def test_a_rebuilt_value_is_the_donors_value(run_with_gaps) -> None:
    """Area-weighted mean of the segments that touch it -- which is what
    "use the neighbouring segments" means for a scalar."""
    sr, cfg = run_with_gaps
    g = gold.run(sr, cfg, log=None)
    rec = g.records["33"]
    donors = sr.filled["33"].donors
    areas = {d: geom.SEGMENTS[d].area_cm2 for d in donors}
    want = (sum(areas[d] * sr.spectra[d].R_ohmic for d in donors)
            / sum(areas.values()))
    assert rec.values["R_ohmic"] == pytest.approx(want, rel=1e-6)


def test_rebuilt_segments_are_still_labelled(run_with_gaps) -> None:
    """Filling every map must not make a reconstruction look measured."""
    sr, cfg = run_with_gaps
    g = gold.run(sr, cfg, log=None)
    for s in MISSING:
        assert g.records[s].cls == "substituted"
        assert "rebuilt from neighbours" in " ".join(g.records[s].flags)
    assert g.records["19"].cls == "measured"


# ---------------------------------------------------------------------------
# the Nyquist cells
# ---------------------------------------------------------------------------


def _cell(marker: str) -> str:
    for chunk in RUNNER.read_text().split("\n# COMMAND ----------\n"):
        if marker in chunk:
            return chunk
    raise AssertionError(f"cell {marker!r} not found")


def test_the_display_filter_is_off(tmp_path) -> None:
    """Every finite silver-accepted point is drawn in its segment's colour;
    nothing is set aside as a grey cross."""
    src = _cell("INTERACTIVE NYQUIST + BODE PER CONDITION")
    assert "DISPLAY_FILTER = False" in src


def test_both_nyquist_cells_read_the_reconstructions(tmp_path) -> None:
    """The aggregate already included them; the segment traces did not."""
    for marker in ("INTERACTIVE NYQUIST + BODE PER CONDITION",
                   "GAMRY vs PIPELINE OVERLAY"):
        assert "spectra_reconstructed.csv" in _cell(marker), marker


def test_the_overlay_reports_the_aggregate_coverage() -> None:
    """Against a whole-cell instrument, an aggregate over 96 % of the plate
    is not the same quantity, and the figure has to say which it is."""
    src = _cell("GAMRY vs PIPELINE OVERLAY")
    assert "area_coverage" in src
    assert "covers" in src


def test_silver_writes_the_file_the_cells_read(run_with_gaps, tmp_path) -> None:
    """The contract between them, checked rather than assumed."""
    sr, cfg = run_with_gaps
    silver.save(sr, cfg.replace(out_dir=tmp_path), log=None)
    rec = tmp_path / "silver" / "spectra_reconstructed.csv"
    assert rec.exists()
    df = pd.read_csv(rec)
    assert set(df["segment"].astype(str)) == set(MISSING)
    for col in ("donors", "hops", "freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2"):
        assert col in df.columns
    # and the aggregate carries the coverage the overlay prints
    agg = pd.read_csv(tmp_path / "silver" / "cell_aggregate.csv")
    assert "area_coverage" in agg.columns
    assert float(agg["area_coverage"].median()) == pytest.approx(1.0, abs=1e-6)
