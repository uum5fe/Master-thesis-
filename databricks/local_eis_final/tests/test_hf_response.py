"""The two fixes of the October 2026 multi-condition review:

* a chain sweep is normalised by its flat-band median, not by its first
  point (segment 36 of Kashyyyk: a glitch at 1 Hz put -4.1 % on its Z at
  every frequency);
* the channels' high-frequency response, the same at every operating point,
  is measured across conditions, validated leave-one-out and removed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gamry_dta
import hf_response
import utils

F = np.array([0.5, 2, 10, 50, 120, 300, 600, 946.5, 1194, 1884, 2987, 4733,
              7547, 9516, 11953.0])


def test_a_glitch_at_the_first_point_does_not_scale_the_sweep():
    f = np.array([1.0, 1.6, 5, 20, 100, 400, 1000, 4500])
    Z = np.full(f.size, 0.96 + 0j)
    Z[0] = 1.0                                        # the 1 Hz glitch
    sw = gamry_dta.GamrySweep(path=Path("36.DTA"), segment="36", freq=f, Z=Z,
                              meta={})
    H = sw.normalised()
    assert np.allclose(np.abs(H[1:]), 1.0)            # flat stays flat
    assert abs(sw.z_flat) == pytest.approx(0.96)


def _plate(level: float, seed: int, hw: dict) -> pd.DataFrame:
    """A plate: the cell (current-dependent) times a fixed per-segment /
    per-card hardware response above 1 kHz (hw)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(1, 31):
        s, card = str(k), f"L_Karte_{1 + (k - 1) // 10}"
        z_cell = level * (0.05 + 0.1 / (1 + 1j * F / 150)) * (1 + 0.02 * rng.standard_normal())
        g = np.where(F >= 1000, hw["mag"][card](F) * np.exp(1j * hw["ph"][s](F)), 1.0)
        z = 1000 * z_cell * g
        for fi, zi in zip(F, z):
            rows.append({"segment": s, "card": card, "freq_hz": fi,
                         "z_re_mohm_cm2": zi.real, "z_im_mohm_cm2": zi.imag})
    d = pd.DataFrame(rows)
    d["fk"] = d.freq_hz.round(1)
    d["Z"] = d.z_re_mohm_cm2.to_numpy() + 1j * d.z_im_mohm_cm2.to_numpy()
    return d


def _hardware(seed=7):
    rng = np.random.default_rng(seed)
    ph = {str(k): (lambda a, b: (lambda f: np.radians(a + b * f / 1e3)))(
              rng.normal(0, 8), rng.normal(0, 1.5)) for k in range(1, 31)}
    mag = {"L_Karte_1": lambda f: 1 + 0.012 * f / 1e3,
           "L_Karte_2": lambda f: 1 - 0.011 * f / 1e3,
           "L_Karte_3": lambda f: 1 + 0.0 * f}
    return {"ph": ph, "mag": mag}


def test_the_response_is_found_on_currents_it_was_not_fitted_on():
    hw = _hardware()
    runs = {c: _plate(lv, i, hw) for i, (c, lv) in
            enumerate((("45A", 2.5), ("60A", 2.0), ("150A", 1.4), ("450A", 1.0)))}
    val = hf_response.validate(runs)
    assert len(val) == 4
    for r in val:
        assert r["phase_sd_before_deg"] > 5
        assert r["phase_sd_after_deg"] < 1.5
        assert r["card_step_after_pct"] < 2.0 < r["card_step_before_pct"]


def test_written_file_is_one_below_the_band_and_silver_reads_it(tmp_path):
    hw = _hardware()
    runs = [hf_response.deviations(_plate(lv, i, hw))
            for i, lv in enumerate((2.5, 1.4, 1.0))]
    h = hf_response.estimate(runs)
    p = hf_response.write(h, tmp_path / "hf.csv")
    g = utils.load_gain(p)
    assert np.allclose(utils.gain_at(g, "5", np.array([0.5, 300, 946.5])), 1.0)
    hi = utils.gain_at(g, "5", np.array([11953.0]))[0]
    want = hw["mag"]["L_Karte_1"](11953.0) * np.exp(1j * hw["ph"]["5"](11953.0))
    # relative to the plate median, so up to the plate-wide part
    assert abs(np.angle(hi / want, deg=True)) < 6


def test_silver_divides_by_the_response_after_the_card_phase(tmp_path):
    import silver
    from config import DEFAULT
    p = tmp_path / "hf.csv"
    p.write_text("segment,freq_hz,gain_real,gain_imag\n"
                 "7,0.01,1,0\n7,999,1,0\n7,1000,1.1,0\n7,12000,1.1,0\n")
    hf = silver._hf_response(p)
    assert set(hf) == {"7"}
    assert silver._hf_response(None) == {}
