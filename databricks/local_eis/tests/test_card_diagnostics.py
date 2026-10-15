"""Card diagnostics against the Gamry: per-card medians, the card-to-Gamry
transfer function and the plate-minus-Gamry series offset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import card_diagnostics as cd

F = np.array([1.0, 10, 100, 300, 1000, 2987, 9516])


def _gamry():
    z = 50 + 100 / (1 + 1j * F / 150)
    return pd.DataFrame({"freq_hz": F * 1.01, "z_re_mohm_cm2": z.real,
                         "z_im_mohm_cm2": z.imag})


def _plate(offset=0.0, gain=None):
    g = _gamry()
    zg = g.z_re_mohm_cm2.to_numpy() + 1j * g.z_im_mohm_cm2.to_numpy()
    rows = []
    for k in range(1, 31):
        card = f"L_Karte_{1 + (k - 1) // 6}"
        a = (gain or {}).get(card[2:], 1.0)
        for fi, zi in zip(F, zg * a + offset):
            rows.append({"segment": str(k), "card": card, "fk": fi,
                         "z_re_mohm_cm2": zi.real, "z_im_mohm_cm2": zi.imag})
    return pd.DataFrame(rows)


def test_card_medians_keep_card_names_and_add_the_plate():
    cm = cd.card_medians(_plate())
    assert set(cm.card) == {f"Karte_{i}" for i in range(1, 6)} | {"plate"}
    assert len(cm) == 6 * F.size


def test_transfer_shows_a_card_gain_on_that_card_only():
    cm = cd.card_medians(_plate(gain={"Karte_3": 0.9}))
    tr = cd.transfer(cm, _gamry())
    assert tr[tr.card == "Karte_3"].H_mag_pct.to_numpy() == pytest.approx(-10, abs=1e-6)
    assert np.allclose(tr[tr.card == "Karte_1"].H_mag_pct, 0, atol=1e-6)
    assert np.allclose(tr.H_phase_deg[tr.card == "Karte_3"], 0, atol=1e-6)


def test_transfer_is_nan_without_a_gamry_point_within_two_percent():
    cm = cd.card_medians(_plate())
    g = _gamry().assign(freq_hz=F * 1.05)
    assert cd.transfer(cm, g).H_mag_pct.isna().all()


def test_series_offset_recovers_an_additive_real_offset():
    o = cd.series_offset(cd.card_medians(_plate(offset=10.0)), _gamry())
    assert o["dRe_mohm_cm2"] == pytest.approx(10.0)
    assert o["dIm_mohm_cm2"] == pytest.approx(0.0, abs=1e-9)


def test_series_offset_tells_a_resistance_from_a_frequency_dependent_offset():
    f = np.geomspace(1, 12000, 40)
    zg = 45 + 100 / (1 + 1j * f / 150) + 1j * 2 * np.pi * f * 6e-4
    g = pd.DataFrame({"freq_hz": f, "z_re_mohm_cm2": zg.real, "z_im_mohm_cm2": zg.imag})

    def plate(dz):
        z = zg + dz
        return pd.DataFrame({"card": "plate", "fk": f, "Z": z})

    o = cd.series_offset(plate(10.0), g)                         # a resistance
    assert o["dRe_at_hfr_mohm_cm2"] == pytest.approx(10.0, abs=0.5)
    o = cd.series_offset(plate(10.0 / (1 + (f / 1500) ** 4)), g)  # gone at the HFR
    assert o["dRe_mohm_cm2"] > 7
    assert abs(o["dRe_at_hfr_mohm_cm2"]) < 3
