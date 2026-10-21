"""Full band (a developer diagnostic since the shipped evaluation stops at
2.4 kHz, config.eval_f_max_hz; these tests lift that limit with
eval_f_max_hz=0): evaluate every step the card can resolve, up to just below its
Nyquist frequency, with R_ohmic read where the measured spectrum crosses the
real axis -- the Gamry and bench-tool definition."""

from __future__ import annotations

import numpy as np
import pytest

import silver
from bronze import BronzeSpectrum
from config import DEFAULT
from silver import SkewModel

FS = 25_000.0
R_TRUE = 45e-3
L_TRUE = 2.3e-7          # ohm*s*cm2: the Gamry's whole-cell value on 2612030


def _skew():
    return SkewModel(card="Karte_1", basis="structural", dt0=0.0, k_slot=0.0,
                     k_nominal=0.0, cost_gain=0.0, n_segments=1,
                     applied=False, note="")


def _z(f):
    w = 2 * np.pi * f
    return (R_TRUE + 20e-3 / (1 + (1j * w * 1.5e-4) ** 0.9)
            + 60e-3 / (1 + (1j * w * 3e-2) ** 0.9) + 1j * w * L_TRUE)


def _sweep():
    """The 2612030 Gamry sweep, 30 kHz .. 0.3 Hz, 10 points per decade."""
    return 30046.88 / 10 ** (np.arange(0, 51) / 10.0)


def _spectrum(f, Z) -> BronzeSpectrum:
    n = f.size
    snr = np.full(n, 30.0)
    return BronzeSpectrum(
        segment="1", card="Karte_1", freq=f, Z_raw=Z, snr_ref_db=snr,
        snr_seg_db=snr, snr_comb_db=snr, thd=np.zeros(n), drift=np.zeros(n),
        n_per_step=np.full(n, 20000), on_grid=np.ones(n, bool), channel_slot=3,
        ref_slot=0, n_ch_on_card=16, fs=FS, K=1.0, K_imputed=False,
        T_degC=60.0, u_dc=0.77, ref_name="UC2")


def _true_intercept():
    f = np.geomspace(100, 12000, 20000)
    z = _z(f)
    i = np.flatnonzero((z.imag[:-1] < 0) & (z.imag[1:] >= 0))[0]
    return float(z.real[i])


def test_the_band_reaches_nyquist_with_full_band_and_stops_coherent_without():
    f = _sweep()
    sp = _spectrum(f, _z(f))
    kept = f[silver.gate_points(sp, DEFAULT)["keep"]]
    assert kept.max() <= 0.16 * FS                       # 4 kHz
    full = DEFAULT.replace(full_band=True, eval_f_max_hz=0, param_method="drt")
    kept = f[silver.gate_points(sp, full)["keep"]]
    assert kept.max() == pytest.approx(11953.1, rel=1e-3)  # below fs/2
    assert not (kept > 0.5 * FS).any()                   # 15..30 kHz alias
    # the blind search keeps its ceiling; a Gamry-known step goes higher
    assert full.f_hi(FS) == pytest.approx(0.45 * FS)
    assert full.f_known_hi(FS) == pytest.approx(0.49 * FS)
    assert DEFAULT.f_known_hi(FS) == pytest.approx(0.45 * FS)


def test_intercept_is_the_real_axis_crossing_and_ignores_the_lf_loop():
    f = _sweep()
    Z = _z(f)
    Z[f < 0.5] += 1j * 5e-3              # a low-frequency inductive loop
    R, sd = silver.r_ohmic_intercept(f, Z, np.full(f.size, 0.01))
    assert R == pytest.approx(_true_intercept(), rel=0.01)
    assert np.isfinite(sd) and sd > 0
    # still capacitive at the top: no crossing, no number
    cap = f <= 1000
    assert np.isnan(silver.r_ohmic_intercept(f[cap], Z[cap])[0])


def test_full_band_r_ohmic_is_the_intercept_and_the_top_band_mean_is_not():
    """With the inductive branch inside the band the top-band mean reads the
    branch, not the cell; the intercept does not care how far the band
    goes."""
    f = _sweep()
    sp = _spectrum(f, _z(f))
    full = DEFAULT.replace(full_band=True, eval_f_max_hz=0, param_method="drt")
    res = silver.process_segment(sp, _skew(), full)
    assert res is not None
    assert res.R_ohmic == pytest.approx(_true_intercept(), rel=0.02)
    assert res.hf_closure == 1.0
    assert silver.r_ohmic_method(full) == "intercept"
    # the shipped default: Bayesian DRT, R_ohmic = the high-frequency
    # intercept (extrapolated when the band ends before the crossing)
    assert silver.r_ohmic_method(DEFAULT) == "hf_intercept"
    assert silver.r_ohmic_method(DEFAULT.replace(param_method="ecm")) == "ecm"
    assert silver.r_ohmic_method(DEFAULT.replace(r_ohmic_method="topband")) == "topband"
    assert silver.r_ohmic_method(DEFAULT.replace(r_ohmic_method="intercept", param_method="drt")
                                 ) == "intercept"


def test_the_channel_lag_fit_stays_below_the_coherent_limit():
    f = _sweep()
    sp = _spectrum(f, _z(f))
    full = DEFAULT.replace(full_band=True, eval_f_max_hz=0, param_method="drt")
    items = silver.channel_lag_items({"1": sp}, {}, full)
    fq, _Z, use = items["1"]
    assert fq[use].max() <= 0.16 * FS


def test_a_window_on_the_wrong_stretch_of_record_is_rejected():
    """Its segment SNR sits far below its neighbours' while the CRLB, with
    tens of thousands of samples, still calls it precise."""
    f = _sweep()
    sp = _spectrum(f, _z(f))
    s = np.full(f.size, 20.0)
    bad = int(np.argmin(np.abs(f - 3.0)))
    s[bad] = 1.0                                  # 2612030 450 A, 3.0 Hz
    sp.snr_seg_db = s
    on = DEFAULT.replace(off_tone_drop_db=12.0)
    assert DEFAULT.off_tone_drop_db is None          # optional, off by default
    g = silver.gate_points(sp, on)
    assert not g["keep"][bad] and g["reason"][bad] == "off_tone"
    assert g["reason"][bad - 1] != "off_tone"
    assert g["reason"][bad + 1] != "off_tone"
    off = silver.gate_points(sp, DEFAULT.replace(off_tone_drop_db=None))
    assert off["reason"][bad] != "off_tone"


def test_each_cards_high_frequency_phase_is_taken_out():
    """Above the coherent limit cards 4-5 read 20 deg more than cards 1-2 on
    2612030; the per-card offset against the plate median is removed, the
    magnitude is not."""
    f = _sweep()
    full = DEFAULT.replace(full_band=True, eval_f_max_hz=0, param_method="drt")
    offs = {"Karte_1": 0.0, "Karte_2": 0.0, "Karte_4": np.radians(20.0)}
    spectra = {}
    n = 0
    for card, d in offs.items():
        for k in range(4):
            n += 1
            Z = _z(f) * (1 + 0.02 * k)
            Z = np.where(f > 0.16 * FS, Z * np.exp(1j * d), Z)
            sp = _spectrum(f, Z)
            sp.segment, sp.card = str(n), card
            spectra[str(n)] = sp
    ph = silver.card_hf_phase(spectra, {}, {}, full)
    assert set(ph) == set(offs)
    f_hi = [k for k in ph["Karte_4"] if k > 0.16 * FS]
    assert f_hi and all(abs(np.degrees(ph["Karte_4"][k]) - 20.0) < 1.0
                        for k in f_hi)
    assert all(abs(ph["Karte_1"][k]) < 1e-6 for k in f_hi)
    res = silver.process_segment(spectra["9"], _skew(), full,
                                 card_phase=ph["Karte_4"])
    i = int(np.argmin(np.abs(res.freq - 7546.88)))
    ref = _z(res.freq[i:i + 1])[0] * 1.0
    assert abs(np.degrees(np.angle(res.Z_corr[i] / ref))) < 1.0
    assert abs(abs(res.Z_corr[i]) / abs(ref) - 1.0) < 1e-6
    assert any(fl.startswith("card_hf_phase") for fl in res.flags)
    assert silver.card_hf_phase(spectra, {}, {}, DEFAULT) == {}


def _gamry(monkeypatch, tmp_path, f, Z):
    import gamry_compare
    from pathlib import Path
    sweep = gamry_compare.CellSweep(path=Path(tmp_path / "450A.DTA"),
                                    freq=f[::-1].copy(), Z_ohm=Z[::-1] / 300.0,
                                    current_a=450.0, started=None)
    monkeypatch.setattr(gamry_compare, "find_cell_sweeps",
                        lambda *a, **k: [sweep])
    return DEFAULT.replace(full_band=True, eval_f_max_hz=0, gamry_dir=tmp_path,
                           condition="450A")


def test_each_cards_phase_is_calibrated_against_the_gamry(monkeypatch,
                                                          tmp_path):
    """2612030: channel_lag aligns every channel to the plate median, so the
    ~11 us lead all of them share survives it, and cards 3-5 add their own
    filter lead above ~3.8 kHz -- the tails that bend down and to the left.
    Per card, the delay is fitted against the Gamry and removed everywhere,
    and from 1 kHz up the card's remaining offset is removed too; phase
    only."""
    f = _sweep()
    lead = {"Karte_1": 11e-6, "Karte_2": 11e-6, "Karte_3": 8e-6}
    spectra = {}
    n = 0
    for card, tau in lead.items():
        for k in range(4):
            n += 1
            Z = _z(f) * (1 + 0.02 * k) * np.exp(2j * np.pi * f * tau)
            if card == "Karte_3":                  # its filter above 3.8 kHz
                Z = np.where(f > 3800, Z * np.exp(1j * np.radians(15.0)), Z)
            sp = _spectrum(f, Z)
            sp.segment, sp.card = str(n), card
            spectra[str(n)] = sp
    cfg = _gamry(monkeypatch, tmp_path, f, _z(f))
    got = silver.gamry_card_phase(spectra, {}, {}, cfg)
    assert got["applied"] and set(got["phase"]) == set(lead)
    for card, tau in lead.items():
        assert abs(got["cards"][card]["tau_us"] - 1e6 * tau) < 0.3
        assert got["cards"][card]["resid_deg"] < 0.5

    for seg in ("1", "9"):
        sp = spectra[seg]
        res = silver.process_segment(sp, _skew(), cfg,
                                     card_phase=got["phase"][sp.card],
                                     card_phase_ref="gamry")
        scale = 1 + 0.02 * ((int(seg) - 1) % 4)
        for f0 in (98.0, 1194.0, 7546.88):
            i = int(np.argmin(np.abs(res.freq - f0)))
            ref = _z(res.freq[i:i + 1])[0] * scale
            assert abs(np.degrees(np.angle(res.Z_corr[i] / ref))) < 1.0, f0
            assert abs(abs(res.Z_corr[i]) / abs(ref) - 1.0) < 1e-6
        assert any(fl.startswith("card_phase_vs_gamry") for fl in res.flags)

    # no sweep for the condition, the coherent band, or switched off: nothing
    for c in (cfg.replace(condition="60A"), cfg.replace(full_band=False),
              cfg.replace(gamry_card_phase=False)):
        r = silver.gamry_card_phase(spectra, {}, {}, c)
        assert not r["applied"] and r["phase"] == {}


def test_a_card_with_an_implausible_delay_is_left_alone(monkeypatch,
                                                         tmp_path):
    """A fit far beyond any chain delay (here 200 us) means the comparison
    went wrong (wrong sweep, wrong condition): that card is not touched."""
    f = _sweep()
    spectra = {}
    for k in range(8):
        card = "Karte_1" if k < 4 else "Karte_2"
        tau = 200e-6 if card == "Karte_2" else 11e-6
        sp = _spectrum(f, _z(f) * np.exp(2j * np.pi * f * tau))
        sp.segment, sp.card = str(k + 1), card
        spectra[sp.segment] = sp
    cfg = _gamry(monkeypatch, tmp_path, f, _z(f))
    got = silver.gamry_card_phase(spectra, {}, {}, cfg)
    assert got["applied"] and set(got["phase"]) == {"Karte_1"}
    assert "not credible" in got["cards"]["Karte_2"]["reason"]


def test_the_inductance_term_follows_the_top_of_the_band():
    """Once the HF phase is corrected to the Gamry's, 1-3 kHz is still
    capacitive and only above ~3.8 kHz is inductive (cards 4-5 keep
    nothing above 5.9 kHz). Averaged over the top 20 %
    of the band the tail disappeared and the KK/DRT models dropped L; the
    top of the band decides now."""
    import eis_measurement_model as mm
    from eis_validation import lin_kk
    f = _sweep()
    f = f[f <= 6000]               # cards 4-5: nothing kept above 5.9 kHz
    Z = _z(f)
    hi = f >= np.percentile(f, 80)
    assert np.mean(Z.imag[hi]) < 0          # the old rule saw no inductance
    assert mm.has_inductive_tail(f, Z)
    assert not mm.has_inductive_tail(f[f < 2000], Z[f < 2000])
    kk = lin_kk(f, Z)
    assert kk["res_max"] < 0.02
    drt = silver.bayesian_drt(f, Z, np.full(f.size, 0.01), DEFAULT)
    assert drt["L"] > 0
    assert abs(drt["R_inf"] - R_TRUE) / R_TRUE < 0.05


def test_the_evaluated_band_stops_at_2_4_khz():
    """The shipped evaluation: the 2390.6 Hz step is the last one kept."""
    f = np.unique(np.r_[_sweep(), 2390.62, 2987.13])
    sp = _spectrum(f, _z(f))
    kept = f[silver.gate_points(sp, DEFAULT)["keep"]]
    assert np.isclose(kept.max(), 2390.62)
    assert DEFAULT.f_eval_max() == 2400.0
    # detection is not limited by it
    assert DEFAULT.f_hi(25_000.0) == pytest.approx(0.45 * 25_000.0)
