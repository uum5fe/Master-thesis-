"""The per-card voltage-gain check and the DC closure, on data with a known answer.

Five cards record one cell voltage on UC2; card 3's voltage chain reads
4 % high and card 4's 2 % low. The check must find exactly that from the
recordings, leave Z alone in "report", remove it in "correct", say so in the
outputs and the plausibility report -- and the DC closure must separate a
current scale from a zero offset over several conditions.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import bronze
import card_gain
import make_synth_famos as M
import plausibility
from config import Config, DEFAULT
from eis_local import Step

FS = 5000.0
FREQS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 120.0, 300.0, 700.0]
DWELL = 2500                       # samples per step (0.5 s)
GAIN = {"Karte_1": 1.0, "Karte_2": 1.0, "Karte_3": 1.04, "Karte_4": 0.98,
        "Karte_5": 1.0}
U_DC = 0.70


def _schedule():
    return [Step(freq=f, start=i * DWELL, stop=(i + 1) * DWELL, amp=0.01,
                 snr_db=40.0, thd=0.0, stationarity=0.0)
            for i, f in enumerate(FREQS)]


def _cell_voltage(n):
    t = np.arange(n) / FS
    v = np.full(n, U_DC)
    for i, f in enumerate(FREQS):
        sl = slice(i * DWELL, (i + 1) * DWELL)
        v[sl] += 0.01 * np.cos(2 * np.pi * f * t[sl] + 0.3)
    return v


def _cards(tmp_path, gains=GAIN):
    n = DWELL * len(FREQS)
    v = _cell_voltage(n)
    rng = np.random.default_rng(0)
    files, cards = [], {}
    for k, (card, g) in enumerate(sorted(gains.items())):
        uc2 = g * v + 2e-5 * rng.standard_normal(n)
        uc1 = 1.0 * v + 2e-5 * rng.standard_normal(n)
        seg = 0.1 + 0.002 * np.cos(np.arange(n))
        fp = M.write_v2(tmp_path / f"Leepa_X_{card}.DAT", ["UC2", "UC1", "1"],
                        np.column_stack([uc2, uc1, seg]), FS)
        files.append(fp)
        cards[fp.stem] = SimpleNamespace(ref_name="UC2")
    return files, cards


# ---------------------------------------------------------------------------
# the UC comparison
# ---------------------------------------------------------------------------


def test_the_card_gains_are_measured_from_the_recordings(tmp_path):
    files, cards = _cards(tmp_path)
    cfg = DEFAULT.replace(card_gain="report", verbose=False)
    refs = card_gain.measure_reference(files, cards, _schedule(), cfg)
    uc2 = {card_gain.short(c): r for c, r in refs["UC2"].items()}
    assert uc2["Karte_3"].gain == pytest.approx(1.04, abs=0.002)
    assert uc2["Karte_4"].gain == pytest.approx(0.98, abs=0.002)
    assert uc2["Karte_1"].gain == pytest.approx(1.00, abs=0.002)
    assert uc2["Karte_3"].status == card_gain.OFF
    assert uc2["Karte_1"].status == card_gain.OK
    assert all(r.used_for_Z for r in uc2.values())
    assert uc2["Karte_3"].dc_ratio == pytest.approx(1.04, abs=0.002)
    # report: measured, not applied
    assert all(r.applied == 1.0 for r in uc2.values())
    # UC1 is wired alike on every card here: all ok, and not Z's reference
    uc1 = refs["UC1"]
    assert all(r.status == card_gain.OK and not r.used_for_Z
               for r in uc1.values())


def test_correct_applies_only_what_is_a_flat_gain(tmp_path):
    gains = dict(GAIN, Karte_5=1.25)          # 25 %: a different tap
    files, cards = _cards(tmp_path, gains)
    cfg = DEFAULT.replace(card_gain="correct", verbose=False)
    refs = card_gain.measure_reference(files, cards, _schedule(), cfg)
    g = {card_gain.short(c): v
         for c, v in card_gain.applied_gains(refs).items()}
    assert g["Karte_3"] == pytest.approx(1.04, abs=0.002)
    assert g["Karte_5"] == 1.0                 # too large: reported only
    st = {card_gain.short(c): r.status for c, r in refs["UC2"].items()}
    assert st["Karte_5"] == card_gain.TOO_LARGE


def test_off_reads_nothing(tmp_path):
    files, cards = _cards(tmp_path)
    cfg = DEFAULT.replace(card_gain="off", verbose=False)
    assert card_gain.measure_reference(files, cards, _schedule(), cfg) == {}


def test_a_band_dependent_gain_is_not_called_a_scale():
    f = np.array(FREQS)
    A = {c: np.exp(1j * 0.3) * np.ones(f.size) for c in GAIN}
    A["Karte_2"] = A["Karte_2"] * np.where(f > 100, 1.06, 1.0)   # a filter
    snr = {c: np.full(f.size, 40.0) for c in GAIN}
    dc = {c: np.full(f.size, U_DC) for c in GAIN}
    out = card_gain.compare_phasors(f, A, snr, dc, DEFAULT)
    assert out["Karte_2"].status == card_gain.NOT_FLAT
    assert not card_gain.correctable(out["Karte_2"])


# ---------------------------------------------------------------------------
# bronze: applied, saved, read back; re-evaluation can switch it
# ---------------------------------------------------------------------------


def test_bronze_hands_each_card_its_gain(tmp_path, monkeypatch):
    files, cards = _cards(tmp_path)
    seen = {}

    def _process_card(fp_, cal, schedule, grid, cfg, log, **kw):
        seen[fp_.stem] = kw.get("v_gain")
        return {}

    cal = SimpleNamespace(has_current_cal=True, seg_c0={}, temp_c0={})
    monkeypatch.setattr(bronze, "discover_files", lambda cfg: files)
    monkeypatch.setattr(bronze, "inventory_channels",
                        lambda files_, cfg, log=None: ({}, cards))
    monkeypatch.setattr(bronze, "PlateCalibration",
                        SimpleNamespace(load=lambda *a: cal))
    monkeypatch.setattr(bronze, "estimate_card_lags", lambda *a, **k: {})
    monkeypatch.setattr(bronze, "plate_temperatures", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(bronze, "consensus_schedule",
                        lambda *a, **k: (_schedule(), {}))
    monkeypatch.setattr(bronze, "process_card", _process_card)
    cfg = Config(dat_dir=tmp_path, out_dir=tmp_path, verbose=False,
                 card_gain="correct")
    br = bronze.run(cfg)
    k3 = next(k for k in seen if k.endswith("Karte_3"))
    assert seen[k3] == pytest.approx(1.04, abs=0.002)
    assert br.card_ref["UC2"]
    seen.clear()
    bronze.run(Config(dat_dir=tmp_path, out_dir=tmp_path, verbose=False,
                      card_gain="report"))
    assert all(v == 1.0 for v in seen.values())


def test_the_table_round_trips_and_reevaluation_switches_it(tmp_path):
    from test_channel_lag_stage import _run_obj, _cfg
    import main
    br = _run_obj()
    br.card_ref = {"UC2": {
        "Karte_1": card_gain.CardReference(
            card="Karte_1", channel="UC2", n_steps=10, gain=1.05,
            gain_lo=1.05, gain_hi=1.05, used_for_Z=True, status=card_gain.OFF),
        "Karte_2": card_gain.CardReference(
            card="Karte_2", channel="UC2", n_steps=10, gain=1.0,
            gain_lo=1.0, gain_hi=1.0, used_for_Z=True, status=card_gain.OK)}}
    run = tmp_path / "X" / "60A"
    cfg = _cfg(out_dir=run, condition="60A", write_png=False, write_html=False)
    cfg.save(run / "config_used.json")
    bronze.save(br, cfg)
    back = bronze.load(run)
    r = back.card_ref["UC2"]["Karte_1"]
    assert r.gain == pytest.approx(1.05) and r.used_for_Z and r.applied == 1.0
    z0 = back.spectra["19"].Z_raw.copy()

    out = tmp_path / "re"
    main.reevaluate(run, out_dir=out, card_gain="correct")
    again = bronze.load(out)
    assert again.card_ref["UC2"]["Karte_1"].applied == pytest.approx(1.05)
    assert np.allclose(again.spectra["19"].Z_raw, z0 / 1.05)
    rows = {r["check"]: r for r in
            __import__("csv").DictReader(open(out / "plausibility.csv"))}
    assert rows["card voltage gain"]["verdict"] == plausibility.PASS

    # ... and back to report: the division is undone
    main.reevaluate(out, out_dir=tmp_path / "re2", card_gain="report")
    back2 = bronze.load(tmp_path / "re2")
    assert np.allclose(back2.spectra["19"].Z_raw, z0)
    rows = {r["check"]: r for r in __import__("csv").DictReader(
        open(tmp_path / "re2" / "plausibility.csv"))}
    assert rows["card voltage gain"]["verdict"] == plausibility.WARN
    assert "Karte_1" in rows["card voltage gain"]["detail"]


# ---------------------------------------------------------------------------
# the impedance, card against plate
# ---------------------------------------------------------------------------


def _spectra(scale3=1.0, extra_lf3=0.0, j3=1.0):
    f = np.geomspace(0.2, 1000, 30)
    w = 2 * np.pi * f
    out = {}
    for k in range(40):
        card = f"Karte_{k // 8 + 1}"
        z = 0.065 + 1.0 / (1 / 0.3 + 2e-3 * (1j * w) ** 0.9)
        j = 0.2
        if card == "Karte_3":
            z = scale3 * z + extra_lf3 * 1.0 / (1 + 1j * w * 0.5)
            j = 0.2 * j3
        out[str(k + 1)] = SimpleNamespace(freq=f, Z_corr=z, card=card, j_dc=j)
    return out


def test_a_card_scale_error_looks_like_one():
    cf = card_gain.impedance_factors(_spectra(scale3=1.05, j3=1 / 1.05))
    assert cf["Karte_3"].verdict == "gain_like"
    assert cf["Karte_3"].level_pct == pytest.approx(5.0, abs=0.3)
    assert cf["Karte_1"].verdict == "consistent"
    c = plausibility.card_factor_check(cf)
    assert c.verdict == plausibility.WARN and "Karte_3" in c.detail


def test_a_real_regional_difference_does_not():
    cf = card_gain.impedance_factors(_spectra(extra_lf3=0.06, j3=0.95))
    assert cf["Karte_3"].verdict == "frequency_dependent"
    assert plausibility.card_factor_check(cf).verdict == plausibility.PASS


# ---------------------------------------------------------------------------
# the current the closure is judged against, and the closure over conditions
# ---------------------------------------------------------------------------


def test_the_reference_current_and_where_it_came_from():
    cfg = DEFAULT.replace(condition="450A")
    assert card_gain.current_of("45A") == 45.0
    assert card_gain.current_of("450A") == 450.0
    assert card_gain.current_of("ALL") != card_gain.current_of("ALL")
    assert plausibility.resolve_current(cfg) == (450.0, "condition name")
    i, src = plausibility.resolve_current(cfg, {"I_S": 449.97,
                                                "I_S_set": 444.1})
    assert (i, src) == (449.97, "bench I_S (measured)")
    i, src = plausibility.resolve_current(cfg.replace(i_setpoint_a=440.0),
                                          {"I_S": 449.97})
    assert src == "given setpoint"
    c = plausibility.current_closure({"1": 0.2}, {"1": 300.0}, 60.0,
                                     source="bench I_S (measured)")
    assert "bench I_S" in c.detail


def test_dc_closure_separates_scale_from_offset():
    rows = [{"condition": f"{i:g}A", "i_ref_A": float(i),
             "i_full_A": 0.976 * i - 1.0} for i in (45, 60, 150, 450)]
    res = card_gain.dc_closure_table(rows)
    assert res["scale_pct"] == pytest.approx(-2.4, abs=1e-6)
    assert res["offset_A"] == pytest.approx(-1.0, abs=1e-6)
    assert rows[0]["dev_pct"] == pytest.approx(100 * ((0.976 * 45 - 1) / 45 - 1))


def test_dc_closure_tool_reads_run_folders(tmp_path, capsys):
    import dc_closure
    import json
    import utils
    for cond, i in (("45A", 45.0), ("150A", 150.0)):
        run = tmp_path / cond
        (run / "bronze").mkdir(parents=True)
        (run / "config_used.json").write_text(json.dumps({"condition": cond}))
        a = 304.92 / 4
        utils.write_table(run / "bronze" / "segment_meta.csv", [
            {"segment": str(k), "card": f"L_Karte_{k}", "area_cm2": a,
             "j_dc_A_cm2": (0.98 * i - 1.0) / 304.92} for k in (1, 2, 3, 4)])
    assert dc_closure.main([str(tmp_path), "-o", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "scale -2.00 %" in out
    assert (tmp_path / "o" / "dc_closure_cards.csv").is_file()


def test_noisy_steps_fall_back_to_the_dc_level():
    """2612030: the UC2 AC response is 0-20 dB per step, so no card had five
    clean steps and every gain came out nan. The DC level decides then."""
    f = np.array(FREQS)
    A = {c: np.exp(1j * 0.3) * np.ones(f.size) for c in GAIN}
    snr = {c: np.full(f.size, 6.0) for c in GAIN}            # all too noisy
    dc = {c: np.full(f.size, 0.815 * g) for c, g in GAIN.items()}
    out = card_gain.compare_phasors(f, A, snr, dc, DEFAULT,
                                    used_for_Z={c: True for c in GAIN})
    assert out["Karte_3"].source == "dc"
    assert out["Karte_3"].gain == pytest.approx(1.04, abs=1e-6)
    assert out["Karte_3"].status == card_gain.OFF
    assert out["Karte_1"].status == card_gain.OK
    c = plausibility.card_voltage_check({"UC2": out})
    assert c.verdict == plausibility.WARN and "DC level" in c.detail


def test_nothing_measurable_is_not_a_pass():
    refs = {"UC2": {c: card_gain.CardReference(card=c, channel="UC2",
                                               used_for_Z=True,
                                               status=card_gain.TOO_FEW)
                    for c in GAIN}}
    assert plausibility.card_voltage_check(refs).verdict == plausibility.NA


def test_re_z_at_one_frequency_does_not_extrapolate():
    import gold
    f = np.array([300.0, 946.5, 1194.0, 1516.5])
    z = np.array([80, 68, 66, 64]) + 0j
    assert gold.re_at(f, z, 1000.0) == pytest.approx(67.6, abs=0.1)
    assert np.isnan(gold.re_at(f, z, 2000.0))
    assert np.isnan(gold.re_at(np.array([100.0, 1500.0]), z[:2], 1000.0))


def test_the_tap_resistance_is_measured_and_subtracted():
    import dc_closure
    import silver
    from test_channel_lag_stage import _run_obj, _cfg
    rows = [{"i_ref_A": i, "uc2_V": 0.82 - 0.0016 - i * 44e-6 - 0.0003 * i / 45,
             "gamry_vdc_V": 0.82 - 0.0003 * i / 45} for i in (45., 60., 150., 450.)]
    se = dc_closure.sense_offset(rows)
    assert se["R_x_uohm"] == pytest.approx(44.0, abs=0.1)
    assert se["R_x_mohm_cm2"] == pytest.approx(13.4, abs=0.1)
    br = _run_obj()
    a = silver.run(br, _cfg(channel_lag="off"))
    b = silver.run(br, _cfg(channel_lag="off", uc_series_mohm_cm2=13.4))
    s = "14"
    assert 1e3 * (a.spectra[s].R_ohmic - b.spectra[s].R_ohmic) == pytest.approx(
        13.4, abs=0.3)
