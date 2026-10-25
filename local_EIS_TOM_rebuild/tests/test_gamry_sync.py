"""The Gamry clock: read its timeline, measure the offset, fix what the blind
detector got wrong -- and refuse when the evidence is not there.

2612030 / 60 A is the model: every FAMOS window from 0.3 Hz to 1.2 kHz ends
at t_gamry + 99.2 s, while 1.5-4.7 kHz were placed tens of seconds away and
the detector's ladder frequencies sat up to 1 % off the Gamry's.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import bronze
import gamry_sync as gs
import ladder_snap
import make_synth_famos as M
from config import DEFAULT
from eis_local import Step

FS = 25_000.0
#: the sweep, in the order the Gamry runs it (descending), dwell in s
SWEEP = [(3010.0, 0.12), (2390.6, 0.15), (1884.2, 0.18), (1516.5, 0.2),
         (1194.0, 0.22), (946.5, 0.25), (751.8, 0.3), (597.4, 0.35),
         (477.4, 0.4), (376.7, 0.45), (298.3, 0.5)]
T0 = 5.0          # FAMOS time of the first step
OFFSET = 0.6      # FAMOS end - Gamry stamp, plus up to 1 s of stamp rounding


def _truth():
    t, out = T0, []
    for f, d in SWEEP:
        out.append((f, t, t + d))
        t += d + 0.05
    return out


def _timeline() -> gs.Timeline:
    tr = _truth()
    return gs.Timeline(path=gs.Path("synthetic_CurrVal_60.dta"),
                       freq=np.array([f for f, _, _ in tr]),
                       t_s=np.array([np.floor(e - OFFSET) for _, _, e in tr]))


def _chans(n_ch=6, seed=3):
    """Segment channels carrying the sweep (and noise), own sample index."""
    rng = np.random.default_rng(seed)
    n = int((T0 + sum(d + 0.05 for _, d in SWEEP) + 1.0) * FS)
    t = np.arange(n) / FS
    tone = np.zeros(n)
    for f, a, b in _truth():
        m = (t >= a) & (t < b)
        tone[m] = np.cos(2 * np.pi * f * t[m])
    return {str(k + 1): 0.13 + 0.004 * (0.8 + 0.1 * k) * tone
            + 0.002 * rng.standard_normal(n) for k in range(n_ch)}


def _step(f, a_s, b_s, snr=15.0, src="detected", fs=FS):
    return Step(freq=f, start=int(a_s * fs), stop=int(b_s * fs), amp=0.01,
                snr_db=snr, thd=0.0, stationarity=0.0, window_source=src)


def _detected():
    """What a blind detector hands over: ladder frequencies up to 1 % off,
    1884 Hz placed ~5 s early (by "interpolation"), 2390 Hz not found."""
    out = []
    for f, a, b in _truth():
        if abs(f - 2390.6) < 1:
            continue
        if abs(f - 1884.2) < 1:
            out.append(_step(f * 1.003, 0.10, 0.28, snr=-12, src="interpolated"))
            continue
        out.append(_step(f * (1.0 - 0.006), a, b))
    return sorted(out, key=lambda s: s.freq)


def _rebuild(st, **kw):
    return ladder_snap._rebuild(Step, st, **kw)


# ---------------------------------------------------------------------------


def test_timeline_reads_freq_and_time_from_a_real_layout(tmp_path):
    p = tmp_path / "V26_092_HFR_102_CurrVal_60.dta"
    p.write_text(
        "EXPLAIN\nTAG\tGALVEIS\nIACREQ\tQUANT\t7\tAC Current (A rms)\n"
        "STARTTIME\tLABEL\t07.09.2026 10:32:26\tStart Time\n"
        "ZCURVE\tTABLE\n"
        "\tPt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\tIdc\tVdc\tIERange\n"
        "\t#\ts\tHz\tohm\tohm\tV\tohm\tdeg\tA\tV\t#\n"
        "\t1\t2\t30046,88\t1,6E-04\t1,4E-04\t1\t2,1E-04\t41,5\t4,7E-03\t0,80\t13\n"
        "\t2\t3\t23859,38\t1,6E-04\t1,1E-04\t1\t2,0E-04\t36,0\t5,3E-03\t0,80\t13\n",
        encoding="latin-1")
    tl = gs.read_timeline(p)
    assert np.allclose(tl.freq, [30046.88, 23859.38])
    assert np.allclose(tl.t_s, [2, 3])
    assert tl.started is not None and tl.started.minute == 32
    assert tl.iac_rms == pytest.approx(7.0)


def test_offset_is_measured_and_every_point_gets_a_verdict():
    res = gs.align(_detected(), _timeline(), FS)
    assert res.ok, res.reason
    assert OFFSET <= res.offset_s < OFFSET + 1.0
    v = {round(r["f_gamry_hz"]): r["verdict"] for r in res.rows}
    assert v[1884] == "misplaced"
    assert v[2391] == "missing"
    assert v[1194] == "ok" and v[298] == "ok"


def test_too_few_confident_steps_is_refused_not_guessed():
    weak = [_rebuild(s, snr_db=-5.0) for s in _detected()]
    res = gs.align(weak, _timeline(), FS)
    assert not res.ok and "confident" in res.reason


def test_exact_frequencies_replace_the_ladder_ones():
    res = gs.align(_detected(), _timeline(), FS)
    steps, n = gs.apply_frequencies(_detected(), res, _rebuild)
    assert n >= 8
    got = sorted(s.freq for s in steps)
    for f, _, _ in _truth():
        if f in (2390.6, 1884.2):
            continue
        assert min(abs(g - f) for g in got) < 1e-9, f


def test_guide_relocates_the_misplaced_and_finds_the_missing():
    res = gs.align(_detected(), _timeline(), FS)
    steps, _ = gs.apply_frequencies(_detected(), res, _rebuild)

    def make(f, a, b, snr):
        return _step(f, a / FS, b / FS, snr=snr, src="gamry")

    steps, rows = gs.relocate(steps, res, _timeline(), _chans(), FS, 0,
                              _rebuild, make)
    truth = {round(f): (a, b) for f, a, b in _truth()}
    for f in (1884, 2391):
        s = min(steps, key=lambda x: abs(x.freq - f))
        assert abs(s.freq - f) < 1.0
        assert s.window_source == "gamry"
        a, b = s.start / FS, s.stop / FS
        ta, tb = truth[f]
        assert ta - 0.01 <= a and b <= tb + 0.01, (f, a, b, ta, tb)
    assert {r["result"] for r in rows} == {"relocated"}


def test_a_slot_without_the_tone_drops_the_wrong_window():
    chans = _chans()
    t = np.arange(len(chans["1"])) / FS
    _f, a, b = [x for x in _truth() if abs(x[0] - 1884.2) < 1][0]
    for k in chans:                       # silence the real 1884 Hz dwell
        chans[k] = np.where((t >= a) & (t < b), 0.13, chans[k])
    res = gs.align(_detected(), _timeline(), FS)
    steps, rows = gs.relocate(
        list(_detected()), res, _timeline(), chans, FS, 0, _rebuild,
        lambda f, a, b, snr: _step(f, a / FS, b / FS, snr=snr, src="gamry"))
    assert not any(abs(s.freq - 1884.2 * 1.003) < 0.1 for s in steps)
    r = [r for r in rows if round(r["f_gamry_hz"]) == 1884][0]
    assert "dropped" in r["result"]


def test_an_interpolated_window_inside_the_tolerance_is_still_searched():
    """2612030 / 45 A and 450 A: the 1884 and 2391 Hz windows were guessed by
    window sanity, landed within the Gamry's time tolerance ("ok"), and still
    missed the tone -- drift-rejected in every segment, band cut at 1.5 kHz."""
    det = []
    for s in _detected():
        if abs(s.freq - 1884.2 * 1.003) < 1:
            f, a, b = [x for x in _truth() if abs(x[0] - 1884.2) < 1][0]
            # half a second late: inside the 1.5 s tolerance, mostly on the
            # next (1516 Hz) dwell
            s = _step(s.freq, a + 0.15, b + 0.15, snr=-12, src="interpolated")
        det.append(s)
    res = gs.align(det, _timeline(), FS)
    r = [r for r in res.rows if round(r["f_gamry_hz"]) == 1884][0]
    assert r["verdict"] == "ok" and r["window_source"] == "interpolated"
    steps, rows = gs.relocate(
        list(det), res, _timeline(), _chans(), FS, 0, _rebuild,
        lambda f, a, b, snr: _step(f, a / FS, b / FS, snr=snr, src="gamry"))
    s = min(steps, key=lambda x: abs(x.freq - 1884.2))
    _f, ta, tb = [x for x in _truth() if abs(x[0] - 1884.2) < 1][0]
    assert s.window_source == "gamry"
    assert ta - 0.01 <= s.start / FS and s.stop / FS <= tb + 0.01
    row = [r for r in rows if round(r["f_gamry_hz"]) == 1884][0]
    assert row["was"] == "interpolated" and row["result"] == "relocated"
    # a detected "ok" window is left alone
    assert not [r for r in rows if r["was"] == "ok"]


def test_a_refused_card_lag_is_corroborated_by_the_gamry_clock():
    tl = _timeline()
    good = _detected()
    lag_s = 8.6                           # card 2's clock runs 8.6 s late
    late = [_rebuild(s, start=s.start + int(lag_s * FS),
                     stop=s.stop + int(lag_s * FS)) for s in good]
    lags = {"K1": {"lag": 0, "applied": True},
            "K2": {"lag": int(lag_s * FS), "applied": False}}
    rows = gs.corroborate_card_lags({"K1": good, "K2": late}, lags,
                                    {"K1": FS, "K2": FS}, tl, 0.0, np.inf)
    r2 = [r for r in rows if r["card"] == "K2"][0]
    assert r2["corroborated"] and r2["gamry_implied_lag_s"] == pytest.approx(
        lag_s, abs=0.5)
    lags["K2"]["lag"] = int(3.0 * FS)     # a wrong measured lag is NOT backed
    rows = gs.corroborate_card_lags({"K1": good, "K2": late}, lags,
                                    {"K1": FS, "K2": FS}, tl, 0.0, np.inf)
    assert not [r for r in rows if r["card"] == "K2"][0]["corroborated"]


def test_gamry_frequencies_count_as_on_grid():
    grid = {"ok": True, "f0": 1000.0, "ratio": 10 ** 0.1,
            "gamry_freqs": [1516.544]}
    assert bronze._on_grid(1516.544, grid, 0.001)
    assert not bronze._on_grid(1530.0, {**grid, "gamry_freqs": None}, 0.001)


# ---------------------------------------------------------------------------
# bronze, on a real FAMOS file
# ---------------------------------------------------------------------------


def test_bronze_guided_schedule_on_a_famos_card(tmp_path):
    chans = _chans()
    names = ["UC2"] + list(chans) + ["Temp_1"]
    n = len(chans["1"])
    data = np.column_stack([0.78 + 0.0 * np.arange(n)] + list(chans.values())
                           + [np.ones(n)])
    fp = M.write_v2(tmp_path / "Leepa_X_Current_60A_Test_01_Karte_1.DAT",
                    names, data, FS)
    cfg = DEFAULT.replace(gamry_sync="guide", f_min_hz=0.1)
    kept, info = bronze._gamry_guided(
        _detected(), _timeline(), [fp], {fp.stem: object()}, cfg,
        {fp.stem: {"lag": 0, "applied": True}}, {fp.stem: FS}, "guide",
        logging.getLogger("t"))
    assert info["ok"] and info["n_frequency_corrected"] >= 8
    srcs = {round(s.freq): s.window_source for s in kept}
    assert srcs[1884] == "gamry" and srcs[2391] == "gamry"
    assert {r["result"] for r in info["relocated"]} == {"relocated"}
    # report mode changes nothing
    kept2, info2 = bronze._gamry_guided(
        _detected(), _timeline(), [fp], {fp.stem: object()}, cfg,
        {fp.stem: {"lag": 0, "applied": True}}, {fp.stem: FS}, "report",
        logging.getLogger("t"))
    assert [s.freq for s in kept2] == [s.freq for s in _detected()]


# ---------------------------------------------------------------------------
# the ex-situ chain (Abgleich bode sweeps) against the in-situ lag
# ---------------------------------------------------------------------------


def test_bode_tau_uses_the_in_situ_model_and_sign():
    import gamry_dta
    f = np.geomspace(1.0, 1e5, 60)[::-1]
    taus = {str(k): t for k, t in enumerate(
        [5e-6, 40e-6, 90e-6, -20e-6, 0.0, 60e-6], start=1)}
    sweeps = {s: gamry_dta.GamrySweep(
        path=gs.Path(f"x_#{s}.DTA"), segment=s, freq=f,
        Z=1e-3 * (1 + 1j * 2 * np.pi * f * t), meta={})
        for s, t in taus.items()}
    got = gamry_dta.chain_tau(sweeps)
    for s, t in taus.items():
        assert got[s] == pytest.approx(t, abs=1.5e-6), s
    same = gamry_dta.compare_chain_tau(got, taus)
    assert same["explained"] > 0.98 and same["r"] > 0.99
    rng = np.random.default_rng(0)
    other = {s: 1e-6 * rng.normal(0, 50) for s in taus}
    assert gamry_dta.compare_chain_tau(got, other)["explained"] < 0.5


def test_the_abgleich_delivery_is_found_by_its_layout(tmp_path):
    import gamry_dta
    good = tmp_path / "R2D2_green_Kashyyyk" / "Abgleichdaten" / "Kashyyyk"
    (good / "bode").mkdir(parents=True)
    (good / "coefficients").mkdir()
    (good / "coefficients" / "curr.csv").write_text("0.45;0.1\n")
    (good / "bode" / "plate_100kHz_1Hz_500mA_#1.DTA").write_text("x")
    half = tmp_path / "other" / "x"
    (half / "bode").mkdir(parents=True)
    (half / "bode" / "a_#2.DTA").write_text("x")
    (tmp_path / "noise" / "bode").mkdir(parents=True)     # no #n sweeps
    hits = gamry_dta.find_abgleich_dirs([tmp_path])
    assert hits == [good, half]                 # complete delivery first
