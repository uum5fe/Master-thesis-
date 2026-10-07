"""The channel-lag stage inside the pipeline, on a plate with a known answer.

A plate whose R_s really varies (55..69 mOhm*cm2 along x) is measured through
chains that lag by 0..110 us, the way 2612030's did. The stage must remove
the lag, keep the real gradient, refuse to "correct" a channel whose phase is
not a lag at all, and say all of it in the outputs and the plausibility
report -- for the FAMOS path (silver) and the CSV path alike.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

import channel_lag                                          # noqa: E402
import plausibility                                         # noqa: E402
import r2d2_geometry as geom                                # noqa: E402
import silver                                               # noqa: E402
from bronze import BronzeRun, BronzeSpectrum                # noqa: E402
from config import DEFAULT                                  # noqa: E402
from eis_local import Step                                  # noqa: E402

FS = 25_000.0
F = np.geomspace(0.2, 1200.0, 40)
SEGS = [str(s) for s in range(1, 25)]
LAG_US = {"15": 81.0, "19": 105.0, "23": 110.0, "11": 58.0, "20": 64.0,
          "6": -35.0, "10": -22.0}
BROKEN = "8"          # its phase is scrambled above 100 Hz: not a lag


def _rs_true(seg: str) -> float:
    """A real gradient along x: 55 at x = 0 to ~69 at the far end."""
    return 55.0 + 14.0 * geom.SEGMENTS[seg].cx_mm / geom.PLATE_W_MM


def _z(seg: str, f=F) -> np.ndarray:
    """mOhm*cm2 -> ohm*cm2, as bronze stores it."""
    w = 2 * np.pi * f
    z = _rs_true(seg) + 1.0 / (1.0 / 150.0 + 2e-3 * (1j * w) ** 0.9)
    tau = 1e-6 * LAG_US.get(seg, 0.0)
    z = z * np.exp(-1j * w * tau)          # the channel samples late
    if seg == BROKEN:
        rng = np.random.default_rng(1)
        hf = f > 100
        z[hf] = z[hf] * np.exp(1j * rng.uniform(-1.2, 1.2, hf.sum()))
    return z / 1e3


def _bronze(seg: str) -> BronzeSpectrum:
    n = F.size
    snr = np.full(n, 40.0)
    return BronzeSpectrum(
        segment=seg, card="Karte_1", freq=F, Z_raw=_z(seg),
        snr_ref_db=snr, snr_seg_db=snr, snr_comb_db=snr, thd=np.zeros(n),
        drift=np.zeros(n),
        n_per_step=np.maximum((FS * np.maximum(8.0 / F, 0.05)).astype(int), 256),
        on_grid=np.ones(n, bool), channel_slot=2, ref_slot=0, n_ch_on_card=16,
        fs=FS, K=1.0, K_imputed=False, T_degC=60.0, u_dc=0.15, ref_name="UC2")


def _run_obj() -> BronzeRun:
    schedule = [Step(freq=float(f), start=i * 1000, stop=i * 1000 + 800,
                     amp=0.01, snr_db=40.0, thd=0.0, stationarity=0.0)
                for i, f in enumerate(F)]
    return BronzeRun(schedule=schedule, channels={},
                     spectra={s: _bronze(s) for s in SEGS}, cards={},
                     grid={}, config_digest="x", input_digest="y", n_files=1,
                     lags={"Karte_1": {"lag": 0, "applied": True, "corr": 1.0,
                                       "prominence": float("nan"),
                                       "rescued": False}})


def _cfg(**kw):
    return DEFAULT.replace(verbose=False, f_min_hz=0.15, f_max_hz=1300.0,
                           skew_model="none", drt_enable=False,
                           fill_missing_from_neighbours=False, **kw)


def _items(cfg):
    sk = {"Karte_1": silver.SkewModel("Karte_1", "slot", 0.0, 0.0, 0.0, 0.0,
                                      len(SEGS), False, "off")}
    spectra = {s: _bronze(s) for s in SEGS}
    return silver.channel_lag_items(spectra, sk, cfg)


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------


def test_lags_are_found_and_the_scrambled_channel_is_refused():
    lags = channel_lag.estimate(_items(_cfg()), _cfg())
    for s, us in LAG_US.items():
        assert lags[s].status == channel_lag.CORRECTED, s
        assert 1e6 * lags[s].applied_s == pytest.approx(us, abs=4.0), s
    assert lags[BROKEN].status == channel_lag.NOT_FIRST_ORDER
    assert lags[BROKEN].applied_s == 0.0
    clean = [s for s in SEGS if s not in LAG_US and s != BROKEN]
    assert all(abs(1e6 * lags[s].tau_s) < 5.0 for s in clean)


def test_report_mode_changes_nothing_and_off_does_nothing():
    lags = channel_lag.estimate(_items(_cfg()), _cfg(channel_lag="report"))
    assert lags["19"].status == channel_lag.REPORT_ONLY
    assert all(g.applied_s == 0.0 for g in lags.values())
    assert channel_lag.estimate(_items(_cfg()), _cfg(channel_lag="off")) == {}


def test_too_few_points_is_said_not_guessed():
    items = _items(_cfg())
    f, z, u = items["3"]
    items["3"] = (f, z, u & (f < 75))           # 2 points left above 50 Hz
    lags = channel_lag.estimate(items, _cfg())
    assert lags["3"].status == channel_lag.TOO_FEW
    assert lags["3"].applied_s == 0.0


# ---------------------------------------------------------------------------
# silver, end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs():
    br = _run_obj()
    on = silver.run(br, _cfg())
    off = silver.run(br, _cfg(channel_lag="off"))
    return on, off


def test_silver_removes_the_lag_and_keeps_the_real_gradient(runs):
    on, off = runs
    err_on = {s: 1e3 * sp.R_ohmic - _rs_true(s) for s, sp in on.spectra.items()
              if s != BROKEN}
    err_off = {s: 1e3 * sp.R_ohmic - _rs_true(s)
               for s, sp in off.spectra.items() if s != BROKEN}
    # without the stage the lagging channels read far low...
    assert min(err_off[s] for s in ("15", "19", "23")) < -8.0
    # ...with it every lagging segment reads like the clean ones (the
    # top-of-band estimator keeps its own common bias), gradient included
    clean = np.median([err_on[s] for s in SEGS
                       if s not in LAG_US and s != BROKEN])
    worst = max(abs(err_on[s] - clean) for s in LAG_US)
    assert worst < 2.0, err_on
    assert on.spectra["19"].chain_tau_applied * 1e6 == pytest.approx(105, abs=4)
    assert any(f.startswith("chain_lag_removed_") for f in on.spectra["19"].flags)


def test_the_unreliable_channel_is_demoted_and_flagged(runs):
    on, _ = runs
    sp = on.spectra.get(BROKEN)
    if sp is not None:                     # it may also fail the model gates
        assert sp.tier == "C"
        assert "chain_phase_not_first_order" in sp.flags
    assert on.channel_lag[BROKEN].status == channel_lag.NOT_FIRST_ORDER


def test_outputs_and_plausibility_carry_it(runs, tmp_path):
    on, _ = runs
    out = silver.save(on, _cfg(out_dir=tmp_path))
    import csv
    rows = list(csv.DictReader(open(out / "channel_lag.csv")))
    assert {r["segment"] for r in rows} == set(SEGS)
    summ = list(csv.DictReader(open(out / "segments_summary.csv")))
    assert "chain_tau_applied_us" in summ[0]
    rep = plausibility.check_run(on, _cfg())
    names = {c.name: c for c in rep.checks}
    assert names["card alignment"].verdict == plausibility.PASS
    assert names["channel lag"].verdict == plausibility.WARN
    assert BROKEN in names["channel lag"].detail


def test_card_alignment_fails_when_a_card_was_not_shifted():
    c = plausibility.card_alignment_check(
        {"Leepa_X_Karte_1": {"applied": False}, "Leepa_X_Karte_3": {"applied": True}})
    assert c.verdict == plausibility.FAIL and "Karte_1" in c.detail


# ---------------------------------------------------------------------------
# the CSV path runs the same stage
# ---------------------------------------------------------------------------


def test_csv_path_applies_the_same_correction(tmp_path):
    import logging
    import csv_pipeline as cp
    spectra = {s: cp.SegmentSpectrum(segment=s, freq=F, Z=_z(s),
                                     sigma_rel=np.full(F.size, 0.01),
                                     snr_db=np.full(F.size, 40.0),
                                     n_used=np.full(F.size, 4096))
               for s in SEGS}
    lags = cp._apply_channel_lag(spectra, _cfg(out_dir=tmp_path),
                                 logging.getLogger("t"))
    assert lags["19"].status == channel_lag.CORRECTED
    w = 2 * np.pi * F
    true19 = (_rs_true("19") + 1.0 / (1.0 / 150.0 + 2e-3 * (1j * w) ** 0.9)) / 1e3
    assert np.allclose(spectra["19"].Z, true19, rtol=0.03)
    assert (tmp_path / "silver" / "channel_lag.csv").is_file()


# ---------------------------------------------------------------------------
# re-evaluating a saved run (bronze.load + main.reevaluate)
# ---------------------------------------------------------------------------


def test_bronze_round_trips_through_its_tables(tmp_path):
    import bronze
    br = _run_obj()
    bronze.save(br, _cfg(out_dir=tmp_path))
    back = bronze.load(tmp_path)
    assert set(back.spectra) == set(br.spectra)
    a, b = br.spectra["19"], back.spectra["19"]
    assert np.allclose(a.Z_raw, b.Z_raw) and np.allclose(a.freq, b.freq)
    assert (b.channel_slot, b.fs, b.card) == (a.channel_slot, a.fs, a.card)
    assert back.lags["Karte_1"]["applied"] is True
    assert len(back.schedule) == len(br.schedule)


def test_reevaluate_runs_silver_and_gold_from_saved_bronze(tmp_path):
    import csv
    import bronze
    import main
    run = tmp_path / "2612030" / "60A"
    cfg = _cfg(out_dir=run, condition="60A", write_png=False,
               write_html=False)
    cfg.save(run / "config_used.json")
    bronze.save(_run_obj(), cfg)
    out = tmp_path / "re" / "2612030" / "60A"
    main.reevaluate(run, out_dir=out)
    rows = {r["segment"]: r for r in
            csv.DictReader(open(out / "silver" / "channel_lag.csv"))}
    assert rows["19"]["status"] == channel_lag.CORRECTED
    assert (out / "gold" / "plate_summary.csv").is_file()
    assert (out / "plausibility.csv").is_file()
    main.reevaluate(run, out_dir=tmp_path / "rep", channel_lag="report")
    rows = {r["segment"]: r for r in csv.DictReader(
        open(tmp_path / "rep" / "silver" / "channel_lag.csv"))}
    assert rows["19"]["status"] == channel_lag.REPORT_ONLY


def test_two_card_groups_far_apart_are_both_brought_onto_the_plate():
    """2612030 / 150 A: cards 1-2 and cards 3-5 sample ~60 us apart (a pure
    delay). With the band at 3.8 kHz a first-order model left them tens of
    degrees apart; the delay model brings every segment onto the plate."""
    f = np.geomspace(50, 3800, 22)
    w = 2 * np.pi * f
    rng = np.random.default_rng(4)
    items, true = {}, {}
    for k in range(30):
        s = str(k + 1)
        tau = (25e-6 if k < 12 else -40e-6) + 3e-6 * rng.standard_normal()
        z = (0.065 + 0.15 / (1 + 1j * w * 0.15 * 0.02)) * np.exp(-1j * w * tau)
        items[s] = (f, z, np.ones(f.size, bool))
        true[s] = tau
    def spread_after(n_iter):
        lags = channel_lag.estimate(items, _cfg().replace(
            channel_lag_iterations=n_iter, f_max_hz=4000.0))
        top = [np.degrees(np.angle(items[s][1][-1] * channel_lag.correction(
            f[-1], lags[s].applied_s))) for s in items]
        return np.ptp(top), lags
    three, lags = spread_after(3)
    assert three < 3.0
    rel = [1e6 * (lags[s].tau_s - true[s]) for s in items]
    assert np.ptp(rel) < 2.0          # one common offset left, nothing else


def test_the_modelled_band_stops_where_the_cards_stop_agreeing():
    """2612030 at fs = 25 kHz: cards coherent to 3.8 kHz, 10-25 deg apart at
    4.7-5.9 kHz. With f_max_hz = 250000 the band ran to 9.5 kHz."""
    from dataclasses import replace
    f = np.array([100.0, 1194.0, 3796.9, 4733.5, 9515.6])
    sp = replace(_bronze("5"), freq=f, Z_raw=np.full(f.size, 0.06 + 0j),
                 snr_ref_db=np.full(f.size, 40.0), snr_seg_db=np.full(f.size, 40.0),
                 snr_comb_db=np.full(f.size, 40.0), thd=np.zeros(f.size),
                 drift=np.zeros(f.size), n_per_step=np.full(f.size, 20000),
                 on_grid=np.ones(f.size, bool))
    cfg = _cfg().replace(f_max_hz=250000.0)
    g = silver.gate_points(sp, cfg)
    assert list(g["freq"][g["keep"]]) == [100.0, 1194.0, 3796.9]
    g = silver.gate_points(sp, cfg.replace(coherent_f_max_frac_fs=0.0))
    assert g["keep"].sum() == 5
