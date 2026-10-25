"""The two fixes of the October 2026 multi-condition review:

* a chain sweep is normalised by its flat-band median, not by its first
  point (segment 36 of Kashyyyk: a glitch at 1 Hz put -4.1 % on its Z at
  every frequency);
* the channels' high-frequency response, the same at every operating point,
  is measured across conditions, validated leave-one-out and removed.
"""

from __future__ import annotations

import json
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
    # rebuilt in the same session: read again, not served from the cache
    p.write_text("segment,freq_hz,gain_real,gain_imag\n"
                 "8,0.01,1,0\n8,999,1,0\n8,1000,1.2,0\n8,12000,1.2,0\n"
                 "9,0.01,1,0\n")
    assert set(silver._hf_response(p)) == {"8", "9"}


# ---------------------------------------------------------------------------
# rolled-off channels: a box of lines that low-passes, across cards
# ---------------------------------------------------------------------------
ROLLED = {"8", "9", "10", "11", "12"}           # cards 1 and 2, like box r5
FC = 3000.0


def _rolled_plate(level: float, seed: int, hw: dict, cut_top=True) -> pd.DataFrame:
    """_plate with ROLLED low-passed (|H| only: the phase is the timing
    stages' and the per-segment phase's) and, like the |Z| outlier gate
    does, their two top steps missing."""
    d = _plate(level, seed, hw)
    lp = np.where(d.segment.isin(ROLLED),
                  1 / np.sqrt(1 + (d.freq_hz.to_numpy() / FC) ** 2), 1.0)
    d["Z"] = d["Z"].to_numpy() * lp
    d["z_re_mohm_cm2"] = d["Z"].to_numpy().real
    d["z_im_mohm_cm2"] = d["Z"].to_numpy().imag
    if cut_top:
        d = d[~(d.segment.isin(ROLLED) & (d.freq_hz > 8000))]
    return d.reset_index(drop=True)


def _rolled_runs(hw=None):
    hw = hw or _hardware()
    return {c: _rolled_plate(lv, i, hw) for i, (c, lv) in
            enumerate((("45A", 2.5), ("60A", 2.0), ("150A", 1.4), ("450A", 1.0)))}


def test_rolled_off_channels_are_found_whatever_card_they_are_on():
    info: dict = {}
    hf_response.estimate([hf_response.deviations(d) for d in _rolled_runs().values()],
                         info=info)
    assert set(info["flagged"]) == ROLLED
    t = info["segments"].set_index("segment")
    assert (t.loc[sorted(ROLLED), "mag_source"] == "segment").all()
    assert (t.drop(sorted(ROLLED))["mag_source"] == "card").all()
    # a first-order low-pass, recovered up to the card's own slope it rides on
    assert np.allclose(t.loc[sorted(ROLLED), "fc_hz"], FC, rtol=0.2)


def test_only_card_hardware_flags_nothing():
    hw = _hardware()
    info: dict = {}
    hf_response.estimate([hf_response.deviations(_plate(lv, i, hw))
                          for i, lv in enumerate((2.5, 1.4, 1.0))], info=info)
    assert info["flagged"] == []


def test_per_segment_magnitude_passes_the_blind_test_where_the_card_fails():
    runs = _rolled_runs()
    seg = hf_response.validate(runs, mag="segment")
    card = hf_response.validate(runs, mag="card")
    for s, c in zip(seg, card):
        assert s["rolled_off_before"] == len(ROLLED)
        assert s["rolled_off_after"] == 0
        assert c["rolled_off_after"] >= 3
        assert s["mag_top_sd_after_pct"] < 0.5 * c["mag_top_sd_after_pct"]
        assert s["phase_sd_after_deg"] < 1.5


def test_the_low_pass_is_written_above_the_last_kept_step():
    """The gate cut the top steps; the file still covers them, so the next
    run can keep them."""
    runs = [hf_response.deviations(d) for d in _rolled_runs().values()]
    h = hf_response.estimate(runs)
    f, g = h["10"]
    assert f.max() == pytest.approx(11953.0)
    # the low-pass times card 1's own rise, against a plate median of ~1
    want = (1 + 0.012 * 11.953) / np.sqrt(1 + (11953.0 / FC) ** 2)
    assert abs(g[-1]) == pytest.approx(want, rel=0.15)


def test_build_writes_the_sources_and_silver_flags_them(tmp_path):
    import silver
    dirs = []
    for name, d in _rolled_runs().items():
        r = tmp_path / name
        (r / "silver").mkdir(parents=True)
        (r / "bronze").mkdir()
        d[["segment", "freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2"]].to_csv(
            r / "silver" / "spectra_clean.csv", index=False)
        d.drop_duplicates("segment")[["segment", "card"]].to_csv(
            r / "bronze" / "segment_meta.csv", index=False)
        (r / "config_used.json").write_text(f'{{"condition": "{name}"}}')
        dirs.append(r)
    out = hf_response.build(dirs, tmp_path / "hf", log=None)
    assert set(out["flagged"]) == ROLLED
    src = hf_response.segment_sources(out["path"])
    assert src["10"] == "segment" and src["20"] == "card"
    assert silver._hf_sources(out["path"])["10"] == "segment"
    assert silver._hf_sources(None) == {}
    # an older file without the table: nothing is per segment
    assert hf_response.segment_sources(tmp_path / "nowhere.csv") == {}


# ---------------------------------------------------------------------------
# build guards (2612087: a build from report-mode runs, whose band ended at
# 1.2-1.5 kHz, replaced the file with one that stopped at 1.5 kHz)
# ---------------------------------------------------------------------------
def _run_dirs(tmp_path, runs: dict, hf_file=None, sha=None):
    dirs = []
    for name, d in runs.items():
        r = tmp_path / name
        (r / "silver").mkdir(parents=True, exist_ok=True)
        (r / "bronze").mkdir(exist_ok=True)
        d[["segment", "freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2"]].to_csv(
            r / "silver" / "spectra_clean.csv", index=False)
        d.drop_duplicates("segment")[["segment", "card"]].to_csv(
            r / "bronze" / "segment_meta.csv", index=False)
        cfg = {"condition": name}
        if hf_file:
            cfg["hf_response_file"] = str(hf_file)
        (r / "config_used.json").write_text(json.dumps(cfg))
        if sha is not None:
            (r / "run_identity.json").write_text(json.dumps(
                {"settings": {"hf_response_file_sha": sha}}))
        dirs.append(r)
    return dirs


def test_runs_whose_band_ends_low_are_refused_and_nothing_is_written(tmp_path):
    hw = _hardware()
    runs = {c: _plate(lv, i, hw) for i, (c, lv) in
            enumerate((("150A", 1.4), ("450A", 1.0)))}
    runs = {c: d[d.freq_hz <= 1500] for c, d in runs.items()}
    out = tmp_path / "hf"
    out.mkdir()
    (out / "hf_response.csv").write_text("keep me")
    with pytest.raises(ValueError, match="ends at 1194 Hz"):
        hf_response.build(_run_dirs(tmp_path, runs), out, log=None)
    assert (out / "hf_response.csv").read_text() == "keep me"


def test_composing_keeps_the_old_file_above_the_new_runs_band():
    f_old = np.array([1000.0, 4000.0, 11953.0])
    old = {"5": (f_old, np.array([1.0, 0.9, 0.8 + 0.1j]))}
    new = {"5": (np.array([1000.0, 1500.0]), np.array([1.0, 1.0]))}
    c = hf_response.compose(new, old)
    f, g = c["5"]
    assert f.max() == pytest.approx(11953.0)
    assert g[-1] == pytest.approx(0.8 + 0.1j)
    # a segment only the old file has is kept as it was
    old["9"] = (f_old, np.array([1.0, 1.1, 1.2]))
    assert np.allclose(hf_response.compose(new, old)["9"][1], [1.0, 1.1, 1.2])


def test_a_file_changed_since_the_runs_is_refused_and_a_replaced_one_kept(tmp_path):
    hw = _hardware()
    runs = {c: _plate(lv, i, hw) for i, (c, lv) in
            enumerate((("150A", 1.4), ("450A", 1.0)))}
    out = tmp_path / "hf"
    first = hf_response.build(_run_dirs(tmp_path / "a", runs), out, log=None)
    sha = hf_response._sha12(first["path"])
    # runs evaluated with that file: composing onto it is fine, and the old
    # one goes to history/
    dirs = _run_dirs(tmp_path / "b", runs, hf_file=first["path"], sha=sha)
    second = hf_response.build(dirs, out, log=None)
    assert (out / "history" / f"{sha}_hf_response.csv").is_file()
    assert second["backup"] == out / "history" / f"{sha}_hf_response.csv"
    # the same runs again: the file is no longer the one they ran with
    new_sha = hf_response._sha12(second["path"])
    with pytest.raises(ValueError, match="changed after they ran"):
        hf_response.build(dirs, out, log=None)
    assert hf_response._sha12(second["path"]) == new_sha
    # a recorded file that is not there
    gone = _run_dirs(tmp_path / "c", runs, hf_file=tmp_path / "nowhere.csv")
    with pytest.raises(ValueError, match="which is not there"):
        hf_response.build(gone, tmp_path / "hf2", log=None)
