"""frequency_response: card delays show before the timing correction and not
after it, a card whose response is not a pure delay is flagged, and an
ex-situ sweep with a step at its normalisation point is caught."""

from __future__ import annotations

import numpy as np
import pandas as pd

import frequency_response as fr
import plausibility as pl

F = np.geomspace(0.2, 3800, 45)
CARDS = {f"Leepa_X_Karte_{c}": [str(s) for s in range(4 * c - 3, 4 * c + 1)]
         for c in (1, 2, 3, 4)}
DELAY = {"Leepa_X_Karte_1": 25e-6, "Leepa_X_Karte_2": -30e-6,
         "Leepa_X_Karte_3": 10e-6, "Leepa_X_Karte_4": -35e-6}


def _cell(seg: str) -> np.ndarray:
    w = 2 * np.pi * F
    # varies from segment to segment, not from card to card
    rct = 0.30 * (1 + 0.03 * (int(seg) % 3 - 1))
    return 0.047 + rct / (1 + (1j * w * 0.002) ** 0.8)


def _run(tmp_path, clean_delay=0.0, hf_step=None, seed=0):
    rng = np.random.default_rng(seed)
    run = tmp_path / "run"
    (run / "silver").mkdir(parents=True)
    (run / "bronze").mkdir()
    w = 2 * np.pi * F
    raw, clean, meta = [], [], []
    for card, segs in CARDS.items():
        for s in segs:
            z = _cell(s) * (1 + 0.04 * rng.standard_normal())   # scale factor
            z = z * (1 + 0.002 * rng.standard_normal(F.size))
            zr = z * np.exp(-1j * w * DELAY[card])
            zc = z * np.exp(-1j * w * (clean_delay if card.endswith(("1", "3"))
                                       else 0.0))
            if hf_step and card.endswith("2"):
                zc = zc * np.where(F >= 2000, 1 + hf_step, 1.0)
            raw += [{"segment": s, "card": card, "freq_hz": f,
                     "z_re_ohm_cm2": v.real, "z_im_ohm_cm2": v.imag}
                    for f, v in zip(F, zr)]
            clean += [{"segment": s, "freq_hz": f,
                       "z_re_mohm_cm2": 1e3 * v.real,
                       "z_im_mohm_cm2": 1e3 * v.imag} for f, v in zip(F, zc)]
            meta.append({"segment": s, "card": card})
    pd.DataFrame(raw).to_csv(run / "bronze" / "raw_spectra.csv", index=False)
    pd.DataFrame(clean).to_csv(run / "silver" / "spectra_clean.csv",
                               index=False)
    pd.DataFrame(meta).to_csv(run / "bronze" / "segment_meta.csv", index=False)
    return run


def _gain(tmp_path, step_seg=None):
    """pipeline convention: gain = 1/H, H a 25 kHz low-pass, normalised at
    the first point; `step_seg` gets a 4 % drop just above it."""
    f = np.geomspace(1, 1e5, 60)
    rows = []
    for s in range(1, 17):
        H = 1 / (1 + 1j * f / (25e3 * (1 + 0.01 * (s - 8))))
        if str(s) == step_seg:
            H = H * np.where(f > 1.3, 0.96, 1.0)
        H = H / H[0]
        g = 1 / H
        rows += [f"{s},{fi:.6g},{gi.real:.8g},{gi.imag:.8g}"
                 for fi, gi in zip(f, g)]
    p = tmp_path / "gain.csv"
    p.write_text("segment,freq_hz,gain_real,gain_imag\n" + "\n".join(rows))
    return p


def _check(cs, name):
    return next(c for c in cs if c.name == name)


def test_card_delays_show_before_and_not_after_the_correction(tmp_path):
    run = _run(tmp_path)
    s = fr.run(run, gain_file=_gain(tmp_path), write_png=False, log=None)
    key = f"{s['insitu']['f_top_hz']:g}"
    raw = s["insitu"]["stages"]["raw"][key]["phase_deg_sd"]
    clean = s["insitu"]["stages"]["clean"][key]["phase_deg_sd"]
    assert raw > 25 and clean < 1
    c = _check(fr.checks(s), "segment frequency response (in-situ)")
    assert c.verdict == pl.PASS
    # the flat per-segment scale factor is not a card step
    for r in s["insitu"]["card_residual"]:
        assert abs(r["mag_step_hf_pct"]) < 2
    assert (run / "frequency_response" / "insitu_response.csv").is_file()


def test_a_delay_left_in_fails(tmp_path):
    s = fr.run(_run(tmp_path, clean_delay=20e-6), write_png=False)
    c = _check(fr.checks(s), "segment frequency response (in-situ)")
    assert c.verdict == pl.FAIL


def test_a_card_step_above_2khz_warns(tmp_path):
    s = fr.run(_run(tmp_path, hf_step=0.10), write_png=False)
    c = _check(fr.checks(s), "segment frequency response (in-situ)")
    assert c.verdict == pl.WARN
    step = {r["card"]: r["mag_step_hf_pct"]
            for r in s["insitu"]["card_residual"]}
    assert step["Karte_2"] > 5
    # the plate median moves with it, so the others shift a little the other way
    assert all(abs(v) < 3 for k, v in step.items() if k != "Karte_2")


def test_exsitu_reads_the_gain_file_back_as_H(tmp_path):
    s = fr.run(_run(tmp_path), gain_file=_gain(tmp_path), write_png=False)
    ex = s["exsitu"]
    assert ex["available"] and ex["n_segments"] == 16
    top = ex["at"]["4000"]
    # 1/(1 + j 4/25): -9.1 deg, |H| 0.988
    assert -10 < top["phase_deg_median"] < -8
    assert 0.98 < top["mag_median"] < 0.995
    assert ex["not_flat"] == {}
    assert _check(fr.checks(s), "chain response (ex-situ)").verdict == pl.PASS


def test_a_step_at_the_normalisation_point_is_caught(tmp_path):
    s = fr.run(_run(tmp_path), gain_file=_gain(tmp_path, step_seg="7"),
               write_png=False)
    assert list(s["exsitu"]["not_flat"]) == ["7"]
    c = _check(fr.checks(s), "chain response (ex-situ)")
    assert c.verdict == pl.WARN and "seg 7" in c.detail


def test_no_gain_file_is_not_applicable(tmp_path):
    s = fr.run(_run(tmp_path), write_png=False)
    assert _check(fr.checks(s), "chain response (ex-situ)").verdict == pl.NA
