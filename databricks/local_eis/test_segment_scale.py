"""segment_scale: a constant per-segment factor is found, a real local
difference is not mistaken for one, and the gain file divides it out."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import segment_scale as ss
import utils

F = np.geomspace(0.2, 3000, 40)
SEGS = [str(s) for s in range(1, 25)]
SCALE = {"5": 1.08, "9": 0.92, "14": 1.05}


def _run(tmp_path, cond, j_scale, seed):
    rng = np.random.default_rng(seed)
    run = tmp_path / cond
    (run / "silver").mkdir(parents=True)
    (run / "bronze").mkdir()
    (run / "config_used.json").write_text(json.dumps({"condition": cond}))
    w = 2 * np.pi * F
    rows, meta = [], []
    for s in SEGS:
        # the cell: R_ct varies along the plate and with load (frequency-
        # dependent), segment 20 carries a real +6 mOhm*cm2 HFR offset
        rct = 0.15 * j_scale * (1 + 0.005 * (int(s) - 12))
        z = 0.065 + (0.006 if s == "20" else 0) + rct / (1 + 1j * w * rct * 0.02)
        z = z * SCALE.get(s, 1.0) * (1 + 0.003 * rng.standard_normal(F.size))
        rows += [{"segment": s, "freq_hz": f, "z_re_mohm_cm2": 1e3 * zz.real,
                  "z_im_mohm_cm2": 1e3 * zz.imag} for f, zz in zip(F, z)]
        meta.append({"segment": s})
    pd.DataFrame(rows).to_csv(run / "silver" / "spectra_clean.csv", index=False)
    pd.DataFrame(meta).to_csv(run / "bronze" / "segment_meta.csv", index=False)
    return run


def test_a_constant_factor_is_found_and_a_real_offset_is_not(tmp_path):
    runs = [_run(tmp_path, c, j, k) for k, (c, j) in
            enumerate([("45A", 1.0), ("150A", 0.7), ("450A", 0.5)])]
    res = ss.analyse(runs)
    t = res["table"]
    for s, k in SCALE.items():
        assert t.loc[s, "scale"] == pytest.approx(k, abs=0.01)
        assert t.loc[s, "verdict"] == "scale_like"
    # a real HFR offset is not flat in ratio: it fades as |Z| grows
    assert t.loc["20", "verdict"] != "scale_like"
    for v in res["loo"].values():
        assert v["after_pct"] < v["spread_pct"]


def test_the_gain_file_divides_the_factor_out(tmp_path):
    runs = [_run(tmp_path, c, j, k) for k, (c, j) in
            enumerate([("45A", 1.0), ("150A", 0.7)])]
    res = ss.analyse(runs)
    p = ss.write_gain(tmp_path / "g.csv", res["table"].scale)
    g = utils.load_gain(p)
    zf = utils.gain_at(g, "5", np.array([100.0]))
    assert abs(zf[0]) == pytest.approx(1.08, abs=0.01)
    assert ss.main([str(tmp_path), "-o", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "segment_scale.csv").is_file()
