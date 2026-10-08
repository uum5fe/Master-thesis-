"""via_resistance: the via coefficient's slope and temperature coefficient
are read back from a synthetic Abgleich calibration, cards come from a run,
and the plots are written."""

from __future__ import annotations

import numpy as np
import pandas as pd

import via_resistance as vr

TC = [(3.188, 0.013132), (3.039, 0.012525), (3.154, 0.013011), (3.299, 0.013561)]
STEPS = [(1, 20), (2, 40), (3, 60), (4, 80), (5, 90), (6, 20)]
ALPHA = 0.0039


def _k(seg: int, T: float) -> float:
    k20 = 2.0 * (1 + 0.05 * (seg % 5 - 2))
    return k20 * (1 + ALPHA * (T - 20))


def _abgleich(tmp_path, n=8):
    ab = tmp_path / "Kashyyyk"
    (ab / "coefficients").mkdir(parents=True)
    (ab / "coefficients" / "temp.csv").write_text(
        "\n".join(f"{a};{b}" for a, b in TC))
    K = [(0.6 * (1 + 0.05 * (s % 5 - 2)), 2.4 * (1 + 0.05 * (s % 5 - 2)))
         for s in range(1, n + 1)]
    (ab / "coefficients" / "curr.csv").write_text(
        "\n".join(f"{a:.6f};{b:.6f}" for a, b in K))
    for step, T in STEPS:
        lines = []
        temps = ";".join(f"temp{i + 1}={a + b * T:.6f}V"
                         for i, (a, b) in enumerate(TC))
        for s in range(1, n + 1):
            pts = "\t".join(f"i_s={i:.6f}A;u_s={0.004 + _k(s, T) * i:.6f}V;"
                            for i in (0.0, 0.25, 0.5, 0.75, 1.0))
            lines.append(f"s{s}:\t{temps};\t{pts}")
        (ab / f"Step{step}_{T}Grad.csv").write_text("\n".join(lines))
    return ab


def _run(tmp_path, n=8):
    rd = tmp_path / "45A"
    (rd / "bronze").mkdir(parents=True)
    pd.DataFrame({"segment": [str(s) for s in range(1, n + 1)],
                  "card": [f"Leepa_X_Karte_{1 + (s - 1) // 4}"
                           for s in range(1, n + 1)],
                  "T_degC": 58.0 + 0.1 * np.arange(n)}).to_csv(
        rd / "bronze" / "segment_meta.csv", index=False)
    return rd


def test_slope_temperature_and_alpha_are_recovered(tmp_path):
    steps = vr.read_steps(_abgleich(tmp_path))
    assert len(steps) == 8 * len(STEPS)
    r = steps[(steps.segment == "3") & (steps.step == 3)].iloc[0]
    assert abs(r.T_degC - 60) < 0.05
    assert abs(r.k_V_per_A - _k(3, 60)) < 1e-4
    seg = vr.analyse(steps)
    assert np.allclose(seg.alpha_pct_per_K, 100 * ALPHA, atol=0.002)
    assert np.allclose(seg.return_drift_pct, 0, atol=1e-3)


def test_cards_and_operating_temperature_from_a_run(tmp_path):
    ab = _abgleich(tmp_path)
    s = vr.run(ab, [_run(tmp_path)], out_dir=tmp_path / "out")
    seg = pd.read_csv(tmp_path / "out" / "via_segments.csv",
                      dtype={"segment": str})
    assert set(seg.card) == {"card 1", "card 2"}
    assert "k_45A" in seg and "T_45A" in seg
    assert abs(s["alpha_median"] - 100 * ALPHA) < 0.002
    # 0.7 K span across the plate -> ~0.27 % in K
    assert abs(s["operating"]["45A"]["K_span_pct"] - 0.7 * 100 * ALPHA) < 0.01
    assert (tmp_path / "out" / "via_vs_temperature.png").is_file()
    assert (tmp_path / "out" / "via_alpha.png").is_file()


def test_no_calibration_files_is_reported_not_raised(tmp_path):
    (tmp_path / "empty").mkdir()
    assert vr.run(tmp_path / "empty", out_dir=tmp_path / "o") == {}
