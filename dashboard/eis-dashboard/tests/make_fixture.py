"""A results tree shaped like a real run, for tests and for eyeballing the app."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

def build(root: Path, leepa="2612030", condition="150A", n_seg=72, seed=3):
    rng = np.random.default_rng(seed)
    run = root / leepa / condition
    for sub in ("bronze", "silver", "gold"):
        (run / sub).mkdir(parents=True, exist_ok=True)

    # 8 x 9 plate of 72 segments
    seg, cx, cy = [], [], []
    for i in range(n_seg):
        seg.append(i + 1); cx.append(20.0 * (i % 8)); cy.append(20.0 * (i // 8))
    R_oh = 45 + 6 * rng.standard_normal(n_seg)
    R_ct = 22 + 4 * rng.standard_normal(n_seg)
    tau_max = np.full(n_seg, 1.06)                 # band reaches 0.15 Hz
    R_mt = 18 + 5 * rng.standard_normal(n_seg)
    # 11 segments lost their low-frequency points -> no slow tau bin at all
    band_limited = rng.choice(n_seg, 11, replace=False)
    tau_max[band_limited] = 0.004
    R_mt[band_limited] = np.nan
    cls = np.array(["measured"] * n_seg, dtype=object)
    cls[rng.choice(n_seg, 5, replace=False)] = "inferred"

    pd.DataFrame({
        "segment": seg, "class": cls, "tier": ["B"] * n_seg,
        "cx_mm": cx, "cy_mm": cy, "area_cm2": [4.235] * n_seg,
        "fault": [""] * n_seg, "flags": [""] * n_seg,
        "R_ohmic": R_oh.round(4), "R_ohmic_sd": (0.05 * R_oh).round(4),
        "R_ct": R_ct.round(4), "R_ct_sd": (0.08 * R_ct).round(4),
        "R_mt": R_mt.round(4), "R_mt_sd": (0.1 * R_mt).round(4),
        "tau_max": tau_max.round(6),
        "j_dc": (0.35 + 0.02 * rng.standard_normal(n_seg)).round(4),
    }).to_csv(run / "gold" / "plate_summary.csv", index=False)

    f = np.geomspace(0.15, 4500, 40)
    rows = []
    for s in seg[:6]:
        Z = 0.045 + 0.025 / (1 + 1j * 2 * math.pi * f * 2e-3)
        for fi, z in zip(f, Z):
            rows.append({"segment": s, "freq_hz": fi,
                         "Z_re": z.real, "Z_im": z.imag})
    pd.DataFrame(rows).to_csv(run / "silver" / "impedance.csv", index=False)

    pd.DataFrame({"reason": ["snr", "snr", "thd", "stationarity", "snr"],
                  "segment": [3, 4, 3, 9, 12],
                  "freq_hz": [1000, 2000, 1500, 12, 3000]}
                 ).to_csv(run / "silver" / "point_rejections.csv", index=False)

    json.dump({"card_lag_s": {"Karte_1": 8.6254, "Karte_2": 8.6275,
                              "Karte_3": -2.8445, "Karte_4": 0.0,
                              "Karte_5": 0.0176},
               "card_lag_corr": {"Karte_1": 0.083, "Karte_2": 0.083,
                                 "Karte_3": 0.995, "Karte_4": 1.0,
                                 "Karte_5": 0.995}},
              (run / "bronze" / "bronze_manifest.json").open("w"), indent=2)
    json.dump({"n_total": 72, "n_measured": 67, "n_inferred": 5, "n_bad": 0,
               "tiers": {"A": 0, "B": 40, "C": 27}},
              (run / "gold" / "gold_manifest.json").open("w"), indent=2)
    json.dump({"condition": condition, "leepa": leepa},
              (run / "run_manifest.json").open("w"), indent=2)
    (run / "run.log").write_text("synthetic fixture\n")
    return run

if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "./results")))
