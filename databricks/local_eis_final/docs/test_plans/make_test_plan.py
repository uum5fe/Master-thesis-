#!/usr/bin/env python3
"""Generate the five-test plan (Gamry + FAMOS) for the R2-D2 gen1 plate.

Base operating table = the bench profile OpConds 2.3 (Polcurve 2.3 611A,
profile ID 12013): pressures, dew points and minimum flows per current.
Flows follow FN = max(minimum flow, lambda * I * k); RH is referenced to the
coolant inlet temperature (65 degC), as in the profile.

    python make_test_plan.py      -> future_test_plan.csv, future_test_plan_overview.csv
"""
import csv
import math
from pathlib import Path

A_CELL = 304.9                       # cm2, active area of the plate
F, VM = 96485.0, 22.414              # C/mol, Nl/mol
K_H2 = VM * 60 / (2 * F)             # Nl/min of H2 per A at lambda 1
K_AIR = VM * 60 / (4 * F) / 0.2095   # Nl/min of air per A at lambda 1
T_COOL, COOL_FLOW, T_GAS = 65.0, 0.6, 70.0
MIN_H2, MIN_AIR = 1.88, 5.31         # Nl/min, profile minimum flows

# OpConds 2.3: current -> (p_out anode, p_out cathode [bar a], DPT anode, DPT cathode [degC])
BASE = {0: (1.40, 1.20, 60.7, 58.2), 15: (1.40, 1.20, 59.9, 57.9), 30: (1.40, 1.20, 59.2, 57.6),
        45: (1.42, 1.22, 58.5, 57.2), 60: (1.46, 1.26, 57.7, 56.9), 90: (1.56, 1.36, 56.1, 56.2),
        150: (1.74, 1.54, 52.4, 54.8), 210: (1.93, 1.73, 48.2, 53.2), 300: (2.20, 2.00, 45.8, 50.8),
        360: (2.20, 2.00, 45.8, 49.0), 450: (2.20, 2.00, 45.8, 46.0), 611: (2.20, 2.00, 45.8, 39.3)}
LAM_H2, LAM_AIR = 1.5, 1.8           # above the minimum-flow region

# whole-cell |Z| at 0.25 Hz [mOhm cm2] from the RO2612030 Gamry sweeps, for the AC amplitude
Z_LF = {15: 700, 30: 480, 45: 389, 60: 326, 90: 270, 150: 212, 210: 180, 300: 160,
        360: 150, 450: 140, 611: 135}
V_AC = 0.008                         # V rms target response (linear regime)


def psat(t):                         # kPa, Buck
    return 0.61121 * math.exp((18.678 - t / 234.5) * (t / (257.14 + t)))


def dpt_for_rh(rh, t_ref=T_COOL):    # degC dew point for RH (%) at t_ref
    target, lo, hi = rh / 100 * psat(t_ref), -20.0, t_ref
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if psat(mid) < target else (lo, mid)
    return round((lo + hi) / 2, 1)


def rh(dpt, t_ref=T_COOL):
    return round(100 * psat(dpt) / psat(t_ref), 1)


def iac(i):                          # A rms: ~8 mV response, peak <= 30 % of DC
    if i <= 0:
        return ""
    z_ohm = Z_LF[i] / 1000 / A_CELL
    return round(min(V_AC / z_ohm, 0.212 * i, 20.0) * 2) / 2


GATE = "G"   # stability gate, defined in the overview file and the document
COLS = ["test_id", "test_name", "step", "phase", "current_A", "j_A_cm2", "direction",
        "hold_s", "eis_gamry", "famos_record", "gamry_Iac_A_rms", "gamry_f_start_Hz",
        "gamry_f_end_Hz", "gamry_points_per_decade", "T_coolant_in_C", "coolant_flow_l_min",
        "T_gas_in_anode_C", "T_gas_in_cathode_C", "DPT_anode_C", "DPT_cathode_C",
        "RH_anode_pct_at_65C", "RH_cathode_pct_at_65C", "p_out_anode_bar_a",
        "p_out_cathode_bar_a", "lambda_H2", "lambda_air", "FN_H2_Nl_min", "FN_air_Nl_min",
        "start_EIS_when", "note"]


def row(tid, name, step, phase, i, direction, hold, eis, *, lam_air=None, lam_h2=None,
        dpt=None, p=None, note=""):
    pa, pc, da, dc = BASE[i]
    if p is not None:
        pa, pc = p
    if dpt is not None:
        da, dc = dpt
    fh = max(MIN_H2, (lam_h2 or LAM_H2) * i * K_H2)
    fa = max(MIN_AIR, (lam_air or LAM_AIR) * i * K_AIR)
    if lam_air is not None:          # a stoichiometry test sets the flow exactly
        fa = lam_air * i * K_AIR
    lh = round(fh / (i * K_H2), 2) if i else ""
    la = round(fa / (i * K_AIR), 2) if i else ""
    return {"test_id": tid, "test_name": name, "step": step, "phase": phase,
            "current_A": i, "j_A_cm2": round(i / A_CELL, 3), "direction": direction,
            "hold_s": hold, "eis_gamry": "yes" if eis else "no",
            "famos_record": "yes" if eis else "no",
            "gamry_Iac_A_rms": iac(i) if eis else "", "gamry_f_start_Hz": 30000 if eis else "",
            "gamry_f_end_Hz": 0.2 if eis else "", "gamry_points_per_decade": 10 if eis else "",
            "T_coolant_in_C": T_COOL, "coolant_flow_l_min": COOL_FLOW,
            "T_gas_in_anode_C": T_GAS, "T_gas_in_cathode_C": T_GAS,
            "DPT_anode_C": da, "DPT_cathode_C": dc, "RH_anode_pct_at_65C": rh(da),
            "RH_cathode_pct_at_65C": rh(dc), "p_out_anode_bar_a": pa, "p_out_cathode_bar_a": pc,
            "lambda_H2": lh, "lambda_air": la, "FN_H2_Nl_min": round(fh, 2),
            "FN_air_Nl_min": round(fa, 2), "start_EIS_when": f"hold done + gate {GATE}" if eis else "",
            "note": note}


rows, overview = [], []


def add(test, *r):
    rows.extend(r)


# T1 -- replicate the CSV profile with FAMOS + Gamry (instrument comparison)
t, n = "T1", "Baseline: OpConds 2.3 profile, EIS descending"
seq = [row(t, n, 1, "conditioning", 300, "-", 600, False, note="profile step 1; record a photo of the plate orientation before start")]
for k, i in enumerate([0, 15, 30, 45, 60, 90, 150, 210, 300, 360, 450], start=2):
    seq.append(row(t, n, k, "pol curve up", i, "ascending", 120 if i <= 45 else 360, False, note="no EIS; U_cell logged"))
for k, i in enumerate([611, 450, 360, 300, 210, 150, 90, 60, 45, 30, 15], start=13):
    seq.append(row(t, n, k, "EIS", i, "descending", 360, True,
                   note="same points as Spectrum1..11 of the CSV delivery; export the bench-tool CSV of the same sweep"))
seq.append(row(t, n, 24, "end", 0, "-", 60, False, note="OCV"))
add(t, *seq)
overview.append([t, n, "Is the FAMOS-vs-CSV difference the instrument or the cell state? Same cell, same profile, same day; FAMOS, Gamry and bench tool on the same sweeps.", "11 EIS, 611 -> 15 A descending", "OpConds 2.3 unchanged"])

# T2 -- hysteresis: ascending then descending EIS
t, n = "T2", "Hysteresis: EIS ascending then descending"
seq = [row(t, n, 1, "conditioning", 300, "-", 600, False, note="then 60 s at OCV")]
k = 2
for i in [45, 150, 300, 450, 611]:
    seq.append(row(t, n, k, "EIS", i, "ascending", 360, True)); k += 1
for i in [450, 300, 150, 45]:
    seq.append(row(t, n, k, "EIS", i, "descending", 360, True, note="pair with the ascending point at the same current")); k += 1
add(t, *seq)
overview.append([t, n, "How much do history and direction change HFR, R_ct, R_mt and the maps? (FAMOS was measured ascending from a cold start, CSV descending after 611 A.)", "9 EIS: 45,150,300,450,611 up, then 450,300,150,45 down", "OpConds 2.3 unchanged"])

# T3 -- air stoichiometry at 450 A and 300 A
t, n = "T3", "Air stoichiometry sweep"
seq, k = [row(t, n, 1, "conditioning", 300, "-", 600, False)], 2
for i in [450, 300]:
    seq.append(row(t, n, k, "EIS reference", i, "-", 360, True, lam_air=1.8, note="reference lambda_air 1.8")); k += 1
    for la in [3.0, 2.5, 2.2, 1.5]:
        seq.append(row(t, n, k, "EIS", i, "lambda high -> low", 360, True, lam_air=la)); k += 1
    seq.append(row(t, n, k, "EIS repeat", i, "-", 360, True, lam_air=1.8, note="repeat of the reference: reproducibility")); k += 1
add(t, *seq)
overview.append([t, n, "How does oxygen supply set R_mt and its map towards the air outlet? (FAMOS 450 A ran at lambda_air 1.53-1.76 instead of 1.8.)", "2 x 6 EIS at 450 A and 300 A: lambda_air 1.8 ref, 3.0, 2.5, 2.2, 1.5, 1.8 repeat", "lambda_air varied; H2 lambda 1.5, pressures and dew points of OpConds 2.3"])

# T4 -- humidity at 450 A and 150 A (both gases, RH referenced to 65 degC)
t, n = "T4", "Humidity sweep"
seq, k = [row(t, n, 1, "conditioning", 300, "-", 600, False)], 2
for i in [450, 150]:
    pa, pc, da, dc = BASE[i]
    seq.append(row(t, n, k, "EIS reference", i, "-", 600, True, note="profile dew points")); k += 1
    ref_rh = rh(dc)
    for r in [r for r in [80, 60, 50, 40, 30] if abs(r - ref_rh) > 5][:4]:
        d = dpt_for_rh(r)
        seq.append(row(t, n, k, "EIS", i, "wet -> dry", 600, True, dpt=(d, d),
                       note="600 s for the membrane to equilibrate; HFR at 1 kHz must be flat over the last 120 s")); k += 1
    seq.append(row(t, n, k, "EIS repeat", i, "-", 600, True, note="back to profile dew points")); k += 1
add(t, *seq)
overview.append([t, n, "How does inlet humidity move HFR (membrane) and R_mt (flooding)? (FAMOS 450 A inlet was ~63 % RH vs 40 % in the CSV profile.)", "2 x 6 EIS: 450 A ref (40 %), RH 80, 60, 50, 30 %; 150 A ref (62 %), RH 80, 50, 40, 30 %; ref repeat each", "both dew points varied together; lambda and pressures of OpConds 2.3"])

# T5 -- pressure at 450 A and 150 A (anode 0.2 bar above cathode)
t, n = "T5", "Pressure sweep"
seq, k = [row(t, n, 1, "conditioning", 300, "-", 600, False)], 2
for i in [450, 150]:
    seq.append(row(t, n, k, "EIS reference", i, "-", 360, True, note="profile pressures")); k += 1
    for pc in [x for x in [1.5, 2.0, 2.5] if abs(x - BASE[i][1]) > 0.1]:
        seq.append(row(t, n, k, "EIS", i, "low -> high", 360, True, p=(round(pc + 0.2, 2), pc),
                       note="check the bench and plate pressure limit before 2.5 bar a")); k += 1
    seq.append(row(t, n, k, "EIS repeat", i, "-", 360, True, note="back to profile pressures")); k += 1
add(t, *seq)
overview.append([t, n, "How does pressure change kinetics and transport, and the R_mt map? Anode always 0.2 bar above cathode.", "450 A: ref 2.0, cathode 1.5 and 2.5 bar a, ref repeat; 150 A: ref 1.54, cathode 2.0 and 2.5 bar a, ref repeat (8 EIS)", "p_out varied; dew points and lambda of OpConds 2.3"])

out = Path(__file__).resolve().parent
with open(out / "future_test_plan.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)

# total time per test: hold + sweep (~4 min) per EIS point
dur = {}
for r in rows:
    dur[r["test_id"]] = dur.get(r["test_id"], 0) + int(r["hold_s"]) + (240 if r["eis_gamry"] == "yes" else 0)
neis = {}
for r in rows:
    neis[r["test_id"]] = neis.get(r["test_id"], 0) + (r["eis_gamry"] == "yes")
gate = ("G = start the Gamry sweep only when, over the last 60 s: coolant inlet within 65 +/- 0.5 degC and "
        "drifting < 0.05 K/min; H2 and air flow within +/- 2 % of set; dew points within +/- 0.5 K; "
        "outlet pressures within +/- 0.02 bar; cell voltage drifting < 1 mV/min. FAMOS records from 30 s "
        "before to 30 s after the sweep.")
with open(out / "future_test_plan_overview.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["test_id", "test_name", "question_answered", "eis_points", "what_is_varied",
                "n_eis", "approx_duration_min", "common_settings"])
    common = ("Gamry galvanostatic 30 kHz -> 0.2 Hz, 10 points/decade, AC amplitude per row (~8 mV "
              "response); FAMOS on all 5 cards; coolant 65 degC inlet, 0.6 l/min; gas inlets 70 degC; "
              "stability gate G before every sweep")
    for o in overview:
        w.writerow(o[:2] + o[2:5] + [neis[o[0]], round(dur[o[0]] / 60), common])
    w.writerow(["G", "stability gate", gate, "", "", "", "", ""])
print(len(rows), "rows;", {k: round(v / 60) for k, v in dur.items()}, "min")
