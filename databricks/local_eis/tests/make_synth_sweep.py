"""Synthetic raw data for the standard pipeline (std_pipeline.py).

Writes, for one condition:
  * FAMOS v1 card files: UC2 + segment channels + 4 temperature channels,
    a DESCENDING stepped sine sweep as the Gamry runs it, every card armed
    at a different moment (its own time offset), each segment channel read
    one ADC slot later than the one before (multiplexer delay);
  * the Gamry .DTA of the sweep (ZCURVE: Time, Freq, Zreal, Zimag);
  * curr.csv / temp.csv.

Each segment's true impedance is R_ohm + jwL + ZARC(R_ct) + ZARC(R_mt), with
R values varying from segment to segment, so the parameters coming back can
be checked one by one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FS = 25_000.0
K0, K1 = 0.45, 0.10          # curr.csv c0, c1  ->  K = c0 + 1e-3*c1*T
T_TRUE = 60.0
TAU_CT, TAU_MT, N_CT, N_MT, L = 1 / (2 * np.pi * 150), 1 / (2 * np.pi * 4), 0.85, 0.95, 2e-8


def zarc(w, R, tau, n):
    return R / (1 + (1j * w * tau) ** n)


def true_z(params, f):
    w = 2 * np.pi * np.asarray(f, float)
    Rs, Rct, Rmt = params
    return Rs + 1j * w * L + zarc(w, Rct, TAU_CT, N_CT) + zarc(w, Rmt, TAU_MT, N_MT)


def sweep(f_hi=2390.0, f_lo=0.5, ppd=10):
    n = int(round(ppd * np.log10(f_hi / f_lo))) + 1
    f = f_hi / 10 ** (np.arange(n) / ppd)
    dwell = np.maximum(0.4, 4.0 / f)              # >= 4 periods, >= 0.4 s
    return f, dwell


def write_famos_v1(path, names, data, fs):
    cp = ",".join(f"7,32,{nm}" for nm in names)
    hdr = (f"|CF,2,1,1;|CK,1,3,1,1;|CD,2,{1.0/fs},1;"
           f"|CR,1,{len(names)},1,0,1;|CP,{cp};|CS,1,{data.nbytes},").encode("latin-1")
    with open(path, "wb") as fh:
        fh.write(hdr)
        fh.write(data.astype("<f4").tobytes())


def write_dta(path, f, t_end, area=304.92, z_cell=None):
    lines = ["EXPLAIN", "TAG\tEISGALV", "TITLE\tLABEL\tsynthetic",
             "ZCURVE\tTABLE",
             "\tPt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\tIdc\tVdc\tIERange",
             "\t#\ts\tHz\tohm\tohm\tV\tohm\t°\tA\tV\t#"]
    for i, (fi, ti, z) in enumerate(zip(f, t_end, z_cell)):
        lines.append(f"\t{i}\t{int(ti)}\t{fi:.6g}\t{z.real:.6e}\t{z.imag:.6e}\t1\t"
                     f"{abs(z):.6e}\t{np.degrees(np.angle(z)):.4f}\t-450\t0.7\t10")
    Path(path).write_text("\n".join(lines) + "\n", encoding="latin-1")


def main(out, leepa="9999999", cond="450A", n_cards=3, segs_per_card=6,
         offsets_s=(0.0, 0.7, 1.9), noise=2e-6, seed=1):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    f, dwell = sweep()
    lead = 1.0                                   # silence before the sweep
    t_start = lead + np.concatenate([[0], np.cumsum(dwell)[:-1]])
    t_stop = t_start + dwell
    T_total = t_stop[-1] + 1.0
    truth = {}
    seg_no = 1
    for c in range(n_cards):
        segs = [str(seg_no + i) for i in range(segs_per_card)]
        seg_no += segs_per_card
        names = ["UC2"] + segs + ["Temp_1", "Temp_2", "Temp_3", "Temp_4"]
        n_ch = len(names)
        off = offsets_s[c % len(offsets_s)]
        n = int((T_total + off) * FS)
        t_rec = np.arange(n) / FS - off          # sweep time of each sample
        data = np.zeros((n, n_ch))
        U_amp = 0.004
        u = np.zeros(n)
        for fi, a, b in zip(f, t_start, t_stop):
            m = (t_rec >= a) & (t_rec < b)
            u[m] = U_amp * np.cos(2 * np.pi * fi * t_rec[m])
        data[:, 0] = 0.7 + u + noise * rng.standard_normal(n)
        K = K0 + 1e-3 * K1 * T_TRUE
        for j, s in enumerate(segs):
            p = (0.050 + 0.010 * rng.random(), 0.050 + 0.030 * rng.random(),
                 0.060 + 0.060 * rng.random())
            truth[s] = p
            slot = j + 1
            ts = t_rec + slot / (n_ch * FS)      # sampled one slot later
            x = np.full(n, K * 1.4)              # DC: j = 1.4 A/cm2
            for fi, a, b in zip(f, t_start, t_stop):
                m = (ts >= a) & (ts < b)
                Z = true_z(p, fi)
                x[m] += K * U_amp / abs(Z) * np.cos(2 * np.pi * fi * ts[m] - np.angle(Z))
            data[:, 1 + j] = x + noise * rng.standard_normal(n)
        for k in range(4):
            data[:, 1 + segs_per_card + k] = 1.0 + 0.01 * T_TRUE
        write_famos_v1(out / f"Leepa_{leepa}_Current_{cond}_Test_01_Karte_{c+1}.DAT",
                       names, data, FS)
    # the Gamry: step END times, 1 s resolution, from the sweep start
    zc = np.array([true_z((0.055, 0.065, 0.09), fi) / 304.92 for fi in f])
    write_dta(out / f"V00_000_HFR_10x_CurrVal_{cond[:-1]}.DTA", f,
              np.floor(t_stop - lead), z_cell=zc)
    (out / "curr.csv").write_text("\n".join(f"{K0};{K1}" for _ in range(72)))
    (out / "temp.csv").write_text("\n".join("1.0;0.01" for _ in range(4)))
    return {"dir": out, "truth": truth, "freq": f, "leepa": leepa, "cond": cond}


if __name__ == "__main__":
    import sys
    print(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth_sweep")["dir"])
