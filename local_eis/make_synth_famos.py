#!/usr/bin/env python3
"""Write a synthetic FAMOS card set so the pipeline can be run end to end.

The synthetic deliberately reproduces the two effects the pipeline exists to
handle: a per-channel multiplexer skew that differs from segment to segment,
and a high-frequency arc that has not closed at f_max.
"""
import numpy as np
from pathlib import Path

def main(out="/tmp/famos"):
    d = Path(out); d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    fs = 10000.0
    freqs = np.logspace(np.log10(1.0), np.log10(3000), 18)
    dwell = int(2.0 * fs)
    n_tot = len(freqs) * dwell + int(0.5 * fs)

    cards, seg_no = [], 1
    for _ in range(5):
        segs = [str(seg_no + i) for i in range(15) if seg_no + i <= 72]
        seg_no += 15
        if segs: cards.append(segs)

    for ci, segs in enumerate(cards):
        names = ["UC1"] + segs + ["temp1", "temp2", "temp3", "temp4"]
        n_ch = len(names)
        t = np.arange(n_tot) / fs
        data = np.zeros((n_tot, n_ch), dtype="<f4")

        uc = np.zeros(n_tot)
        for k, f in enumerate(freqs):
            a, b = k * dwell, (k + 1) * dwell
            uc[a:b] = 0.004 * np.cos(2 * np.pi * f * t[a:b])
        data[:, 0] = uc + 2e-4 * rng.normal(size=n_tot)

        for j, s in enumerate(segs):
            slot = j + 1                       # conversion order after UC1
            Rs = 0.060 + 0.006 * rng.normal()
            K = 0.45
            x = np.full(n_tot, K * 0.5)        # DC operating point
            for k, f in enumerate(freqs):
                a, b = k * dwell, (k + 1) * dwell
                w = 2 * np.pi * f
                Z = (Rs + 1j * w * 2e-7 + 0.30 / (1 + 1j * w * 0.02) ** 0.85
                     + 0.012 / (1 + 1j * w * 3.5e-5))
                amp = K * 0.004 / abs(Z)
                # The segment channel is converted LATER than the reference,
                # so its samples are taken at t + tau.  Fitting both against
                # the same cos(w t) basis puts A_seg = U_seg * exp(+j w tau),
                # hence a measured Z = Z_true * exp(-j w tau).  Getting this
                # sign backwards is exactly what the slot-ratio check in
                # silver.fit_structural_skew is there to catch: it returned a
                # ratio of -1.0 and refused the fit.
                ph = -np.angle(Z) + w * (slot / (n_ch * fs))   # mux skew
                x[a:b] += amp * np.cos(w * t[a:b] + ph)
            data[:, 1 + j] = x + 5e-5 * rng.normal(size=n_tot)

        for j in range(4):
            data[:, 1 + len(segs) + j] = 1.0 + 0.01 * 58.4

        # all names live inside ONE |CP field, comma separated: the reader
        # captures up to the first semicolon, then findall's 7,32,<name>
        cp = ",".join(f"7,32,{nm}" for nm in names)
        hdr = (f"|CF,2,1,1;|CK,1,3,1,1;|CD,2,{1.0/fs},1;"
               f"|CR,1,{n_ch},1,0,1;|CP,{cp};|CS,1,{data.nbytes},"
               ).encode("latin-1")
        p = d / f"Leepa_2611976_Current_450A_Test_01_Karte_{ci+1}.DAT"
        with open(p, "wb") as fh:
            fh.write(hdr); fh.write(data.tobytes())
        print(f"  {p.name}: {n_ch} ch, {n_tot} samples, segments {segs[0]}..{segs[-1]}")

    Path("/tmp/curr.csv").write_text("\n".join("0.45;0.10" for _ in range(72)))
    Path("/tmp/temp.csv").write_text("\n".join("1.0;0.01" for _ in range(4)))
    print("  /tmp/curr.csv, /tmp/temp.csv")

if __name__ == "__main__":
    main()
