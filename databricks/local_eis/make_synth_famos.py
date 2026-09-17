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

def write_v2(path, names, data, fs, bytes_per_val=8):
    """Write a FAMOS **v2** file: one metadata block per channel, float64.

    Exists so the v2 reader is tested against a byte layout rather than
    against a description of one. The campaign files this mirrors are on a
    share the test suite cannot reach, and a reader verified only by "it did
    not raise on the one file I had" is not verified.

    Layout, per the recordings it was written from:

        |CF,2,...           file format, version 2
        |CK,...             key block
        per channel:  |CG  |CD (dt)  |CP (.., bytes/value, ..)  |Cb  |CR  |CN
        |CS,<ver>,<bytes>,  then the interleaved samples

    |CN carries the name as field 6, counting from zero.
    """
    import numpy as np
    from pathlib import Path

    d = np.asarray(data, dtype="<f8" if bytes_per_val == 8 else "<f4")
    n_samples, n_ch = d.shape
    assert n_ch == len(names)

    parts = [b"|CF,2,1,1;", b"|CK,1,3,1,1;"]
    for i, nm in enumerate(names):
        nm_b = nm.encode("latin-1")
        parts.append(b"|CG,1,1,1;")
        parts.append(f"|CD,2,16,{1.0 / fs:.10g},1,0;".encode("latin-1"))
        # |CP,<ver>,<len>,<buffer>,<bytes per value>,<numeric type>,...
        parts.append(f"|CP,1,14,1,{bytes_per_val},7,0,1;".encode("latin-1"))
        parts.append(b"|Cb,1,10,1,0,0,0,0;")
        parts.append(b"|CR,1,12,1,1,0,1,;")
        # |CN,<ver>,<len>,<idx>,0,0,<name len>,<NAME>,<comment len>,;
        parts.append(b"|CN,1,20," + str(i).encode() + b",0,0,"
                     + str(len(nm_b)).encode() + b"," + nm_b + b",0,;")
    raw = d.tobytes()
    parts.append(f"|CS,1,{len(raw)},".encode("latin-1"))
    header = b"".join(parts)
    Path(path).write_bytes(header + raw)
    return Path(path)


if __name__ == "__main__":
    main()
