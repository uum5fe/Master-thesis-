"""polcurve: Gamry Vdc/HFR parsing, FAMOS current density, the plot."""
import json

import numpy as np

import polcurve


def _dta(path, setpoint, vdc, r_hf=1.5e-4, r_pol=6e-4, idc=0.0):
    f = np.logspace(4, -1, 30)
    w = 2 * np.pi * f
    Z = r_hf + r_pol / (1 + 1j * w * 0.02) + 1j * w * 2e-9
    rows = "".join(
        f"\t{i}\t{i}\t{fi:.6g}\t{z.real:.6e}\t{z.imag:.6e}\t1\t{abs(z):.6e}"
        f"\t{np.degrees(np.angle(z)):.4f}\t{idc:.4e}\t{vdc:.5f}\t10\n"
        for i, (fi, z) in enumerate(zip(f, Z)))
    path.write_text(
        "EXPLAIN\nTAG\tEISGALV\nIDCREQ\tQUANT\t0.00000E+000\tDC Current (A)\n"
        "ZCURVE\tTABLE\n\tPt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\tIdc\tVdc\tIERange\n"
        "\t#\ts\tHz\tohm\tohm\tV\tohm\t°\tA\tV\t#\n" + rows, encoding="latin-1")
    return path


def test_gamry_points_use_vdc_and_the_set_point(tmp_path):
    files = [_dta(tmp_path / f"V26_092_HFR_10{i}_CurrVal_{a}.dta", a, v)
             for i, (a, v) in enumerate([(45, 0.86), (150, 0.78), (450, 0.66)])]
    curve, rows = polcurve.gamry_polcurve(files, area_cm2=300.0)
    assert np.allclose(curve.j, [0.15, 0.5, 1.5])
    assert np.allclose(curve.v_mV, [860, 780, 660])
    assert all(r["current_from"] == "file name" for r in rows)
    # HFR 1.5e-4 ohm * 300 cm2 = 45 mOhm*cm2
    assert np.allclose(curve.extra["hfr_mohm_cm2"], 45.0, rtol=0.05)
    irf = curve.ir_free()
    assert np.all(irf.v_mV > curve.v_mV)


def test_measured_idc_wins_when_it_carries_the_load(tmp_path):
    f = _dta(tmp_path / "x_CurrVal_100.dta", 100, 0.8, idc=-98.0)
    _, rows = polcurve.gamry_polcurve([f], area_cm2=100.0)
    assert rows[0]["current_from"] == "Idc" and np.isclose(rows[0]["j_A_cm2"], 0.98)


def test_famos_bands_from_a_gold_folder(tmp_path):
    g = tmp_path / "gold"
    g.mkdir()
    (g / "gold_manifest.json").write_text(json.dumps(
        {"dc_closure": {"I_measured_A": 150.0, "area_measured_cm2": 300.0}}))
    (g / "plate_summary.csv").write_text(
        "segment,class,cx_mm,area_cm2,j_dc\n1,measured,20,2,0.4\n"
        "2,measured,120,2,0.5\n3,measured,200,2,0.6\n4,substituted,200,2,9\n")
    j = polcurve.famos_current_density(tmp_path)
    assert np.isclose(j["plate"], 0.5)
    assert np.isclose(j["air outlet band"], 0.4) and np.isclose(j["air inlet band"], 0.6)
    curves, rows = polcurve.famos_polcurves({"150A": ([], tmp_path)},
                                            v_override={"150A": 0.78})
    assert np.isclose(curves[0].v_mV[0], 780) and len(curves) == 4


def test_plot_in_bench_style(tmp_path):
    import matplotlib.pyplot as plt
    c1 = polcurve.PolCurve("A", np.array([0.1, 0.5, 1.5]), np.array([850, 760, 660]))
    c2 = polcurve.PolCurve("B", np.array([0.1, 0.5, 1.5]), np.array([840, 740, 630]), True)
    fig = polcurve.plot_polcurves([c1, c2], title="Comparison Polcurves")
    fig.savefig(tmp_path / "pc.png", dpi=40)
    plt.close(fig)
    assert (tmp_path / "pc.png").stat().st_size > 0


def test_load_curve_csv_converts_volts(tmp_path):
    p = tmp_path / "ref.csv"
    p.write_text("Current density [A/cm2];Voltage [V]\n0.1;0.85\n1.0;0.72\n")
    c = polcurve.load_curve_csv(p, "ref")
    assert np.allclose(c.v_mV, [850, 720])
