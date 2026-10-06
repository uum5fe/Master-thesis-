"""ecm_drt: the revised DRT-informed ECM fit (method B)."""
import numpy as np

import ecm_drt

F = np.logspace(np.log10(0.19), np.log10(3000), 40)


def _spec(R1, t1, n1, R2=None, t2=None, n2=1.0, Rs=60.0, noise=0.002, seed=1):
    w = 2 * np.pi * F
    Z = Rs + ecm_drt.z_zarc(w, R1, t1, n1)
    if R2:
        Z = Z + ecm_drt.z_zarc(w, R2, t2, n2)
    rng = np.random.default_rng(seed)
    Z = Z * (1 + noise * (rng.normal(size=F.size) + 1j * rng.normal(size=F.size)))
    return Z, np.full(F.size, noise)


def test_two_arcs_recovered_and_sorted_fast_slow():
    Z, sig = _spec(R2=60.0, t2=2e-3, n2=0.9, R1=250.0, t1=0.03, n1=1.0)
    r = ecm_drt.fit_drt_ecm(F, Z, sig, rs_fixed=60.0)
    assert r["ok"] and r["n_arcs"] == 2
    assert r["tau_fast"] < r["tau_slow"]                      # point 1
    assert abs(r["R_fast"] / 60 - 1) < 0.1 and abs(r["R_slow"] / 250 - 1) < 0.05
    assert abs(r["R_pol"] / 310 - 1) < 0.03
    assert 0.2 < r["chi2_nu"] < 5                              # point 2: sigma weights
    assert r["clean"], r["flags"]


def test_one_arc_spectrum_gets_one_arc():                      # point 3
    Z, sig = _spec(R1=200.0, t1=5e-3, n1=0.9)
    r = ecm_drt.fit_drt_ecm(F, Z, sig, rs_fixed=60.0)
    assert r["n_arcs"] == 1 and r["R_slow"] == 0.0


def test_without_sigma_no_chi2_is_claimed():                   # point 2
    Z, _ = _spec(R1=200.0, t1=5e-3, n1=0.9)
    r = ecm_drt.fit_drt_ecm(F, Z, None)
    assert r["weight"] == "modulus" and np.isnan(r["chi2_nu"])
    assert r["rms_rel_resid"] < 0.02


def test_top_band_and_mains_points_are_left_out():             # points 5, 6
    f = np.array(list(F) + [149.8])
    Z, sig = _spec(R1=200.0, t1=5e-3, n1=0.9)
    Z = np.append(Z, Z[20] * 1.5)                              # a bad 150 Hz step
    sig = np.append(sig, 0.002)
    r = ecm_drt.fit_drt_ecm(f, Z, sig, rs_fixed=60.0, f_max_fit_hz=1000.0)
    assert r["f_fit_hz"][1] <= 1000.0
    assert any(abs(x - 149.8) < 1 for x in r["dropped_mains_hz"])
    assert r["Rs"] == 60.0 and r["Rs_fixed"]
    assert not ecm_drt.mains_mask(np.array([750.0, 946.0])).any()


def test_clean_test_flags_tiny_arcs_and_tau_outside_band():   # point 4
    fake = {"Rs": 1.0, "L": 0.0, "pa": np.array([100.0, 1e-3, 0.9, 0.5, 1e2, 0.9]),
            "layout": ("zarc", "zarc"), "at_bound": [], "nfree": 7, "rss": 1.0,
            "aicc": 0.0, "res": type("R", (), {"success": True})()}
    f = np.logspace(-1, 3, 20)
    Zf = ecm_drt.z_model(1.0, 0.0, fake["pa"], fake["layout"], f)
    r = ecm_drt._report(fake, f, Zf, np.abs(Zf), "modulus", [], None, 1e3,
                        1 / (2 * np.pi * 1e3), 1 / (2 * np.pi * 0.1), [fake])
    assert any("< 1 % of R_pol" in x for x in r["flags"])
    assert any("tau outside band" in x for x in r["flags"])
    assert not r["clean"]


def test_summary_is_median_over_all_fitted():                  # point 7
    fits = {i: {"ok": True, "clean": i < 2, "n_arcs": 2, "fast_element": "zarc",
                "Rs": float(i), "R_pol": 10.0 * i} for i in range(5)}
    s = ecm_drt.summarize(fits)
    assert s["n_fitted"] == 5 and s["n_clean"] == 2
    assert s["Rs"][0] == 2.0 and s["R_pol"][0] == 20.0


def test_method_a_params_map_to_fast_slow():
    p = {"Rs": 60, "R1": 250, "tau1": 0.03, "n1": 1, "R2": 60, "tau2": 2e-3, "n2": 0.9}
    a = ecm_drt.arcs_from_params(p, 2)
    assert a["R_fast"] == 60 and a["R_slow"] == 250 and a["R_pol"] == 310
