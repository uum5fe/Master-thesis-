"""The Gamry overlay on the Impedance Spectra tab.

Unlike the Reference tab, which crops both instruments to their overlapping
band because that is the only band the CROSS-CHECK is meaningful over, this
overlay draws the Gamry sweep on its own full band -- the point of it is to
show how much further the whole-cell instrument reached than the local,
externally-sampled measurement did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from app.views import spectra


def _pipeline():
    root = Path(__file__).resolve().parents[1] / "local_eis"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gamry_compare
    return gamry_compare


A_CELL = 304.92


def _cell(freq, r_ohm=60e-3, r_ct=300e-3, tau=2e-3, n=0.85, L=2.6e-7):
    w = 2 * np.pi * np.asarray(freq, float)
    return 1j * w * L + r_ohm + r_ct / (1.0 + (1j * w * tau) ** n)


def _write_gamry(path, freq, Z_ohm):
    head = ["EXPLAIN", "TAG\tGALVEIS",
            "STARTTIME\tLABEL\t16.07.2026 07:47:03\tStart Time",
            "ZCURVE\tTABLE",
            "\tPt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\tIdc\tVdc\tIERange",
            "\t#\ts\tHz\tohm\tohm\tV\tohm\tdeg\tA\tV\t#"]
    g = lambda v: f"{v:.6E}".replace(".", ",")
    for i, (f, z) in enumerate(zip(freq, Z_ohm), start=1):
        head.append("\t".join(["", str(i), "1", g(f), g(z.real), g(z.imag), "1",
                               g(abs(z)), g(np.degrees(np.angle(z))),
                               g(4e-3), g(0.77), "13"]))
    path.write_text("\n".join(head) + "\n", encoding="latin-1")


def _run_with_gamry_sweep(tmp_path):
    """A local run whose band stops well short of the Gamry sweep's."""
    import pandas as pd

    results = tmp_path / "Daten" / "EIS_Results" / "2611976" / "45A"
    gamry = tmp_path / "EIS_Daten_Gamry_Tom"
    (results / "silver").mkdir(parents=True)
    (results / "gold").mkdir(parents=True)
    gamry.mkdir(parents=True)

    # detect_layout() needs one of these two before the run is discovered
    # at all -- see app/data/loaders.py.
    pd.DataFrame({"segment": ["1"], "R_ohmic": [0.06]}).to_csv(
        results / "gold" / "plate_summary.csv", index=False)

    freq_local = np.logspace(np.log10(0.3), np.log10(2987), 27)   # stops early
    Z = _cell(freq_local)
    pd.DataFrame({"freq_hz": freq_local,
                  "z_re_mohm_cm2": 1e3 * Z.real,
                  "z_im_mohm_cm2": 1e3 * Z.imag}).to_csv(
        results / "silver" / "cell_aggregate.csv", index=False)

    freq_gamry = np.logspace(np.log10(0.3), np.log10(30000), 61)  # reaches far higher
    _write_gamry(gamry / "cell_CurrVal_45.dta", freq_gamry, _cell(freq_gamry) / A_CELL)
    return results, gamry


def _catalog_and_settings(tmp_path, results, gamry, monkeypatch):
    from app.data.sources import Catalog
    from app.settings import Settings

    settings = Settings(results_roots=[str(results.parent.parent)],
                        gamry_roots=[str(gamry)], famos_roots=[], csv_roots=[])
    catalog = Catalog(settings).refresh(kinds=("results",))
    monkeypatch.setattr(spectra.store, "current_catalog", lambda: catalog)
    monkeypatch.setattr(spectra, "SETTINGS", settings)
    return catalog


def test_the_gamry_curve_reaches_past_the_locals_own_ceiling(tmp_path, monkeypatch):
    results, gamry = _run_with_gamry_sweep(tmp_path)
    _catalog_and_settings(tmp_path, results, gamry, monkeypatch)

    sel = {"kind": "results", "measurement_id": "2611976", "condition": "45A",
          "plate_key": "gen1_r2d2_72"}
    curve = spectra._gamry_curve(sel)
    assert curve is not None
    assert curve["f"].max() > 20_000, (
        "the whole point is showing frequencies the local band never reached")


def test_the_gamry_curve_uses_the_selected_plates_area_not_the_default(
        tmp_path, monkeypatch):
    """Same latent bug as the Reference tab: `selection["plate_key"]`, not
    `selection.get("plate")`, is what the sidebar actually sends (app.py's
    `_selection` callback). Both registered plates share one total area
    today, so this only shows up as a wrong REGISTRY LOOKUP, not yet a wrong
    number -- but it is the lookup that has to be right.
    """
    results, gamry = _run_with_gamry_sweep(tmp_path)
    _catalog_and_settings(tmp_path, results, gamry, monkeypatch)

    seen: list[str] = []
    import app.plates.registry as reg
    real = reg.get
    monkeypatch.setattr(reg, "get", lambda key: seen.append(key) or real(key))

    spectra._gamry_curve({"kind": "results", "measurement_id": "2611976",
                          "condition": "45A", "plate_key": "gen2_r2d2_naboo_72"})
    assert seen == ["gen2_r2d2_naboo_72"]


def test_no_matching_condition_returns_none_not_a_crash(tmp_path, monkeypatch):
    results, gamry = _run_with_gamry_sweep(tmp_path)
    _catalog_and_settings(tmp_path, results, gamry, monkeypatch)

    sel = {"kind": "results", "measurement_id": "2611976", "condition": "999A",
          "plate_key": "gen1_r2d2_72"}
    assert spectra._gamry_curve(sel) is None


def test_no_selection_returns_none() -> None:
    assert spectra._gamry_curve(None) is None
    assert spectra._gamry_curve({}) is None
