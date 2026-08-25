"""Operating conditions over the plate.

The three fields are not equally measured, and the tests that matter here are
the ones that check the tab SAYS so -- a modelled humidity field drawn in the
same colours as a measured temperature field is the failure mode this tab has
to avoid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from app.views import operating


def _pc():
    root = Path(__file__).resolve().parents[1] / "local_eis"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import plate_conditions
    return plate_conditions


SENSORS = {"temp1": 0.0, "temp2": 84.0, "temp3": 168.0, "temp4": 252.0}


@pytest.fixture()
def plate():
    cent = {str(i): (float(x), 60.0)
            for i, x in enumerate(np.linspace(0, 252, 12), start=1)}
    return cent, {k: 25.41 for k in cent}


@pytest.fixture()
def ports():
    PC = _pc()
    return PC.PortState(
        current_a=450.0, n_cells=1.0, t_in_c=70.0, t_out_c=62.0,
        p_in_bara=1.5, p_out_bara=1.39, dew_in_c=55.0,
        air_dry_flow_nl_min=15.0,
        plate_t={"temp1": 58.0, "temp2": 62.0, "temp3": 66.0, "temp4": 70.0})


# ---------------------------------------------------------------------------
# the physics
# ---------------------------------------------------------------------------

def test_saturation_pressure_matches_the_steam_tables():
    """p_sat is the denominator of every humidity number here."""
    PC = _pc()
    for t_c, want_pa in ((0.0, 611.2), (40.0, 7384.0), (60.0, 19946.0),
                         (80.0, 47414.0), (100.0, 101325.0)):
        got = float(PC.saturation_pressure_pa(t_c))
        assert abs(got / want_pa - 1.0) < 0.005, f"{t_c} C: {got:.0f} Pa"


def test_temperature_comes_from_the_plate_sensors_when_there_are_any(plate, ports):
    PC = _pc()
    cent, areas = plate
    t = PC.condition_fields(cent, areas, ports, SENSORS)["temperature"]
    assert t.measured
    assert "plate sensors" in t.provenance
    assert min(t.values.values()) == pytest.approx(58.0, abs=0.1)
    assert max(t.values.values()) == pytest.approx(70.0, abs=0.1)


def test_without_plate_sensors_temperature_says_it_is_only_the_ports(plate, ports):
    """A port-to-port gradient is not a measurement of the plate."""
    PC = _pc()
    cent, areas = plate
    bare = PC.PortState(**{**ports.__dict__, "plate_t": {}})
    t = PC.condition_fields(cent, areas, bare, SENSORS)["temperature"]
    assert not t.measured
    assert "inlet and outlet" in t.provenance
    assert any("not a measurement of the plate" in n for n in t.notes)


def test_humidity_is_never_labelled_measured(plate, ports):
    """Only the INLET humidity is measured. The field is a water balance."""
    PC = _pc()
    cent, areas = plate
    h = PC.condition_fields(cent, areas, ports, SENSORS)["humidity"]
    assert not h.measured
    assert h.provenance.startswith("modelled")


def test_water_accumulates_downstream_even_when_rh_turns_over(plate, ports):
    """What the balance conserves is the vapour fraction, not RH.

    RH = x_v * p / p_sat(T), and this plate is 12 K hotter at the outlet than
    at the inlet. Downstream x_v rises and p falls, but p_sat rises faster
    still, so RH can peak mid-plate and come back down. That is a real effect
    worth seeing on the map -- the driest part of a cell is not always the
    inlet -- so the test asserts the monotone quantity rather than forcing the
    field to be monotone.
    """
    PC = _pc()
    cent, areas = plate
    f = PC.condition_fields(cent, areas, ports, SENSORS)
    h, t, p = f["humidity"], f["temperature"], f["pressure"]

    x_v = [h.values[k] / 100.0 * float(PC.saturation_pressure_pa(t.values[k]))
           / (p.values[k] * 1e5) for k in sorted(cent, key=int)]
    assert all(np.diff(x_v) > 0), "product water must accumulate downstream"

    # and more current must add more water, everywhere
    hot = PC.condition_fields(
        cent, areas, PC.PortState(**{**ports.__dict__, "current_a": 900.0}),
        SENSORS)["humidity"]
    assert hot.values["12"] > h.values["12"]
    assert hot.values["6"] > h.values["6"]


def test_a_starved_cathode_is_called_out_rather_than_drawn(plate, ports):
    """lambda below 1 is impossible: the log is wrong, not the cell."""
    PC = _pc()
    cent, areas = plate
    starved = PC.condition_fields(
        cent, areas,
        PC.PortState(**{**ports.__dict__, "air_dry_flow_nl_min": 2.0}),
        SENSORS)["humidity"]
    assert starved.stoichiometry < 1.0
    assert any("BELOW 1" in n for n in starved.notes)
    assert any("unusable" in n for n in starved.notes)


def test_oversaturation_is_reported_not_clipped(plate, ports):
    PC = _pc()
    cent, areas = plate
    wet = PC.condition_fields(
        cent, areas,
        PC.PortState(**{**ports.__dict__, "air_dry_flow_nl_min": 8.0}),
        SENSORS)["humidity"]
    assert max(wet.values.values()) > 100.0, "clipping would hide flooding"
    assert any("above 100" in n for n in wet.notes)


def test_the_measured_current_map_changes_the_humidity_field(plate, ports):
    """Assuming uniform current would bake in what a local EIS disproves."""
    PC = _pc()
    cent, areas = plate
    flat = PC.condition_fields(cent, areas, ports, SENSORS)["humidity"]
    skewed = {k: (4.0 if int(k) <= 6 else 0.3) for k in cent}
    tilted = PC.condition_fields(cent, areas, ports, SENSORS,
                                 j_dc=skewed)["humidity"]
    assert any("assumed uniform" in n for n in flat.notes)
    assert any("measured local map" in n for n in tilted.notes)
    assert abs(tilted.values["6"] - flat.values["6"]) > 1.0


def test_which_end_is_the_inlet_is_a_choice_that_changes_the_answer(plate, ports):
    PC = _pc()
    cent, areas = plate
    a = PC.condition_fields(cent, areas, ports, SENSORS, inlet="min")["humidity"]
    b = PC.condition_fields(cent, areas, ports, SENSORS, inlet="max")["humidity"]
    assert abs(a.values["1"] - b.values["1"]) > 1.0


def test_a_missing_input_disables_the_field_instead_of_guessing(plate, ports):
    PC = _pc()
    cent, areas = plate
    for attr in ("current_a", "air_dry_flow_nl_min", "p_in_bara"):
        blind = PC.PortState(**{**ports.__dict__, attr: float("nan"),
                                **({"air_flow_nl_min": float("nan")}
                                   if attr == "air_dry_flow_nl_min" else {})})
        h = PC.condition_fields(cent, areas, blind, SENSORS)["humidity"]
        assert h.provenance == "unavailable", attr
        assert all(not np.isfinite(v) for v in h.values.values())


# ---------------------------------------------------------------------------
# the tab
# ---------------------------------------------------------------------------

def test_the_tab_explains_itself_when_there_is_no_bench_log(tmp_path,
                                                            monkeypatch):
    """Without an MF4 there is no operating point, and it must say which file."""
    import pandas as pd
    from app.data.sources import Catalog
    from app.settings import Settings

    run = tmp_path / "results" / "2611976" / "450A"
    (run / "gold").mkdir(parents=True)
    pd.DataFrame({"segment": ["1"], "R_ohmic": [0.06]}).to_csv(
        run / "gold" / "plate_summary.csv", index=False)
    catalog = Catalog(Settings(results_roots=[str(tmp_path / "results")],
                               famos_roots=[], csv_roots=[])).refresh(
        kinds=("results",))
    monkeypatch.setattr(operating.store, "current_catalog", lambda: catalog)

    sel = {"kind": "results", "measurement_id": "2611976",
           "condition": "450A", "plate": "gen1_r2d2_72"}
    fields, _meta, problem = operating.fields_for(sel)
    assert fields is None
    assert ".mf4" in problem.lower()

    fig, prof, status, ports = operating.render(sel)
    assert not fig.data and not prof.data
    assert ports is None
    assert "mf4" in str(status).lower()


def test_an_empty_selection_does_not_raise():
    fields, _meta, problem = operating.fields_for({})
    assert fields is None and problem


# ---------------------------------------------------------------------------
# finding the bench log when it is not beside the run
# ---------------------------------------------------------------------------

def _split_share(tmp_path):
    """Results and reference on different branches, as on the real share."""
    import pandas as pd

    results = tmp_path / "Daten" / "EIS_Results" / "2611976" / "45A"
    gamry = tmp_path / "EIS_Daten_Gamry_Tom"
    (results / "gold").mkdir(parents=True)
    gamry.mkdir(parents=True)
    pd.DataFrame({"segment": ["1"], "R_ohmic": [0.06]}).to_csv(
        results / "gold" / "plate_summary.csv", index=False)
    (gamry / "run_2611976.mf4").write_bytes(b"not a real mf4")
    return results, gamry


def test_the_configured_gamry_root_is_searched_not_just_the_run(tmp_path,
                                                                monkeypatch):
    """Walking up from the run cannot reach a different branch of the share.

    The results live at .../Daten/EIS_Results/<order>/<condition> and the
    sweeps at .../EIS_Daten_Gamry_Tom. No number of parents joins those, which
    is why this tab said "no bench log" while EIS_GAMRY_ROOT was set right.
    """
    from app.data.sources import Catalog
    from app.settings import Settings

    results, gamry = _split_share(tmp_path)
    settings = Settings(results_roots=[str(tmp_path / "Daten" / "EIS_Results")],
                        gamry_roots=[str(gamry)], famos_roots=[], csv_roots=[])
    catalog = Catalog(settings).refresh(kinds=("results",))
    monkeypatch.setattr(operating.store, "current_catalog", lambda: catalog)
    monkeypatch.setattr(operating, "SETTINGS", settings)

    run = catalog.runs[0]
    roots = [str(r) for r in operating._search_roots(run)]
    assert str(gamry) in roots, roots
    # the run's own folder is still tried first
    assert roots[0] == str(results)


def test_the_message_lists_where_it_looked(tmp_path, monkeypatch):
    """"Not found" is only useful with the search path attached."""
    from app.data.sources import Catalog
    from app.settings import Settings

    results, _gamry = _split_share(tmp_path)
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    settings = Settings(results_roots=[str(tmp_path / "Daten" / "EIS_Results")],
                        gamry_roots=[str(empty)], famos_roots=[], csv_roots=[])
    catalog = Catalog(settings).refresh(kinds=("results",))
    monkeypatch.setattr(operating.store, "current_catalog", lambda: catalog)
    monkeypatch.setattr(operating, "SETTINGS", settings)

    _fields, _meta, problem = operating.fields_for(
        {"kind": "results", "measurement_id": "2611976", "condition": "45A",
         "plate": "gen1_r2d2_72"})
    assert "EIS_GAMRY_ROOT" in problem
    assert str(empty) in problem, "it must say where it actually looked"
