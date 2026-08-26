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


def test_a_missing_asammdf_names_the_interpreter_to_install_into(tmp_path,
                                                                 monkeypatch):
    """"pip install asammdf" is ambiguous on a machine with several Pythons.

    This one has a Microsoft Store placeholder, a Python Install Manager shim
    and possibly a .venv. Installing into the wrong one leaves the message on
    screen unchanged and looks like the fix did not work, so the message
    quotes sys.executable.
    """
    import sys

    from app.data.sources import Catalog
    from app.settings import Settings

    results, gamry = _split_share(tmp_path)
    settings = Settings(results_roots=[str(tmp_path / "Daten" / "EIS_Results")],
                        gamry_roots=[str(gamry)], famos_roots=[], csv_roots=[])
    catalog = Catalog(settings).refresh(kinds=("results",))
    monkeypatch.setattr(operating.store, "current_catalog", lambda: catalog)
    monkeypatch.setattr(operating, "SETTINGS", settings)

    operating._pipeline_on_path()
    import gamry_compare as GC

    def _no_asammdf(_path):
        raise ImportError("No module named 'asammdf'")

    monkeypatch.setattr(GC, "read_bench_log", _no_asammdf)

    _fields, _meta, problem = operating.fields_for(
        {"kind": "results", "measurement_id": "2611976", "condition": "45A",
         "plate": "gen1_r2d2_72"})

    assert "asammdf" in problem
    assert sys.executable in problem, "it must name the interpreter"
    assert "-m pip install" in problem
    # and it should say the log WAS found, so this does not read as a
    # configuration problem when it is only a missing package
    assert "run_2611976.mf4" in problem


def test_asammdf_is_a_listed_dependency():
    """The Operating map cannot work without it, so it belongs in the file."""
    text = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text()
    assert "asammdf" in text


# ---------------------------------------------------------------------------
# the right cell's bench log, not merely the first one
# ---------------------------------------------------------------------------

def _two_cells(tmp_path):
    """A Gamry parent holding one folder per cell, as on the real share."""
    import pandas as pd

    results = tmp_path / "Daten" / "EIS_Results" / "2611976" / "45A"
    (results / "gold").mkdir(parents=True)
    pd.DataFrame({"segment": ["1"], "R_ohmic": [0.06]}).to_csv(
        results / "gold" / "plate_summary.csv", index=False)

    parent = tmp_path / "EIS_Daten_Gamry_Tom"
    wrong, right = parent / "RO2611959-01", parent / "RO2611976-01"
    wrong.mkdir(parents=True)
    right.mkdir(parents=True)
    # The wrong one sorts first, which is how it used to win.
    (wrong / "20260703_0848_1268_70758_RO2611959-01_V26_087_EIS_Test_3.mf4"
     ).write_bytes(b"x")
    (right / "20260716_0742_1269_70758_RO2611976-01_V26_088_lokale_EIS.mf4"
     ).write_bytes(b"x")
    return results, parent, right


def test_the_bench_log_of_another_cell_is_never_used(tmp_path):
    """Not a near miss: a different cell at a different operating point.

    Every field derived from it -- temperature, pressure, humidity -- would be
    fiction, and nothing on screen would say so.
    """
    operating._pipeline_on_path()
    import gamry_compare as GC

    _results, parent, right = _two_cells(tmp_path)

    # Pointed at the parent, it picks the one that names this cell.
    assert GC.find_bench_log(parent, order_id="2611976").parent == right
    # Pointed at the wrong cell's folder, it refuses rather than substituting.
    assert GC.find_bench_log(parent / "RO2611959-01", order_id="2611976") is None
    # And a file that names no cell is still allowed: silence is not a denial.
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    (quiet / "bench.mf4").write_bytes(b"x")
    assert GC.find_bench_log(quiet, order_id="2611976") is not None


def test_the_cells_own_folder_is_searched_first(tmp_path, monkeypatch):
    from app.data.sources import Catalog
    from app.settings import Settings

    results, parent, right = _two_cells(tmp_path)
    settings = Settings(results_roots=[str(tmp_path / "Daten" / "EIS_Results")],
                        gamry_roots=[str(parent)], famos_roots=[], csv_roots=[])
    catalog = Catalog(settings).refresh(kinds=("results",))
    monkeypatch.setattr(operating.store, "current_catalog", lambda: catalog)
    monkeypatch.setattr(operating, "SETTINGS", settings)

    roots = [str(r) for r in operating._search_roots(catalog.runs[0])]
    assert roots.index(str(right)) < roots.index(str(parent))


def test_a_sweep_from_another_session_is_refused(tmp_path):
    """13 days into a 30 minute recording is not an operating point.

    A zero-order hold is right for a channel logged on change and wrong for a
    timestamp outside the recording: it reports a number from another day as
    though it were this run's state.
    """
    from datetime import datetime, timedelta

    operating._pipeline_on_path()
    import gamry_compare as GC
    import numpy as np

    t0 = datetime(2026, 7, 16, 7, 42)
    log = GC.BenchLog(path=Path("bench.mf4"), t0=t0,
                      series={"I_S": (np.linspace(0, 1800, 50),
                                      np.full(50, 45.0))})

    inside = log.state_at(t0 + timedelta(seconds=900))
    assert inside["I_S"] == pytest.approx(45.0)
    assert not inside.get("out_of_record")

    far = log.state_at(t0 + timedelta(days=13))
    assert far.get("out_of_record") == 1.0
    assert "I_S" not in far, "no value may be reported from outside the record"


def test_one_configured_parent_serves_every_cell(tmp_path, monkeypatch):
    """The point of the design: configure the parent once, never touch it again.

    Adding a campaign means creating a folder named after its order id and
    dropping the files in. Selecting that order in the sidebar then finds them,
    with no settings change. Folder naming is deliberately loose -- RO2611976-01,
    RO2611976, 2611976 and Leepa_2611976_campaign all work -- because a
    convention nobody enforces is a convention that will be broken.
    """
    import pandas as pd
    from app.data.sources import Catalog
    from app.settings import Settings

    operating._pipeline_on_path()
    import gamry_compare as GC

    res_root = tmp_path / "Daten" / "EIS_Results"
    parent = tmp_path / "EIS_Daten_Gamry_Tom"
    cells = {"2611976": "RO2611976-01", "2611959": "RO2611959",
             "2612101": "2612101", "2612222": "Leepa_2612222_campaign"}
    for order, folder in cells.items():
        gold = res_root / order / "45A" / "gold"
        gold.mkdir(parents=True)
        pd.DataFrame({"segment": ["1"], "R_ohmic": [0.06]}).to_csv(
            gold / "plate_summary.csv", index=False)
        cell_dir = parent / folder
        cell_dir.mkdir(parents=True)
        (cell_dir / f"20260716_0742_RO{order}-01_V26_lokale_EIS.mf4"
         ).write_bytes(b"x")

    settings = Settings(results_roots=[str(res_root)],
                        gamry_roots=[str(parent)],   # the parent, once
                        famos_roots=[], csv_roots=[])
    catalog = Catalog(settings).refresh(kinds=("results",))
    monkeypatch.setattr(operating.store, "current_catalog", lambda: catalog)
    monkeypatch.setattr(operating, "SETTINGS", settings)

    def _log_for(run):
        for folder in operating._search_roots(run):
            found = GC.find_bench_log(folder, order_id=run.measurement_id)
            if found is not None:
                return found
        return None

    for run in catalog.runs:
        found = _log_for(run)
        assert found is not None, run.measurement_id
        assert run.measurement_id in found.name, (
            f"{run.measurement_id} got {found.name}")

    # And a cell with results but no folder takes nobody else's log.
    gold = res_root / "2699999" / "45A" / "gold"
    gold.mkdir(parents=True)
    pd.DataFrame({"segment": ["1"], "R_ohmic": [0.06]}).to_csv(
        gold / "plate_summary.csv", index=False)
    catalog = Catalog(settings).refresh(kinds=("results",))
    monkeypatch.setattr(operating.store, "current_catalog", lambda: catalog)
    orphan = next(r for r in catalog.runs if r.measurement_id == "2699999")
    assert _log_for(orphan) is None
