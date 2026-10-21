"""A plate mounted turned round: the segments move, the ports do not.

The gases and the coolant are fixed to the cell (H2 in at x = 0, air in at
x = 252). A plate put in rotated 180 deg puts its "inlet" segments at the
outlet. Everything that says WHERE a segment is -- the maps, the inlet ->
outlet Nyquist colour, along-flow statistics -- has to follow the mounting;
everything that says WHAT a segment is -- its number, area, calibration row,
the temperature of the sensors beside it -- must not change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import r2d2_geometry as geom

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _standard_afterwards():
    yield
    geom.use_plate(geom.DEFAULT_PLATE)


@pytest.mark.parametrize("plate", ["gen1", "gen2"])
def test_rotated_plate_tiles_the_grid_with_the_same_areas(plate):
    std = {k: s.area_cm2 for k, s in geom.use_plate(plate).segments.items()}
    p = geom.use_plate(plate, "rot180")
    assert p.orientation == "rot180" and "rotated 180" in p.title
    assert not geom.self_check(verbose=False)["problems"]
    assert {k: s.area_cm2 for k, s in p.segments.items()} == std
    # the drawing's corner pad (1, 1) is the cell's opposite corner
    owner = {pad: n for n, s in p.segments.items() for pad in s.pads}
    drawn = {pad: n for n, s in geom.plate_frame(plate).segments.items()
             for pad in s.pads}
    assert owner[(45, 20)] == drawn[(1, 1)] == "37"
    assert owner[(1, 1)] == drawn[(45, 20)] == "72"


def test_rotated_matches_the_drawing_read_upside_down():
    """gen1 rotated: 72 top left, 42 top right, 67 bottom left, 37 bottom
    right -- the coordinate drawing turned upside down."""
    geom.use_plate("gen1", "rot180")
    lab = geom.label_xy()
    W, H = geom.PLATE_W_MM, geom.PLATE_H_MM
    assert lab["72"][0] < 10 and lab["72"][1] < 10
    assert lab["42"][0] > W - 10 and lab["42"][1] < 10
    assert lab["67"][0] < 10 and lab["67"][1] > H - 10
    assert lab["37"][0] > W - 10 and lab["37"][1] > H - 10


def test_plate_lookup_by_name_is_the_mounted_plate():
    p = geom.use_plate("gen2", "rot180")
    assert geom.plate("gen2") is p is geom.ACTIVE_PLATE
    assert geom.plate_frame().orientation == "standard"
    assert geom.use_plate("gen2").orientation == "standard"   # default


@pytest.mark.parametrize("plate", ["gen1", "gen2"])
def test_sensors_move_with_the_plate_so_temperatures_do_not_change(plate):
    T = {"temp1": 55.0, "temp2": 57.0, "temp3": 60.0, "temp4": 62.0}
    geom.use_plate(plate)
    std = geom.segment_temperatures(T)
    geom.use_plate(plate, "rot180")
    assert geom.TEMP_SENSOR_X_MM["temp1"] == pytest.approx(geom.PLATE_W_MM)
    assert geom.TEMP_SENSOR_X_MM["temp4"] == pytest.approx(0.0)
    rot = geom.segment_temperatures(T)
    for k in std:
        assert rot[k] == pytest.approx(std[k], abs=1e-9), k


def test_inlet_outlet_colour_follows_the_mounting():
    import nyquist
    geom.use_plate("gen1")
    std = nyquist.flow_position(geom.SEGMENTS)
    geom.use_plate("gen1", "rot180")
    rot = nyquist.flow_position(geom.SEGMENTS)
    # air enters at x = 252: segment 67 (pad column 45) is at the air inlet
    # as drawn and at the air outlet when the plate is turned
    assert std["67"] < 0.05 and rot["67"] > 0.95
    assert std["37"] > 0.95 and rot["37"] < 0.05
    for k in std:
        assert std[k] + rot[k] == pytest.approx(1.0, abs=1e-9)


def test_maps_say_how_the_plate_was_mounted():
    import plate_style
    geom.use_plate("gen1")
    assert "mounted" not in plate_style.flow_note()
    geom.use_plate("gen1", "rot180")
    assert "rotated 180" in plate_style.flow_note()


def test_csv_tool_positions_stay_on_the_drawing():
    import csv_source
    geom.use_plate("gen1")
    std = csv_source.mirror_map_x()
    geom.use_plate("gen1", "rot180")
    assert csv_source.mirror_map_x() == std


def test_orientation_spellings():
    assert geom.orientation_key(None) == "standard"
    assert geom.orientation_key("Rotated 180") == "rot180"
    with pytest.raises(ValueError):
        geom.orientation_key("sideways")


def test_config_and_cli_carry_the_orientation():
    from config import DEFAULT, Config
    assert DEFAULT.plate_orientation == "standard"
    cfg = Config.from_cli(["--plate", "gen2", "--plate-orientation", "rot180"])
    assert (cfg.plate, cfg.plate_orientation) == ("gen2", "rot180")
    assert cfg.to_dict()["plate_orientation"] == "rot180"


def test_gen2_naboo_calibration_ships_with_the_package():
    import site_paths
    r = site_paths.resolve("gen2", root=ROOT / "no_volume", package_dir=ROOT)
    assert r["missing"] == []
    assert r["curr"].parent.name == "gen2" and r["abgleich"] is not None
    rows = [ln for ln in r["curr"].read_text().splitlines() if ln.strip()]
    assert len(rows) == 72
    assert r["gain"] is not None and "plate median" in r["gain"].read_text()


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------

def _session(tmp_path, **w):
    from test_runner_session import _DB
    import runner
    famos = tmp_path / "Famos"
    famos.mkdir(parents=True)
    (famos / "Leepa_RO2612086_Current_60A_Test_01_Karte_1.DAT").write_bytes(b"x")
    (tmp_path / "Gamry").mkdir()
    return runner.Session({"dbutils": _DB(w), "spark": None,
                           "displayHTML": lambda _h: None},
                          famos_root=famos, gamry_root=tmp_path / "Gamry",
                          results_root=tmp_path / "EIS_Results",
                          config_root=ROOT / "EIS_config",
                          scratch=tmp_path / "scratch")


@pytest.mark.parametrize("plate, orient, folder", [
    ("gen1", "standard", "2612086"),
    ("gen1", "rot180", "2612086_rot180"),
    ("gen2", "standard", "2612086_gen2"),
    ("gen2", "rot180", "2612086_gen2_rot180"),
])
def test_runner_drop_down_binds_the_mounting_and_its_own_folder(
        tmp_path, plate, orient, folder):
    R = _session(tmp_path, leepa_id="2612086", plate=plate,
                 plate_orientation=orient)
    sel = R.widgets()
    assert sel["plate_orientation"] == orient
    assert geom.ACTIVE_PLATE.key == plate
    assert geom.active_orientation() == orient
    assert R.order_dir().name == folder
    cfg = R._config("60A", tmp_path / "out")
    assert (cfg.plate, cfg.plate_orientation) == (plate, orient)
    assert R.identity(cfg)["settings"]["plate_orientation"] == orient


def test_results_saved_before_the_option_are_standard(tmp_path):
    R = _session(tmp_path, leepa_id="2612086")
    R.widgets()
    ident = R.identity(R._config("60A", tmp_path / "out"))
    old = dict(ident["settings"])
    old.pop("plate_orientation")
    saved = tmp_path / "saved"
    saved.mkdir()
    import runner
    (saved / "run_identity.json").write_text(json.dumps(
        {"digest": "older", "settings": old,
         "runner": runner.settings.RUNNER_VERSION, "written": "then"}))
    assert "OTHER settings" not in R._identity_note(saved, ident)
    R2 = _session(tmp_path / "b", leepa_id="2612086",
                  plate_orientation="rot180")
    R2.widgets()
    ident2 = R2.identity(R2._config("60A", tmp_path / "out"))
    assert "plate_orientation" in R2._identity_note(saved, ident2)


@pytest.mark.parametrize("plate", ["gen1", "gen2"])
def test_tiles_turn_with_the_plate(plate):
    """Segment k still stands for its own 5x5 tile, wherever it sat."""
    import plate_stats
    geom.use_plate(plate)
    std = {k: plate_stats.covered(k) for k in plate_stats.REPRESENTATIVE}
    geom.use_plate(plate, "rot180")
    for k in plate_stats.REPRESENTATIVE:
        c0, c1, r0, r1 = plate_stats.tile_of(k)
        assert (geom.LABEL_COL[k], geom.LABEL_ROW[k]) == ((c0 + c1) // 2,
                                                          (r0 + r1) // 2)
        assert plate_stats.covered(k) == std[k]
    assert plate_stats.tile_of("1") == (41, 45, 16, 20)
