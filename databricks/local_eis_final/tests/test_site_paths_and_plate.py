"""Shared plate files, the plate drop-down, and a cache that is found again.

Three things a colleague's first run depends on:

  * the calibration is read from a shared folder, never from one person's
    workspace, and a gen2 run never borrows a gen1 file;
  * the plate widget reaches the whole run (calibration, geometry, cache key);
  * the cache key does not change between sessions on its own -- it did,
    because the high-frequency response file was rebuilt on every "Run all".
"""

from __future__ import annotations

from pathlib import Path

import pytest

import r2d2_geometry as geom
import site_paths

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "Local EIS Pipeline Runner.py"


def _write(p: Path, text: str = "1;2\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


# ---------------------------------------------------------------- site_paths

def test_files_come_from_the_shared_plate_folder(tmp_path):
    root = tmp_path / "EIS_config"
    for n in ("curr.csv", "temp.csv", "chain_gain.csv"):
        _write(root / "gen1" / n)
    (root / "gen1" / "Abgleich" / "bode").mkdir(parents=True)
    r = site_paths.resolve("gen1", root=root)
    assert r["missing"] == []
    assert r["curr"] == root / "gen1" / "curr.csv"
    assert r["gain"] == root / "gen1" / "chain_gain.csv"
    assert r["abgleich"] == root / "gen1" / "Abgleich"
    assert r["target_dir"] == root / "gen1"


def test_gen2_never_borrows_a_gen1_file(tmp_path):
    root = tmp_path / "EIS_config"
    home = tmp_path / "home"
    pkg = tmp_path / "pkg"
    for n in ("curr.csv", "temp.csv", "chain_gain.csv"):
        _write(root / "gen1" / n)
    for d in (home, pkg):
        _write(d / "curr.csv")
        _write(d / "temp.csv")
    r = site_paths.resolve("gen2", root=root, package_dir=pkg, user_home=home)
    assert r["curr"] is None and r["temp"] is None and r["gain"] is None
    assert set(r["missing"]) == {"curr", "temp"}
    msg = site_paths.missing_message(r)
    assert str(root / "gen2") in msg and "curr.csv" in msg


def test_gen2_files_are_found_in_their_own_folder(tmp_path):
    root = tmp_path / "EIS_config"
    _write(root / "gen2" / "curr.csv")
    _write(root / "gen2" / "temp.csv")
    r = site_paths.resolve("gen2", root=root)
    assert r["missing"] == [] and r["curr"].parent.name == "gen2"


def test_gen1_still_runs_from_the_old_locations(tmp_path):
    """Until the files are on the Volume: the current user's own folder, then
    the package folder (older zips carry curr.csv at the top level)."""
    home = tmp_path / "home"
    pkg = tmp_path / "pkg"
    _write(home / "curr.csv")
    _write(pkg / "temp.csv")
    _write(pkg / "chain_gain_gen1.csv")
    r = site_paths.resolve("gen1", root=tmp_path / "nothing", package_dir=pkg,
                           user_home=home)
    assert r["curr"] == home / "curr.csv"
    assert r["temp"] == pkg / "temp.csv"
    assert r["gain"] == pkg / "chain_gain_gen1.csv"
    # but a new gain file is always written to the shared folder
    assert r["target_dir"] == tmp_path / "nothing" / "gen1"


def test_the_volume_wins_over_the_package(tmp_path):
    root = tmp_path / "EIS_config"
    pkg = tmp_path / "pkg"
    _write(root / "gen1" / "curr.csv", "volume\n")
    _write(pkg / "EIS_config" / "gen1" / "curr.csv", "package\n")
    r = site_paths.resolve("gen1", root=root, package_dir=pkg)
    assert r["curr"].read_text() == "volume\n"


def test_an_unknown_plate_is_refused():
    with pytest.raises(ValueError):
        site_paths.resolve("gen3")


@pytest.mark.parametrize("path,plate,ok", [
    ("/V/R2D2_green_Kashyyyk/Abgleichdaten/Kashyyyk", "gen1", True),
    ("/V/R2D2_green_Kashyyyk/Abgleichdaten/Kashyyyk", "gen2", False),
    ("/V/R2D2_blue_Naboo/Abgleichdaten/Naboo", "gen2", True),
    ("/V/R2D2_blue_Naboo/Abgleichdaten/Naboo", "gen1", False),
    ("/V/EIS_config/gen2/Abgleich", "gen2", True),
    ("/V/EIS_config/gen2/Abgleich", "gen1", False),
])
def test_an_abgleich_search_keeps_to_its_plate(path, plate, ok):
    assert site_paths.matches_plate(path, plate) is ok


def test_the_environment_variable_moves_the_shared_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("EIS_CONFIG_ROOT", str(tmp_path / "elsewhere"))
    assert site_paths.config_root() == tmp_path / "elsewhere"


# ---------------------------------------------------------------- no personal paths

def test_no_module_or_notebook_reads_a_personal_workspace_path():
    hits = []
    for p in ROOT.rglob("*.py"):
        if "tests" in p.parts:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "uum5fe" in line:
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    assert not hits, hits


# ---------------------------------------------------------------- the plate widget

def _runner_cells() -> list[str]:
    return RUNNER.read_text(encoding="utf-8").split("\n# COMMAND ----------\n")


def test_the_run_uses_the_plate_widget_not_a_hardcoded_gen1():
    src = RUNNER.read_text(encoding="utf-8")
    assert "dbutils.widgets.dropdown('plate', 'gen1', ['gen1', 'gen2']" in src
    assert "plate='gen1'" not in src
    assert "plate=PLATE," in src
    assert "PLATE = 'gen1'" not in src


def test_the_widgets_cell_resolves_the_selected_plate(tmp_path):
    """Run the real widgets cell with gen2 selected."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rw", Path(__file__).with_name("test_runner_widgets_cell.py"))
    rw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rw)
    vol = tmp_path
    (vol / "Famos").mkdir()
    (vol / "Gamry").mkdir()
    for card in (1, 2):
        (vol / "Famos" / f"Leepa_RO2612030_Current_45A_Test_1_Karte_{card}.DAT").touch()
    _write(vol / "EIS_config" / "gen2" / "curr.csv")
    _write(vol / "EIS_config" / "gen2" / "temp.csv")
    try:
        ns = rw._run_cell(vol, {"plate": "gen2"})
        assert ns["PLATE"] == "gen2" and ns["PLATE_TAG"] == "_gen2"
        assert ns["PLATE_FILES"]["missing"] == []
        assert ns["PLATE_FILES"]["curr"] == vol / "EIS_config" / "gen2" / "curr.csv"
        assert geom.active_plate() == "gen2"
        ns1 = rw._run_cell(vol)
        assert ns1["PLATE"] == "gen1" and ns1["PLATE_TAG"] == ""
        assert geom.active_plate() == "gen1"
    finally:
        geom.use_plate(geom.DEFAULT_PLATE)


# ---------------------------------------------------------------- the cache key

def test_the_plate_is_in_the_cache_key_for_gen2_only():
    src = RUNNER.read_text(encoding="utf-8")
    block = src[src.index("def _run_identity("):src.index("def _cache_dir(")]
    assert "payload['plate'] = _pl" in block
    assert "if _pl != 'gen1':" in block


def test_the_hf_response_is_built_once_not_on_every_run():
    """Rebuilding it composed old x residual, so its content -- which is in the
    cache key -- changed on every 'Run all' and no session found its cache."""
    cell = next(c for c in _runner_cells()
                if "High-frequency response across conditions" in c)
    assert "HF_RESPONSE_REBUILD = False" in cell
    assert "_hr_existing is not None and not HF_RESPONSE_REBUILD" in cell


def test_the_chain_gain_file_is_not_rewritten_on_every_run():
    cell = next(c for c in _runner_cells() if "Chain response: build" in c)
    assert "CHAIN_GAIN_REBUILD = False" in cell
    assert "Path(CHAIN_GAIN_DEFAULT).is_file() and not CHAIN_GAIN_REBUILD" in cell


def test_a_gen2_display_never_falls_back_to_a_gen1_result():
    src = RUNNER.read_text(encoding="utf-8")
    block = src[src.index("def result_dir("):]
    block = block[:block.index("\ndef ", 10)]
    assert "_pl == str(globals().get('PLATE', 'gen1'))" in block
