"""The plate calibration as versioned Delta tables (runner/calibration.py),
against the in-memory Spark of test_tables_results_auto (the same code runs
on a real Delta Lake in production)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from runner import auto, calibration, tables
from runner import settings as S

from test_runner_session import _session, volume  # noqa: F401
from test_tables_results_auto import FakeSpark

ROOT = Path(__file__).resolve().parents[1]


def _folder(tmp_path, plate="gen1"):
    """A copy of the shipped calibration of `plate`, with a small bode/ so
    the Abgleich counts as a delivery."""
    d = tmp_path / f"cal_{plate}"
    shutil.copytree(ROOT / "EIS_config" / plate, d)
    (d / "Abgleich" / "bode").mkdir(parents=True, exist_ok=True)
    (d / "Abgleich" / "bode" / "seg_100kHz_1Hz_500mA_#1.DTA").write_text("ZCURVE\n")
    (d / "Abgleich" / "bode" / "seg_100kHz_1Hz_500mA_#1_Raw.DTA").write_text("raw\n")
    return d


def test_publish_materialize_round_trip(tmp_path):
    sp = FakeSpark()
    d = _folder(tmp_path)
    v = calibration.publish(sp, "gen1", d, say=lambda *a: None)
    assert v == 1
    row = calibration.versions(sp, "gen1").iloc[0]
    assert (row.valid_from, row.has_abgleich) == ("2000-01-01", True)
    coeffs = tables.read(sp, calibration.COEFFS)
    assert (coeffs.kind == "curr").sum() == 72 and (coeffs.kind == "temp").sum() == 4
    assert len(tables.read(sp, calibration.GAIN)) > 100
    paths = set(tables.read(sp, calibration.FILES)["path"])
    assert "Abgleich/bode/seg_100kHz_1Hz_500mA_#1.DTA" in paths
    assert not any(p.endswith("_Raw.DTA") for p in paths)      # raw sweeps left out
    res = calibration.materialize(sp, "gen1", 1, tmp_path / "m", say=lambda *a: None)
    for kind, name in (("curr", "curr.csv"), ("temp", "temp.csv"),
                       ("gain", "chain_gain.csv")):
        assert res[kind].read_bytes() == (d / name).read_bytes()  # byte for byte
    assert res["abgleich"] is not None and res["missing"] == []
    assert (res["source"], res["calibration_version"]) == ("tables", 1)
    # the same folder again: no new version
    assert calibration.publish(sp, "gen1", d, say=lambda *a: None) == 1
    assert len(calibration.versions(sp, "gen1")) == 1


def test_recalibration_is_a_new_version_chosen_by_date_or_pin(tmp_path):
    sp = FakeSpark()
    d = _folder(tmp_path)
    calibration.publish(sp, "gen1", d, say=lambda *a: None)
    lines = (d / "curr.csv").read_text().splitlines()
    c0, c1 = lines[0].split(";")
    lines[0] = f"{float(c0) * 1.01:.6f};{c1}"                    # segment 1 recalibrated
    (d / "curr.csv").write_text("\n".join(lines) + "\n")
    assert calibration.publish(sp, "gen1", d, valid_from="2026-10-01",
                               say=lambda *a: None) == 2
    pick = lambda **k: int(calibration.select(sp, "gen1", pins={}, **k)["version"])  # noqa: E731
    assert pick(when="2026-09-07") == 1               # measured before the recalibration
    assert pick(when="2026-10-21") == 2
    assert pick() == 2                                # date unknown: the newest
    assert int(calibration.select(sp, "gen1", pins={"gen1": 1})["version"]) == 1
    assert int(calibration.select(sp, "gen1", order="2612090",
                                  pins={"gen1": 1, "2612090": 2})["version"]) == 2
    with pytest.raises(ValueError):
        calibration.select(sp, "gen1", pins={"gen1": 7})
    assert calibration.select(sp, "gen2", pins={}) is None
    # both versions materialize side by side, each with its own bytes
    a = calibration.materialize(sp, "gen1", 1, tmp_path / "m", say=lambda *a: None)
    b = calibration.materialize(sp, "gen1", 2, tmp_path / "m", say=lambda *a: None)
    assert a["curr"].read_bytes() != b["curr"].read_bytes()
    assert b["curr"].read_bytes() == (d / "curr.csv").read_bytes()


def test_a_wrong_folder_is_refused_with_nothing_written(tmp_path):
    sp = FakeSpark()
    d = _folder(tmp_path)
    (d / "curr.csv").write_text("\n".join((d / "curr.csv").read_text()
                                          .splitlines()[:70]) + "\n")
    with pytest.raises(ValueError, match="70 rows"):
        calibration.publish(sp, "gen1", d, say=lambda *a: None)
    assert not sp.tables
    (d / "temp.csv").unlink()
    assert calibration.check_folder(d, "gen1") == ["temp.csv missing"]


def test_a_stray_file_is_not_reused_for_another_version(tmp_path):
    sp = FakeSpark()
    d = _folder(tmp_path)
    (d / "chain_gain.csv").unlink()
    calibration.publish(sp, "gen1", d, say=lambda *a: None)
    res = calibration.materialize(sp, "gen1", 1, tmp_path / "m", say=lambda *a: None)
    assert res["gain"] is None
    (res["target_dir"] / "chain_gain.csv").write_text("0,1,1,0\n")  # built later
    again = calibration.materialize(sp, "gen1", 1, tmp_path / "m", say=lambda *a: None)
    assert again["gain"] is None                     # version 1 has no chain gain


def test_session_takes_the_calibration_from_the_tables(volume, monkeypatch):
    monkeypatch.setattr(S, "DATAGO_FIRST", False)
    monkeypatch.setattr(S, "CALIBRATION_SOURCE", "tables")
    monkeypatch.setattr(S, "CALIBRATION_VERSION", {})
    sp = FakeSpark()
    # nothing published: the Volume / package files, said once
    R, _ = _session(volume, leepa_id="2612086", conditions="60A", test_id="02")
    R.spark = sp
    sel = R.widgets()
    assert sel["plate_files"].get("source", "volume") == "volume"
    ident_vol = R.identity(R._config("60A", volume / "x"))
    # published: the same bytes come from the tables
    calibration.publish(sp, "gen1", ROOT / "EIS_config" / "gen1", say=lambda *a: None)
    sel = R.widgets()
    assert (sel["plate_files"]["source"], sel["plate_files"]["calibration_version"]) == (
        "tables", 1)
    assert "scratch" in str(sel["gain_file"])
    ident = R.identity(R._config("60A", volume / "x"))
    assert ident["digest"] == ident_vol["digest"]          # same bytes, same digest
    assert ident["calibration"]["version"] == 1
    assert ident["calibration"]["curr_sha"] == ident_vol["calibration"]["curr_sha"]


def test_a_recalibration_is_announced_and_planned_but_old_runs_are_not(volume, monkeypatch,
                                                                       tmp_path):
    from test_runner_session import _fake_result
    monkeypatch.setattr(S, "DATAGO_FIRST", False)
    monkeypatch.setattr(S, "CALIBRATION_SOURCE", "tables")
    monkeypatch.setattr(S, "CALIBRATION_VERSION", {})
    sp = FakeSpark()
    calibration.publish(sp, "gen1", ROOT / "EIS_config" / "gen1", say=lambda *a: None)
    R, _ = _session(volume, leepa_id="2612086", conditions="60A", test_id="02")
    R.spark = sp
    R.widgets()
    saved = R.saved_dir("60A")
    _fake_result(saved)
    ident = R.identity(R._config("60A", volume / "x"))
    (saved / "run_identity.json").write_text(json.dumps(
        {**ident, "runner": S.RUNNER_VERSION, "written": "2026-10-21T10:00:00"}))
    tables.publish(sp, saved, say=lambda *a: None)
    # a run saved BEFORE the calibration was recorded: not planned again
    old = tables.read(sp, "runs")
    old_no_cal = old.assign(calibration_json="")
    sp.tables[tables.table_name("runs")] = old_no_cal
    plan = auto.plan(R, orders=["2612086"], say=lambda *a: None)
    assert "2612086/60A_Test02" not in {t["run_key"] for t in plan}
    sp.tables[tables.table_name("runs")] = old
    # recalibration published: announced under cache, planned by the job
    d = _folder(tmp_path)
    lines = (d / "curr.csv").read_text().splitlines()
    c0, c1 = lines[0].split(";")
    lines[0] = f"{float(c0) * 1.01:.6f};{c1}"
    (d / "curr.csv").write_text("\n".join(lines) + "\n")
    calibration.publish(sp, "gen1", d, valid_from="2000-01-01", say=lambda *a: None)
    R.widgets()
    new = R.identity(R._config("60A", volume / "x"))
    assert new["digest"] == ident["digest"] and new["calibration"]["version"] == 2
    assert "ANOTHER calibration" in R._identity_note(saved, new)
    plan = auto.plan(R, orders=["2612086"], say=lambda *a: None)
    got = {t["run_key"]: t["reason"] for t in plan}
    assert got["2612086/60A_Test02"] == "calibration changed"


def test_a_chain_gain_built_from_the_tables_becomes_a_new_version(volume, monkeypatch,
                                                                   tmp_path):
    import gamry_dta
    monkeypatch.setattr(S, "DATAGO_FIRST", False)
    monkeypatch.setattr(S, "CALIBRATION_SOURCE", "tables")
    monkeypatch.setattr(S, "CALIBRATION_VERSION", {})
    sp = FakeSpark()
    d = _folder(tmp_path)
    (d / "chain_gain.csv").unlink()
    calibration.publish(sp, "gen1", d, valid_from="2000-01-01", say=lambda *a: None)
    monkeypatch.setattr(gamry_dta, "read_bode_folder", lambda folder: {"1": None})
    monkeypatch.setattr(gamry_dta, "cross_check_abgleich", lambda sw, curr: {"ok": True})
    monkeypatch.setattr(gamry_dta, "write_gain_csv",
                        lambda sw, out, curr_csv=None, shared=False:
                        Path(out).write_text("segment,freq_hz,gain_real,gain_imag\n"
                                             "1,10,1,0\n1,1000,0.99,-0.04\n"))
    R, _ = _session(volume, leepa_id="2612086", conditions="60A", test_id="02")
    R.spark = sp
    sel = R.widgets()
    v = calibration.versions(sp, "gen1")
    assert list(v["version"]) == [1, 2] and list(v["valid_from"]) == ["2000-01-01"] * 2
    assert "chain_gain.csv built" in v["note"].iloc[-1]
    assert sel["plate_files"]["calibration_version"] == 2
    assert Path(sel["gain_file"]).read_text().startswith("segment,freq_hz")
    # the next session takes version 2 and builds nothing
    R2, _ = _session(volume, leepa_id="2612086", conditions="60A", test_id="02")
    R2.spark = sp
    assert R2.widgets()["plate_files"]["calibration_version"] == 2
    assert len(calibration.versions(sp, "gen1")) == 2
