"""The configuration check.

"No runs discovered" has several causes that look identical from the browser.
These tests pin the check to naming the right one, because a diagnostic that is
merely reassuring is worse than none.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.diagnose import report                      # noqa: E402
from app.settings import Settings                    # noqa: E402


def _settings(**kwargs) -> Settings:
    base = dict(results_roots=[], famos_roots=[], datago_metadata_table="",
                warehouse_id="", scratch_results_root=str(Path("/nonexistent")))
    return Settings(**{**base, **kwargs})


def _make_run(root: Path, order: str, condition: str) -> None:
    run = root / order / condition
    (run / "gold").mkdir(parents=True)
    pd.DataFrame({"segment": ["1"], "class": ["measured"], "R_ohmic": [80.0],
                  "cx_mm": [14.0], "cy_mm": [15.1], "area_cm2": [5.082]}).to_csv(
        run / "gold" / "plate_summary.csv", index=False)


def test_reports_a_working_setup(tmp_path):
    _make_run(tmp_path, "2611976", "45A")
    text = report(_settings(results_roots=[str(tmp_path)]))
    assert "2611976: 45A" in text
    assert "1 run(s)" in text


def test_names_a_missing_folder(tmp_path):
    text = report(_settings(famos_roots=[str(tmp_path / "gone")]))
    assert "does not exist" in text
    assert "OneDrive" in text          # the likely cause on this user's machine


def test_names_an_unrecognised_dat_filename(tmp_path):
    (tmp_path / "Cell2611976_45A_run1.DAT").touch()
    text = report(_settings(famos_roots=[str(tmp_path)]))
    assert "1 .DAT file(s) found" in text
    assert "not recognised" in text
    assert "EIS_FAMOS_REGEX" in text


def test_recognises_the_two_known_conventions(tmp_path):
    (tmp_path / "Leepa_2611976_Current_45A_Test_01_Karte_1.DAT").touch()
    (tmp_path / "RO2611976-01_Current_450A_Test_01_Karte_1.DAT").touch()
    text = report(_settings(famos_roots=[str(tmp_path)]))
    assert "2 recognised" in text
    assert "order 2611976  condition 45A" in text
    assert "order 2611976  condition 450A" in text
    assert "not recognised" not in text


def test_names_a_results_root_with_the_wrong_layout(tmp_path):
    (tmp_path / "some_folder" / "whatever").mkdir(parents=True)
    text = report(_settings(results_roots=[str(tmp_path)]))
    assert "some_folder: nothing recognisable inside" in text
    assert "plate_summary.csv" in text
    assert "nothing — every dropdown will be empty" in text


def test_warns_when_the_root_is_itself_a_single_run(tmp_path):
    _make_run(tmp_path, "2611976", "45A")
    text = report(_settings(results_roots=[str(tmp_path / "2611976" / "45A")]))
    assert "IS itself one run" in text


def test_lists_subfolders_when_a_famos_root_holds_no_recordings(tmp_path):
    (tmp_path / "2611976_16_07").mkdir()
    text = report(_settings(famos_roots=[str(tmp_path)]))
    assert "no .DAT files" in text
    assert "2611976_16_07" in text          # points at where to look next


def test_flags_a_stray_env_txt(tmp_path, monkeypatch):
    """Notepad appends .txt and Explorer hides it; this is the usual outcome."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.txt").write_text("EIS_RESULTS_ROOT=C:\\x\n")
    import app.diagnose as diagnose
    monkeypatch.setattr(diagnose, "DOTENV_LOADED", "")
    text = diagnose.report(_settings())
    assert ".env.txt" in text
    assert "File name extensions" in text
