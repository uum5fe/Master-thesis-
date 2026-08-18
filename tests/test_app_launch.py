"""The launcher.

A server that starts in silence looks like a server that did not start, and the
address it binds to is not the address to visit. Both of those are worth a test
rather than a comment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_dashboard                                     # noqa: E402
from app.app import banner                               # noqa: E402


def test_banner_shows_a_loopback_url_not_the_bind_address():
    text = banner("0.0.0.0", 8050)
    assert "http://127.0.0.1:8050" in text
    assert "http://0.0.0.0:8050" not in text             # not a destination
    assert "0.0.0.0:8050" in text                        # still says what it bound


def test_banner_explains_an_empty_catalogue(monkeypatch):
    from app.services import store
    monkeypatch.setattr(store, "current_catalog",
                        lambda: type("C", (), {"runs": [], "messages": []})())
    text = banner("0.0.0.0", 8050)
    assert "No measurements found" in text
    assert "EIS_RESULTS_ROOT" in text


def test_banner_counts_what_it_found(monkeypatch):
    from app.services import store
    from app.data.sources import RunRef
    runs = [RunRef("results", "2611976", "45A"), RunRef("results", "2611976", "450A")]
    monkeypatch.setattr(store, "current_catalog",
                        lambda: type("C", (), {"runs": runs, "messages": []})())
    text = banner("127.0.0.1", 8060)
    assert "Found 2 run(s)" in text
    assert "2611976" in text


def test_launcher_maps_flags_onto_environment(tmp_path, monkeypatch):
    results = tmp_path / "results"
    famos = tmp_path / "famos"
    results.mkdir()
    famos.mkdir()

    captured = {}
    monkeypatch.setattr(
        "app.app.serve",
        lambda **kwargs: captured.update(kwargs | {
            "EIS_RESULTS_ROOT": os.environ.get("EIS_RESULTS_ROOT"),
            "EIS_FAMOS_ROOT": os.environ.get("EIS_FAMOS_ROOT")}))

    assert run_dashboard.main(["--results", str(results), "--famos", str(famos),
                               "--port", "8061"]) == 0
    assert captured["EIS_RESULTS_ROOT"] == str(results.resolve())
    assert captured["EIS_FAMOS_ROOT"] == str(famos.resolve())
    assert captured["port"] == 8061
    assert captured["open_browser"] is False


def test_launcher_rejects_a_path_that_does_not_exist(tmp_path, capsys):
    assert run_dashboard.main(["--results", str(tmp_path / "nope")]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_launcher_is_runnable_from_any_working_directory(tmp_path, monkeypatch):
    """It inserts the project root itself, so cwd must not matter."""
    monkeypatch.chdir(tmp_path)
    assert str(ROOT) in sys.path
    import importlib
    importlib.reload(run_dashboard)
    assert str(ROOT) in sys.path
