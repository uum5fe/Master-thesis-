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


# ---------------------------------------------------------------------------
# started from a notebook cell on a cluster driver
# ---------------------------------------------------------------------------

class _FakeConf:
    def __init__(self, values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _FakeSpark:
    def __init__(self, values):
        self.conf = _FakeConf(values)


def _fake_pyspark(monkeypatch, session):
    """Stand in for pyspark, which is not installed outside a cluster."""
    import types
    module = types.ModuleType("pyspark.sql")
    module.SparkSession = type("SparkSession", (),
                               {"getActiveSession": staticmethod(lambda: session)})
    parent = types.ModuleType("pyspark")
    parent.sql = module
    monkeypatch.setitem(sys.modules, "pyspark", parent)
    monkeypatch.setitem(sys.modules, "pyspark.sql", module)


def test_driver_proxy_url_is_built_from_the_cluster_tags(monkeypatch):
    from app import app as app_module
    _fake_pyspark(monkeypatch, _FakeSpark({
        "spark.databricks.clusterUsageTags.clusterOwnerOrgId": "1234567890",
        "spark.databricks.clusterUsageTags.clusterId": "0101-abc-xyz",
        "spark.databricks.workspaceUrl": "example.cloud.databricks.com",
    }))
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    assert app_module.driver_proxy_url(8050) == (
        "https://example.cloud.databricks.com/driver-proxy/o/1234567890/"
        "0101-abc-xyz/8050/")


def test_no_proxy_url_is_invented_when_the_tags_are_missing(monkeypatch):
    from app import app as app_module
    _fake_pyspark(monkeypatch, _FakeSpark({}))
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    # Printing a link that 404s is worse than printing none.
    assert app_module.driver_proxy_url(8050) == ""


def test_no_proxy_url_without_a_spark_session(monkeypatch):
    from app import app as app_module
    _fake_pyspark(monkeypatch, None)
    assert app_module.driver_proxy_url(8050) == ""


def test_banner_in_a_notebook_prefers_the_proxy_link(monkeypatch):
    from app import app as app_module
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    monkeypatch.setattr(app_module, "driver_proxy_url",
                        lambda port: f"https://ws/driver-proxy/o/1/c/{port}/")
    text = app_module.banner("0.0.0.0", 8050)
    assert "https://ws/driver-proxy/o/1/c/8050/" in text
    assert "Open this link:   http://127.0.0.1" not in text
    assert "stay busy" in text                       # the cell does not finish


def test_banner_in_a_notebook_says_so_when_there_is_no_proxy(monkeypatch):
    from app import app as app_module
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    monkeypatch.setattr(app_module, "driver_proxy_url", lambda port: "")
    text = app_module.banner("0.0.0.0", 8050)
    assert "your browser cannot reach" in text
    assert "Databricks App" in text


def test_banner_outside_a_notebook_is_unchanged(monkeypatch):
    from app import app as app_module
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    text = app_module.banner("0.0.0.0", 8050)
    assert "http://127.0.0.1:8050" in text
    assert "driver-proxy" not in text
    assert "stay busy" not in text


# ---------------------------------------------------------------------------
# source integrity
# ---------------------------------------------------------------------------

def test_every_app_module_compiles():
    """Catches the commonest local edit that breaks the app.

    Anything inserted above ``from __future__ import annotations`` - an
    ``import sys``, a ``sys.path.append``, a stray cell marker - is a
    SyntaxError, and the message ("must occur at the beginning of the file")
    points at the future import rather than at the inserted line. Compiling
    every module here turns that into one obvious failure.
    """
    failures = []
    for path in sorted((ROOT / "app").rglob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    assert not failures, "\n".join(failures)


def test_future_imports_are_the_first_statement():
    for path in sorted((ROOT / "app").rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        future = [i for i, line in enumerate(lines)
                  if line.startswith("from __future__")]
        if not future:
            continue
        before = [line for line in lines[:future[0]]
                  if line.strip() and not line.lstrip().startswith("#")]
        # Only the module docstring may precede it.
        joined = "\n".join(before).strip()
        assert not joined or joined.startswith(('"""', "'''")), (
            f"{path.relative_to(ROOT)}: code before the future import: {before[:3]}")
