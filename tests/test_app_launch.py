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
        # Only the module docstring may precede it - possibly a raw one.
        joined = "\n".join(before).strip().lstrip("rRuUbBfF")
        assert not joined or joined.lstrip().startswith(("\"", "'")), (
            f"{path.relative_to(ROOT)}: code before the future import: {before[:3]}")


# ---------------------------------------------------------------------------
# .env — the one place a path is written down
# ---------------------------------------------------------------------------

def test_dotenv_reads_a_windows_path_with_spaces(tmp_path, monkeypatch):
    from app import settings
    env = tmp_path / ".env"
    env.write_text(
        "# where the recordings live\n"
        "EIS_FAMOS_ROOT=C:\\Users\\uum5fe\\OneDrive - Bosch Group\\Local_Eis\n",
        encoding="utf-8")
    monkeypatch.delenv("EIS_FAMOS_ROOT", raising=False)
    settings.load_dotenv(env)
    # Backslashes are not escapes and spaces do not need quoting.
    assert os.environ["EIS_FAMOS_ROOT"] == \
        "C:\\Users\\uum5fe\\OneDrive - Bosch Group\\Local_Eis"


def test_dotenv_tolerates_quotes_export_and_a_bom(tmp_path, monkeypatch):
    from app import settings
    env = tmp_path / ".env"
    env.write_text('\ufeffexport EIS_TITLE="Local EIS Viewer"\n'
                   "EIS_DEFAULT_PLATE='gen1_r2d2_72'\n", encoding="utf-8")
    for key in ("EIS_TITLE", "EIS_DEFAULT_PLATE"):
        monkeypatch.delenv(key, raising=False)
    settings.load_dotenv(env)
    assert os.environ["EIS_TITLE"] == "Local EIS Viewer"
    assert os.environ["EIS_DEFAULT_PLATE"] == "gen1_r2d2_72"


def test_the_real_environment_beats_the_file(tmp_path, monkeypatch):
    """A --famos flag must not need the file edited and changed back."""
    from app import settings
    env = tmp_path / ".env"
    env.write_text("EIS_FAMOS_ROOT=C:\\from-the-file\n", encoding="utf-8")
    monkeypatch.setenv("EIS_FAMOS_ROOT", "D:\\from-the-flag")
    settings.load_dotenv(env)
    assert os.environ["EIS_FAMOS_ROOT"] == "D:\\from-the-flag"


def test_blank_and_comment_lines_are_ignored(tmp_path, monkeypatch):
    from app import settings
    env = tmp_path / ".env"
    env.write_text("\n# just a comment\n\n   \nEIS_TITLE=x\n", encoding="utf-8")
    monkeypatch.delenv("EIS_TITLE", raising=False)
    settings.load_dotenv(env)
    assert os.environ["EIS_TITLE"] == "x"


def test_example_env_documents_every_setting_the_app_reads():
    """The example file is the documentation; it must not fall behind."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for name in ("EIS_FAMOS_ROOT", "EIS_RESULTS_ROOT", "EIS_CURR_CAL",
                 "EIS_PLATE_SPEC_DIR", "EIS_ALLOW_INLINE_PIPELINE",
                 "EIS_FAMOS_REGEX"):
        assert name in text, f"{name} is not mentioned in .env.example"


def test_dotenv_preserves_a_unc_network_path(tmp_path, monkeypatch):
    r"""\\server\share paths must survive verbatim - leading backslashes and all."""
    from app import settings
    unc = (r"\\bosch.com\DfsRB\DfsDE\LOC\Fe\ILM\A_ILM_DSETD\Gruppenablage"
           r"\EAT3\Charan\Lokale_EIS\Daten\2611976_16_07")
    env = tmp_path / ".env"
    env.write_text(f"EIS_FAMOS_ROOT={unc}\n", encoding="utf-8")
    monkeypatch.delenv("EIS_FAMOS_ROOT", raising=False)
    settings.load_dotenv(env)
    assert os.environ["EIS_FAMOS_ROOT"] == unc
    assert os.environ["EIS_FAMOS_ROOT"].startswith("\\\\")


def test_a_unc_path_is_not_split_into_several_roots(monkeypatch):
    """os.pathsep is ';' on Windows, so a UNC path is one root, not many."""
    from app import settings
    unc = r"\\bosch.com\DfsRB\Charan\Lokale_EIS\Daten"
    monkeypatch.setenv("EIS_FAMOS_ROOT", unc)
    assert settings.Settings().famos_roots == [unc]


# ---------------------------------------------------------------------------
# creating the settings file
# ---------------------------------------------------------------------------

class _InitArgs:
    def __init__(self, **kwargs):
        self.famos = kwargs.get("famos")
        self.results = kwargs.get("results")
        self.plate_specs = kwargs.get("plate_specs")


def test_init_writes_an_env_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_dashboard, "ROOT", tmp_path)
    (tmp_path / ".env.example").write_text(
        "# comment\nEIS_FAMOS_ROOT=C:\\placeholder\n", encoding="utf-8")

    assert run_dashboard.write_env_file(_InitArgs()) == 0
    assert (tmp_path / ".env").is_file()
    # The absolute path is printed: which file is read must never be in doubt.
    assert str(tmp_path / ".env") in capsys.readouterr().out


def test_init_fills_in_paths_given_on_the_command_line(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dashboard, "ROOT", tmp_path)
    (tmp_path / ".env.example").write_text(
        "EIS_FAMOS_ROOT=C:\\placeholder\n# EIS_RESULTS_ROOT=C:\\other\n",
        encoding="utf-8")

    unc = r"\\bosch.com\DfsRB\Charan\Lokale_EIS\Daten\2611976_16_07"
    run_dashboard.write_env_file(_InitArgs(famos=unc, results=r"C:\results"))

    written = (tmp_path / ".env").read_text(encoding="utf-8")
    active = [l for l in written.splitlines() if l.strip()
              and not l.strip().startswith("#")]
    assert f"EIS_FAMOS_ROOT={unc}" in active            # UNC written verbatim
    assert r"EIS_RESULTS_ROOT=C:\results" in active     # added even if commented out


def test_init_does_not_need_the_folders_to_exist(tmp_path, monkeypatch):
    """The share may be offline, or the paths may be for another machine."""
    monkeypatch.setattr(run_dashboard, "ROOT", tmp_path)
    (tmp_path / ".env.example").write_text("", encoding="utf-8")
    assert run_dashboard.write_env_file(
        _InitArgs(famos=r"\\server\share\nothing-here")) == 0
    assert r"\\server\share\nothing-here" in (tmp_path / ".env").read_text()


def test_init_refuses_to_clobber_an_existing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_dashboard, "ROOT", tmp_path)
    (tmp_path / ".env.example").write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("EIS_FAMOS_ROOT=keep me\n", encoding="utf-8")

    run_dashboard.write_env_file(_InitArgs(famos=r"C:\other"))
    assert "keep me" in (tmp_path / ".env").read_text()
    assert "already exists" in capsys.readouterr().out


def test_force_replaces_it(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dashboard, "ROOT", tmp_path)
    (tmp_path / ".env.example").write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("EIS_FAMOS_ROOT=old\n", encoding="utf-8")

    run_dashboard.write_env_file(_InitArgs(famos=r"C:\new"), force=True)
    assert "old" not in (tmp_path / ".env").read_text()


# ---------------------------------------------------------------------------
# the sidebar reflects the deployment, not Databricks
# ---------------------------------------------------------------------------

def _reload_app():
    import importlib
    from app import app as app_module, settings as settings_module
    importlib.reload(settings_module)
    importlib.reload(app_module)
    return app_module


def test_datago_is_not_offered_when_it_is_not_configured(monkeypatch):
    for key in ("EIS_DATAGO_METADATA_TABLE", "DATABRICKS_WAREHOUSE_ID",
                "DATABRICKS_HOST"):
        monkeypatch.delenv(key, raising=False)
    module = _reload_app()
    options = module.location_options()
    # Offering a source that cannot work invites picking it and wondering why
    # nothing appears.
    assert [o["value"] for o in options] == ["volumes"]
    assert options[0]["label"] == "Local or network folder"


def test_databricks_vocabulary_returns_where_it_means_something(monkeypatch):
    monkeypatch.setenv("EIS_DATAGO_METADATA_TABLE", "cat.sch.meta")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "abc123")
    module = _reload_app()
    options = module.location_options()
    assert [o["value"] for o in options] == ["volumes", "datago"]
    assert options[0]["label"] == "Volumes / file system"


def test_the_hint_names_the_folders_actually_being_read(monkeypatch, tmp_path):
    for key in ("EIS_DATAGO_METADATA_TABLE", "DATABRICKS_WAREHOUSE_ID",
                "DATABRICKS_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EIS_RESULTS_ROOT", str(tmp_path / "results"))
    module = _reload_app()
    hint = module.location_hint(module.location_options())
    assert "Reading:" in hint and "results" in hint


def test_the_hint_says_what_to_do_when_nothing_is_configured(monkeypatch):
    for key in ("EIS_RESULTS_ROOT", "EIS_FAMOS_ROOT",
                "EIS_DATAGO_METADATA_TABLE", "DATABRICKS_WAREHOUSE_ID",
                "DATABRICKS_HOST"):
        monkeypatch.delenv(key, raising=False)
    module = _reload_app()
    hint = module.location_hint(module.location_options())
    assert "--init" in hint
    _reload_app()


def test_settings_summary_hides_databricks_rows_when_unconfigured(monkeypatch):
    from app.settings import Settings
    rows = dict(Settings(datago_metadata_table="", warehouse_id="").summary())
    assert "datago" not in rows
    assert "SQL warehouse" not in rows
    assert "Pipeline" in rows and "FAMOS roots" in rows

    rows = dict(Settings(datago_metadata_table="cat.sch.meta").summary())
    assert rows["datago"] == "cat.sch.meta"


def test_request_logging_is_quieted_but_errors_are_not(monkeypatch):
    """A quiet log that also hides failures would be worse than a noisy one."""
    import logging
    from app import app as app_module

    logger = logging.getLogger("werkzeug")
    monkeypatch.setattr(logger, "level", logging.INFO)
    app_module.quiet_request_log()

    assert not logger.isEnabledFor(logging.INFO)        # access lines: gone
    assert logger.isEnabledFor(logging.WARNING)         # warnings: kept
    assert logger.isEnabledFor(logging.ERROR)           # errors: kept
