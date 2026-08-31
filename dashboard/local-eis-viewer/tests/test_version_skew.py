"""A folder holding some new files and some old ones.

This is the failure the check exists for, reported as an AttributeError at
the moment of the call:

    AttributeError: module 'app.services.runner' has no attribute 'run_pipeline'

Too late, because run_evaluation stages gigabytes before it reaches that
call, and too vague, because the attribute it names is the symptom rather
than the cause.
"""

from __future__ import annotations

import sys
import types

import pytest

from app import version


def test_a_complete_copy_passes() -> None:
    version.check()


def test_a_stale_module_is_named_along_with_the_attribute() -> None:
    """The message has to say WHICH file is old and what it is missing."""
    stale = types.ModuleType("app.services.runner_stale")
    stale.run_famos = lambda *a, **k: None      # the pre-rename name only
    stale.__file__ = r"C:\...\app\services\runner.py"
    sys.modules["app.services.runner_stale"] = stale
    try:
        with pytest.raises(version.VersionSkew) as excinfo:
            version.check({"app.services.runner_stale":
                           ("run_pipeline", "build_command")})
    finally:
        del sys.modules["app.services.runner_stale"]

    message = str(excinfo.value)
    assert "run_pipeline" in message and "build_command" in message
    assert "runner.py" in message, "the message must name the file"
    assert "git pull" in message, "and say how to fix it"


def test_a_missing_package_is_not_reported_as_a_stale_copy() -> None:
    """The wrong diagnosis is worse than none.

    Both a half-copied folder and an uninstalled dependency surface here as
    an ImportError.  Telling someone to re-copy `app\\` when the real answer
    is `pip install pandas` sends them to replace files that were never the
    problem, and the check is then confidently wrong.
    """
    broken = types.ModuleType("app_needs_a_package")
    broken.__file__ = "app_needs_a_package.py"

    class _Loader:
        def find_module(self, name, path=None):
            return None

    def _raise(*a, **k):
        raise ImportError("No module named 'somethirdparty'",
                          name="somethirdparty")

    import importlib
    real = importlib.import_module
    importlib.import_module = _raise
    try:
        with pytest.raises(version.MissingDependency) as excinfo:
            version.check({"app_needs_a_package": ("anything",)})
    finally:
        importlib.import_module = real
    assert "pip install" in str(excinfo.value)
    assert "re-copying" in str(excinfo.value)


def test_a_missing_project_module_is_reported_as_skew() -> None:
    """An `app.*` module that will not import IS a copy problem."""
    import importlib
    real = importlib.import_module

    def _raise(*a, **k):
        raise ImportError("No module named 'app.services.brand_new'",
                          name="app.services.brand_new")

    importlib.import_module = _raise
    try:
        with pytest.raises(version.VersionSkew):
            version.check({"app.services.brand_new": ("thing",)})
    finally:
        importlib.import_module = real


def test_the_report_lists_every_module_and_its_file() -> None:
    """`--check-install` output, for a bug report that arrives without one."""
    report = version.report()
    assert report["api_version"] == version.API_VERSION
    for module in version.REQUIRED:
        assert module in report["modules"]
        assert report["modules"][module]["ok"] is True


def test_the_runner_still_answers_to_its_old_name() -> None:
    """`run_famos` predates the CSV path; copies in the wild still call it."""
    from app.services import runner
    assert runner.run_famos is runner.run_pipeline
