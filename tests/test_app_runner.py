r"""Driving the bundled bronze/silver/gold pipeline.

The pipeline is invoked as a subprocess, so these tests stand a fake ``main.py``
in for it: that exercises the command line, the progress reporting and the
failure path without spending ten minutes reading synthetic recordings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.sources import RunRef                    # noqa: E402
from app.services import runner                        # noqa: E402
from app.settings import Settings                      # noqa: E402


def _settings(tmp_path, **kwargs) -> Settings:
    base = dict(results_roots=[str(tmp_path / "results")], famos_roots=[],
                curr_cal=str(tmp_path / "curr.csv"),
                temp_cal=str(tmp_path / "temp.csv"),
                pipeline_dir=str(tmp_path / "pipeline"),
                allow_inline_pipeline=True,
                scratch_results_root=str(tmp_path / "scratch"))
    return Settings(**{**base, **kwargs})


def _ref(tmp_path) -> RunRef:
    dat = tmp_path / "famos"
    dat.mkdir(exist_ok=True)
    files = []
    for card in range(1, 6):
        path = dat / f"Leepa_2611976_Current_45A_Test_01_Karte_{card}.DAT"
        path.touch()
        files.append(str(path))
    return RunRef(kind="famos", measurement_id="2611976", condition="45A",
                  path=str(dat), layout="famos", files=tuple(files))


def _fake_pipeline(tmp_path, body: str) -> None:
    directory = tmp_path / "pipeline"
    directory.mkdir(exist_ok=True)
    (directory / "main.py").write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# the bundled pipeline
# ---------------------------------------------------------------------------

def test_the_pipeline_is_bundled():
    """local_eis/ holds the same modules that ran on Databricks."""
    for name in ("main.py", "bronze.py", "silver.py", "gold.py", "config.py",
                 "r2d2_geometry.py", "utils.py", "eis_local.py"):
        assert (runner.BUNDLED_PIPELINE / name).is_file(), name


def test_bundled_pipeline_is_the_default(tmp_path):
    settings = _settings(tmp_path, pipeline_dir="")
    assert runner.pipeline_dir(settings) == runner.BUNDLED_PIPELINE


# ---------------------------------------------------------------------------
# where output goes
# ---------------------------------------------------------------------------

def test_output_lands_in_the_results_layout(tmp_path):
    settings = _settings(tmp_path)
    out = runner.output_dir(_ref(tmp_path), settings)
    # <results root>/<order id>/<condition> - exactly where the viewer looks,
    # so a finished run needs no copying afterwards.
    assert out == tmp_path / "results" / "2611976" / "45A"


def test_falls_back_to_scratch_when_no_results_root(tmp_path):
    settings = _settings(tmp_path, results_roots=[])
    out = runner.output_dir(_ref(tmp_path), settings)
    assert out == tmp_path / "scratch" / "2611976" / "45A"


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def test_command_carries_every_required_argument(tmp_path):
    settings = _settings(tmp_path)
    argv = runner.build_command(_ref(tmp_path), Path("/out"), settings)
    assert argv[1] == "main.py"
    for flag in ("--dat", "--curr-cal", "--leepa", "--condition", "--out"):
        assert flag in argv
    assert argv[argv.index("--leepa") + 1] == "2611976"
    assert argv[argv.index("--condition") + 1] == "45A"
    assert "--temp-cal" in argv


def test_optional_arguments_are_omitted_when_unset(tmp_path):
    settings = _settings(tmp_path, temp_cal="", areas_file="")
    argv = runner.build_command(_ref(tmp_path), Path("/out"), settings)
    assert "--temp-cal" not in argv
    assert "--areas" not in argv
    assert "--equal-areas" not in argv


def test_equal_areas_and_no_png_are_passed_through(tmp_path):
    argv = runner.build_command(_ref(tmp_path), Path("/out"), _settings(tmp_path),
                                equal_areas=True, no_png=True)
    assert "--equal-areas" in argv and "--no-png" in argv


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def test_missing_shunt_calibration_stops_the_run(tmp_path):
    settings = _settings(tmp_path, curr_cal="")
    problems = runner.preflight(_ref(tmp_path), settings)
    # Not a warning: without it the impedances are in shunt volts per amp.
    assert any("EIS_CURR_CAL" in p for p in problems)


def test_a_missing_pipeline_is_reported(tmp_path):
    settings = _settings(tmp_path, pipeline_dir=str(tmp_path / "nowhere"))
    (tmp_path / "curr.csv").touch()
    problems = runner.preflight(_ref(tmp_path), settings)
    assert any("EIS_PIPELINE_DIR" in p for p in problems)


def test_disabled_inline_running_is_reported(tmp_path):
    _fake_pipeline(tmp_path, "")
    (tmp_path / "curr.csv").touch()
    settings = _settings(tmp_path, allow_inline_pipeline=False)
    problems = runner.preflight(_ref(tmp_path), settings)
    assert any("EIS_ALLOW_INLINE_PIPELINE" in p for p in problems)


def test_a_complete_configuration_has_no_problems(tmp_path):
    _fake_pipeline(tmp_path, "")
    (tmp_path / "curr.csv").touch()
    assert runner.preflight(_ref(tmp_path), _settings(tmp_path)) == []


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def test_a_successful_run_reports_each_stage(tmp_path):
    _fake_pipeline(tmp_path, "\n".join([
        "print('BRONZE  --  raw ingestion')",
        "print('SILVER  --  correction, modelling, validation')",
        "print('GOLD  --  the plate as a field')",
        "print('DONE')",
    ]))
    (tmp_path / "curr.csv").touch()
    seen = []
    out = runner.run_famos(lambda done, total, message="": seen.append(done),
                           _ref(tmp_path), settings=_settings(tmp_path))
    assert out.endswith(str(Path("2611976") / "45A"))
    assert Path(out).is_dir()
    assert seen[-1] == len(runner.STAGES)        # finished
    assert sorted(set(seen)) == [0, 1, 2, 3, 4]  # and passed through each stage


def test_a_failing_run_raises_with_the_last_output(tmp_path):
    _fake_pipeline(tmp_path, "\n".join([
        "import sys",
        "print('BRONZE  --  raw ingestion')",
        "print('error: --curr-cal is required', file=sys.stdout)",
        "sys.exit(2)",
    ]))
    (tmp_path / "curr.csv").touch()
    with pytest.raises(runner.PipelineUnavailable) as exc:
        runner.run_famos(lambda *a, **k: None, _ref(tmp_path),
                         settings=_settings(tmp_path))
    assert "code 2" in str(exc.value)
    assert "--curr-cal is required" in str(exc.value)   # the cause, not just the code


def test_output_is_collected_for_the_log(tmp_path):
    _fake_pipeline(tmp_path, "print('BRONZE'); print('hello from the pipeline')")
    (tmp_path / "curr.csv").touch()
    lines: list[str] = []
    runner.run_famos(lambda *a, **k: None, _ref(tmp_path),
                     settings=_settings(tmp_path), log_lines=lines)
    assert "hello from the pipeline" in lines
