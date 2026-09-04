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


# ---------------------------------------------------------------------------
# what the command actually carries
# ---------------------------------------------------------------------------

def test_the_command_names_the_plate_it_was_given(tmp_path):
    """The generation used to be dropped between the UI and the pipeline.

    `run_famos` took a geometry and never passed it on, so every run evaluated
    with the pipeline's default plate whatever was picked in the sidebar. The
    two generations have different segment areas, so that is wrong in every
    area-weighted result -- the cell aggregate, the DC closure, the maps.
    """
    from app.plates import registry
    from app.services.runner import build_command
    from app.data.sources import RunRef
    from app.settings import Settings

    ref = RunRef(kind="famos", measurement_id="2611976", condition="45A",
                 path=str(tmp_path), files=("a.DAT",))
    settings = Settings(curr_cal=str(tmp_path / "curr.csv"), temp_cal="",
                        gamry_roots=[], famos_roots=[], results_roots=[])

    for key, want in (("gen1_r2d2_72", "gen1"),
                      ("gen2_r2d2_naboo_72", "gen2")):
        argv = build_command(ref, tmp_path / "out", settings,
                             geom=registry.get(key))
        assert "--plate" in argv
        assert argv[argv.index("--plate") + 1] == want

    # and with no geometry it does not invent one
    argv = build_command(ref, tmp_path / "out", settings)
    assert "--plate" not in argv


def test_the_command_points_at_the_gamry_folder_when_there_is_one(tmp_path):
    """Without --gamry the whole-cell comparison silently never runs."""
    from app.services.runner import build_command
    from app.data.sources import RunRef
    from app.settings import Settings

    sweeps = tmp_path / "hfr"
    sweeps.mkdir()
    ref = RunRef(kind="famos", measurement_id="2611976", condition="45A",
                 path=str(tmp_path), files=("a.DAT",))
    settings = Settings(curr_cal=str(tmp_path / "curr.csv"), temp_cal="",
                        gamry_roots=[str(sweeps)], famos_roots=[],
                        results_roots=[])

    # an empty folder is not a reference
    assert "--gamry" not in build_command(ref, tmp_path / "out", settings)

    (sweeps / "cell_CurrVal_45.dta").write_text("EXPLAIN\nZCURVE\tTABLE\n")
    argv = build_command(ref, tmp_path / "out", settings)
    assert argv[argv.index("--gamry") + 1] == str(sweeps)


# ---------------------------------------------------------------------------
# a CSV selection is not a folder of FAMOS cards
# ---------------------------------------------------------------------------

def test_a_csv_selection_is_sent_to_the_csv_reader(tmp_path):
    """The bug behind "the gen2 CSV files do not evaluate".

    build_command emitted --dat for every source kind, so a csvlog selection
    handed its folder to the FAMOS reader. main.py then printed "source:
    FAMOS", found no .DAT files, and produced a run with no spectra -- which
    looks exactly like the CSV path being broken, when it was never entered.
    """
    from app.data.sources import RunRef

    ref = RunRef(kind="csvlog", measurement_id="FC2600265-02",
                 condition="Spectrum10_65degC_450A",
                 path=str(tmp_path / "sweep"), files=("p1.csv",))
    argv = runner.build_command(ref, tmp_path / "out")

    assert "--source" in argv and argv[argv.index("--source") + 1] == "csv"
    assert "--csv" in argv and argv[argv.index("--csv") + 1] == str(tmp_path / "sweep")
    assert "--dat" not in argv, "a CSV folder must never reach the FAMOS reader"


def test_a_famos_selection_still_uses_dat(tmp_path):
    from app.data.sources import RunRef

    ref = RunRef(kind="famos", measurement_id="2611976", condition="45A",
                 path=str(tmp_path / "cards"), files=("a.DAT",))
    argv = runner.build_command(ref, tmp_path / "out")

    assert "--dat" in argv and argv[argv.index("--dat") + 1] == str(tmp_path / "cards")
    assert "--source" not in argv and "--csv" not in argv


def test_a_csv_run_is_not_blocked_for_want_of_a_shunt_calibration(tmp_path):
    """The R2-D2 logger already applied its coefficients.

    csv_pipeline says so explicitly and declines to apply curr.csv twice, so
    demanding the file before a CSV run refuses a run that would have worked.
    """
    from app.data.sources import RunRef
    from app.settings import Settings

    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "main.py").write_text("")
    settings = Settings(pipeline_dir=str(pipeline), curr_cal="",
                        allow_inline_pipeline=True)

    csv_ref = RunRef(kind="csvlog", measurement_id="x", condition="y",
                     path=str(tmp_path), files=("p1.csv",))
    assert runner.preflight(csv_ref, settings) == []

    famos_ref = RunRef(kind="famos", measurement_id="x", condition="y",
                       path=str(tmp_path), files=("a.DAT",))
    problems = runner.preflight(famos_ref, settings)
    assert any("EIS_CURR_CAL" in p for p in problems), (
        "FAMOS has no other absolute scale, so it must still be required")


def test_the_empty_selection_message_names_the_right_extension(tmp_path):
    from app.data.sources import RunRef
    from app.settings import Settings

    settings = Settings(curr_cal="", allow_inline_pipeline=True)
    csv_ref = RunRef(kind="csvlog", measurement_id="x", condition="y", files=())
    assert any("no CSV files" in p
               for p in runner.preflight(csv_ref, settings))
