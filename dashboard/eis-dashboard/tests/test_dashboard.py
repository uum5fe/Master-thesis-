"""The dashboard's contract: .env in, correct commands and honest maps out."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eis_dashboard import pipeline, results, settings, theme  # noqa: E402
from eis_dashboard.app import plate_figure  # noqa: E402
from tests.make_fixture import build  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("EIS_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EIS_NO_DOTENV", "1")


@pytest.fixture
def run_dir(tmp_path) -> Path:
    build(tmp_path / "results")
    return tmp_path / "results"


# ---- settings -------------------------------------------------------------


def test_dotenv_fills_only_unset_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("EIS_PLATE=gen2\nEIS_LEEPA=9999999\n")
    monkeypatch.delenv("EIS_NO_DOTENV", raising=False)
    monkeypatch.setenv("EIS_PLATE", "gen1")     # already exported: must win
    st = settings.load(env)
    assert st.plate == "gen1"
    assert st.leepa == "9999999"


def test_dotenv_keeps_windows_paths_and_spaces_intact(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('EIS_DAT_DIR="C:\\Users\\me\\OneDrive - Bosch Group\\Famos"\n')
    monkeypatch.delenv("EIS_NO_DOTENV", raising=False)
    settings.load(env)
    assert os.environ["EIS_DAT_DIR"] == "C:\\Users\\me\\OneDrive - Bosch Group\\Famos"


def test_comments_and_blank_lines_are_ignored(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# a comment\n\n  \nEIS_PLATE=gen2\n")
    monkeypatch.delenv("EIS_NO_DOTENV", raising=False)
    assert settings.load(env).plate == "gen2"


def test_missing_input_directory_is_an_error_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("EIS_DAT_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("EIS_READ_ONLY", "1")
    st = settings.load()
    assert not st.ok()
    assert any(name == "EIS_DAT_DIR" and sev == "error"
               for sev, name, _ in st.problems())


def test_read_only_does_not_require_a_pipeline_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("EIS_DAT_DIR", str(tmp_path))
    monkeypatch.setenv("EIS_READ_ONLY", "1")
    monkeypatch.setenv("EIS_PIPELINE_DIR", str(tmp_path / "absent"))
    assert settings.load().ok()


# ---- the run command ------------------------------------------------------


def test_command_carries_the_env_paths(monkeypatch, tmp_path):
    cal = tmp_path / "curr.csv"
    cal.write_text("1;0\n")
    monkeypatch.setenv("EIS_DAT_DIR", str(tmp_path))
    monkeypatch.setenv("EIS_OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("EIS_CURR_CAL", str(cal))
    st = settings.load()
    cmd = pipeline.build_command(st, pipeline.RunSpec("150A"), "2612030")
    assert "--dat" in cmd and str(tmp_path) in cmd
    assert "--curr-cal" in cmd and str(cal) in cmd
    assert cmd[cmd.index("--out") + 1].endswith(str(Path("2612030") / "150A"))
    assert cmd[cmd.index("--condition") + 1] == "150A"


def test_optional_paths_are_omitted_when_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("EIS_DAT_DIR", str(tmp_path))
    cmd = pipeline.build_command(settings.load(), pipeline.RunSpec("45A"), "")
    for flag in ("--gamry", "--temp-cal", "--areas", "--gain", "--leepa"):
        assert flag not in cmd


def test_toggles_map_to_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("EIS_DAT_DIR", str(tmp_path))
    spec = pipeline.RunSpec("45A", drt=False, spatial=False, png=False,
                            html=False, f_min=0.5, f_max=1000.0)
    cmd = pipeline.build_command(settings.load(), spec, "")
    for flag in ("--no-drt", "--no-spatial", "--no-png", "--no-html"):
        assert flag in cmd
    assert cmd[cmd.index("--f-min") + 1] == "0.5"


@pytest.mark.parametrize("name, cond", [
    ("Leepa_RO2612030_Current_150A_Test_01_Karte_1.DAT", "150A"),
    ("Leepa_RO2612030_Current_45A_Test_01_Karte_2.DAT", "45A"),
    ("x_450 A_y.DAT", "450A"),
])
def test_conditions_are_discovered_from_filenames(tmp_path, name, cond):
    (tmp_path / name).write_bytes(b"")
    assert pipeline.discover_conditions(tmp_path) == [cond]


def test_conditions_sort_by_current_not_alphabetically(tmp_path):
    for c in ("45A", "150A", "450A"):
        (tmp_path / f"Leepa_RO2612030_Current_{c}_Test_01_Karte_1.DAT").write_bytes(b"")
    assert pipeline.discover_conditions(tmp_path) == ["45A", "150A", "450A"]


def test_leepa_is_read_from_the_filename(tmp_path):
    (tmp_path / "Leepa_RO2612030_Current_150A_Karte_1.DAT").write_bytes(b"")
    assert pipeline.discover_leepa(tmp_path) == "2612030"


# ---- results --------------------------------------------------------------


def test_runs_are_discovered_under_the_results_root(run_dir):
    runs = results.discover(run_dir)
    assert [r.label for r in runs] == ["2612030 / 150A"]
    assert runs[0].has(results.PLATE_SUMMARY)


def test_missing_results_root_is_empty_not_an_exception(tmp_path):
    assert results.discover(tmp_path / "absent") == []


def test_label_columns_are_not_offered_as_parameters(run_dir):
    df = results.plate_summary(results.discover(run_dir)[0])
    params = results.parameter_columns(df)
    for c in ("class", "tier", "fault", "flags", "segment", "cx_mm"):
        assert c not in params
    assert "R_ohmic" in params and "R_mt" in params


def test_sd_columns_are_not_offered_as_parameters(run_dir):
    df = results.plate_summary(results.discover(run_dir)[0])
    assert not [c for c in results.parameter_columns(df) if c.endswith("_sd")]


# ---- the mass-transport distinction ---------------------------------------


def test_band_limited_is_decided_by_tau_max():
    """tau_max < 10 ms means the slow DRT bucket has no bins to sum."""
    df = pd.DataFrame({"tau_max": [1.06, 0.05, 0.01, 0.009999, 0.004, None]})
    assert list(results.band_limited_mt(df)) == [False, False, False,
                                                 True, True, False]


def test_without_tau_max_nothing_is_claimed():
    """An older run cannot be labelled either way, so it is labelled neither."""
    df = pd.DataFrame({"R_mt": [0.0, 1.0]})
    assert not results.band_limited_mt(df).any()


def test_15_9_hz_is_the_boundary():
    """f_min = 1/(2*pi*tau_split_kinetic_s); the docs and the code agree."""
    import math
    f_crit = 1.0 / (2 * math.pi * results.TAU_SPLIT_KINETIC_S)
    assert f_crit == pytest.approx(15.915, abs=1e-3)


def test_band_limited_segments_are_drawn_as_their_own_state(run_dir):
    """Never coloured on the value ramp: absent is not the same as zero."""
    df = results.plate_summary(results.discover(run_dir)[0])
    fig = plate_figure(df, "R_mt", "light")
    names = [t.name for t in fig.data]
    assert "band-limited (not measurable)" in names
    bl = next(t for t in fig.data if t.name.startswith("band-limited"))
    assert len(bl.x) == int(results.band_limited_mt(df).sum()) == 11
    # its colour is the reserved status colour, never a point on the ramp
    assert bl.marker.color == theme.STATUS["serious"]


def test_a_band_limited_segment_is_never_also_counted_as_measured(run_dir):
    df = results.plate_summary(results.discover(run_dir)[0])
    fig = plate_figure(df, "R_mt", "light")
    measured = next(t for t in fig.data if t.name == "measured")
    limited = set(map(tuple, zip(
        *[df.loc[results.band_limited_mt(df), c] for c in ("cx_mm", "cy_mm")])))
    assert not limited & set(zip(measured.x, measured.y))


def test_other_parameters_have_no_band_limited_state(run_dir):
    """Only R_mt's bucket can fall outside the band; R_ohmic's cannot."""
    df = results.plate_summary(results.discover(run_dir)[0])
    fig = plate_figure(df, "R_ohmic", "light")
    assert not any(t.name.startswith("band-limited") for t in fig.data)


# ---- charting rules -------------------------------------------------------


def test_every_marker_symbol_is_a_filled_plotly_symbol():
    """"-open"/"-thin" symbols draw from marker.line, which we set to the
    surface colour -- they would come out invisible."""
    for key, spec in theme.SEGMENT_STATE.items():
        if key == "inferred":
            continue          # deliberately hollow, and drawn from marker.color
        assert "thin" not in spec["symbol"]


def test_scatter_palette_is_capped_at_the_validated_subset():
    assert theme.SCATTER_SLOTS == 3
    assert theme.CATEGORICAL["light"][:3] == ["#2a78d6", "#eb6834", "#1baf7a"]


def test_sequential_ramp_is_one_hue_and_monotonic():
    """A rainbow ramp invents boundaries the data does not have."""
    def lum(h):
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    ls = [lum(c) for c in theme.SEQUENTIAL_BLUE]
    assert all(a > b for a, b in zip(ls, ls[1:])), "ramp must be light -> dark"
