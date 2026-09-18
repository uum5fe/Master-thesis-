"""Which cell's reference sweep is this, and how would you know?

A whole-cell sweep is named V26_092_HFR_101_CurrVal_45.dta. It carries the
BUILD TOKEN and no order number, so the order filter that protects the bench
logs cannot see it at all. The measurement file for the same cell,
..._RO2612030-01_V26_092_lokale_EIS_6_Boxen_3.mf4, carries both -- which is
the only thing that makes order -> build -> sweep resolvable.

This matters because the failure is silent. Sweeps are keyed by CURRENT, and
every campaign has a 45 A: reading a shared folder whole means the last file
read replaces the earlier one, and the comparison then runs one cell's local
aggregate against another cell's reference with nothing out of place to see.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import gamry_compare as GC
from test_gamry_compare import cell_asr, write_dta


RUNNER = Path(__file__).with_name("Local EIS Pipeline Runner.py")


# ---------------------------------------------------------------------------
# reading the build token out of a name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, want", [
    ("V26_092_HFR_101_CurrVal_45.dta", "V26_092"),
    ("V26_088_HFR_101_CurrVal_45.dta", "V26_088"),
    ("v26_086_hfr_101_currval_45.dta", "V26_086"),
    ("V26-092_HFR_1.dta", "V26_092"),
    ("20260907_1021_1273_70758_RO2612030-01_V26_092_lokale_EIS_6_Boxen_3.mf4",
     "V26_092"),
    ("20260716_0742_1269_70758_RO2611976-01_V26_088_lokale_EIS_6_Boxen.mf4",
     "V26_088"),
    ("Leepa_RO2612030_Current_45A_Test_1_Karte_1.DAT", None),
    ("HFR_101_CurrVal_450.dta", None),
])
def test_the_build_token_is_read_from_the_name(name, want) -> None:
    assert GC.version_of(name) == want


def test_a_word_boundary_would_not_have_worked() -> None:
    r"""Worth pinning: "_" is a word character.

    \bV\d\d_\d\d\d\b finds nothing in "_V26_092_HFR", because there is no
    boundary between "_" and "V" nor between "092" and "_" -- the token is
    surrounded by underscores in every real filename.
    """
    assert GC.version_of("RO2612030-01_V26_092_lokale") == "V26_092"
    assert re.search(r"\bV\d{2}_\d{3}\b", "RO2612030-01_V26_092_lokale") is None


@pytest.mark.parametrize("name, version, want", [
    ("V26_092_HFR_101_CurrVal_45.dta", "V26_092", True),
    ("V26_092_HFR_101_CurrVal_45.dta", "V26_088", False),
    ("HFR_101_CurrVal_45.dta", "V26_092", None),     # silent, may be ours
    ("V26_092_HFR_101_CurrVal_45.dta", None, None),  # nothing to compare to
    ("V26_092_HFR_101_CurrVal_45.dta", "v26_092", True),
])
def test_naming_another_build_is_a_refusal_not_a_near_miss(name, version, want):
    assert GC.names_version(name, version) is want


# ---------------------------------------------------------------------------
# picking sweeps out of a shared folder
# ---------------------------------------------------------------------------


def _campaign(folder, build, currents=(45, 60, 150, 450)):
    freq = np.logspace(np.log10(0.3), np.log10(30000), 40)
    for i, cur in enumerate(currents, start=1):
        write_dta(folder / f"{build}_HFR_10{i}_CurrVal_{cur}.dta",
                  freq, cell_asr(freq), cur)


def test_two_campaigns_in_one_folder_stay_apart(tmp_path) -> None:
    """The defect in one line: both campaigns have a 45 A."""
    _campaign(tmp_path, "V26_092")
    _campaign(tmp_path, "V26_088")

    ours = GC.find_cell_sweeps(tmp_path, version="V26_092")
    assert len(ours) == 4
    assert all("V26_092" in s.path.name for s in ours)

    theirs = GC.find_cell_sweeps(tmp_path, version="V26_088")
    assert all("V26_088" in s.path.name for s in theirs)


def test_without_the_build_the_currents_collide(tmp_path) -> None:
    """What the old code did, stated as a test so it cannot come back quietly.

    Eight files, four currents. Keyed by condition, four of them vanish --
    and which four depends on sort order, not on which cell was measured.
    """
    _campaign(tmp_path, "V26_092")
    _campaign(tmp_path, "V26_088")

    everything = GC.find_cell_sweeps(tmp_path)
    assert len(everything) == 8
    keyed = {s.condition: s for s in everything}
    assert len(keyed) == 4                      # half were silently dropped
    assert len({GC.version_of(s.path.name) for s in keyed.values()}) >= 1


def test_a_folder_with_one_campaign_needs_no_version(tmp_path) -> None:
    """The common case must not require configuration."""
    _campaign(tmp_path, "V26_092")
    assert len(GC.find_cell_sweeps(tmp_path)) == 4
    assert len(GC.find_cell_sweeps(tmp_path, version="V26_092")) == 4


def test_a_sweep_that_names_no_build_is_a_last_resort(tmp_path) -> None:
    """Silent about the build, so it may be ours -- used only if nothing else."""
    freq = np.logspace(np.log10(0.3), np.log10(30000), 40)
    write_dta(tmp_path / "HFR_101_CurrVal_45.dta", freq, cell_asr(freq), 45)
    found = GC.find_cell_sweeps(tmp_path, version="V26_092")
    assert len(found) == 1

    _campaign(tmp_path, "V26_092", currents=(45,))
    found = GC.find_cell_sweeps(tmp_path, version="V26_092")
    assert [f.path.name for f in found] == ["V26_092_HFR_101_CurrVal_45.dta"]


def test_the_bench_log_is_picked_by_build_too(tmp_path) -> None:
    """Two campaigns' .mf4 files sit in the same folder as their sweeps."""
    a = tmp_path / "20260907_RO2612030-01_V26_092_lokale_EIS.mf4"
    b = tmp_path / "20260716_RO2611976-01_V26_088_lokale_EIS.mf4"
    a.touch(); b.touch()

    assert GC.find_bench_log(tmp_path, version="V26_092") == a
    assert GC.find_bench_log(tmp_path, version="V26_088") == b
    # the order number still wins when the name carries one
    assert GC.find_bench_log(tmp_path, order_id="2611976") == b


# ---------------------------------------------------------------------------
# the runner: order -> build -> files
# ---------------------------------------------------------------------------


def _runner_ns(tmp_path, ev_root):
    """The runner's Gamry resolver, lifted out by name."""
    src = RUNNER.read_text()
    ns = {"Path": Path, "gamry_compare": GC, "print": lambda *a, **k: None}
    ns["_EV_ROOT"] = ev_root
    ns["FAMOS_ROOT"] = ev_root / "Famos"
    ns["LEEPA"] = "2612030"
    for name in ("GAMRY_SEARCH_ROOT_TEMPLATES", "GAMRY_VERSION_SOURCES",
                 "PLATE_VERSION_OVERRIDE"):
        m = re.search(rf"^{name}(: [^=]+)? = (\[.*?\n\]|\{{\}})", src, re.S | re.M)
        assert m, f"{name} is gone from the runner"
        exec(m.group(0), ns)
    for name in ("gamry_search_roots", "plate_version", "gamry_root_for",
                 "gamry_files"):
        m = re.search(rf"^def {name}\b.*?(?=^\S|\Z)", src, re.S | re.M)
        assert m, f"{name} is gone from the runner"
        exec(m.group(0), ns)
    return ns


@pytest.fixture()
def share(tmp_path):
    """A shared drive: two campaigns' sweeps and both measurement files."""
    ev = tmp_path / "ev"
    (ev / "Gamry").mkdir(parents=True)
    (ev / "Famos").mkdir()
    _campaign(ev / "Gamry", "V26_092")
    _campaign(ev / "Gamry", "V26_088")
    (ev / "Gamry" / ("20260907_1021_1273_70758_RO2612030-01_V26_092"
                     "_lokale_EIS_6_Boxen_3.mf4")).touch()
    (ev / "Gamry" / ("20260716_0742_1269_70758_RO2611976-01_V26_088"
                     "_lokale_EIS_6_Boxen.mf4")).touch()
    return ev


def test_the_order_selects_its_own_build(share, tmp_path) -> None:
    ns = _runner_ns(tmp_path, share)
    assert ns["plate_version"]("2612030") == "V26_092"
    assert ns["plate_version"]("2611976") == "V26_088"


def test_the_order_selects_its_own_sweeps(share, tmp_path) -> None:
    """Change the dashboard's order, get a different set of sweeps."""
    ns = _runner_ns(tmp_path, share)
    for leepa, build in (("2612030", "V26_092"), ("2611976", "V26_088")):
        files, version = ns["gamry_files"](leepa)
        assert version == build
        assert len(files) == 4
        assert all(build in f.name for f in files)


def test_an_unresolvable_order_refuses_rather_than_mixing(share, tmp_path):
    """Handing back both campaigns is the silent failure, so it is refused."""
    ns = _runner_ns(tmp_path, share)
    files, version = ns["gamry_files"]("9999999")
    assert files == [] and version is None


def test_one_campaign_and_no_measurement_file_still_works(tmp_path) -> None:
    """The simple case stays simple: nothing to disambiguate, nothing refused."""
    ev = tmp_path / "ev"
    (ev / "Gamry").mkdir(parents=True)
    (ev / "Famos").mkdir()
    _campaign(ev / "Gamry", "V26_092")
    ns = _runner_ns(tmp_path, ev)
    files, version = ns["gamry_files"]("9999999")
    assert len(files) == 4 and version == "V26_092"


def test_the_override_is_the_last_word(share, tmp_path) -> None:
    ns = _runner_ns(tmp_path, share)
    ns["PLATE_VERSION_OVERRIDE"]["2612030"] = "V26_088"
    assert ns["plate_version"]("2612030") == "V26_088"
    files, _v = ns["gamry_files"]("2612030")
    assert all("V26_088" in f.name for f in files)


def test_a_per_order_folder_is_preferred_over_the_shared_one(share, tmp_path):
    """A folder named for the order is unambiguous, so it wins."""
    own = share / "RO2612030_Gamry"
    own.mkdir()
    _campaign(own, "V26_092", currents=(45,))
    ns = _runner_ns(tmp_path, share)
    assert ns["gamry_root_for"]("2612030") == own
    files, _v = ns["gamry_files"]("2612030")
    assert len(files) == 1
