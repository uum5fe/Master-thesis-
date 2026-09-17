"""Which run do the display cells show?

The heat map cell once listed every condition directory on the Volume and drew
a map for each, ignoring the condition widget entirely. With one figure per
condition the last one drawn is the one on screen, and 60A sorts last among
150A, 450A, 45A, 60A -- so selecting 450 A produced a 60 A map whose title
honestly said 60A.

Three cells each carried their own copy of the cache path shape, so when the
run cell's layout changed they kept reading the old one. These tests pin the
single resolver they now share. The notebook cannot be imported (dbutils,
Volume paths), so the functions are lifted out of it by name and executed --
which also means a rename that breaks the display cells breaks these tests.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import pytest

from config import Config, DEFAULT


RUNNER = Path(__file__).with_name("Local EIS Pipeline Runner.py")
FUNCS = ("_run_identity", "_cache_dir", "selected_conditions", "result_dir",
         "_legacy_snr_names", "describe_source")


def _namespace(tmp_path, condition="450A", mode="default", snr=0.0):
    """The runner's resolver functions, with the notebook globals they read."""
    src = RUNNER.read_text()
    ns = dict(json=json, os=os, re=re, shutil=shutil, time=time, Path=Path,
              Config=Config, DEFAULT=DEFAULT)
    ns["_CACHE_VOL"] = tmp_path / "EIS_Results"
    ns["_TMP_BASE"] = tmp_path / "tmp"
    ns["CONDITIONS"] = ["150A", "450A", "45A", "60A"]
    ns["EVALUATION_MODE"] = mode
    ns["F_MIN"], ns["F_MAX"], ns["MIN_SNR_DB"] = 0.15, 4500.0, snr
    ns["_w"] = lambda n, d="": {"condition": condition}.get(n, d)

    keys = re.search(r"_CACHE_IDENTITY_KEYS = \((.*?)\n\)", src, re.S)
    assert keys, "the cache identity key list moved or was renamed"
    ns["_CACHE_IDENTITY_KEYS"] = eval(keys.group(0).split("=", 1)[1], ns)

    for name in FUNCS:
        m = re.search(rf"^def {name}\b.*?(?=^\S|\Z)", src, re.S | re.M)
        assert m, f"{name} is gone from the runner"
        exec(m.group(0), ns)
    return ns


def _plant(ns, leepa, cond, rel="", stage="gold"):
    d = ns["_CACHE_VOL"] / leepa / cond / rel if rel else \
        ns["_CACHE_VOL"] / leepa / cond
    (d / stage).mkdir(parents=True, exist_ok=True)
    (d / stage / "plate_summary.csv").write_text("segment,class\n1,measured\n")
    return d


# ---------------------------------------------------------------------------
# the condition widget decides, not the filesystem
# ---------------------------------------------------------------------------


def test_only_the_selected_condition_is_returned(tmp_path) -> None:
    ns = _namespace(tmp_path, condition="450A")
    assert ns["selected_conditions"]() == ["450A"]


def test_all_means_all(tmp_path) -> None:
    ns = _namespace(tmp_path, condition="ALL")
    assert ns["selected_conditions"]() == ["150A", "450A", "45A", "60A"]


def test_a_condition_with_no_results_is_not_substituted(tmp_path) -> None:
    """The failure this whole file exists for.

    Every OTHER condition has results. The selected one has none. The answer
    must be "nothing", never a neighbour's map.
    """
    ns = _namespace(tmp_path, condition="450A")
    for cond in ("150A", "45A", "60A"):
        _plant(ns, "2612030", cond, "mode_default/snr_0.0_" + ns["_run_identity"]())

    d, prov = ns["result_dir"]("2612030", "450A")
    assert d is None
    assert prov == "not found"
    assert "NO RESULTS" in ns["describe_source"]("450A", d, prov)


# ---------------------------------------------------------------------------
# which run, of the several that may be on disk
# ---------------------------------------------------------------------------


def test_this_sessions_run_outranks_the_cache(tmp_path) -> None:
    """A fresh run may not have reached the Volume at all if caching is off."""
    ns = _namespace(tmp_path)
    cached = _plant(ns, "2612030", "450A",
                    "mode_default/snr_0.0_" + ns["_run_identity"]())
    live = ns["_TMP_BASE"] / "2612030" / "450A"
    (live / "gold").mkdir(parents=True)
    (live / "gold" / "plate_summary.csv").write_text("segment,class\n1,measured\n")
    ns["PIPELINE_RESULTS"] = {"450A": {"out_dir": live}}

    d, prov = ns["result_dir"]("2612030", "450A")
    assert d == live and "session" in prov
    assert d != cached


def test_a_condition_loaded_from_cache_is_not_passed_off_as_a_fresh_run(tmp_path) -> None:
    """PIPELINE_RESULTS holds cache hits too, flagged with cached=True."""
    ns = _namespace(tmp_path)
    cached = _plant(ns, "2612030", "450A",
                    "mode_default/snr_0.0_" + ns["_run_identity"]())
    ns["PIPELINE_RESULTS"] = {"450A": {"out_dir": cached, "cached": True}}

    d, prov = ns["result_dir"]("2612030", "450A")
    assert prov.startswith("cache:")


def test_a_skipped_condition_does_not_crash_the_lookup(tmp_path) -> None:
    ns = _namespace(tmp_path)
    ns["PIPELINE_RESULTS"] = {"450A": None}
    assert ns["result_dir"]("2612030", "450A")[0] is None


# ---------------------------------------------------------------------------
# the cache key names the settings
# ---------------------------------------------------------------------------


def test_default_and_permissive_are_different_entries(tmp_path) -> None:
    """The two used to overwrite each other; whichever ran last was served."""
    ns = _namespace(tmp_path)
    a = ns["_cache_dir"]("2612030", "450A", 0.0, "default")
    b = ns["_cache_dir"]("2612030", "450A", 0.0, "permissive")
    assert a != b
    assert "mode_default" in str(a) and "mode_permissive" in str(b)


def test_the_band_changes_the_cache_entry(tmp_path) -> None:
    """A cache keyed on SNR alone served results from another band."""
    a = _namespace(tmp_path)["_run_identity"]()
    ns_b = _namespace(tmp_path)
    ns_b["F_MAX"] = 2000.0
    assert ns_b["_run_identity"]() != a


def test_the_same_settings_give_the_same_entry(tmp_path) -> None:
    """A key that moved on every run would never hit."""
    assert (_namespace(tmp_path)["_run_identity"]()
            == _namespace(tmp_path)["_run_identity"]())


# ---------------------------------------------------------------------------
# older caches are readable, but never silently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", ["snr_0.0", "snr_0", ""])
def test_an_older_layout_is_found_and_labelled_legacy(tmp_path, rel) -> None:
    ns = _namespace(tmp_path)
    _plant(ns, "2612030", "450A", rel)
    d, prov = ns["result_dir"]("2612030", "450A")
    assert d is not None
    assert prov.startswith("LEGACY")
    assert "NOT the current settings" in ns["describe_source"]("450A", d, prov)


def test_the_current_entry_wins_over_an_older_one(tmp_path) -> None:
    ns = _namespace(tmp_path)
    _plant(ns, "2612030", "450A", "snr_0.0")
    cur = _plant(ns, "2612030", "450A",
                 "mode_default/snr_0.0_" + ns["_run_identity"]())
    d, prov = ns["result_dir"]("2612030", "450A")
    assert d == cur and prov.startswith("cache:")


def test_a_directory_with_no_stage_folders_is_not_a_result(tmp_path) -> None:
    """An empty directory left by a failed run is not a cache hit."""
    ns = _namespace(tmp_path)
    (ns["_CACHE_VOL"] / "2612030" / "450A" / "mode_default").mkdir(parents=True)
    assert ns["result_dir"]("2612030", "450A")[0] is None
