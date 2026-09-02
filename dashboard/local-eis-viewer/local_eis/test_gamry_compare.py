"""The whole-cell cross-check: does it recover an error we planted?

A comparison that always agrees is worthless, so every test here injects a
known defect and asks whether the metric that should see it does, and whether
the metrics that should not stay quiet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import gamry_compare as GC
import utils


A_CELL = 304.92


# ---------------------------------------------------------------------------
# a synthetic cell: R_ohm in series with one depressed arc
# ---------------------------------------------------------------------------

def cell_asr(freq, r_ohm=60e-3, r_ct=300e-3, tau=2e-3, n=0.85, L=2.6e-7):
    """R_ohm, one depressed arc, and the lead inductance.

    The inductance is not decoration: without it Z'' is capacitive at every
    frequency and the curve never crosses the real axis, so there is no HFR to
    find. Real cells cross -- the delivered Gamry file is inductive at 30 kHz
    -- and the crossing is what HFR means.
    """
    w = 2 * np.pi * np.asarray(freq, float)
    return 1j * w * L + r_ohm + r_ct / (1.0 + (1j * w * tau) ** n)


def write_dta(path, freq, Z_ohm, current, start="16.07.2026 07:53:44"):
    """A Gamry ZCURVE table, German decimal commas and all."""
    head = ["EXPLAIN", "TAG\tGALVEIS",
            "IDCREQ\tQUANT\t0\tDC Current (A)",
            "IACREQ\tQUANT\t7\tAC Current (A rms)",
            f"STARTTIME\tLABEL\t{start}\tStart Time",
            "ZCURVE\tTABLE",
            "\tPt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\tIdc\tVdc\tIERange",
            "\t#\ts\tHz\tohm\tohm\tV\tohm\tdeg\tA\tV\t#"]
    def g(v):                       # Gamry writes 2,105402E-04
        return f"{v:.6E}".replace(".", ",")
    for i, (f, z) in enumerate(zip(freq, Z_ohm), start=1):
        head.append("\t".join([
            "", str(i), "1", g(f), g(z.real), g(z.imag), "1",
            g(abs(z)), g(np.degrees(np.angle(z))), g(4e-3), g(0.77), "13"]))
    path.write_text("\n".join(head) + "\n", encoding="latin-1")
    return path


@pytest.fixture()
def sweep(tmp_path):
    freq = np.logspace(np.log10(0.3), np.log10(30000), 10 * 5 + 1)
    write_dta(tmp_path / f"V26_088_HFR_102_CurrVal_60.dta",
              freq, cell_asr(freq) / A_CELL, 60)
    return GC.read_cell_sweep(tmp_path / "V26_088_HFR_102_CurrVal_60.dta")


# ---------------------------------------------------------------------------


def test_a_gamry_file_is_read_with_its_condition_and_its_clock(sweep):
    assert sweep.condition == "60A"
    assert sweep.current_a == 60.0
    assert sweep.started is not None and sweep.started.hour == 7
    assert sweep.freq.size == 51
    # ohms in the file, ohm.cm2 once multiplied by the area
    assert np.max(sweep.asr(A_CELL).real) == pytest.approx(360e-3, rel=0.05)


def test_a_perfect_local_result_shows_no_difference(sweep):
    """The floor: identical physics in must give zero out."""
    f = np.logspace(np.log10(0.3), np.log10(4500), 60)
    c = GC.compare(f, cell_asr(f), sweep, A_CELL)
    # 60 asked for, minus any endpoint that rounds just outside the reference
    # band -- dropping those is the conservative behaviour, not a fault.
    assert 58 <= c.n_points <= 60
    assert abs(c.mag_rel_median) < 2e-3
    assert np.max(np.abs(c.phase_diff_deg)) < 0.2
    assert c.rms_rel < 5e-3


def test_a_scale_error_shows_up_as_a_flat_magnitude_offset(sweep):
    """An area error or a common shunt error scales every point equally."""
    f = np.logspace(np.log10(0.3), np.log10(4500), 60)
    c = GC.compare(f, 1.05 * cell_asr(f), sweep, A_CELL)
    assert c.mag_rel_median == pytest.approx(0.05, abs=2e-3)
    # and it is NOT mistaken for a phase problem
    assert np.max(np.abs(c.phase_diff_deg)) < 0.2


def test_an_uncorrected_chain_response_shows_up_in_the_phase(sweep):
    """A one-pole lag leaves the magnitude nearly alone and bends the phase.

    This is the failure mode the chain response causes, and it has to be
    distinguishable from a scale error or the correction cannot be aimed.
    """
    f = np.logspace(np.log10(0.3), np.log10(4500), 60)
    lag = np.exp(-1j * 2 * np.pi * f * 7e-6)          # 7 us of pure delay
    c = GC.compare(f, cell_asr(f) * lag, sweep, A_CELL, chain_applied=False)
    assert abs(c.mag_rel_median) < 2e-3               # magnitude untouched
    top = c.phase_diff_deg[np.argmax(c.freq)]
    assert top == pytest.approx(-np.degrees(2 * np.pi * 4500 * 7e-6), abs=1.0)
    assert any("chain response NOT applied" in n for n in c.notes)


def test_only_the_overlapping_band_is_compared(sweep):
    """Nothing is extrapolated past the band an instrument actually covered."""
    f = np.logspace(np.log10(0.15), np.log10(4500), 60)   # wider at the bottom
    c = GC.compare(f, cell_asr(f), sweep, A_CELL)
    assert c.freq.min() >= sweep.freq.min() - 1e-9
    assert c.freq.max() <= sweep.freq.max() + 1e-9
    assert c.n_points < 60


def test_hfr_is_refused_when_it_lies_above_the_evaluated_band(sweep):
    """The intercept is not always inside the pipeline's band.

    Reporting the low-frequency crossing instead would substitute the
    polarisation resistance for the ohmic one -- a factor of several, silently.
    """
    # The reference reaches 30 kHz and crosses; the local side stops at 200 Hz.
    f = np.logspace(np.log10(0.3), np.log10(200), 40)
    c = GC.compare(f, cell_asr(f), sweep, A_CELL)
    assert not np.isfinite(c.hfr_local)
    assert np.isfinite(c.hfr_ref)
    assert any("above the evaluated band" in n for n in c.notes)

    # Given the full band it is found, and both sides agree on it.
    #
    # The intercept is NOT R_ohm: at the crossing the arc still contributes
    # real part, so the true value is R_ohm + Re(arc(f_cross)) -- a few per
    # cent above. What is asserted is that the estimator finds the model's own
    # intercept, computed here on a grid fine enough to be the answer.
    fine = np.logspace(np.log10(0.3), np.log10(30000), 20001)
    truth = GC._hf_intercept(fine, cell_asr(fine))
    assert truth > 60e-3                       # and it really is above R_ohm

    f = np.logspace(np.log10(0.3), np.log10(30000), 60)
    c = GC.compare(f, cell_asr(f), sweep, A_CELL)
    assert c.hfr_local == pytest.approx(truth, rel=5e-3)
    assert c.hfr_rel == pytest.approx(0.0, abs=0.01)


def test_conditions_are_ordered_by_current_not_alphabetically(tmp_path):
    freq = np.logspace(np.log10(0.3), np.log10(30000), 40)
    for i in (45, 60, 150, 450):
        write_dta(tmp_path / f"V26_088_HFR_1_CurrVal_{i}.dta",
                  freq, cell_asr(freq) / A_CELL, i)
        d = tmp_path / "res" / f"{i}A" / "silver"
        d.mkdir(parents=True)
        utils.write_table(d / "cell_aggregate.csv", [{
            "freq_hz": round(float(f), 6),
            "z_re_mohm_cm2": round(1e3 * z.real, 5),
            "z_im_mohm_cm2": round(1e3 * z.imag, 5),
        } for f, z in zip(freq, cell_asr(freq))])

    class _Quiet:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass

    out = GC.run(tmp_path / "res", tmp_path, A_CELL,
                 out_dir=tmp_path / "res", log=_Quiet())
    assert [c.condition for c in out] == ["45A", "60A", "150A", "450A"]
    assert (tmp_path / "res" / "gamry_comparison.csv").is_file()


def test_a_condition_with_only_one_half_is_skipped_not_guessed(tmp_path):
    """A comparison against a missing half is not weak, it does not exist."""
    freq = np.logspace(np.log10(0.3), np.log10(30000), 40)
    write_dta(tmp_path / "x_CurrVal_60.dta", freq, cell_asr(freq) / A_CELL, 60)
    d = tmp_path / "res" / "450A" / "silver"      # a local result with no sweep
    d.mkdir(parents=True)
    utils.write_table(d / "cell_aggregate.csv", [{
        "freq_hz": round(float(f), 6),
        "z_re_mohm_cm2": round(1e3 * z.real, 5),
        "z_im_mohm_cm2": round(1e3 * z.imag, 5),
    } for f, z in zip(freq, cell_asr(freq))])

    said: list[str] = []
    noted: list[str] = []

    class _Log:
        def info(self, m, *a, **k): noted.append(str(m))
        def warning(self, m, *a, **k): said.append(str(m))

    out = GC.run(tmp_path / "res", tmp_path, A_CELL,
                 out_dir=tmp_path / "res", log=_Log())
    assert out == []

    # A local result with no sweep is a gap: something was evaluated and
    # cannot be checked.  That warns.
    assert any("450A" in m and "no whole-cell sweep" in m for m in said)

    # A sweep with no local result is the ordinary case -- one run evaluates
    # one condition while the campaign folder holds every sweep -- so it is
    # still reported, but not as an alarm.
    assert any("60A" in m for m in noted)
    assert not any("60A" in m for m in said)


def test_per_segment_chain_files_are_not_read_as_cell_references(tmp_path):
    """`bode/..._#12.DTA` is a shunt amplifier, not a cell."""
    freq = np.logspace(np.log10(1), np.log10(1e5), 30)
    bode = tmp_path / "bode"; bode.mkdir()
    write_dta(bode / "plate_100kHz_1Hz_500mA_#12.DTA", freq,
              np.ones_like(freq) * (1 + 0j), 0)
    write_dta(tmp_path / "cell_CurrVal_60.dta", freq,
              cell_asr(freq) / A_CELL, 60)
    found = GC.find_cell_sweeps(tmp_path)
    assert [s.condition for s in found] == ["60A"]


# ---------------------------------------------------------------------------
# the campaign's own case: NEITHER sweep reaches the intercept
# ---------------------------------------------------------------------------

def test_a_truncated_reference_is_reported_as_missing_on_both_sides(tmp_path):
    """V26_086 stops at 3.0 kHz, below where this cell crosses the real axis.

    The old note blamed the local side alone, which sent the reader off to
    raise f_max -- a change that cannot help, because the reference has no
    intercept either.
    """
    freq = np.logspace(np.log10(0.324), np.log10(2987), 27)
    write_dta(tmp_path / "V26_086_HFR_101_CurrVal_45.dta",
              freq, cell_asr(freq) / A_CELL, 45)
    sweep = GC.read_cell_sweep(tmp_path / "V26_086_HFR_101_CurrVal_45.dta")

    c = GC.compare(freq, cell_asr(freq), sweep, A_CELL)
    assert not np.isfinite(c.hfr_local) and not np.isfinite(c.hfr_ref)
    assert any("EITHER side" in n for n in c.notes)
    assert not any("raise cfg.f_max_hz" in n for n in c.notes)


def test_the_extrapolation_lands_near_the_intercept_the_sweep_never_reached():
    fine = np.logspace(np.log10(0.3), np.log10(3e5), 4001)
    truth = GC._hf_intercept(fine, cell_asr(fine))
    assert np.isfinite(truth)

    for top, tol in ((2987.0, 0.05), (4500.0, 0.03)):
        f = np.logspace(np.log10(0.324), np.log10(top), 27)
        assert not np.isfinite(GC._hf_intercept(f, cell_asr(f)))
        got = GC._hf_extrapolated(f, cell_asr(f))
        assert got == pytest.approx(truth, rel=tol), f"at {top} Hz"
        assert got < truth, "the unclosed arc biases the extrapolation LOW"


def test_the_extrapolation_is_not_reported_when_the_intercept_was_measured(sweep):
    """`sweep` runs to 30 kHz, where the cell is inductive: HFR is real."""
    f = np.logspace(np.log10(0.5), np.log10(20000), 40)
    c = GC.compare(f, cell_asr(f), sweep, A_CELL)
    assert np.isfinite(c.hfr_local) and np.isfinite(c.hfr_ref)
    assert not np.isfinite(c.hfr_local_fit), "no fallback when the truth is there"


def test_a_common_scale_error_survives_the_extrapolation(tmp_path):
    """The point of the fallback: the DIFFERENCE is still recoverable.

    Both sides carry the same bias from the unclosed arc, so a 5 % error
    planted on the local side must still read as 5 %, not as the bias.
    """
    freq = np.logspace(np.log10(0.324), np.log10(2987), 27)
    write_dta(tmp_path / "V26_086_HFR_101_CurrVal_45.dta",
              freq, cell_asr(freq) / A_CELL, 45)
    sweep = GC.read_cell_sweep(tmp_path / "V26_086_HFR_101_CurrVal_45.dta")

    c = GC.compare(freq, 1.05 * cell_asr(freq), sweep, A_CELL)
    assert c.hfr_fit_rel == pytest.approx(0.05, abs=0.005)


# ---------------------------------------------------------------------------
# "I installed it and it still says not installed"
# ---------------------------------------------------------------------------

def test_the_asammdf_advice_names_the_interpreter_on_its_own_line():
    """The old message buried the path mid-sentence, where the console cut it.

    The user's terminal showed `... To have it: "C:\\Use` and stopped, so the
    one piece of information that resolves the problem -- WHICH python -- was
    the piece that got truncated.
    """
    import sys
    lines = GC._asammdf_advice(ImportError("No module named 'asammdf'"))
    assert any(line.strip().startswith("interpreter:") and sys.executable
               in line for line in lines)
    assert any(sys.executable in line and "--no-deps asammdf" in line
               for line in lines)


def test_a_broken_asammdf_is_not_reported_as_a_missing_one():
    """asammdf built against another numpy raises ImportError too.

    Telling someone to install what they already installed sends them round
    the same loop, so the two cases have to read differently.
    """
    exc = ImportError("numpy.core.multiarray failed to import")
    exc.name = "numpy.core.multiarray"
    lines = GC._asammdf_advice(exc)
    assert "not installed" not in " ".join(lines)
    assert any("cannot import" in line for line in lines)
    assert any("force-reinstall" in line for line in lines)


# ---------------------------------------------------------------------------
# the "zstd" package has no Windows wheel past Python 3.10
# ---------------------------------------------------------------------------

def test_the_vendor_shim_is_reached_for_only_when_the_real_package_is_absent(
        monkeypatch):
    """`_ensure_zstd_importable` must never shadow a real installation.

    Linux and macOS still get "zstd" wheels; only Windows on Python 3.11+
    is stuck, so the vendored stand-in has to be a fallback, not a
    replacement.
    """
    import importlib.util
    import sys

    vendor = str(Path(GC.__file__).resolve().parent / "_vendor")
    sys.path[:] = [p for p in sys.path if p != vendor]

    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: object() if name == "zstd" else None)
    GC._ensure_zstd_importable()
    assert vendor not in sys.path, "the real package must win when present"


def test_the_vendor_shim_is_added_when_zstd_is_missing(monkeypatch):
    import importlib.util
    import sys

    vendor = str(Path(GC.__file__).resolve().parent / "_vendor")
    sys.path[:] = [p for p in sys.path if p != vendor]

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    GC._ensure_zstd_importable()
    assert vendor in sys.path
    sys.path.remove(vendor)


def test_the_shim_re_exports_exactly_what_asammdf_imports_from_zstd():
    """asammdf imports BOTH names at module load -- `compress` from
    v4_blocks.py (its MDF4 writer, never exercised by this read-only
    pipeline) and `decompress` from both utils.py and v4_blocks.py -- and
    the import runs regardless of which functions ever get called. Shipping
    only `decompress` looks right in isolation and then fails the moment
    `import asammdf` actually runs, which is exactly the shape of bug a
    quick look at one file misses and an end-to-end import does not.
    """
    pytest.importorskip("zstandard")
    import importlib.util

    vendor = Path(GC.__file__).resolve().parent / "_vendor" / "zstd.py"
    spec = importlib.util.spec_from_file_location("_shim_under_test", vendor)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    public = {n for n in vars(mod) if not n.startswith("_")}
    assert public == {"compress", "decompress", "annotations"}
    import zstandard
    assert mod.decompress is zstandard.decompress
    assert mod.compress is zstandard.compress


def test_a_missing_zstandard_is_diagnosed_by_name_not_blamed_on_numpy():
    """If even the shim's own dependency is absent, the advice must say
    that -- not repeat the generic "probably numpy" guess, which would send
    the reader reinstalling the wrong package.
    """
    exc = ImportError("No module named 'zstandard'")
    exc.name = "zstandard"
    lines = GC._asammdf_advice(exc)
    assert any("zstandard" in line and "pip install zstandard" in line
              for line in lines)
    assert "numpy" not in " ".join(lines)


def test_the_not_installed_advice_never_says_plain_pip_install_asammdf():
    """That is the exact command that fails to build "zstd" on Windows;
    telling the reader to run it again would send them in a circle.
    """
    lines = GC._asammdf_advice(ImportError("No module named 'asammdf'"))
    fix_lines = [ln for ln in lines if "pip install" in ln]
    assert any("--no-deps asammdf" in ln for ln in fix_lines)
    assert not any(ln.strip().rstrip('"').endswith("pip install asammdf")
                  for ln in fix_lines), ("must not tell the reader to re-run "
                  "the exact command that failed")
    assert any("zstandard" in ln for ln in fix_lines)


# ---------------------------------------------------------------------------
# reading the setpoint a sweep was taken at
# ---------------------------------------------------------------------------

def test_the_bench_naming_convention_is_read():
    assert GC._setpoint("RO2612025_CurrVal_150_EIS.dta", {}) == 150.0
    assert GC._setpoint("CurrVal-60.DTA", {}) == 60.0


def test_a_hand_saved_sweep_is_read_from_its_name():
    """A sweep saved by hand is called what the operator called it."""
    assert GC._setpoint("45A.DTA", {}) == 45.0
    assert GC._setpoint("RO2612025_150A.DTA", {}) == 150.0
    assert GC._setpoint("450A.DTA", {}) == 450.0


def test_the_title_label_is_read_when_the_name_says_nothing():
    """Gamry stores the Test Identifier box in the header as TITLE.

    On the Reference 3000 sets taken for this cell that is the ONLY place
    the setpoint appears -- the files are called 45A.DTA and 60A.DTA, but a
    campaign that renames them by date would lose it entirely otherwise.
    """
    assert GC._setpoint("sweep_01.DTA", {"TITLE": "450A"}) == 450.0
    assert GC._setpoint("sweep_01.DTA", {"TITLE": "60A"}) == 60.0


def test_idcreq_is_not_used_even_though_it_looks_right():
    """It reads 0 on every galvanostatic sweep driven through a booster.

    The DC comes from the load bank, so the potentiostat truthfully records
    that IT requested none. Reading it would label every one of these
    sweeps 0 A -- and they would all then match each other.
    """
    meta = {"TITLE": "450A", "IDCREQ": "0,00000E+000"}
    assert GC._setpoint("sweep.DTA", meta) == 450.0


def test_a_sweep_with_no_setpoint_anywhere_reports_none():
    """Guessing here would attach a reference curve to the wrong run."""
    assert GC._setpoint("sweep_01.DTA", {}) is None
    assert GC._setpoint("bode_scan.DTA", {"TITLE": "warm-up"}) is None


def test_a_number_that_is_not_a_setpoint_is_not_read_as_one():
    assert GC._setpoint("45Amps_run2.DTA", {}) is None
    assert GC._setpoint("1A5_cell.DTA", {}) is None


def test_the_real_reference_3000_files_are_read(tmp_path):
    """End to end on the four sweeps taken for this cell."""
    pytest.importorskip("numpy")
    header = (
        "EXPLAIN\nTAG\tPWR800_GALVEIS\n"
        "TITLE\tLABEL\t{title}\tTest &Identifier\n"
        "DATE\tLABEL\t27.8.2026\tDate\n"
        "TIME\tLABEL\t11:25:35\tTime\n"
        "IDCREQ\tQUANT\t0,00000E+000\tDC &Current (A)\n"
        "ZCURVE\tTABLE\n"
        "\tPt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\tIdc\tVdc\tIERange\n"
        "\t#\ts\tHz\tohm\tohm\tV\tohm\t\xb0\tA\tV\t#\n"
        "\t0\t3\t30046,88\t0,0001715\t0,0001388\t1\t0,0002206\t38,9\t0,004\t0,82\t13\n"
        "\t1\t5\t1000,00\t0,0002715\t-0,0001388\t1\t0,0003\t-27,1\t0,004\t0,82\t13\n"
        "\t2\t7\t10,00\t0,0012913\t-0,0000477\t1\t0,0012\t-2,1\t0,004\t0,82\t13\n"
    )
    for title in ("45A", "60A", "150A", "450A"):
        (tmp_path / f"{title}.DTA").write_text(
            header.format(title=title), encoding="latin-1")

    sweeps = GC.find_cell_sweeps(tmp_path)
    assert sorted(s.condition for s in sweeps) == ["150A", "450A", "45A", "60A"]
    assert {s.current_a for s in sweeps} == {45.0, 60.0, 150.0, 450.0}
