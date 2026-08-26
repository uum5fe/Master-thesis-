"""The whole-cell cross-check: does it recover an error we planted?

A comparison that always agrees is worthless, so every test here injects a
known defect and asks whether the metric that should see it does, and whether
the metrics that should not stay quiet.
"""

from __future__ import annotations

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

    class _Log:
        def info(self, m, *a, **k): pass
        def warning(self, m, *a, **k): said.append(str(m))

    out = GC.run(tmp_path / "res", tmp_path, A_CELL,
                 out_dir=tmp_path / "res", log=_Log())
    assert out == []
    assert any("450A" in m and "no whole-cell sweep" in m for m in said)
    assert any("60A" in m and "no local result" in m for m in said)


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
