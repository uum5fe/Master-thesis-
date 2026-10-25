"""A frequency in the schedule is not a frequency that was measured.

A BronzeRun carries one schedule for the whole plate, and a reader naturally
takes a step in it as evidence that the plate was measured at that frequency.
It is not: it says the plate was EXCITED there. How much of the plate came
back with a finite phasor is a separate fact, and an aggregate computed
without it can be a plate-wide number resting on four segments in one corner.

The same distinction applies to dwell windows: `repair_windows` replaces a
window that cannot belong to a monotonic sweep with the one the sweep would
have placed there. That is a prediction, and a point built on a predicted
window has to be marked as such.
"""

from __future__ import annotations

import numpy as np
import pytest

import bronze
import ladder_snap
import r2d2_geometry as geom
from bronze import BronzeRun, BronzeSpectrum, CardInfo
from eis_local import Step


FS = 10_000.0


def _spectrum(seg: str, valid: np.ndarray) -> BronzeSpectrum:
    n = len(valid)
    z = np.where(valid, 60e-3 + 1j * 5e-3, np.nan + 1j * np.nan)
    return BronzeSpectrum(
        segment=seg, card="Karte_1", freq=np.geomspace(1.0, 100.0, n),
        Z_raw=z, snr_ref_db=np.full(n, 20.0), snr_seg_db=np.full(n, 20.0),
        snr_comb_db=np.full(n, 20.0), thd=np.zeros(n), drift=np.zeros(n),
        n_per_step=np.where(valid, 2048, 0), on_grid=np.ones(n, bool),
        channel_slot=1, ref_slot=0, n_ch_on_card=16, fs=FS, K=1.0,
        K_imputed=False, T_degC=60.0, u_dc=0.77, ref_name="UC2")


def _run(valid_by_seg: dict[str, np.ndarray], n_steps: int) -> BronzeRun:
    freqs = np.geomspace(1.0, 100.0, n_steps)
    schedule = [Step(freq=float(f), start=i * 1000, stop=i * 1000 + 800,
                     amp=0.01, snr_db=20.0, thd=0.0, stationarity=0.0)
                for i, f in enumerate(freqs)]
    return BronzeRun(
        schedule=schedule, channels={},
        spectra={s: _spectrum(s, v) for s, v in valid_by_seg.items()},
        cards={}, grid={}, config_digest="x", input_digest="y", n_files=1)


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_full_coverage_reads_as_one() -> None:
    n = 5
    segs = {s: np.ones(n, bool) for s in list(geom.SEGMENTS)[:6]}
    rows = bronze.bronze_frequency_coverage(_run(segs, n))
    assert len(rows) == n
    assert all(r["segment_fraction"] == 1.0 for r in rows)
    assert all(r["area_coverage"] == 1.0 for r in rows)


def test_a_frequency_only_some_segments_kept_is_reported_as_partial() -> None:
    """The number an aggregate needs and the schedule cannot give."""
    n = 5
    names = list(geom.SEGMENTS)[:4]
    segs = {s: np.ones(n, bool) for s in names}
    for s in names[:3]:                      # 3 of 4 lose the top frequency
        segs[s][-1] = False
    rows = bronze.bronze_frequency_coverage(_run(segs, n))
    assert rows[-1]["n_valid_segments"] == 1
    assert rows[-1]["segment_fraction"] == pytest.approx(0.25)
    assert rows[0]["segment_fraction"] == 1.0


def test_area_coverage_is_weighted_not_counted() -> None:
    """The distinction that makes this worth computing.

    The plate's segments span 0.678 to 8.470 cm^2, a factor of 12.5, so the
    fraction of SEGMENTS that survived and the fraction of AREA they cover
    are different numbers. Keeping the smallest of two segments gives a
    segment fraction of 0.5 and an area coverage that is nothing like it.
    """
    n = 3
    by_area = sorted(geom.SEGMENTS, key=lambda s: geom.SEGMENTS[s].area_cm2)
    small, large = by_area[0], by_area[-1]
    a_small = geom.SEGMENTS[small].area_cm2
    a_large = geom.SEGMENTS[large].area_cm2
    assert a_large > 2 * a_small, "fixture assumes a real area spread"

    segs = {small: np.ones(n, bool), large: np.ones(n, bool)}
    segs[large][-1] = False                  # the big one loses the top step
    rows = bronze.bronze_frequency_coverage(_run(segs, n))

    assert rows[-1]["segment_fraction"] == pytest.approx(0.5)
    assert rows[-1]["area_coverage"] == pytest.approx(
        a_small / (a_small + a_large), abs=1e-4)
    assert rows[-1]["area_coverage"] < rows[-1]["segment_fraction"]


def test_the_summary_names_the_band_that_rests_on_most_of_the_plate() -> None:
    n = 6
    names = list(geom.SEGMENTS)[:4]
    segs = {s: np.ones(n, bool) for s in names}
    for s in names:                          # everything above step 3 is thin
        segs[s][4:] = False
    segs[names[0]][4:] = True                # one segment keeps them
    rows = bronze.bronze_frequency_coverage(_run(segs, n))
    summary = bronze.coverage_summary(rows, min_area_coverage=0.80)

    assert summary["ok"]
    assert summary["n_steps_above_threshold"] == 4
    assert summary["f_hi_covered_hz"] == pytest.approx(rows[3]["freq_hz"])
    assert summary["worst_area_coverage"] < 0.8


def test_coverage_of_nothing_is_reported_not_crashed() -> None:
    run = _run({}, 3)
    assert bronze.bronze_frequency_coverage(run) == []
    assert bronze.coverage_summary([])["ok"] is False


# ---------------------------------------------------------------------------
# repaired-window provenance
# ---------------------------------------------------------------------------


def _descending_sweep(n_steps=12, dwell=2400, gap=600):
    """A sweep whose windows march in time as the frequency falls."""
    freqs = np.geomspace(200.0, 1.0, n_steps)
    steps = []
    for i, f in enumerate(freqs):
        a = i * (dwell + gap)
        steps.append(Step(freq=float(f), start=a, stop=a + dwell, amp=0.01,
                          snr_db=20.0, thd=0.0, stationarity=0.0))
    return steps


def test_an_untouched_window_stays_marked_as_detected() -> None:
    steps = _descending_sweep()
    out, info = ladder_snap.repair_windows(steps, FS)
    assert info["ok"]
    assert info["n_repaired"] == 0
    assert all(s.window_source == "detected" for s in out)
    assert not any(s.window_repaired for s in out)


def test_a_repaired_window_is_marked_interpolated() -> None:
    """The point of the whole field: a predicted window says so."""
    steps = _descending_sweep()
    # one step claims a window far outside the sweep, as the 1501 Hz rung did
    bad = 5
    steps[bad] = Step(freq=steps[bad].freq, start=900_000, stop=900_360,
                      amp=0.01, snr_db=20.0, thd=0.0, stationarity=0.0)
    out, info = ladder_snap.repair_windows(steps, FS)

    assert info["ok"] and info["n_repaired"] == 1
    repaired = [s for s in out if s.window_repaired]
    assert len(repaired) == 1
    assert repaired[0].window_source == "interpolated"
    assert repaired[0].freq == pytest.approx(steps[bad].freq)
    # and it really was moved back into the sweep
    assert repaired[0].start < 100_000


def test_the_snap_does_not_erase_provenance() -> None:
    """Rebuilding a Step field by field is how a new field gets lost.

    `snap_steps` runs before `repair_windows` in the consensus, but nothing
    guarantees that order forever, and a rebuild that silently resets
    `window_source` to "detected" would turn a predicted window back into a
    measured-looking one.
    """
    s = Step(freq=10.0, start=0, stop=1000, amp=1.0, snr_db=9.0, thd=0.0,
             stationarity=0.0, window_source="interpolated")
    rebuilt = ladder_snap._rebuild(Step, s, start=50)
    assert rebuilt.window_source == "interpolated"
    assert rebuilt.start == 50
    assert rebuilt.freq == 10.0


def test_a_step_alike_without_the_field_still_rebuilds() -> None:
    """_rebuild must not require a Step that knows about window_source."""
    from dataclasses import dataclass

    @dataclass
    class Old:
        freq: float
        start: int
        stop: int
        amp: float
        snr_db: float
        thd: float
        stationarity: float

    o = Old(1.0, 0, 10, 1.0, 1.0, 0.0, 0.0)
    assert ladder_snap._rebuild(Old, o, start=5).start == 5
