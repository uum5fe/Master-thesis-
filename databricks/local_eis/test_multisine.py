"""Stepped multisine: a group of tones applied at once, then the next group.

The estimator is not interchangeable with the stepped-sine one.  Fitting a
group one tone at a time puts each tone's neighbours into its own residual,
and the residual is the SNR, and the SNR is the weight -- so the error does
not show up as a bad spectrum, it shows up as a spectrum whose uncertainties
are fiction.
"""

from __future__ import annotations

import numpy as np
import pytest

import eis_local as E


FS = 100_000.0


def multisine(tones, seconds, fs=FS, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    y = sum(np.sin(2 * np.pi * f * t + p)
            for f, p in zip(tones, rng.uniform(0, 2 * np.pi, len(tones))))
    return y + noise * rng.standard_normal(n)


def stepped_multisine(groups, durations, fs=FS, noise=0.05, gap_s=0.02, seed=3):
    rng = np.random.default_rng(seed)
    chunks = [np.zeros(int(0.05 * fs))]
    for g, T in zip(groups, durations):
        n = int(T * fs)
        t = np.arange(n) / fs
        chunks.append(sum(np.sin(2 * np.pi * f * t + p)
                          for f, p in zip(g, rng.uniform(0, 2 * np.pi, len(g)))))
        chunks.append(np.zeros(int(gap_s * fs)))
    x = np.concatenate(chunks)
    return x + noise * rng.standard_normal(len(x))


GROUPS = [np.array([1, 2, 3, 5, 8], float) * 10.0,
          np.array([1, 2, 3, 5, 8], float) * 100.0,
          np.array([1, 2, 3, 5, 8], float) * 500.0]
DURATIONS = [2.0, 0.5, 0.3]


# ---------------------------------------------------------------------------
# the SNR a single-tone fit reports is a tone count, not a measurement
# ---------------------------------------------------------------------------

def test_single_tone_fit_reports_the_group_size_not_the_noise() -> None:
    """fit3's "SNR" on a multisine is -10*log10(N-1), whatever the noise.

    Worth pinning as a fact about the OLD estimator: this is the number a
    quality gate was being set against, and it explains why lowering the
    gate to -30 dB did not bring the weak tones in -- the number never
    described the tones.
    """
    tones = np.array([1, 2, 3, 5, 8, 13, 21, 34], float) * 100.0
    y = multisine(tones, 0.2, noise=0.01)          # true noise floor ~ -40 dB
    reported = [E.fit3(y, FS, f)[2] for f in tones]
    expected = -10 * np.log10(len(tones) - 1)
    assert np.allclose(reported, expected, atol=0.5), (
        f"expected the group-size floor {expected:.1f} dB, got {reported}")


def test_the_joint_fit_recovers_the_real_noise_floor() -> None:
    """The same dwell, fitted jointly, reports the SNR that is actually there."""
    tones = np.array([1, 2, 3, 5, 8, 13, 21, 34], float) * 100.0
    sigma, amp = 0.01, 1.0
    y = multisine(tones, 0.2, noise=sigma)
    phasors, _r, snr, info = E.fit_multitone(y, FS, tones)

    truth = 10 * np.log10((amp ** 2 / 2) / sigma ** 2)
    assert info["ok"] and info["resolvable"]
    assert np.allclose(np.abs(phasors), amp, rtol=0.02)
    assert np.allclose(snr, truth, atol=1.0), (
        f"want {truth:.1f} dB, got {np.round(snr, 1)}")
    assert info["sigma"] == pytest.approx(sigma, rel=0.1)


def test_unresolvable_tones_are_flagged_not_silently_split() -> None:
    """Two tones closer than 1/T cannot be separated over that window."""
    y = multisine([100.0, 100.5], 0.5)             # 0.5 Hz apart, T = 0.5 s
    _ph, _r, _snr, info = E.fit_multitone(y, FS, [100.0, 100.5])
    assert not info["resolvable"]


# ---------------------------------------------------------------------------
# finding the schedule
# ---------------------------------------------------------------------------

def test_every_tone_of_a_stepped_multisine_is_found() -> None:
    x = stepped_multisine(GROUPS, DURATIONS)
    steps = E.detect_multisine_schedule(x, FS, f_lo=5.0, f_hi=45_000.0,
                                        verbose=False)
    got = np.array([s.freq for s in steps])
    for f in np.unique(np.concatenate(GROUPS)):
        assert np.any(np.abs(got / f - 1) < 0.02), f"missed {f:.1f} Hz"


def test_the_spurious_tones_are_far_below_the_real_ones() -> None:
    """A few noise peaks survive; the SNR has to separate them decisively.

    That separation is the whole benefit of a joint fit: with a single-tone
    fit every tone in the group reports the same fictitious SNR, so no gate
    can tell a real tone from a noise peak.
    """
    x = stepped_multisine(GROUPS, DURATIONS)
    steps = E.detect_multisine_schedule(x, FS, f_lo=5.0, f_hi=45_000.0,
                                        verbose=False)
    real = np.unique(np.concatenate(GROUPS))
    hits = [s for s in steps if np.any(np.abs(real / s.freq - 1) < 0.02)]
    spur = [s for s in steps if not np.any(np.abs(real / s.freq - 1) < 0.02)]
    assert hits
    if spur:
        assert min(s.snr_db for s in hits) > max(s.snr_db for s in spur) + 20, (
            "real and spurious tones must not overlap in SNR")


def test_the_dwells_are_segmented_by_content_not_by_silence() -> None:
    """Three groups, three dwells -- with gaps far shorter than a frame."""
    x = stepped_multisine(GROUPS, DURATIONS, gap_s=0.002)
    dwells = E.detect_dwells(x, FS)
    assert len(dwells) == len(GROUPS), (
        f"want {len(GROUPS)} dwells, got {len(dwells)}: "
        + str([(round(a / FS, 3), round(b / FS, 3)) for a, b in dwells]))


def test_a_record_that_is_one_long_multisine_is_one_dwell() -> None:
    """The degenerate case still has to come out right."""
    x = multisine(np.array([1, 2, 3, 5, 8], float) * 50.0, 1.0)
    assert len(E.detect_dwells(x, FS)) == 1


# ---------------------------------------------------------------------------
# choosing the estimator
# ---------------------------------------------------------------------------

def test_a_stepped_multisine_is_recognised() -> None:
    x = stepped_multisine(GROUPS, DURATIONS)
    assert E.classify_excitation(x, FS, 5.0, 45_000.0)["kind"] == "multisine"


def test_a_stepped_sine_is_still_recognised_as_one() -> None:
    """The classifier must not push a plain sweep down the multisine path."""
    rng = np.random.default_rng(0)
    chunks = [np.zeros(int(0.05 * FS))]
    for f in np.geomspace(100.0, 4000.0, 12):
        n = int(round(20 / f * FS))
        t = np.arange(n) / FS
        chunks.append(np.sin(2 * np.pi * f * t))
        chunks.append(np.zeros(int(0.01 * FS)))
    x = np.concatenate(chunks)
    x = x + 0.05 * rng.standard_normal(len(x))
    assert E.classify_excitation(x, FS, 50.0, 45_000.0)["kind"] == "stepped"


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def test_simultaneous_tones_are_grouped_by_their_windows() -> None:
    """Simultaneity is already in the windows; nothing needs configuring."""
    steps = [E.Step(freq=f, start=0, stop=1000, amp=1.0, snr_db=20.0,
                    thd=0.0, stationarity=0.0) for f in (10.0, 20.0, 30.0)]
    steps += [E.Step(freq=f, start=2000, stop=3000, amp=1.0, snr_db=20.0,
                     thd=0.0, stationarity=0.0) for f in (100.0, 200.0)]
    groups = E.group_simultaneous(steps)
    assert sorted(len(g) for g in groups) == [2, 3]


def test_a_stepped_sine_groups_into_singletons() -> None:
    steps = [E.Step(freq=10.0 * (i + 1), start=1000 * i, stop=1000 * i + 900,
                    amp=1.0, snr_db=20.0, thd=0.0, stationarity=0.0)
             for i in range(5)]
    assert all(len(g) == 1 for g in E.group_simultaneous(steps))
