"""The top of the band is in the file; the detector was reading the wrong channel.

These tests pin the four claims hf_schedule.py makes, each against the failure
it was written for rather than against a golden number:

  1. a phasor's precision is N*gamma, so a short high-frequency dwell that a
     flat SNR gate discards can still be a 2 % measurement;
  2. stacking the current-carrying channels must survive a reversed sense
     pair, or the array gain goes the wrong way;
  3. the fitted ladder must not be a subdivision of the true one, and its
     spacing must snap to the integer a generator would have been asked for;
  4. a predicted rung is accepted on evidence independent of the ladder, and
     an off-ladder interferer is rejected without any amplitude test.

The synthetics are deliberately small -- a few seconds at a few kHz -- because
what is being tested is the logic, not the estimator, which IEEE 1057 and
test_pipeline already cover.
"""

from __future__ import annotations

import numpy as np
import pytest

import hf_schedule as H
from eis_local import Step


# ---------------------------------------------------------------------------
# 1. N*gamma, not gamma
# ---------------------------------------------------------------------------


def test_a_short_high_frequency_dwell_is_kept_on_its_precision():
    """The 11.95 kHz step from card 4: -1.5 dB, N*gamma = 2459, 2.0 % phasor.

    The flat 5 dB gate threw it away.  Nothing about it is bad except the
    quantity it was judged on.
    """
    snr_db = -1.5
    n = int(round(2459 / 10 ** (snr_db / 10)))
    assert snr_db < 5.0                              # the old gate refused it
    assert H.crlb_usable(snr_db, n, sigma_rel_max=0.60)
    assert np.sqrt(1 / 2459) == pytest.approx(0.0202, abs=1e-4)


def test_a_long_dwell_does_not_rescue_a_step_with_no_tone():
    """sigma_rel alone would accept pure noise given enough samples.

    That is what SNR_ABSOLUTE_FLOOR_DB in eis_local is for: it is a backstop
    against absence, not a quality gate.
    """
    assert H.crlb_usable(-40.0, 10 ** 7, 0.60)       # the CRLB is happy
    nothing = Step(freq=1000.0, start=0, stop=10 ** 7, amp=1e-9,
                   snr_db=-40.0, thd=np.nan, stationarity=np.nan)
    assert not nothing.valid(min_snr=5.0)            # Step.valid is not

    real = Step(freq=11950.0, start=0, stop=3473, amp=1.0,
                snr_db=-1.5, thd=np.nan, stationarity=np.nan)
    assert real.valid(min_snr=5.0)


# ---------------------------------------------------------------------------
# 2. The stack must survive a reversed channel
# ---------------------------------------------------------------------------


def test_a_reversed_sense_pair_adds_instead_of_subtracting():
    rng = np.random.default_rng(0)
    fs, f, n = 2000.0, 50.0, 4000
    t = np.arange(n) / fs
    tone = np.sin(2 * np.pi * f * t)
    chans = {}
    for i in range(8):
        s = -1.0 if i in (2, 5) else 1.0
        chans[str(i)] = s * tone + rng.normal(0, 1.0, n)

    ref, info = H.polarity_aligned_reference(chans)
    assert info["n_channels"] == 8
    assert info["n_flipped"] in (2, 6)               # either sign convention

    def tone_frac(x):
        X = np.fft.rfft(x - x.mean())
        k = int(round(f / (fs / len(x))))
        return float(np.abs(X[k]) ** 2 / np.sum(np.abs(X) ** 2))

    # the stack must carry more tone than any single channel ...
    assert tone_frac(ref) > tone_frac(chans["0"])
    # ... and, the point of the polarity check, more than the same channels
    # summed blind, where the two reversed ones subtract
    blind = sum((chans[k] - chans[k].mean()) / chans[k].std()
                for k in chans)
    assert tone_frac(ref) > 1.5 * tone_frac(blind)


def test_a_dead_channel_does_not_poison_the_stack():
    rng = np.random.default_rng(1)
    n = 2000
    chans = {"1": rng.normal(0, 1, n), "2": np.zeros(n),
             "3": rng.normal(0, 1, n)}
    ref, info = H.polarity_aligned_reference(chans)
    assert info["n_channels"] == 2                   # the flat one is dropped
    assert np.all(np.isfinite(ref))


# ---------------------------------------------------------------------------
# 3. The ladder
# ---------------------------------------------------------------------------


def _ladder_freqs(f_hi=1000.0, ppd=10, n=21):
    return f_hi * 10 ** (-np.arange(n) / ppd)


def test_the_spacing_snaps_to_the_integer_a_generator_was_asked_for():
    """A free fit returns 10.06 points/decade and that error compounds.

    On card 4 the free fit's prediction error grew monotonically to +5.8 %
    over fifteen rungs; snapped, it stayed inside +/-1 %.
    """
    f = _ladder_freqs() * (1 + 0.0004 * np.arange(21))   # a little drift
    lad = H.fit_ladder(f, snap=True)
    assert lad.ok
    assert lad.ppd == 10.0
    assert lad.snapped
    assert lad.ppd_free != 10.0

    free = H.fit_ladder(f, snap=False)
    assert free.ok and not free.snapped


def test_the_ladder_is_not_a_subdivision_of_itself():
    """Every step of a 10 ppd sweep also lies on a 20 ppd ladder.

    A fit that scores ties by "most steps on the grid" therefore prefers the
    subdivision, invents the rungs in between, and at the top of the band
    those sit a fraction of a DFT bin from a real tone -- so the acceptance
    tests see the neighbour's leakage and pass.  This is what the gcd guard
    is for.
    """
    lad = H.fit_ladder(_ladder_freqs(ppd=10, n=21))
    assert lad.ok
    assert lad.ppd == pytest.approx(10.0, abs=0.01)
    assert lad.ratio == pytest.approx(10 ** 0.1, rel=1e-3)


def test_a_ladder_with_holes_is_still_the_coarse_ladder():
    """The confident steps are the ones that passed an SNR gate, so the set
    has real holes in it.  Comparing the fit against their median gap -- the
    obvious guard -- reads the spacing as twice too coarse and refuses the
    correct ladder.  The gcd guard does not, because the holes are irregular.
    """
    f = _ladder_freqs(ppd=10, n=21)
    keep = np.array([0, 1, 3, 4, 5, 8, 9, 12, 13, 14, 17, 20])
    lad = H.fit_ladder(f[keep])
    assert lad.ok
    assert lad.ppd == pytest.approx(10.0, abs=0.01)


def test_an_off_ladder_interferer_is_rejected_without_an_amplitude_test():
    """Card 4 carries a continuous 6.79 kHz interferer holding 12.9 % of the
    stacked trace's ac power -- stronger than several genuine rungs.  It sits
    9.9 % off the ladder, which is the one case a loosened SNR gate provably
    cannot handle.
    """
    lad = H.fit_ladder(_ladder_freqs(f_hi=10_000.0, ppd=10, n=21))
    assert lad.ok
    on = H.prune_off_ladder([lad.freq_of(3), lad.freq_of(3) * 1.099], lad)
    assert on[0] and not on[1]


# ---------------------------------------------------------------------------
# 4. The acceptance tests
# ---------------------------------------------------------------------------


def _array(f, fs, n, m=12, amp=1.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    tone = amp * np.sin(2 * np.pi * f * t)
    return {str(i): tone + rng.normal(0, sigma, n) for i in range(m)}


def test_the_cfar_finds_a_tone_that_falls_between_two_bins():
    """The straddle pair is floor and floor+1, not round and round+1.

    Rounding first picks the wrong pair whenever the fractional part of
    f/df exceeds 0.5, and then the test is run on a background bin.  A rung
    at 11 dB of in-bin SNR reported -0.04 dB that way.
    """
    fs, n = 20_000.0, 512
    df = fs / n
    f = (33 + 0.7) * df                       # deliberately past the half bin
    chans = _array(f, fs, n, sigma=3.0)
    res = H.cfar_channel_count(chans, fs, f, 0, n)
    assert res.m == 12
    assert res.ratio_db > 6.0
    assert res.accept()


def test_the_cfar_does_not_fire_on_noise():
    fs, n = 20_000.0, 512
    rng = np.random.default_rng(3)
    chans = {str(i): rng.normal(0, 1.0, n) for i in range(12)}
    fired = sum(H.cfar_channel_count(chans, fs, (30 + 0.3) * (fs / n) * (1 + j),
                                     0, n).accept() for j in range(20))
    assert fired <= 2                          # exact rate is 1/(1+n_bg) each


def test_the_rank_one_test_separates_common_from_per_channel_structure():
    """A real tone is common to every channel; a ground loop on one front end
    is not.  Per-channel SNR provably cannot tell those apart.
    """
    fs, n, m = 10_000.0, 4096, 12
    f = 300.0
    t = np.arange(n) / fs

    rng = np.random.default_rng(4)
    common = {str(i): 0.5 * np.sin(2 * np.pi * f * t) + rng.normal(0, 1, n)
              for i in range(m)}
    res_c = H.rank1_statistic(common, fs, f, 0, n)
    assert res_c.accept()
    assert res_c.participation > res_c.m / 2      # the whole array sees it

    rng = np.random.default_rng(4)
    one = {str(i): rng.normal(0, 1, n) for i in range(m)}
    one["0"] = one["0"] + 4.0 * np.sin(2 * np.pi * f * t)   # much stronger
    res_1 = H.rank1_statistic(one, fs, f, 0, n)
    assert not res_1.accept()
    assert res_1.participation < 2.0              # one channel carries it all


def test_a_rung_too_close_to_its_neighbour_to_resolve_is_never_accepted():
    """Both acceptance tests read a periodogram whose bin width is fs/N.  A
    rung within a few bins of its ladder neighbour cannot be separated from
    it, so a strong real tone one rung over passes both tests -- on every
    channel at once, which is exactly what makes it look convincing.
    """
    fs = 20_000.0
    lad = H.fit_ladder(_ladder_freqs(f_hi=8000.0, ppd=10, n=21))
    assert lad.ok
    f = 4000.0
    n_short = 32                        # df = 625 Hz; the neighbour is 1036 Hz
    n_ok = 2048                         # df = 9.8 Hz
    assert (lad.ratio - 1.0) * f < 3.0 * (fs / n_short)
    assert (lad.ratio - 1.0) * f > 3.0 * (fs / n_ok)


# ---------------------------------------------------------------------------
# 5. End to end, on the failure that motivated all of it
# ---------------------------------------------------------------------------


def _galvanostatic_card(fs=4000.0, f_hi=500.0, f_lo=5.0, ppd=10,
                        n_cyc=8.0, d_min=0.01, settle=0.02, m=12,
                        sigma=0.4, sigma_uc=0.05, seed=11):
    """A sweep whose REFERENCE amplitude falls with |Z_cell| while the segment
    channels stay flat -- the field symptom in miniature.

    The two noise floors are set to the same order of magnitude in volts,
    which is the point: it is |Z_cell| falling across the band, not a worse
    converter, that takes the reference channel below the segments.  Here
    the reference runs from ~11 dB at the bottom of the band to ~0 dB at the
    top while the stack stays flat at ~19 dB, so the crossover lands inside
    the band exactly as it does on card 4.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(ppd * np.log10(f_hi / f_lo))) + 1
    freqs = f_hi * 10 ** (-np.arange(n_steps) / ppd)
    uc, segs = [], [[] for _ in range(m)]
    for f in freqs:
        n = int(max(d_min, n_cyc / f) * fs)
        t = np.arange(n) / fs
        tone = np.sin(2 * np.pi * f * t)
        z = 0.045 + 0.355 / (1 + (f / 3.0) ** 0.8)     # 400 -> 45 mOhm
        uc.append(z * tone)
        for s in segs:
            s.append(tone)                              # current is imposed
        g = int(settle * fs)
        uc.append(np.zeros(g))
        for s in segs:
            s.append(np.zeros(g))
    uc = np.concatenate(uc)
    N = uc.size
    uc = uc + rng.normal(0, sigma_uc, N)
    chans = {}
    for i, parts in enumerate(segs):
        x = np.concatenate(parts) + rng.normal(0, sigma, N)
        if i == 3:
            x = -x
        chans[str(i)] = x
    return freqs, uc, chans, fs


def test_the_ensemble_reaches_higher_than_the_cell_voltage_channel():
    from eis_local import detect_schedule

    freqs, uc, chans, fs = _galvanostatic_card()

    def hits(steps):
        got = [s.freq for s in steps]
        true = sum(1 for ft in freqs
                   if any(abs(v / ft - 1) < 0.02 for v in got))
        spur = sum(1 for v in got
                   if not any(abs(v / ft - 1) < 0.02 for ft in freqs))
        return true, spur, (max(got) if got else 0.0)

    old = hits(detect_schedule(uc, fs, ppd=12, f_lo=1.0, f_hi=0.45 * fs,
                               min_snr_db=5.0, verbose=False))
    new = hits(H.recover_schedule(chans, fs, f_lo=1.0, f_hi=0.45 * fs,
                                  ppd=12, min_snr_db=5.0).as_steps())

    assert new[0] >= old[0]          # at least as many true steps ...
    assert new[2] >= old[2]          # ... reaching at least as high ...
    assert new[1] <= old[1]          # ... and no more spurious ones


def test_the_extension_invents_nothing_outside_the_recorded_band():
    """The strongest evidence that the predict-and-verify layer is honest:
    it predicts rungs across the whole configured band, and every one of them
    outside the sweep that was actually recorded must be refused.
    """
    freqs, _uc, chans, fs = _galvanostatic_card()
    hf = H.recover_schedule(chans, fs, f_lo=0.5, f_hi=0.45 * fs, ppd=12,
                            min_snr_db=5.0, extend=True)
    assert hf.ladder.ok
    assert hf.n_rejected > 0                 # rungs WERE predicted and refused
    for s in hf.predicted:
        assert any(abs(s.freq / ft - 1) < 0.03 for ft in freqs), (
            f"invented a step at {s.freq:.2f} Hz that the record does not "
            f"contain")


def test_the_cell_voltage_channel_is_pooled_in_not_replaced():
    """Neither trace dominates the other everywhere.

    On the synthetic card set the reference amplitude is flat in frequency,
    and there the old path locates the top rungs MORE precisely than the
    stack does -- 730.3 / 1169.6 / 1873.2 / 3000.0 Hz exactly, against the
    stack's 807 / 1182 / 1972 / 3070.  Replacing the reference would throw
    that away.  Pooling both candidate sets and letting the ladder and the
    array tests arbitrate cannot lose anything the shipped path found.
    """
    freqs, uc, chans, fs = _galvanostatic_card()

    def top(steps):
        return max((s.freq for s in steps), default=0.0)

    without = H.recover_schedule(chans, fs, f_lo=1.0, f_hi=0.45 * fs,
                                 ppd=12, min_snr_db=5.0)
    with_uc = H.recover_schedule(chans, fs, f_lo=1.0, f_hi=0.45 * fs,
                                 ppd=12, min_snr_db=5.0, uc_ref=uc)
    assert len(with_uc.as_steps()) >= len(without.as_steps())
    assert top(with_uc.as_steps()) >= top(without.as_steps())
    # and still nothing invented
    for s in with_uc.predicted:
        assert any(abs(s.freq / ft - 1) < 0.03 for ft in freqs)


def test_a_fixed_time_dwell_is_recognised_as_such():
    """A generator asked for a fixed TIME per step, not a fixed cycle count.

    Assuming max(d_min, n_cyc/f) on such a record mispredicts a
    high-frequency step's start by the length of the sweep, and the search
    bracket then lands on a different tone entirely.
    """
    fs = 1000.0
    f = _ladder_freqs(f_hi=200.0, ppd=10, n=15)
    n = int(2.0 * fs)                              # every dwell the same
    starts = np.arange(len(f))[::-1] * n           # sweep runs downward
    lad = H.fit_ladder(f, starts, starts + n, fs=fs)
    assert lad.ok
    assert lad.dwell_mode == "fixed"
    assert lad.dwell_s(200.0) == pytest.approx(2.0, rel=0.05)
    assert lad.dwell_s(2.0) == pytest.approx(2.0, rel=0.05)
