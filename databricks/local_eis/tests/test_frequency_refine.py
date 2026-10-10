"""Blind runs: the ladder snap puts every step on a perfect 10/decade grid,
but the Gamry rounds its frequencies (2612030: up to -1.1 / +0.7 % off that
grid). refine_frequencies finds each step's true frequency on the segment
channels, and every fit then uses that one frequency."""

from __future__ import annotations

import numpy as np

import bronze as B
import make_synth_famos as M
from bronze import CardInfo
from config import DEFAULT
from eis_local import FamosFile, Step

FS = 25_000.0
#: Gamry's real frequencies around 1-4 kHz on 2612030, and the ideal ladder
GAMRY = [3796.875, 2987.132, 2390.625, 1884.191, 1516.544]
LADDER = [30046.88 * 10 ** (-k / 10) for k in (9, 10, 11, 12, 13)]


def _card(tmp_path):
    dwell, gap = 0.3, 0.05
    n = int(len(GAMRY) * (dwell + gap) * FS) + 1000
    t = np.arange(n) / FS
    tone = np.zeros(n)
    win = []
    for k, f in enumerate(GAMRY):
        a = int(k * (dwell + gap) * FS)
        b = a + int(dwell * FS)
        tone[a:b] = np.sin(2 * np.pi * f * t[a:b])
        win.append((a + 200, b - 200))
    rng = np.random.default_rng(0)
    data = np.column_stack(
        [0.7 + 0.0005 * tone + 1e-4 * rng.standard_normal(n)]
        + [0.13 + 0.004 * tone + 4e-4 * rng.standard_normal(n)
           for _ in range(6)])
    p = M.write_v2(tmp_path / "Karte_1.DAT",
                   ["UC2", "1", "2", "3", "4", "5", "6"], data, FS)
    fam = FamosFile(p)
    cards = {p.stem: CardInfo(path=p, stem=p.stem, fs=fam.fs, n_ch=fam.n_ch,
                              n_samples=fam.n_samples,
                              duration_s=fam.n_samples / fam.fs,
                              ref_name="UC2", ref_slot=0, n_segments=6)}
    steps = [Step(freq=f, start=a, stop=b, amp=0.0, snr_db=10.0, thd=0.0,
                  stationarity=0.0) for f, (a, b) in zip(LADDER, win)]
    return [p], cards, steps


def test_ladder_steps_move_onto_the_gamrys_frequencies(tmp_path):
    files, cards, steps = _card(tmp_path)
    err0 = [abs(s.freq / g - 1) for s, g in zip(steps, GAMRY)]
    assert max(err0) > 0.005                      # the snapped ladder is off
    out, info = B.refine_frequencies(files, cards, steps, {}, DEFAULT,
                                     lags={})
    assert info["n_refined"] == len(GAMRY)
    for s, g in zip(out, GAMRY):
        assert abs(s.freq / g - 1) < 2e-4, (s.freq, g)


def test_gamry_frequencies_are_not_touched(tmp_path):
    files, cards, steps = _card(tmp_path)
    steps = [B.ladder_snap._rebuild(Step, s, freq=g)
             for s, g in zip(steps, GAMRY)]
    out, info = B.refine_frequencies(files, cards, steps,
                                     {"gamry_freqs": GAMRY}, DEFAULT, lags={})
    assert info["n_refined"] == 0 and info["n_exact"] == len(GAMRY)
    assert [s.freq for s in out] == GAMRY
