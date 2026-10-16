"""One stretch of the record is one step of the sweep.

A stepped sweep holds ONE frequency at a time, so its dwells are disjoint in
time by construction.  Nothing in `detect_schedule` enforced that: `_dedupe`
collapses candidates whose FREQUENCIES agree, which is a different question
entirely, and both the grid scan and `fill_gaps` can return the same stretch
of record for many different trial frequencies.

Measured on RO2611976-01 at 45 A: 142 reported steps sat on 41 distinct
windows, 111 of them (78 %) sharing one.  A single 118-second window carried
57 "steps" from 0.157 Hz to 3.9 kHz.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 6. One stretch of the record is one step of the sweep
# ---------------------------------------------------------------------------


def test_two_steps_cannot_occupy_the_same_stretch_of_record():
    """The invariant that 142 reported steps on RO2611976-01 violated.

    A stepped sweep holds one frequency at a time, so its dwells are disjoint
    in time by construction.  On that card the grid scan and fill_gaps
    between them put 142 steps on 41 distinct windows -- one 118-SECOND
    window carrying 57 of them, from 0.157 Hz to 3.9 kHz.  Silver then spent
    its gates rejecting them one at a time, which is why 56 % of its points
    came back `not_finite` and the lost band looked like a gating problem.
    """
    from eis_local import Step, _collapse_overlapping

    def s(f, a, b, snr):
        return Step(freq=f, start=a, stop=b, amp=1.0, snr_db=snr,
                    thd=np.nan, stationarity=np.nan)

    # one real dwell, and four frequencies fitted to the same long stretch
    steps = [s(100.0, 1000, 1400, 20.0),
             s(0.157, 5000, 25000, -26.0), s(3933.0, 5000, 25000, -22.0),
             s(1.2, 5000, 25000, -30.0), s(2.4, 5000, 25000, -18.0),
             s(500.0, 30000, 30400, 15.0)]
    out = _collapse_overlapping(steps)
    assert len(out) == 3
    kept = sorted(x.freq for x in out)
    assert kept == [2.4, 100.0, 500.0]      # the strongest of the shared four


def test_collapsing_never_removes_a_disjoint_dwell():
    """It cannot cost a real step, because real steps do not overlap."""
    from eis_local import Step, _collapse_overlapping

    steps = [Step(freq=10.0 * 1.26 ** k, start=1000 * k, stop=1000 * k + 900,
                  amp=1.0, snr_db=5.0 + k, thd=np.nan, stationarity=np.nan)
             for k in range(12)]
    assert len(_collapse_overlapping(steps)) == 12


def test_a_touching_window_is_not_an_overlap():
    """Consecutive dwells share an edge; that must not collapse them."""
    from eis_local import Step, _collapse_overlapping

    a = Step(freq=10.0, start=0, stop=1000, amp=1.0, snr_db=10.0,
             thd=np.nan, stationarity=np.nan)
    b = Step(freq=12.6, start=1000, stop=2000, amp=1.0, snr_db=9.0,
             thd=np.nan, stationarity=np.nan)
    assert len(_collapse_overlapping([a, b])) == 2
