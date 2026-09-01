"""Do these five cards belong to one measurement?

The cards are armed by hand, one after another, so nothing about a folder of
.DAT files guarantees they recorded the same event.  The cross-correlation
that aligns them cannot answer this: it only ever sees sample indices, so
handed two unrelated records it returns the lag of best agreement between
them, scored against that same unrelated background.  The answer looks like
an answer.  The |NT header stamps are the independent evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import bronze as B
import eis_local as E
from config import DEFAULT


def card(stem, start, duration_s=120.0, fs=10_000.0, n_segments=15):
    return B.CardInfo(
        path=Path(f"{stem}.DAT"), stem=stem, fs=fs, n_ch=16,
        n_samples=int(duration_s * fs), duration_s=duration_s,
        ref_name="UC1", ref_slot=0, n_segments=n_segments,
        start_time=start)


T0 = datetime(2025, 7, 16, 7, 45, 46)


def five_cards(offsets_s, duration_s=120.0, fs=10_000.0):
    return {f"card{i+1}": card(f"card{i+1}", T0 + timedelta(seconds=o),
                               duration_s, fs)
            for i, o in enumerate(offsets_s)}


# ---------------------------------------------------------------------------
# the |NT stamp
# ---------------------------------------------------------------------------

def test_the_trigger_stamp_is_read() -> None:
    assert E.parse_start_time("|NT,20,16,7,2025,7,45,46;") == T0


def test_a_missing_stamp_is_none_rather_than_a_guess() -> None:
    assert E.parse_start_time("|CD,8,0.0001;") is None


# ---------------------------------------------------------------------------
# what the check is for
# ---------------------------------------------------------------------------

def test_the_real_45a_stagger_passes() -> None:
    """The stamps that shipped with the 45 A set: 07:45:46/46/49/48/48."""
    report = B.timebase_report(five_cards([0, 0, 3, 2, 2]), DEFAULT)
    assert report.ok, report.problems
    assert report.header_spread_s == pytest.approx(3.0)


def test_cards_from_different_runs_are_refused() -> None:
    """An hour apart is not a stagger, it is two measurements."""
    report = B.timebase_report(five_cards([0, 0, 3, 3600, 2]), DEFAULT)
    assert not report.ok
    assert any("cannot be one event" in p for p in report.problems)


def test_a_stagger_beyond_the_search_window_is_refused() -> None:
    """A true offset outside align_max_lag_s cannot be found.

    The peak the correlation does return in that case is inside the window
    by construction, so it is not the offset -- and it is reported with the
    same confidence as a correct one.
    """
    cfg = DEFAULT.replace(align_max_lag_s=12.0)
    report = B.timebase_report(five_cards([0, 0, 3, 40, 2], duration_s=600.0),
                               cfg)
    assert not report.ok
    assert any("align_max_lag_s" in p for p in report.problems)


def test_cards_that_never_overlap_are_refused() -> None:
    """Started one by one, an early card can stop before a late one starts."""
    report = B.timebase_report(
        five_cards([0, 100, 200, 300, 400], duration_s=60.0), DEFAULT)
    assert not report.ok
    assert any("never all recording at once" in p for p in report.problems)


def test_a_mixed_sample_rate_is_refused() -> None:
    """A lag in SAMPLES is not the same amount of time on two rates.

    estimate_card_lags measures the lag on one card and process_card applies
    it to another; if the rates differ that lag means a different duration
    on each, and every dwell window it places is wrong by the ratio.
    """
    cards = five_cards([0, 0, 1, 1, 1])
    cards["card3"] = card("card3", T0 + timedelta(seconds=1), fs=100_000.0)
    report = B.timebase_report(cards, DEFAULT)
    assert not report.ok
    assert any("share a sample rate" in p for p in report.problems)


def test_no_stamps_is_a_note_not_a_refusal() -> None:
    """Some dialects carry no |NT. That is a loss of evidence, not a fault."""
    cards = {k: card(k, None) for k in ("a", "b")}
    report = B.timebase_report(cards, DEFAULT)
    assert report.ok
    assert any("cannot be cross-checked" in n for n in report.notes)


def test_the_report_survives_a_round_trip_to_the_manifest() -> None:
    import json
    report = B.timebase_report(five_cards([0, 0, 3, 2, 2]), DEFAULT)
    assert json.loads(json.dumps(report.summary()))["ok"] is True


# ---------------------------------------------------------------------------
# the measured lag is cross-checked against the arming ORDER
# ---------------------------------------------------------------------------

def test_a_lag_that_contradicts_the_arming_order_is_refused() -> None:
    """Prominence says the peak is sharp, not that it is the right peak.

    A sweep looks periodic enough that the correlation can lock one dwell
    over and report it confidently.  The stamps are far too coarse to check
    a lag's VALUE -- on the 45 A set they understated the true stagger by
    2.7 s -- but they cannot get the ORDER wrong.
    """
    cards = five_cards([0, 5])                    # card2 armed 5 s later
    fs = 10_000.0
    tb = B.timebase_report(cards, DEFAULT)
    lags = {"card1": {"lag": 0, "corr": 1.0, "prominence": np.inf,
                      "applied": True},
            # armed later means a NEGATIVE lag; +5 s has the wrong sign
            "card2": {"lag": int(5 * fs), "corr": 0.3, "prominence": 100.0,
                      "applied": True}}
    B._check_against_headers(lags, "card1", cards, fs, tb, DEFAULT,
                             B.utils.get_logger(False))
    assert not lags["card2"]["applied"]
    assert "contradicts" in lags["card2"]["refused_reason"]


def test_a_lag_the_coarse_stamps_merely_understate_is_kept() -> None:
    """The 45 A case: stamps 3 s apart, truth 5.712 s. That must NOT refuse."""
    cards = five_cards([0, 3])
    fs = 10_000.0
    tb = B.timebase_report(cards, DEFAULT)
    lags = {"card1": {"lag": 0, "corr": 1.0, "prominence": np.inf,
                      "applied": True},
            "card2": {"lag": int(-5.712 * fs), "corr": 0.276,
                      "prominence": 250.0, "applied": True}}
    B._check_against_headers(lags, "card1", cards, fs, tb, DEFAULT,
                             B.utils.get_logger(False))
    assert lags["card2"]["applied"], (
        "a 2.7 s disagreement is what the coarse stamps do; refusing it "
        "would throw away the correct lag the 45 A set depends on")


# ---------------------------------------------------------------------------
# the refusal itself
# ---------------------------------------------------------------------------

def test_the_gate_is_readable_when_the_time_base_is_bad() -> None:
    """`cfg.require_timebase` is only reached when the check has FAILED.

    `if not timebase.ok and cfg.require_timebase:` short-circuits while the
    time base is fine, so a missing setting stays invisible through every
    run on consistent cards and raises AttributeError on the first run that
    the check was written to stop -- replacing the explanation with a
    traceback exactly when the explanation is the point.
    """
    assert DEFAULT.require_timebase is True
    assert DEFAULT.replace(require_timebase=False).require_timebase is False


def test_the_override_exists_on_the_command_line() -> None:
    """The refusal names --no-require-timebase; it has to be a real flag."""
    import main
    parser = main.build_parser()
    args = parser.parse_args(["--dat", ".", "--no-require-timebase"])
    assert args.require_timebase is False
    assert parser.parse_args(["--dat", "."]).require_timebase is None


def test_a_bad_time_base_stops_the_run_by_default() -> None:
    cards = five_cards([0, 100, 200, 300, 400], duration_s=60.0)
    report = B.timebase_report(cards, DEFAULT)
    assert not report.ok
    assert not report.ok and DEFAULT.require_timebase, (
        "both halves of the gate must be true for the run to stop")


# ---------------------------------------------------------------------------
# evaluating the part of a folder that CAN be evaluated
# ---------------------------------------------------------------------------

def test_the_45a_shape_splits_into_three_groups() -> None:
    """A folder is not a measurement.

    The 45 A set: Karte_1 at 100 kHz armed 15 min early, Karte_2/3 at
    100 kHz, Karte_4/5 at 50 kHz, all four of the latter armed together.
    Three evaluable subsets, and no way to make one plate of them.
    """
    cards = {
        "Karte_1": card("Karte_1", T0, 177.8, fs=100_000.0),
        "Karte_2": card("Karte_2", T0 + timedelta(seconds=899), 538.4,
                        fs=100_000.0),
        "Karte_3": card("Karte_3", T0 + timedelta(seconds=899), 538.4,
                        fs=100_000.0),
        "Karte_4": card("Karte_4", T0 + timedelta(seconds=899), 538.4,
                        fs=50_000.0),
        "Karte_5": card("Karte_5", T0 + timedelta(seconds=899), 538.4,
                        fs=50_000.0),
    }
    groups = B.consistent_groups(cards)
    assert len(groups) == 3
    sets = sorted(tuple(g["cards"]) for g in groups)
    assert sets == [("Karte_1",), ("Karte_2", "Karte_3"),
                    ("Karte_4", "Karte_5")]


def test_grouping_by_rate_alone_would_be_wrong() -> None:
    """Karte_1 shares 100 kHz with Karte_2 and Karte_3 and is not with them.

    Both conditions are needed and neither implies the other, which is the
    whole reason this is not a one-line filter on the sample rate.
    """
    cards = {
        "Karte_1": card("Karte_1", T0, 177.8, fs=100_000.0),
        "Karte_2": card("Karte_2", T0 + timedelta(seconds=899), 538.4,
                        fs=100_000.0),
    }
    groups = B.consistent_groups(cards)
    assert len(groups) == 2, "same rate, different runs -> different groups"


def test_cards_recorded_together_at_one_rate_stay_one_group() -> None:
    cards = {f"Karte_{i}": card(f"Karte_{i}", T0 + timedelta(seconds=i * 2),
                                300.0, fs=100_000.0) for i in range(1, 6)}
    groups = B.consistent_groups(cards)
    assert len(groups) == 1 and groups[0]["n_cards"] == 5


def test_a_group_reports_how_much_of_the_plate_it_covers() -> None:
    """Each group is a partial plate, and the number is the point."""
    cards = {
        "Karte_4": card("Karte_4", T0, 538.4, fs=50_000.0),
        "Karte_5": card("Karte_5", T0, 538.4, fs=50_000.0),
    }
    group = B.consistent_groups(cards)[0]
    assert group["n_segments"] == 30          # 15 per card in this fixture
    assert group["overlap_s"] == pytest.approx(538.4)


def test_unstamped_cards_are_grouped_by_rate_and_marked() -> None:
    """Without |NT there is no way to place them in time, and it says so."""
    cards = {"a": card("a", None, 100.0, fs=100_000.0),
             "b": card("b", None, 100.0, fs=100_000.0)}
    group = B.consistent_groups(cards)[0]
    assert group["timed"] is False and group["n_cards"] == 2
