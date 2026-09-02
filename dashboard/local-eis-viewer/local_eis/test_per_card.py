"""Evaluating every card, each on its own clock and at its own rate.

Z_k(f) = U_cell(f) / I_k(f), and every card records its own copy of the cell
voltage. When both come from the same card, a bulk timing offset of that card
multiplies numerator and denominator alike and CANCELS EXACTLY -- it is
absent from the ratio, not corrected out of it. So a folder whose cards
cannot be evaluated together can still be evaluated card by card.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import evaluate_per_card as P


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------

def test_a_windows_unc_path_survives_being_read(tmp_path) -> None:
    r"""Every backslash in \\bosch.com\DfsRB is part of the path.

    Anything that treats one as an escape turns the share into a path that
    does not exist, and the failure reads as "no cards found" -- which points
    at the network rather than at the parser.
    """
    unc = r"\\bosch.com\DfsRB\DfsDE\LOC\Fe\ILM\Charan\Lokale_EIS\Daten\famos"
    (tmp_path / ".env").write_text(
        f"EIS_FAMOS_ROOT={unc}\n"
        f"EIS_RESULTS_ROOT={unc}\\EIS_Results\n"
        f"# a comment\n\n"
        f'EIS_CURR_CAL="{unc}\\curr.csv"\n', encoding="utf-8")
    env = P.load_env(tmp_path / ".env")
    assert env["EIS_FAMOS_ROOT"] == unc
    assert env["EIS_CURR_CAL"] == unc + r"\curr.csv", "quotes stripped, path kept"
    assert env["EIS_RESULTS_ROOT"].endswith(r"\EIS_Results")


def test_a_missing_env_says_where_it_looked(tmp_path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        P.load_env(tmp_path / "nope.env")
    assert "Looked in" in str(excinfo.value)


# ---------------------------------------------------------------------------
# the per-rate ceiling -- the branch this whole approach turns on
# ---------------------------------------------------------------------------

def test_an_interferer_folds_to_a_different_place_at_each_rate() -> None:
    """45996 Hz is out of band at 100 kHz and mid-band at 50 kHz.

    One ceiling cannot serve both: the value that protects the 50 kHz cards
    throws away three quarters of the band on the 100 kHz ones.
    """
    hi, why_hi = P.ceiling_for(100_000.0, 45_996.0, None)
    lo, why_lo = P.ceiling_for(50_000.0, 45_996.0, None)

    assert hi is None, "at 100 kHz it is above the band; nothing to avoid"
    assert "outside the band" in why_hi
    assert lo == pytest.approx(0.8 * 4004.0, rel=1e-3)
    assert "folds to 4004" in why_lo


def test_no_interferer_means_each_card_uses_its_own_rate() -> None:
    assert P.ceiling_for(100_000.0, None, None)[0] is None
    assert P.ceiling_for(50_000.0, None, None)[0] is None


def test_an_explicit_ceiling_overrides_the_branch() -> None:
    value, why = P.ceiling_for(50_000.0, 45_996.0, 2500.0)
    assert value == 2500.0 and "every card" in why


def test_an_interferer_already_inside_the_band_is_still_avoided() -> None:
    """A 3 kHz interferer at 100 kHz does not fold; it is simply there."""
    value, why = P.ceiling_for(100_000.0, 3000.0, None)
    assert value == pytest.approx(2400.0)
    assert "folds to 3000" in why


# ---------------------------------------------------------------------------
# finding the cards
# ---------------------------------------------------------------------------

def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"|CF,2,1,1;")


def test_cards_are_found_in_per_condition_subfolders(tmp_path) -> None:
    """A campaign root usually holds one sub-folder per condition.

    bronze's own discovery globs a single directory because its caller hands
    it one condition's folder; searching only the root finds nothing and
    reports "no FAMOS files", which reads as a missing share.
    """
    for cond in ("45A", "150A"):
        for card in range(1, 6):
            _touch(tmp_path / "2612025" / cond /
                   f"RO2612025-01_Current_{cond}_Test_01_Karte_{card}.DAT")

    found = P.cards_of(tmp_path, "45A", "2612025")
    assert len(found) == 5
    assert all("_Current_45A_" in f.name for f in found)
    assert P.conditions_of(tmp_path, "2612025") == ["150A", "45A"]


def test_another_order_id_in_the_same_root_is_not_picked_up(tmp_path) -> None:
    _touch(tmp_path / "RO2612025-01_Current_45A_Test_01_Karte_1.DAT")
    _touch(tmp_path / "Leepa_2611976_Current_45A_Test_01_Karte_1.DAT")
    found = P.cards_of(tmp_path, "45A", "2612025")
    assert len(found) == 1 and "2612025" in found[0].name


def test_both_filename_conventions_are_recognised(tmp_path) -> None:
    _touch(tmp_path / "RO2612025-01_Current_45A_Test_01_Karte_1.DAT")
    _touch(tmp_path / "Leepa_2612025_Current_45A_Test_01_Karte_2.DAT")
    assert len(P.cards_of(tmp_path, "45A", "2612025")) == 2


def test_the_card_tag_identifies_one_card(tmp_path) -> None:
    name = Path("RO2612025-01_Current_45A_Test_01_Karte_3.DAT")
    assert P.card_tag(name) == "Karte_3"


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------

def _write(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_disjoint_cards_concatenate(tmp_path) -> None:
    """Each card is wired to its own block of the plate."""
    for tag, segs in (("Karte_1", ["1", "2"]), ("Karte_2", ["3", "4"])):
        _write(tmp_path / tag / "gold" / "plate_summary.csv",
               [{"segment": s, "class": "measured", "R_ohmic": "60"}
                for s in segs])
        _write(tmp_path / tag / "silver" / "spectra_clean.csv",
               [{"segment": s, "freq_hz": "10", "z_re_mohm_cm2": "60",
                 "z_im_mohm_cm2": "-5"} for s in segs])

    stats = P.merge(tmp_path, ["Karte_1", "Karte_2"], tmp_path / "merged")
    assert stats["n_segments"] == 4 and stats["n_points"] == 4
    assert not stats["duplicates"]
    assert (tmp_path / "merged" / "gold" / "plate_summary.csv").is_file()


def test_a_segment_claimed_by_two_cards_is_reported(tmp_path) -> None:
    """It should not happen, and if it does it must not be silent."""
    for tag in ("Karte_1", "Karte_2"):
        _write(tmp_path / tag / "gold" / "plate_summary.csv",
               [{"segment": "7", "class": "measured", "R_ohmic": "60"}])
        _write(tmp_path / tag / "silver" / "spectra_clean.csv",
               [{"segment": "7", "freq_hz": "10", "z_re_mohm_cm2": "60",
                 "z_im_mohm_cm2": "-5"}])
    stats = P.merge(tmp_path, ["Karte_1", "Karte_2"], tmp_path / "merged")
    assert stats["n_segments"] == 1
    assert stats["duplicates"] == ["7"]


def test_inferred_segments_are_not_merged_in(tmp_path) -> None:
    """Only what a card MEASURED belongs to that card.

    Each per-card run infers the 50-odd segments it does not carry from its
    own spatial field. Concatenating those would fill the plate with five
    cards' guesses about each other.
    """
    _write(tmp_path / "Karte_1" / "gold" / "plate_summary.csv",
           [{"segment": "1", "class": "measured", "R_ohmic": "60"},
            {"segment": "2", "class": "inferred", "R_ohmic": "61"}])
    _write(tmp_path / "Karte_1" / "silver" / "spectra_clean.csv",
           [{"segment": "1", "freq_hz": "10", "z_re_mohm_cm2": "60",
             "z_im_mohm_cm2": "-5"}])
    stats = P.merge(tmp_path, ["Karte_1"], tmp_path / "merged")
    assert stats["n_segments"] == 1, "the inferred segment must not be kept"


# ---------------------------------------------------------------------------
# the plotted band
# ---------------------------------------------------------------------------

def _spectra(path: Path, rows) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["segment", "freq_hz", "z_re_mohm_cm2",
                            "z_im_mohm_cm2", "card"])
        writer.writeheader()
        writer.writerows(rows)


def test_the_band_restricts_the_plot_and_not_the_data(tmp_path) -> None:
    """Narrowing the view must be reversible; narrowing the analysis is not.

    Filtering at plot time means a different band is one more command, not
    another pass over the cards.
    """
    pytest.importorskip("matplotlib")
    rows = [{"segment": "1", "freq_hz": f, "z_re_mohm_cm2": 60.0,
             "z_im_mohm_cm2": -5.0, "card": "Karte_1"}
            for f in (10.0, 100.0, 1000.0, 4500.0, 20000.0)]
    _spectra(tmp_path / "silver" / "spectra_clean.csv", rows)

    image = P.plot(tmp_path, "test", None, 5000.0)
    assert image is not None and image.is_file()
    # the source table is untouched
    kept = _read_csv(tmp_path / "silver" / "spectra_clean.csv")
    assert len(kept) == 5


def _read_csv(path):
    import csv
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def test_a_band_with_nothing_in_it_reports_what_the_cards_reach(tmp_path,
                                                                capsys):
    """Silence is not an answer; how far each card got is.

    A card whose ceiling sits below the window has nothing to draw, and the
    reason is its ceiling -- on this plate, set beneath where an interferer
    folds into its band.
    """
    pytest.importorskip("matplotlib")
    rows = [{"segment": "1", "freq_hz": 100.0, "z_re_mohm_cm2": 60.0,
             "z_im_mohm_cm2": -5.0, "card": "Karte_4"}]
    _spectra(tmp_path / "silver" / "spectra_clean.csv", rows)

    assert P.plot(tmp_path, "test", 4000.0, 5000.0) is None
    out = capsys.readouterr().out
    assert "no point falls inside" in out
    assert "Karte_4 to 100 Hz" in out


def test_a_card_absent_from_the_band_is_named_on_the_figure(tmp_path, capsys):
    """A legend that simply lacks a card looks like a card that was not run."""
    pytest.importorskip("matplotlib")
    rows = [{"segment": "1", "freq_hz": 4500.0, "z_re_mohm_cm2": 60.0,
             "z_im_mohm_cm2": -5.0, "card": "Karte_2"},
            {"segment": "9", "freq_hz": 900.0, "z_re_mohm_cm2": 61.0,
             "z_im_mohm_cm2": -6.0, "card": "Karte_4"}]
    _spectra(tmp_path / "silver" / "spectra_clean.csv", rows)

    assert P.plot(tmp_path, "test", 4000.0, 5000.0) is not None
    out = capsys.readouterr().out
    assert "NOT in the band" in out and "Karte_4 reaches only 900 Hz" in out


# ---------------------------------------------------------------------------
# grouping the cards
# ---------------------------------------------------------------------------

def _stub_rates(monkeypatch, rates: dict[str, float]) -> None:
    """Report a rate per card without opening a FAMOS file."""
    monkeypatch.setattr(P, "card_rate",
                        lambda card: rates[P.card_tag(card)])


def _cards(tmp_path, *names: str) -> list[Path]:
    out = []
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        out.append(p)
    return out


def test_cards_of_one_rate_are_evaluated_together(tmp_path, monkeypatch):
    """The whole point: the vote survives inside a rate.

    Splitting card by card would throw away the consensus schedule and the
    cross-card agreement even between cards that share a clock and could
    perfectly well have kept both.
    """
    cards = _cards(tmp_path,
                   *[f"RO26_Current_45A_Test_01_Karte_{i}.DAT"
                     for i in (1, 2, 3, 4, 5)])
    _stub_rates(monkeypatch, {"Karte_1": 100_000.0, "Karte_2": 100_000.0,
                              "Karte_3": 100_000.0, "Karte_4": 50_000.0,
                              "Karte_5": 50_000.0})
    groups, failed = P.group_cards(cards, "rate")
    assert not failed
    assert [(g.tag, g.fs, g.card_tags) for g in groups] == [
        ("100kHz", 100_000.0, ["Karte_1", "Karte_2", "Karte_3"]),
        ("50kHz", 50_000.0, ["Karte_4", "Karte_5"]),
    ]


def test_no_group_ever_mixes_two_rates(tmp_path, monkeypatch):
    """A lag in samples is not the same time on two cards clocked apart.

    This is the invariant the whole partition exists to hold; bronze refuses
    a mixed selection outright, so a group that mixed rates would not give a
    poor answer, it would give none.
    """
    cards = _cards(tmp_path, *[f"x_Karte_{i}.DAT" for i in range(1, 6)])
    _stub_rates(monkeypatch, {"Karte_1": 100_000.0, "Karte_2": 50_000.0,
                              "Karte_3": 100_000.0, "Karte_4": 20_000.0,
                              "Karte_5": 50_000.0})
    for mode in ("rate", "card"):
        groups, _ = P.group_cards(cards, mode)
        for g in groups:
            rates = {P.card_rate(c) for c in g.cards}
            assert rates == {g.fs}, f"{mode}: {g.tag} mixes {rates}"
        assert sum(len(g.cards) for g in groups) == 5, "no card is dropped"


def test_group_card_isolates_every_card(tmp_path, monkeypatch):
    """The fallback: no card has to agree with any other."""
    cards = _cards(tmp_path, *[f"x_Karte_{i}.DAT" for i in (1, 2, 3)])
    _stub_rates(monkeypatch, dict.fromkeys(
        ("Karte_1", "Karte_2", "Karte_3"), 100_000.0))
    groups, _ = P.group_cards(cards, "card")
    assert [g.card_tags for g in groups] == [["Karte_1"], ["Karte_2"],
                                             ["Karte_3"]]
    assert sorted(g.tag for g in groups) == ["Karte_1", "Karte_2", "Karte_3"]


def test_an_unreadable_header_is_reported_not_guessed(tmp_path, monkeypatch):
    """A card whose rate is unknown cannot be placed in any group."""
    cards = _cards(tmp_path, "x_Karte_1.DAT", "x_Karte_2.DAT")

    def rate(card):
        if "Karte_2" in card.name:
            raise ValueError("incomplete FAMOS header")
        return 100_000.0

    monkeypatch.setattr(P, "card_rate", rate)
    groups, failed = P.group_cards(cards, "rate")
    assert [g.card_tags for g in groups] == [["Karte_1"]]
    assert len(failed) == 1 and failed[0][0] == "Karte_2"
    assert "incomplete FAMOS header" in failed[0][1]


def test_two_folders_at_one_rate_get_distinct_tags(tmp_path, monkeypatch):
    """bronze globs a single directory, so a group cannot span two.

    The rate alone would name both groups "100kHz" and the second would
    overwrite the first's results.
    """
    cards = _cards(tmp_path, "a/x_Karte_1.DAT", "b/x_Karte_2.DAT")
    _stub_rates(monkeypatch, {"Karte_1": 100_000.0, "Karte_2": 100_000.0})
    groups, _ = P.group_cards(cards, "rate")
    assert len(groups) == 2, "one group cannot cover two directories"
    assert len({g.tag for g in groups}) == 2, "the tags must not collide"
    assert len({g.directory for g in groups}) == 2


def test_the_group_run_restricts_discovery_to_its_members(tmp_path, monkeypatch):
    """--cards is what keeps a 50 kHz card out of the 100 kHz run."""
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv

        class R:
            returncode = 0
            stdout = stderr = ""
        return R()

    monkeypatch.setattr(P.subprocess, "run", fake_run)
    cards = _cards(tmp_path, "x_Karte_1.DAT", "x_Karte_3.DAT")
    group = P.Group(tag="100kHz", fs=100_000.0, cards=tuple(cards),
                    directory=tmp_path)
    ok, why = P.run_group(group, "2612025", "45A", tmp_path / "out",
                          {"EIS_CURR_CAL": "cal.csv"}, 4000.0, None, [])
    assert ok, why
    argv = seen["argv"]
    assert argv[argv.index("--cards") + 1] == "Karte_1,Karte_3"
    assert argv[argv.index("--dat") + 1] == str(tmp_path)
    assert argv[argv.index("--label") + 1] == "45A_100kHz"
    assert argv[argv.index("--f-max") + 1] == "4000.0"


def test_the_merge_keeps_the_physical_card_of_each_segment(tmp_path):
    """In a rate group the run tag is "100kHz", which is not a card.

    Silver's own summary is the only table that says which card a segment
    was wired to, and losing it would make the merged plate unable to answer
    "which card measured this segment?" -- the first question asked of a
    suspect result.
    """
    _write(tmp_path / "100kHz" / "gold" / "plate_summary.csv",
           [{"segment": s, "class": "measured", "R_ohmic": "60"}
            for s in ("1", "40")])
    _write(tmp_path / "100kHz" / "silver" / "spectra_clean.csv",
           [{"segment": s, "freq_hz": "10", "z_re_mohm_cm2": "60",
             "z_im_mohm_cm2": "-5"} for s in ("1", "40")])
    _write(tmp_path / "100kHz" / "silver" / "segments_summary.csv",
           [{"segment": "1", "card": "Karte_1"},
            {"segment": "40", "card": "Karte_3"}])

    P.merge(tmp_path, ["100kHz"], tmp_path / "merged")
    rows = _read_csv(tmp_path / "merged" / "gold" / "plate_summary.csv")
    assert {r["segment"]: r["card"] for r in rows} == {"1": "Karte_1",
                                                       "40": "Karte_3"}
    assert {r["run"] for r in rows} == {"100kHz"}, "the run is recorded too"

    spectra = _read_csv(tmp_path / "merged" / "silver" / "spectra_clean.csv")
    assert {r["segment"]: r["card"] for r in spectra} == {"1": "Karte_1",
                                                          "40": "Karte_3"}


def test_the_card_falls_back_to_the_run_tag(tmp_path):
    """A run with no segments_summary must still populate the column."""
    _write(tmp_path / "Karte_1" / "gold" / "plate_summary.csv",
           [{"segment": "1", "class": "measured", "R_ohmic": "60"}])
    _write(tmp_path / "Karte_1" / "silver" / "spectra_clean.csv",
           [{"segment": "1", "freq_hz": "10", "z_re_mohm_cm2": "60",
             "z_im_mohm_cm2": "-5"}])
    P.merge(tmp_path, ["Karte_1"], tmp_path / "merged")
    rows = _read_csv(tmp_path / "merged" / "gold" / "plate_summary.csv")
    assert rows[0]["card"] == "Karte_1" and rows[0]["run"] == "Karte_1"


# ---------------------------------------------------------------------------
# the cell aggregate the merged plate implies
# ---------------------------------------------------------------------------

def _plate(tmp_path, tag, rows, areas) -> None:
    """One group's output: spectra, the gold table, and the wiring."""
    _write(tmp_path / tag / "gold" / "plate_summary.csv",
           [{"segment": s, "class": "measured", "area_cm2": str(areas[s])}
            for s in areas])
    _write(tmp_path / tag / "silver" / "spectra_clean.csv", rows)


def test_the_cell_curve_is_the_harmonic_mean_not_the_arithmetic_one(tmp_path):
    """Segments sit in PARALLEL across one cell voltage, so admittances add.

    Two equal-area segments at 100 and 300 mOhm*cm2 give a cell ASR of
    2A / (A/100 + A/300) = 150, not the arithmetic 200. Getting this wrong
    would make the cell curve agree with the Gamry sweep only by accident.
    """
    _plate(tmp_path, "100kHz",
           [{"segment": "1", "freq_hz": "10", "z_re_mohm_cm2": "100",
             "z_im_mohm_cm2": "0"},
            {"segment": "2", "freq_hz": "10", "z_re_mohm_cm2": "300",
             "z_im_mohm_cm2": "0"}],
           {"1": 8.0, "2": 8.0})
    P.merge(tmp_path, ["100kHz"], tmp_path / "merged")
    cell = _read_csv(tmp_path / "merged" / "silver" / "cell_aggregate.csv")
    assert len(cell) == 1
    assert float(cell[0]["z_re_mohm_cm2"]) == pytest.approx(150.0, rel=1e-6)
    assert cell[0]["n_segments"] == "2"


def test_the_cell_curve_is_weighted_by_area(tmp_path):
    """A bigger segment carries more admittance, so it moves the curve more.

    Areas 12 and 4 at 100 and 300 mOhm*cm2 give 16/(12/100 + 4/300) = 120.
    """
    _plate(tmp_path, "100kHz",
           [{"segment": "1", "freq_hz": "10", "z_re_mohm_cm2": "100",
             "z_im_mohm_cm2": "0"},
            {"segment": "2", "freq_hz": "10", "z_re_mohm_cm2": "300",
             "z_im_mohm_cm2": "0"}],
           {"1": 12.0, "2": 4.0})
    P.merge(tmp_path, ["100kHz"], tmp_path / "merged")
    cell = _read_csv(tmp_path / "merged" / "silver" / "cell_aggregate.csv")
    assert float(cell[0]["z_re_mohm_cm2"]) == pytest.approx(120.0, rel=1e-6)


def test_the_cell_curve_spans_the_groups(tmp_path):
    """Neither group covers the cell on its own; the merge does.

    This is why it is recomputed here rather than taken from a group: on a
    plate split by rate, each group's own cell_aggregate.csv is an aggregate
    over that group's segments only.
    """
    _plate(tmp_path, "100kHz",
           [{"segment": "1", "freq_hz": "10", "z_re_mohm_cm2": "100",
             "z_im_mohm_cm2": "0"}], {"1": 8.0})
    _plate(tmp_path, "50kHz",
           [{"segment": "40", "freq_hz": "10", "z_re_mohm_cm2": "300",
             "z_im_mohm_cm2": "0"}], {"40": 8.0})
    P.merge(tmp_path, ["100kHz", "50kHz"], tmp_path / "merged")
    cell = _read_csv(tmp_path / "merged" / "silver" / "cell_aggregate.csv")
    assert cell[0]["n_segments"] == "2", "both groups contribute"
    assert float(cell[0]["z_re_mohm_cm2"]) == pytest.approx(150.0, rel=1e-6)


def test_a_frequency_only_one_group_reached_is_dropped(tmp_path):
    """Groups detect their own steps, so their grids need not coincide.

    A point held up by a small minority of the plate is those segments'
    spectrum wearing the cell's name, and it must not be published as the
    cell's -- nor interpolated across a gap the data does not cover.
    """
    _plate(tmp_path, "100kHz",
           [{"segment": str(s), "freq_hz": f, "z_re_mohm_cm2": "100",
             "z_im_mohm_cm2": "0"}
            for s in (1, 2, 3) for f in ("10", "100")], {"1": 8.0, "2": 8.0,
                                                         "3": 8.0})
    _plate(tmp_path, "50kHz",
           [{"segment": "40", "freq_hz": f, "z_re_mohm_cm2": "300",
             "z_im_mohm_cm2": "0"} for f in ("10", "4000")], {"40": 8.0})
    P.merge(tmp_path, ["100kHz", "50kHz"], tmp_path / "merged")
    cell = _read_csv(tmp_path / "merged" / "silver" / "cell_aggregate.csv")
    have = {r["freq_hz"] for r in cell}
    assert "4000.0" not in have and "4000" not in have, \
        "one segment out of four does not make a cell point"
    assert {float(f) for f in have} == {10.0, 100.0}


def test_a_segment_with_no_area_cannot_be_weighted(tmp_path):
    """Silently treating a missing area as zero, or as one, both lie."""
    _plate(tmp_path, "100kHz",
           [{"segment": "1", "freq_hz": "10", "z_re_mohm_cm2": "100",
             "z_im_mohm_cm2": "0"},
            {"segment": "2", "freq_hz": "10", "z_re_mohm_cm2": "300",
             "z_im_mohm_cm2": "0"},
            {"segment": "3", "freq_hz": "10", "z_re_mohm_cm2": "100",
             "z_im_mohm_cm2": "0"}],
           {"1": 8.0, "2": 8.0, "3": 8.0})
    # blank the area of segment 3 the way an unfitted row is written
    gold = tmp_path / "100kHz" / "gold" / "plate_summary.csv"
    gold.write_text(gold.read_text().replace("3,measured,8.0",
                                             "3,measured,"))
    P.merge(tmp_path, ["100kHz"], tmp_path / "merged")
    cell = _read_csv(tmp_path / "merged" / "silver" / "cell_aggregate.csv")
    assert cell[0]["n_segments"] == "2", "the unweighable segment is left out"
    assert float(cell[0]["z_re_mohm_cm2"]) == pytest.approx(150.0, rel=1e-6)
