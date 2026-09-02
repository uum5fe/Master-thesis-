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
