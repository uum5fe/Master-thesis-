"""Renaming FAMOS channels from a hand-written list: does it refuse well?

The names come from a person typing them, so the tests weigh the refusals as
heavily as the renames: a list one name short would otherwise rename fifteen
channels correctly and leave the sixteenth carrying its old name, which is
exactly the kind of result nobody notices.

And because this rewrites a measurement file, `rename` is held to the only
standard that matters for that: not one sample byte moves.

    python -m pytest test_famos_rename.py -q
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

import famos_rename as F


DT = 2e-5


def key(name: bytes, version: int, body: bytes) -> bytes:
    return b"|%s,%d,%d,%s;\r\n" % (name, version, len(body), body)


def plant(path, names, n=64, dtype="<f8", width=8, fmt=8, stride=None,
          with_names=True):
    """A FAMOS file with a KNOWN channel table, for the parser to recover."""
    nch = len(names)
    frame = stride if stride else width * nch

    out = [key(b"CF", 2, b"1"), key(b"CK", 1, b"1,1")]
    for i, name in enumerate(names):
        nb = name.encode("latin-1")
        out.append(key(b"CD", 1, b"%.6e,1,1,s,0,0,0" % DT))
        out.append(key(b"CP", 1, b"1,%d,%d,%d,0,%d,1,%d"
                       % (width, fmt, width * 8, i * width, frame - width)))
        out.append(key(b"Cb", 1, b"1,0,%d,1,0,%d,0,%d,1,0,0,"
                       % (i + 1, n * frame, n * frame)))
        out.append(key(b"CR", 1, b"0,0,0,1,0,"))
        if with_names:
            out.append(key(b"CN", 1, b"1,0,0,%d,%s,0," % (len(nb), nb)))

    rng = np.random.default_rng(0)
    data = np.zeros((n, nch))
    for c in range(nch):
        data[:, c] = 0.15 + 0.01 * rng.standard_normal(n)

    body = b"1," + data.astype(dtype).tobytes()
    path.write_bytes(b"".join(out) + b"|CS,1,%d,%s;" % (len(body), body))
    return data


@pytest.fixture
def card(tmp_path):
    """Sixteen channels named after their slot, as DASYLab writes them."""
    p = tmp_path / "2026_08_27_KANAL_6479RO2612025_60A.DDF_1.DAT"
    planted = plant(p, [str(i) for i in range(16)])
    return p, planted


SEGMENTS = [str(s) for s in range(64, 80)]


# ---------------------------------------------------------------------------
# The header
# ---------------------------------------------------------------------------


def test_the_channel_table_comes_out_of_the_keys(card):
    p, _ = card
    head = F.read_header(p)

    assert head.n_channels == 16
    assert head.numpy_dtype() == "<f8"
    assert head.frame_bytes == 128
    assert head.n_frames == 64
    assert [c.byte_offset for c in head.channels] == list(range(0, 128, 8))
    assert head.names == [str(i) for i in range(16)]
    assert head.fs == pytest.approx(1.0 / DT)


def test_the_samples_are_where_the_header_says_they_are(card):
    p, planted = card
    head = F.read_header(p)

    assert np.array_equal(np.asarray(F.read_data(head)), planted)


def test_a_name_containing_a_comma_survives(tmp_path):
    """The reason the keys are parsed by declared length, not split on ','."""
    p = tmp_path / "card.DAT"
    plant(p, ["U,ref", "1;2", "plain"])

    assert F.read_header(p).names == ["U,ref", "1;2", "plain"]


def test_a_frame_that_is_not_a_plain_interleave_is_refused(tmp_path):
    """A stride wider than the samples means padding this tool cannot read."""
    p = tmp_path / "card.DAT"
    plant(p, ["0", "1"], stride=64)

    with pytest.raises(F.FamosError, match="not a plain interleave"):
        F.read_header(p)


def test_a_file_that_is_not_famos_is_refused(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_bytes(b"DASYLab V14\r\nchannels\t16\r\n" + b"\x00" * 512)

    with pytest.raises(F.FamosError, match="not a FAMOS file"):
        F.read_header(p)


# ---------------------------------------------------------------------------
# The names you supply -- nothing is inferred
# ---------------------------------------------------------------------------


def test_the_names_are_taken_literally():
    assert F.parse_names_arg("UC1,temp1,temp2") == ["UC1", "temp1", "temp2"]


def test_a_numeric_range_is_a_shorthand_for_typing_them_out():
    assert F.parse_names_arg("64-79") == SEGMENTS
    assert F.parse_names_arg("UC1,65-67") == ["UC1", "65", "66", "67"]
    with pytest.raises(F.FamosError, match="backwards"):
        F.parse_names_arg("79-64")


def test_the_file_name_is_never_read_for_names(card):
    """The card is called KANAL_6479 and that must buy it exactly nothing."""
    p, _ = card

    assert F.read_header(p).names == [str(i) for i in range(16)]
    with pytest.raises(F.FamosError, match="16 names given|0 names"):
        F.rename(p, p.parent / "out.DAT", [])


def test_a_short_list_is_refused_not_zipped_off(card):
    """zip() would rename fifteen and leave the sixteenth quietly wrong."""
    p, _ = card

    with pytest.raises(F.FamosError, match="15 names given for 16 channels"):
        F.check_names(SEGMENTS[:-1], F.read_header(p))


def test_a_long_list_is_refused(card):
    p, _ = card

    with pytest.raises(F.FamosError, match="17 names given"):
        F.check_names(SEGMENTS + ["80"], F.read_header(p))


def test_a_duplicate_name_is_refused(card):
    p, _ = card
    names = list(SEGMENTS)
    names[3] = names[2]

    with pytest.raises(F.FamosError, match="channels 2 and 3"):
        F.check_names(names, F.read_header(p))


def test_a_blank_name_is_refused(card):
    p, _ = card
    names = list(SEGMENTS)
    names[4] = "   "

    with pytest.raises(F.FamosError, match=r"channel\(s\) 4"):
        F.check_names(names, F.read_header(p))


def test_a_name_a_famos_header_cannot_hold_is_refused(card):
    p, _ = card
    names = list(SEGMENTS)
    names[0] = "seg中"

    with pytest.raises(F.FamosError, match="cannot hold"):
        F.check_names(names, F.read_header(p))


# ---------------------------------------------------------------------------
# The template round trip
# ---------------------------------------------------------------------------


def test_the_template_starts_from_the_names_already_there(card, tmp_path):
    """So editing four channels does not mean retyping the other twelve."""
    p, _ = card
    out = F.write_template(F.read_header(p), tmp_path / "names.csv")

    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))

    assert [r["new_name"] for r in rows] == [str(i) for i in range(16)]
    assert [r["label_in_file"] for r in rows] == [str(i) for i in range(16)]


def test_an_edited_template_reads_back_in_order(card, tmp_path):
    p, _ = card
    path = tmp_path / "names.csv"
    F.write_template(F.read_header(p), path)

    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    for i, row in enumerate(rows[1:]):
        row[2] = str(64 + i)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    assert F.read_names_file(path) == SEGMENTS


def test_a_plain_list_of_names_is_read_too(tmp_path):
    path = tmp_path / "names.txt"
    path.write_text("# the card, top to bottom\nUC1\n65\n66\n")

    assert F.read_names_file(path) == ["UC1", "65", "66"]


def test_a_template_name_with_a_comma_survives_the_csv(card, tmp_path):
    p, _ = card
    path = tmp_path / "names.csv"
    rows = [F.TEMPLATE_COLUMNS] + [[i, str(i), f"U,{i}"] for i in range(16)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    assert F.read_names_file(path) == [f"U,{i}" for i in range(16)]


# ---------------------------------------------------------------------------
# The renamed file
# ---------------------------------------------------------------------------


def test_rename_moves_the_names_and_not_one_sample_byte(card, tmp_path):
    p, planted = card
    out = tmp_path / "renamed.DAT"

    head = F.rename(p, out, SEGMENTS)

    assert head.names == SEGMENTS
    # The names grew, so the data moved -- and must still read identically.
    assert head.data_offset > F.read_header(p).data_offset
    assert np.array_equal(np.asarray(F.read_data(head)), planted)
    assert F.data_digest(head) == F.data_digest(F.read_header(p))


def test_rename_takes_any_names_not_just_numbers(card, tmp_path):
    """Channel 0 may be a reference tap rather than a segment."""
    p, _ = card
    names = ["UC1"] + [str(s) for s in range(65, 80)]

    head = F.rename(p, tmp_path / "renamed.DAT", names)

    assert head.names == names


def test_a_renamed_channel_is_readable_by_its_new_name(card, tmp_path):
    p, planted = card
    head = F.rename(p, tmp_path / "renamed.DAT", SEGMENTS)

    got = np.asarray(F.read_data(head, channels=["67"]))

    assert np.array_equal(got[:, 0], planted[:, 3])


def test_asking_for_a_channel_that_is_not_there_is_refused(card, tmp_path):
    p, _ = card
    head = F.rename(p, tmp_path / "renamed.DAT", SEGMENTS)

    with pytest.raises(F.FamosError, match="no channel named '3'"):
        F.read_data(head, channels=["3"])


def test_rename_leaves_the_source_alone(card, tmp_path):
    p, _ = card
    before = p.read_bytes()

    F.rename(p, tmp_path / "renamed.DAT", SEGMENTS)

    assert p.read_bytes() == before


def test_rename_will_not_overwrite_its_own_input(card):
    p, _ = card

    with pytest.raises(F.FamosError, match="refusing to write over"):
        F.rename(p, p, SEGMENTS)


def test_a_file_without_name_keys_is_refused(tmp_path):
    p = tmp_path / "nameless.DAT"
    plant(p, ["0", "1"], with_names=False)

    with pytest.raises(F.FamosError, match="no \\|CN key"):
        F.rename(p, tmp_path / "out.DAT", ["64", "65"])


def test_a_rewritten_key_declares_its_own_new_length():
    """The length field is what every later key's position depends on."""
    built = F.build_cn_key(1, "64", "")

    assert built == b"|CN,1,13,1,0,0,2,64,0,;"
    assert len(built) == len(b"|CN,1,13,") + 13 + 1


def test_renaming_twice_lands_where_renaming_once_would(card, tmp_path):
    """Names get longer and shorter; the header must stay self-consistent."""
    p, planted = card
    once = F.rename(p, tmp_path / "a.DAT", SEGMENTS)
    twice = F.rename(once.path, tmp_path / "b.DAT",
                     [f"segment_{s}" for s in range(64, 80)])
    back = F.rename(twice.path, tmp_path / "c.DAT", SEGMENTS)

    assert back.names == SEGMENTS
    assert np.array_equal(np.asarray(F.read_data(back)), planted)


# ---------------------------------------------------------------------------
# The CSV export
# ---------------------------------------------------------------------------


def test_the_export_carries_the_new_names_and_the_samples(card, tmp_path):
    p, planted = card
    head = F.rename(p, tmp_path / "renamed.DAT", SEGMENTS)
    out = tmp_path / "data.csv"

    F.export_csv(head, out, channels=["64", "79"])

    rows = list(csv.reader(out.open(encoding="utf-8")))
    assert rows[0] == ["time_s", "64", "79"]
    assert float(rows[1][1]) == pytest.approx(planted[0, 0])
    assert float(rows[1][2]) == pytest.approx(planted[0, 15])
    assert float(rows[2][0]) == pytest.approx(DT)


def test_the_export_can_be_thinned(card, tmp_path):
    p, _ = card
    head = F.rename(p, tmp_path / "renamed.DAT", SEGMENTS)
    out = tmp_path / "data.csv"

    n = F.export_csv(head, out, step=8)

    assert n == 8                                   # 64 frames, every 8th
    assert len(list(csv.reader(out.open(encoding="utf-8")))) == n + 1


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_cli_renames_end_to_end(card, tmp_path, capsys):
    p, _ = card
    out = tmp_path / "renamed.DAT"

    rc = F.main(["apply", str(p), "--names", "64-79", "--out", str(out)])

    assert rc == 0
    assert "identical to the source" in capsys.readouterr().out
    assert F.read_header(out).names == SEGMENTS


def test_a_dry_run_writes_nothing(card, tmp_path):
    p, _ = card
    out = tmp_path / "renamed.DAT"

    rc = F.main(["apply", str(p), "--names", "64-79", "--out", str(out),
                 "--dry-run"])

    assert rc == 0
    assert not out.exists()


def test_the_cli_reports_a_refusal_as_a_failure(card, tmp_path, capsys):
    p, _ = card
    out = tmp_path / "renamed.DAT"

    rc = F.main(["apply", str(p), "--names", "64-70", "--out", str(out)])

    assert rc == 2
    assert "7 names given for 16 channels" in capsys.readouterr().err
    assert not out.exists()


def test_verify_against_the_original_compares_the_samples(card, tmp_path,
                                                          capsys):
    p, _ = card
    out = tmp_path / "renamed.DAT"
    F.rename(p, out, SEGMENTS)

    rc = F.main(["verify", str(out), "--against", str(p)])

    assert rc == 0
    assert "samples    : identical" in capsys.readouterr().out
