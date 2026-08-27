"""Segment numbers on FAMOS channels: does it map, and does it refuse?

The mistake this module exists to prevent is silent -- a spectrum filed
under segment 3 that belongs to segment 67 reads as a perfectly good
measurement.  So the tests weigh the refusals as heavily as the mappings,
and `relabel` is held to the only standard that matters for a rewrite of a
measurement file: not one sample byte moves.
"""

from __future__ import annotations

import numpy as np
import pytest

import famos_segments as F


DT = 2e-5


def key(name: bytes, version: int, body: bytes) -> bytes:
    return b"|%s,%d,%d,%s;\r\n" % (name, version, len(body), body)


def plant(path, names, n=64, dtype="<f8", width=8, fmt=8, stride=None):
    """A FAMOS file with a KNOWN channel table, for the parser to recover."""
    nch = len(names)
    frame = stride if stride else width * nch

    out = [key(b"CF", 2, b"1"), key(b"CK", 1, b"1,1")]
    for i, name in enumerate(names):
        nb = name.encode("latin-1")
        out.append(key(b"CD", 1, b"%.6e,1,1,s,0,0,0" % DT))
        out.append(key(b"CP", 1, b"1,%d,%d,%d,0,%d,1,%d"
                       % (width, fmt, width * 8, i * width, frame - width)))
        out.append(key(b"Cb", 1, b"1,0,%d,1,0,%d,0,%d,1,0,0," % (i + 1, n * frame, n * frame)))
        out.append(key(b"CR", 1, b"0,0,0,1,0,"))
        out.append(key(b"CN", 1, b"1,0,0,%d,%s,0," % (len(nb), nb)))

    rng = np.random.default_rng(0)
    data = np.zeros((n, nch))
    for c in range(nch):
        data[:, c] = 0.15 + 0.01 * rng.standard_normal(n)
    blob = data.astype(dtype).tobytes()

    body = b"1," + blob
    path.write_bytes(b"".join(out) + b"|CS,1,%d,%s;" % (len(body), body))
    return data


# ---------------------------------------------------------------------------
# The header
# ---------------------------------------------------------------------------


def test_the_channel_table_comes_out_of_the_keys(tmp_path):
    p = tmp_path / "2026_08_27_KANAL_6479RO.DDF_1.DAT"
    plant(p, [str(i) for i in range(16)])

    head = F.read_header(p)

    assert head.n_channels == 16
    assert head.numpy_dtype() == "<f8"
    assert head.frame_bytes == 128
    assert head.n_frames == 64
    assert [c.byte_offset for c in head.channels] == list(range(0, 128, 8))
    assert [c.label for c in head.channels] == [str(i) for i in range(16)]
    assert head.fs == pytest.approx(1.0 / DT)


def test_the_samples_are_where_the_header_says_they_are(tmp_path):
    p = tmp_path / "2026_08_27_KANAL_6479RO.DDF_1.DAT"
    planted = plant(p, [str(i) for i in range(16)])

    head = F.read_header(p)
    mm = np.memmap(p, dtype=head.numpy_dtype(), mode="r",
                   offset=head.data_offset,
                   shape=(head.n_frames, head.n_channels))

    assert np.array_equal(np.asarray(mm), planted)


def test_a_name_containing_a_comma_survives(tmp_path):
    """The reason the keys are parsed by declared length, not split on ','."""
    p = tmp_path / "card.DAT"
    plant(p, ["U,ref", "1;2", "plain"])

    assert [c.label for c in F.read_header(p).channels] == \
        ["U,ref", "1;2", "plain"]


def test_a_frame_that_is_not_a_plain_interleave_is_refused(tmp_path):
    """A stride wider than the samples means padding this script cannot read."""
    p = tmp_path / "card.DAT"
    plant(p, ["0", "1"], stride=64)

    with pytest.raises(F.FamosError, match="not a plain"):
        F.read_header(p)


def test_a_file_that_is_not_famos_is_refused(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_bytes(b"DASYLab V14\r\nchannels\t16\r\n" + b"\x00" * 512)

    with pytest.raises(F.FamosError, match="not a FAMOS file"):
        F.read_header(p)


# ---------------------------------------------------------------------------
# Where the segment numbers come from
# ---------------------------------------------------------------------------


def test_the_file_name_carries_the_range():
    assert F.segments_from_filename("2026_08_27_KANAL_6479RO2612025_60A.DAT",
                                    16) == list(range(64, 80))
    assert F.segments_from_filename("KANAL_0116_card.DAT", 16) == \
        list(range(1, 17))
    assert F.segments_from_filename("KANAL_64-79.DAT", 16) == \
        list(range(64, 80))


def test_a_range_that_does_not_fit_the_channels_is_not_evidence():
    # 64..79 is sixteen segments; a file with eight channels is not this card.
    assert F.segments_from_filename("KANAL_6479RO.DAT", 8) is None
    # An odd run of digits has no two-bound reading.
    assert F.segments_from_filename("KANAL_647RO.DAT", 16) is None
    assert F.segments_from_filename("2026_08_27_RO2612025.DAT", 16) is None


def test_an_unreadable_name_refuses_rather_than_guessing(tmp_path):
    p = tmp_path / "card.DAT"
    plant(p, [str(i) for i in range(16)])

    with pytest.raises(F.FamosError, match="--segments 64-79"):
        F.resolve_segments(F.read_header(p))


def test_the_segment_spec_reads_ranges_and_lists():
    assert F.parse_segment_spec("64-79") == list(range(64, 80))
    assert F.parse_segment_spec("1,2,5-7") == [1, 2, 5, 6, 7]
    with pytest.raises(F.FamosError, match="backwards"):
        F.parse_segment_spec("79-64")
    with pytest.raises(F.FamosError, match="cannot read"):
        F.parse_segment_spec("sixty-four")


def test_a_count_mismatch_is_refused_not_zipped_short(tmp_path):
    """zip() would pair the first seven and drop nine channels in silence."""
    p = tmp_path / "card.DAT"
    plant(p, [str(i) for i in range(16)])

    with pytest.raises(F.FamosError, match="refusing to pair"):
        F.resolve_segments(F.read_header(p), "64-70")


def test_reverse_pairs_the_highest_segment_first(tmp_path):
    p = tmp_path / "card.DAT"
    plant(p, [str(i) for i in range(16)])

    segs, _ = F.resolve_segments(F.read_header(p), "64-79", reverse=True)

    assert segs[0] == 79 and segs[-1] == 64


def test_the_map_is_positional_and_ascending(tmp_path):
    p = tmp_path / "2026_08_27_KANAL_6479RO2612025_60A.DDF_1.DAT"
    plant(p, [str(i) for i in range(16)])

    assert F.segment_map(p) == {str(i): 64 + i for i in range(16)}


# ---------------------------------------------------------------------------
# The corrected file
# ---------------------------------------------------------------------------


def test_relabel_moves_the_names_and_not_one_sample_byte(tmp_path):
    src = tmp_path / "2026_08_27_KANAL_6479RO2612025_60A.DDF_1.DAT"
    planted = plant(src, [str(i) for i in range(16)])
    out = tmp_path / "fixed.DAT"

    head = F.relabel(src, out)

    assert [c.label for c in head.channels] == [str(s) for s in range(64, 80)]
    # The names grew, so the data moved -- and must still read identically.
    assert head.data_offset > F.read_header(src).data_offset
    mm = np.memmap(out, dtype=head.numpy_dtype(), mode="r",
                   offset=head.data_offset,
                   shape=(head.n_frames, head.n_channels))
    assert np.array_equal(np.asarray(mm), planted)


def test_relabel_leaves_the_source_alone(tmp_path):
    src = tmp_path / "2026_08_27_KANAL_6479RO2612025_60A.DDF_1.DAT"
    plant(src, [str(i) for i in range(16)])
    before = src.read_bytes()

    F.relabel(src, tmp_path / "fixed.DAT")

    assert src.read_bytes() == before


def test_relabel_will_not_overwrite_its_own_input(tmp_path):
    src = tmp_path / "2026_08_27_KANAL_6479RO2612025_60A.DDF_1.DAT"
    plant(src, [str(i) for i in range(16)])

    with pytest.raises(F.FamosError, match="onto itself"):
        F.relabel(src, src)


def test_a_relabelled_file_names_its_segments_for_the_pipeline(tmp_path):
    """eis_local.FamosFile.segment_names keeps the bare-digit names only."""
    import re

    src = tmp_path / "2026_08_27_KANAL_6479RO2612025_60A.DDF_1.DAT"
    plant(src, [str(i) for i in range(16)])
    out = tmp_path / "fixed.DAT"

    head = F.relabel(src, out)
    kept = [c.label for c in head.channels if re.fullmatch(r"\d+", c.label)]

    assert kept == [str(s) for s in range(64, 80)]


def test_a_prefix_is_carried_into_the_names(tmp_path):
    src = tmp_path / "2026_08_27_KANAL_6479RO2612025_60A.DDF_1.DAT"
    plant(src, [str(i) for i in range(16)])

    head = F.relabel(src, tmp_path / "fixed.DAT", prefix="Seg")

    assert head.channels[0].label == "Seg64"


def test_a_rewritten_key_declares_its_own_new_length():
    """The length field is what every later key's position depends on."""
    built = F.build_cn_key(1, "64", "")

    assert built == b"|CN,1,13,1,0,0,2,64,0,;"
    assert len(built) == len(b"|CN,1,13,") + 13 + 1
