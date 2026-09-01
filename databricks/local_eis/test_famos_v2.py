"""Reading the standard FAMOS layout, and choosing between the two.

The simplified reader does not fail on a standard-layout file. It matches
the wrong fields -- every key's third field is a byte count there, so the
value it wants sits one place along -- and returns 1/21 Hz as the sample
rate. So the dispatch cannot be "try one, fall back if it raises": it has
to be "accept the one whose answer is self-consistent".
"""

from __future__ import annotations

import numpy as np
import pytest

import eis_local as E
from famos_keys import FamosStructureError
from famos_v2 import FamosFileV2


def key(name, ver, *fields) -> bytes:
    body = ",".join(str(f) for f in fields).encode("latin-1")
    return f"|{name},{ver},{len(body)},".encode("latin-1") + body + b";"


def contiguous(path, channels=("UC1", "1", "2"), n=500, fs=100_000.0,
               dtype="<f4", fmt=7, stamp=True):
    """Standard layout: one |CS per component, channels not interleaved."""
    parts = [key("CF", 2, 1), b"\r\n", key("CK", 1, 1, 1), b"\r\n",
             key("NO", 1, 1, 0, 12, "Messung_1"), b"\r\n"]
    if stamp:
        parts += [key("NT", 1, 16, 7, 2025, 7, 45, 46.0), b"\r\n"]
    parts += [key("CG", 1, 1, 1, len(channels)), b"\r\n"]
    width = np.dtype(dtype).itemsize
    for i, name in enumerate(channels):
        payload = (np.arange(n) + i * 1000).astype(dtype).tobytes()
        parts += [key("CC", 1, 1, 1), b"\r\n",
                  key("CD", 2, 1.0 / fs, 1, 0, "s", 0, 0), b"\r\n",
                  key("CR", 1, 1, 1.0, 0.0, 0, "V"), b"\r\n",
                  key("CN", 1, 0, 0, 0, len(name), name), b"\r\n",
                  key("CP", 1, 1, width, fmt, 0, 0, 0, 0), b"\r\n",
                  f"|CS,1,{len(payload)},".encode("latin-1"), payload, b";",
                  b"\r\n"]
    path.write_bytes(b"".join(parts))
    return path


def interleaved(path, channels=("UC1", "1", "2"), n=500, fs=100_000.0,
                dtype="<f8", fmt=8):
    """Standard keys, but one shared |CS: the components are interleaved."""
    width = np.dtype(dtype).itemsize
    parts = [key("CF", 2, 1), b"\r\n", key("CK", 1, 1, 1), b"\r\n",
             key("CG", 1, 1, 1, len(channels)), b"\r\n"]
    for i, name in enumerate(channels):
        parts += [key("CC", 1, 1, 1), b"\r\n",
                  key("CD", 2, 1.0 / fs, 1, 0, "s", 0, 0), b"\r\n",
                  key("CN", 1, 0, 0, 0, len(name), name), b"\r\n",
                  # field 5 is this channel's byte offset inside the row
                  key("CP", 1, i + 1, width, fmt, 0, 0, i * width, 0, 0),
                  b"\r\n"]
    block = np.zeros((n, len(channels)), dtype=dtype)
    for i in range(len(channels)):
        block[:, i] = np.arange(n) + i * 1000
    raw = block.tobytes()
    parts += [f"|CS,1,{len(raw)},".encode("latin-1"), raw, b";"]
    path.write_bytes(b"".join(parts))
    return path


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def test_a_contiguous_card_reads(tmp_path) -> None:
    fam = FamosFileV2(contiguous(tmp_path / "c.DAT"))
    assert fam.fs == pytest.approx(100_000.0)
    assert fam.names == ["UC1", "1", "2"]
    assert fam.n_samples == 500
    assert not fam.interleaved
    assert np.allclose(fam.channel("1"), np.arange(500) + 1000)


def test_an_interleaved_card_reads(tmp_path) -> None:
    fam = FamosFileV2(interleaved(tmp_path / "i.DAT"))
    assert fam.interleaved and fam.n_samples == 500
    assert np.allclose(fam.channel("2"), np.arange(500) + 2000)


def test_the_layout_is_read_from_the_block_count_not_assumed(tmp_path) -> None:
    """One |CS means shared; one each means contiguous.

    Nothing else tells them apart, and they need completely different reads.
    Assuming interleaved on a contiguous file returns numbers -- every
    channel a stride through the first channel's samples.
    """
    assert FamosFileV2(contiguous(tmp_path / "c.DAT")).interleaved is False
    assert FamosFileV2(interleaved(tmp_path / "i.DAT")).interleaved is True


def test_the_trigger_stamp_survives(tmp_path) -> None:
    """bronze reads fam.start_time; a reader without it crashes the run."""
    fam = FamosFileV2(contiguous(tmp_path / "c.DAT"))
    assert fam.start_time is not None
    assert fam.start_time.year == 2025 and fam.start_time.minute == 45


def test_a_bounded_read_is_supported(tmp_path) -> None:
    """bronze calls channel(name, 0, n); a two-argument reader is a TypeError."""
    fam = FamosFileV2(contiguous(tmp_path / "c.DAT"))
    assert np.allclose(fam.channel("1", 0, 10), np.arange(10) + 1000)
    assert np.allclose(fam.channel("1", 100, 110), np.arange(100, 110) + 1000)
    assert len(fam.channel("1", 0, 10_000)) == 500


def test_names_with_a_comma_are_not_torn_in_half(tmp_path) -> None:
    """The name is read by its declared length, not by splitting on commas."""
    fam = FamosFileV2(contiguous(tmp_path / "c.DAT",
                                 channels=("UC1", "seg,7", "2")))
    assert "seg,7" in fam.names


# ---------------------------------------------------------------------------
# what it refuses
# ---------------------------------------------------------------------------

def test_an_unknown_sample_format_is_refused_not_defaulted(tmp_path) -> None:
    """Eight bytes is float64 or int64; reading one as the other is silent."""
    path = contiguous(tmp_path / "odd.DAT", dtype="<f8", fmt=99)
    with pytest.raises(FamosStructureError) as excinfo:
        FamosFileV2(path).channel("UC1")
    assert "not a combination this reader will guess at" in str(excinfo.value)


def test_mixed_sample_rates_are_refused(tmp_path) -> None:
    """There is no single fs for a card whose components disagree."""
    parts = [key("CF", 2, 1), key("CG", 1, 1, 1, 2)]
    for i, (name, fs) in enumerate((("UC1", 100_000.0), ("1", 10_000.0))):
        payload = np.zeros(100, dtype="<f4").tobytes()
        parts += [key("CC", 1, 1, 1), key("CD", 2, 1.0 / fs, 1, 0, "s", 0, 0),
                  key("CN", 1, 0, 0, 0, len(name), name),
                  key("CP", 1, 1, 4, 7, 0, 0, 0, 0),
                  f"|CS,1,{len(payload)},".encode("latin-1"), payload, b";"]
    (tmp_path / "mix.DAT").write_bytes(b"".join(parts))
    with pytest.raises(FamosStructureError) as excinfo:
        FamosFileV2(tmp_path / "mix.DAT")
    assert "different sample rates" in str(excinfo.value)


def test_a_non_identity_calibration_is_reported_not_applied(tmp_path) -> None:
    """A factor read from a guessed field position rescales every impedance.

    The shunt calibration is meant to be the only absolute scale in the
    chain, so a second one applied on a guess is exactly the failure the
    whole design avoids.
    """
    parts = [key("CF", 2, 1), key("CG", 1, 1, 1, 1)]
    payload = np.ones(100, dtype="<f4").tobytes()
    parts += [key("CC", 1, 1, 1), key("CD", 2, 1e-5, 1, 0, "s", 0, 0),
              key("CR", 1, 1, 2.5, 0.0, 0, "V"),
              key("CN", 1, 0, 0, 0, 3, "UC1"),
              key("CP", 1, 1, 4, 7, 0, 0, 0, 0),
              f"|CS,1,{len(payload)},".encode("latin-1"), payload, b";"]
    (tmp_path / "cal.DAT").write_bytes(b"".join(parts))
    fam = FamosFileV2(tmp_path / "cal.DAT")
    assert any("REPORTED, NOT APPLIED" in n for n in fam.notes)
    assert np.allclose(fam.channel("UC1"), 1.0), "the factor must not be applied"


# ---------------------------------------------------------------------------
# choosing the reader
# ---------------------------------------------------------------------------

def test_the_simplified_reader_does_not_raise_on_a_standard_file(tmp_path):
    """The premise of the whole dispatch, pinned.

    If this ever starts raising, "try v1, fall back on ValueError" would
    become safe -- and until it does, it is not.
    """
    path = interleaved(tmp_path / "i.DAT")
    try:
        fam = E.FamosFile(path)
    except ValueError:
        pytest.skip("v1 now raises on this file; the trap no longer applies")
    assert fam.fs != pytest.approx(100_000.0), (
        f"v1 returned {fam.fs} Hz, which happens to be right; this test "
        f"needs a file where it is wrong")


def test_the_dispatch_picks_by_self_consistency(tmp_path) -> None:
    """Not by which reader failed to raise."""
    fam = E.open_famos(interleaved(tmp_path / "i.DAT"))
    assert isinstance(fam, FamosFileV2)
    assert fam.fs == pytest.approx(100_000.0)
    assert fam.names == ["UC1", "1", "2"]


def test_a_simplified_card_still_goes_to_the_simplified_reader(tmp_path):
    """The existing recordings must not be re-routed."""
    n_ch, n, fs = 6, 400, 10_000.0
    names = ["UC1"] + [str(i) for i in range(1, n_ch)]
    cp = ",".join(f"7,32,{nm}" for nm in names)
    data = np.zeros((n, n_ch), dtype="<f4")
    (tmp_path / "v1.DAT").write_bytes(
        (f"|CF,2,1,1;|CK,1,3,1,1;|CD,2,{1.0 / fs},1;|CR,1,{n_ch},1,0,1;"
         f"|CP,{cp};|CS,1,{data.nbytes},").encode("latin-1") + data.tobytes())
    fam = E.open_famos(tmp_path / "v1.DAT")
    assert isinstance(fam, E.FamosFile)
    assert fam.fs == pytest.approx(10_000.0) and fam.n_ch == n_ch


def test_a_file_neither_can_read_names_both_attempts(tmp_path) -> None:
    path = tmp_path / "other.DAT"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 4000)
    with pytest.raises(ValueError) as excinfo:
        E.open_famos(path)
    message = str(excinfo.value)
    assert "simplified layout" in message and "standard layout" in message
    assert "famos_keys.py" in message


def test_the_reader_describes_what_it_decided(tmp_path) -> None:
    """A layout decision the operator cannot see is one they cannot check."""
    text = FamosFileV2(contiguous(tmp_path / "c.DAT")).describe()
    assert "100000 Hz" in text and "contiguous" in text


# ---------------------------------------------------------------------------
# the DASYLab 14.2 layout the 2612025 cards actually use
# ---------------------------------------------------------------------------

def dasylab(path, names=("UC2", "UC1", "1", "2"), n=400, fs=100_000.0):
    """A byte-faithful replica of the header dumped from a 150 A card.

        |NO,1,27,0,7,DASYLab,12,V 14.2.0.889
        |CD,1,25,1.000000e-005,1,1,s,0,0,0
        |CP,1,18,1,8,8,64,0,0,1,120        <- offset 0, then 8, then 16
        |Cb,1,40,1,0,1,1,0,3946840064,0,3946840064,1,0,0,
        |CR,1,10,0,0,0,1,0,                <- transform flag 0
        |CN,1,14,1,0,0,3,UC2,0,
        |CS,1,3946840066,1,<raw>           <- note the "1," before the data
    """
    def k(name, ver, body):
        b = body.encode("latin-1")
        return f"|{name},{ver},{len(b)},".encode("latin-1") + b + b";"

    itemsize, n_ch = 8, len(names)
    row, buf = n_ch * itemsize, n * n_ch * itemsize
    parts = [k("CF", 2, "1"), k("CK", 1, "1,1"),
             k("NO", 1, "0,7,DASYLab,12,V 14.2.0.889"),
             k("CB", 1, "1,7,Messung,0,")]
    for i, nm in enumerate(names):
        parts += [k("CG", 1, "1,1,1"),
                  k("CD", 1, "1.000000e-005,1,1,s,0,0,0"),
                  k("NT", 1, "27, 8,2026,11,36,19.00"),
                  k("CC", 1, "1,1"),
                  k("CP", 1, f"{i+1},8,8,64,0,{i*itemsize},1,{row-itemsize}"),
                  k("Cb", 1, f"1,0,{i+1},1,0,{buf},0,{buf},1,0,0,"),
                  k("CR", 1, "0,0,0,1,0,"),
                  k("CN", 1, f"1,0,0,{len(nm)},{nm},0,")]
    block = np.zeros((n, n_ch), dtype="<f8")
    for i in range(n_ch):
        block[:, i] = np.arange(n) + i * 1000
    payload = b"1," + block.tobytes()
    parts += [f"|CS,1,{len(payload)},".encode("latin-1"), payload, b";"]
    path.write_bytes(b"".join(parts))
    return path


def test_the_dasylab_card_reads_correctly(tmp_path) -> None:
    fam = FamosFileV2(dasylab(tmp_path / "d.DAT"))
    assert fam.fs == pytest.approx(100_000.0)
    assert fam.names == ["UC2", "UC1", "1", "2"]
    assert fam.n_samples == 400
    assert fam.interleaved
    assert fam.start_time.year == 2026 and fam.start_time.day == 27
    for i, name in enumerate(fam.names):
        assert np.allclose(fam.channel(name, 0, 4), np.arange(4) + i * 1000), (
            f"{name} did not read its own samples")


def test_the_samples_key_index_prefix_is_skipped(tmp_path) -> None:
    """|CS content is "<index>,<raw>", so the payload is not at its start.

    On the real cards that prefix is exactly the two bytes by which the
    declared |CS length exceeds the |Cb buffer length. Two bytes of skew on
    a float64 stream is not an error -- it shifts every sample of every
    channel by a fraction of a value and returns noise.
    """
    fam = FamosFileV2(dasylab(tmp_path / "d.DAT"))
    assert fam.channel("UC2", 0, 1)[0] == pytest.approx(0.0)
    assert not any("left over" in n for n in fam.notes), fam.notes


def test_the_channel_name_is_not_the_first_number_that_fits(tmp_path) -> None:
    """|CN,1,0,0,3,UC2,0 also contains "1" followed by "0", one char long.

    A scan that takes the first such pair names every channel "0".
    """
    fam = FamosFileV2(dasylab(tmp_path / "d.DAT"))
    assert "0" not in fam.names
    assert fam.names[0] == "UC2"


def test_a_transform_flag_of_zero_is_not_a_calibration(tmp_path) -> None:
    """DASYLab writes |CR,1,10,0,0,0,1,0 -- flag 0, factor and offset unused.

    Reading those as a scaling reports "factor 0" on every channel of an
    ordinary file, and applying it would zero the plate.
    """
    fam = FamosFileV2(dasylab(tmp_path / "d.DAT"))
    assert all(c.identity_calibration for c in fam.components)
    assert not any("NOT APPLIED" in n for n in fam.notes), fam.notes


def test_the_column_comes_from_the_cp_offset(tmp_path) -> None:
    """Assuming index*itemsize is right only while the components are ordered.

    Here they are not: the file lists UC2 first but gives it the second slot
    in the row. Reading by list position returns its neighbour's samples --
    silently, because both are perfectly ordinary numbers.
    """
    def k(name, ver, body):
        b = body.encode("latin-1")
        return f"|{name},{ver},{len(b)},".encode("latin-1") + b + b";"

    names, n, itemsize = ["UC2", "UC1"], 64, 8
    row = len(names) * itemsize
    slot = {"UC2": 1, "UC1": 0}            # deliberately not list order
    parts = [k("CF", 2, "1"), k("CK", 1, "1,1")]
    for i, nm in enumerate(names):
        parts += [k("CG", 1, "1,1,1"),
                  k("CD", 1, "1.000000e-005,1,1,s,0,0,0"),
                  k("CC", 1, "1,1"),
                  k("CP", 1, f"{i+1},8,8,64,0,{slot[nm] * itemsize},1,0"),
                  k("Cb", 1, f"1,0,{i+1},1,0,{n * row},0,{n * row},1,0,0,"),
                  k("CR", 1, "0,0,0,1,0,"),
                  k("CN", 1, f"1,0,0,{len(nm)},{nm},0,")]
    block = np.zeros((n, len(names)), dtype="<f8")
    for column in range(len(names)):
        block[:, column] = np.arange(n) + column * 1000
    payload = b"1," + block.tobytes()
    parts += [f"|CS,1,{len(payload)},".encode("latin-1"), payload, b";"]
    (tmp_path / "shuffled.DAT").write_bytes(b"".join(parts))

    fam = FamosFileV2(tmp_path / "shuffled.DAT")
    assert np.allclose(fam.channel("UC2", 0, 2), [1000, 1001]), (
        "UC2 holds the second slot and must read it")
    assert np.allclose(fam.channel("UC1", 0, 2), [0, 1])


def test_the_dispatch_routes_a_dasylab_card_to_v2(tmp_path) -> None:
    fam = E.open_famos(dasylab(tmp_path / "d.DAT"))
    assert isinstance(fam, FamosFileV2)
    assert fam.fs == pytest.approx(100_000.0) and fam.names[1] == "UC1"
