"""Reading the FAMOS key structure instead of searching it.

Every FAMOS key is  |<KK>,<KeyVersion>,<Length>,<Length bytes>;  so the file
says where each key ends. That makes the structure of a 4 GB card readable
in milliseconds -- a data block declares its size, so it is seeked past.

It also removes a whole class of error. Searching a header with regular
expressions works only for the dialect it was written against: against a
standard header, where every key's third field is a BYTE COUNT, the
pipeline reader's `\\|CD,\\d+,([\\d.eE+-]+)` captures the LENGTH of the |CD
block and reports 1/21 Hz as the sample rate. It does not fail -- it returns
a plausible number that is not the sample rate.
"""

from __future__ import annotations

import numpy as np
import pytest

import famos_keys as FK


def key(name, ver, *fields) -> bytes:
    body = ",".join(str(f) for f in fields).encode("latin-1")
    return f"|{name},{ver},{len(body)},".encode("latin-1") + body + b";"


def standard_card(path, n=500, fs=100_000.0, channels=("UC1", "1", "2")):
    """A standard-layout file: one component per channel, each with its |CS."""
    parts = [key("CF", 2, 1), b"\r\n", key("CK", 1, 1, 1), b"\r\n",
             key("NO", 1, 1, 0, 12, "Messung_1"), b"\r\n",
             key("CG", 1, 1, 1, len(channels)), b"\r\n"]
    for i, name in enumerate(channels):
        payload = (np.arange(n, dtype="<f4") + i).tobytes()
        parts += [key("CC", 1, 1, 1), b"\r\n",
                  key("CD", 2, 1.0 / fs, 1, 0, "s", 0, 0), b"\r\n",
                  key("CR", 1, 1, 1.0, 0.0, 0, "V"), b"\r\n",
                  key("CN", 1, 0, 0, 0, len(name), name), b"\r\n",
                  key("CP", 1, 1, 4, 7, 0, 0, 0, 0), b"\r\n",
                  f"|CS,1,{len(payload)},".encode("latin-1"), payload, b";",
                  b"\r\n"]
    path.write_bytes(b"".join(parts))
    return path


def test_the_whole_structure_is_walked(tmp_path) -> None:
    keys = list(FK.walk(standard_card(tmp_path / "c.DAT")))
    counts = {}
    for entry in keys:
        counts[entry["key"]] = counts.get(entry["key"], 0) + 1
    assert counts["CS"] == 3 and counts["CN"] == 3
    assert counts["CF"] == 1 and counts["NO"] == 1
    assert all(entry["terminated"] for entry in keys)


def test_the_real_sample_interval_is_visible(tmp_path) -> None:
    """What the regex reader gets wrong, read correctly.

    The regex captures |CD's length field; the walker hands back the content,
    whose first field is the actual dx.
    """
    keys = list(FK.walk(standard_card(tmp_path / "c.DAT", fs=100_000.0)))
    cd = next(e for e in keys if e["key"] == "CD")
    dx = float(cd["content"].split(",")[0])
    assert 1.0 / dx == pytest.approx(100_000.0)

    import re
    raw = (tmp_path / "c.DAT").read_bytes().decode("latin-1")
    captured = re.search(r"\|CD,\d+,([\d.eE+-]+)", raw).group(1)
    assert float(captured) != pytest.approx(dx), (
        "this test is only meaningful if the regex really does capture "
        "something else; it captured the same value")


def test_channel_names_come_from_CN(tmp_path) -> None:
    keys = list(FK.walk(standard_card(tmp_path / "c.DAT",
                                      channels=("UC1", "7", "42"))))
    names = [e["content"].split(",")[-1] for e in keys if e["key"] == "CN"]
    assert names == ["UC1", "7", "42"]


def test_a_large_payload_is_seeked_past_not_read(tmp_path) -> None:
    """A 4 GB card must cost no more than a small card to enumerate."""
    path = tmp_path / "big.DAT"
    standard_card(path, n=400_000)
    keys = list(FK.walk(path, preview=64))
    data = [e for e in keys if e["key"] == "CS"]
    assert len(data) == 3
    assert all(e["truncated"] for e in data)
    assert all(len(e["content"]) <= 64 for e in data), (
        "the walker must preview a data block, never materialise it")
    assert data[0]["length"] == 400_000 * 4


def test_several_payload_blocks_mean_contiguous_channels(tmp_path) -> None:
    """Which decides how a channel is read: contiguous, not interleaved."""
    keys = list(FK.walk(standard_card(tmp_path / "c.DAT")))
    assert len([e for e in keys if e["key"] == "CS"]) == 3


def test_a_dialect_whose_lengths_are_not_byte_counts_is_refused(tmp_path):
    """It says which key and which byte, rather than guessing.

    The pipeline's other dialect writes fields where the standard writes a
    length, so the terminator does not land where the length says it will.
    That is a fact worth reporting precisely -- it is what tells the two
    layouts apart.
    """
    path = tmp_path / "other.DAT"
    path.write_bytes(b"|CF,2,1,1;|CK,1,3,1,1;|CD,2,0.0001,1;" + b"\x00" * 100)
    with pytest.raises(FK.FamosStructureError) as excinfo:
        list(FK.walk(path))
    message = str(excinfo.value)
    assert "|CD" in message, "the message must name the key that disagreed"
    assert "byte count" in message


def test_a_file_that_is_not_famos_at_all_is_refused(tmp_path) -> None:
    path = tmp_path / "hdf.DAT"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 200)
    with pytest.raises(FK.FamosStructureError) as excinfo:
        list(FK.walk(path))
    assert "expected '|'" in str(excinfo.value)


def test_the_command_runs_and_reports_the_payload(tmp_path, capsys) -> None:
    standard_card(tmp_path / "c.DAT")
    assert FK.main([str(tmp_path / "c.DAT")]) == 0
    out = capsys.readouterr().out
    assert "3 |CS block(s)" in out
    assert "not interleaved" in out
    assert "|CN" in out
