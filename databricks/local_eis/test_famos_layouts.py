"""Two FAMOS layouts reach this pipeline, and they share almost nothing.

The imc-written cards from the rig and a DASYLab FAMOS export are both
`.DAT`, both start `|CF,`, and are otherwise different files. Reading one
with the other's assumptions is not an error -- it is a header full of
plausible wrong numbers and a data block of alternating zeros and nonsense,
which still plots. So each layout is pinned here against a file built to
match a real one byte for byte.
"""

from __future__ import annotations

import numpy as np
import pytest

from eis_local import FamosFile


FS_DASY = 100_000.0


def dasylab_file(path, n=2000, ch=16, names=None):
    """A DASYLab FAMOS export, rebuilt from a real header dump.

    Keeps the three things that break a parser written for the imc cards:
    proper |KK,<ver>,<len>,<body>; keys, space-padded integers, and
    length-prefixed channel names in per-channel |CN keys.
    """
    names = names or [str(c) for c in range(ch)]
    hdr = bytearray(b"|CF,2,1,1;|CK,1,3,1,1;")
    hdr += b"|NO,1,27,0,7,DASYLab,12,V 14.2.0.889;"
    for c in range(ch):
        hdr += b"|CD,1,25,1.000000e-005,1,1,s,0,0,0;"
        cp = f"1,8,8,64,0,{c * 8},1,{(ch - 1) * 8}"
        hdr += b"|CP,1," + str(len(cp)).encode() + b"," + cp.encode() + b";"
        cb = "1,0,%d,1,0,         0,0,         0,1,0,0," % (c + 1)
        hdr += b"|Cb,1," + str(len(cb)).encode() + b"," + cb.encode() + b";"
        hdr += b"|CR,1,10,0,0,0,1,0,;"
        cn = f"1,0,0,{len(names[c])},{names[c]},0,"
        hdr += b"|CN,1," + str(len(cn)).encode() + b"," + cn.encode() + b";"
    hdr += b"|CS,1,                   0,1,"          # space padded, length 0

    t = np.arange(n) / FS_DASY
    data = np.zeros((n, ch))
    data[:, 0] = 0.805 + 0.004 * np.sin(2 * np.pi * 298.3 * t)
    for c in range(1, ch):
        data[:, c] = 0.15 + 0.002 * c
    path.write_bytes(bytes(hdr) + data.astype("<f8").tobytes())
    return data


def test_a_dasylab_export_is_read_at_all(tmp_path):
    """It used to raise "incomplete FAMOS header" and stop there."""
    f = tmp_path / "d.DAT"
    dasylab_file(f)
    head = FamosFile(f)

    assert head.n_ch == 16
    assert head.names == [str(c) for c in range(16)]


def test_the_samples_are_float64_not_float32(tmp_path):
    """The failure with no error message.

    float64 data read as float32 gives alternating zeros and large nonsense
    -- 0.805 V becomes [0.0, 1.83]. Nothing raises; the plate map just shows
    a different cell.
    """
    f = tmp_path / "d.DAT"
    truth = dasylab_file(f)
    head = FamosFile(f)

    assert head.dtype == np.dtype("<f8")
    got = head.channel("0")
    assert got[0] == pytest.approx(truth[0, 0], abs=1e-12)
    assert 0.75 < got[0] < 0.85, "a cell voltage, not a reinterpreted one"


def test_the_sample_rate_is_the_interval_not_the_key_length(tmp_path):
    """|CD,1,25,1.000000e-005 -- 25 is the BODY LENGTH.

    Reading it as the interval gives 1/25 Hz, and every dwell window derived
    from it lands somewhere unrelated.
    """
    f = tmp_path / "d.DAT"
    dasylab_file(f)
    assert FamosFile(f).fs == pytest.approx(FS_DASY)


def test_the_channel_count_is_not_taken_from_a_length_field(tmp_path):
    """|CR,1,10,... -- 10 is a length. The file has 16 channels."""
    f = tmp_path / "d.DAT"
    dasylab_file(f, ch=16)
    assert FamosFile(f).n_ch == 16


def test_a_two_digit_name_survives_the_length_prefix(tmp_path):
    """Names are length-prefixed, so "10" is two bytes, not a parse error.

    Walking to the next comma works by luck for "0".."9" and then reads the
    wrong field -- which is exactly where the rename tool crashed.
    """
    f = tmp_path / "d.DAT"
    names = ["UC2", "UC1"] + [str(n) for n in range(1, 15)]
    dasylab_file(f, names=names)
    head = FamosFile(f)

    assert head.names == names
    assert head.segment_names == [str(n) for n in range(1, 15)]
    assert head.uc_names == ["UC2", "UC1"]


def test_the_space_padded_cs_length_does_not_defeat_the_offset(tmp_path):
    """|CS,1,<19 spaces>0, -- the declared length is a placeholder."""
    f = tmp_path / "d.DAT"
    dasylab_file(f, n=2000, ch=16)
    head = FamosFile(f)

    assert head.n_samples == 2000, "the count comes from the file size"


def test_a_file_that_is_neither_layout_still_raises(tmp_path):
    """Tolerance must not become "read anything and hope"."""
    f = tmp_path / "junk.DAT"
    f.write_bytes(b"|CF,2,1,1;" + b"\x00" * 400)
    with pytest.raises(ValueError, match="incomplete FAMOS header"):
        FamosFile(f)
