"""famos_convert: a FAMOS-v2 card (DASYLab 14, buffer column from the |CP
byte offset) rewritten as FAMOS v1 keeps every channel, its samples, the
slot order, the sample rate and the trigger time."""

from __future__ import annotations

import numpy as np
import pytest

import make_synth_famos as M
from eis_local import FamosFile, famos_start_time

import famos_convert

NAMES = ["UC2", "1", "2", "3", "4", "Temp1"]
COLUMNS = [0, 3, 1, 5, 2, 4]           # |CN order != buffer order
FS = 25000.0


def _v2(tmp_path, n=5003):
    rng = np.random.default_rng(3)
    d = rng.standard_normal((n, len(NAMES))) * [0.7, 0.01, 0.02, 0.03, 0.04, 60]
    p = M.write_v2(tmp_path / "RO1_Current_450A_Test_01_Karte_3.DAT", NAMES, d,
                   FS, columns=COLUMNS)
    raw = p.read_bytes()
    k = raw.index(b"|CK,1,3,1,1;") + len(b"|CK,1,3,1,1;")
    p.write_bytes(raw[:k] + b"|NT,1,22,22,10,2026, 9,41,07.00;" + raw[k:])
    return p, d


def test_v2_card_becomes_v1_with_everything_kept(tmp_path):
    src, d = _v2(tmp_path)
    a = FamosFile(src)
    assert type(a).__name__ == "FamosV2"
    out = famos_convert.convert(src, chunk_rows=1000, log=None)   # chunked
    b = FamosFile(out)
    assert type(b).__name__ == "FamosV1"
    assert b.n_samples == d.shape[0] and b.fs == pytest.approx(FS)
    for i, nm in enumerate(NAMES):
        assert np.allclose(b.channel(nm), d[:, i].astype(np.float32), rtol=0, atol=0)
    # the slot order is the v2 buffer order, not its |CN order
    assert sorted(b.names, key=b.position) == sorted(a.names, key=a.position)
    assert [b.position(nm) for nm in NAMES] == COLUMNS
    assert famos_start_time(out) == famos_start_time(src)


def test_the_original_is_never_overwritten(tmp_path):
    src, _ = _v2(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        famos_convert.convert(src, src, log=None)


def test_a_damaged_copy_is_caught(tmp_path):
    src, _ = _v2(tmp_path)
    out = famos_convert.convert(src, tmp_path / "v1.DAT", check=False, log=None)
    raw = bytearray(out.read_bytes())
    raw[-4 * 6 * 10:] = b"\0" * (4 * 6 * 10)          # zero the last rows
    out.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="differ"):
        famos_convert.verify(src, out, n_probe=2, probe_len=5003, log=None)
