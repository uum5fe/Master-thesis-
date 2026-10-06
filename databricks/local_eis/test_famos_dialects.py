"""Two FAMOS dialects, and the one that fails silently.

RO2611976-01 is written in one dialect, RO2612025-01 in another. The reader
knew only the first, and the second is not merely unsupported -- it is
DANGEROUS, because the v1 reader does not raise on it. Handed a v2 file it
returns, without complaint, a sample rate of 0.0625 Hz, a channel count taken
out of a calibration field, and no channel names at all.

That is why these tests exist and why they write real files rather than
mocking a header: a reader verified only by "it did not raise on the one file
I had" is not verified. `make_synth_famos.write_v2` emits the v2 byte layout,
so the reader is checked against bytes.
"""

from __future__ import annotations

import numpy as np
import pytest

import make_synth_famos as M
from eis_local import FamosFile, FamosV1, FamosV2, _famos_plausible


NAMES = ["UC2", "UC1", "1", "2", "3", "14", "Temp_1"]


def _signal(n=8192, fs=50_000.0):
    t = np.arange(n) / fs
    cols = [0.78 + 0.001 * np.sin(2 * np.pi * 137 * t),
            0.78 + 0.001 * np.sin(2 * np.pi * 137 * t + 0.1)]
    cols += [0.13 + 0.01 * np.sin(2 * np.pi * 137 * t + k) for k in range(4)]
    cols += [1.0 + 0.0 * t]
    return np.column_stack(cols), fs


# ---------------------------------------------------------------------------
# v2 is read exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [8, 4])
def test_a_v2_file_round_trips_every_channel(tmp_path, width):
    d, fs = _signal()
    fam = FamosFile(M.write_v2(tmp_path / "v2.DAT", NAMES, d, fs,
                               bytes_per_val=width))
    assert isinstance(fam, FamosV2)
    assert fam.names == NAMES
    assert fam.n_ch == len(NAMES)
    assert fam.n_samples == d.shape[0]
    assert fam.fs == pytest.approx(fs, rel=1e-9)
    tol = 0.0 if width == 8 else 1e-7
    for i, name in enumerate(NAMES):
        assert np.max(np.abs(fam.channel(name) - d[:, i])) <= tol


def test_the_data_offset_is_confirmed_against_the_declared_byte_count(tmp_path):
    """Counting a fixed number of commas is not safe.

    A 0x2C byte inside a float64 sample is indistinguishable from a
    delimiter, so a fixed comma count that is one too many lands INSIDE the
    data and shifts every channel. Measured before this was fixed: a
    20,000-sample file read back 19,998 samples. |CS declares the byte count,
    and the right offset is the one where what remains equals what was
    declared -- so the offset is checked, not assumed.
    """
    d, fs = _signal()
    fam = FamosFile(M.write_v2(tmp_path / "v2.DAT", NAMES, d, fs))
    assert "matches exactly" in fam._offset_note
    assert fam.n_samples == d.shape[0]          # not one row short


def test_v2_channel_kinds_are_classified_like_v1(tmp_path):
    d, fs = _signal()
    fam = FamosFile(M.write_v2(tmp_path / "v2.DAT", NAMES, d, fs))
    assert fam.segment_names == ["1", "2", "3", "14"]
    assert fam.uc_names == ["UC2", "UC1"]
    assert fam.temp_names == ["Temp_1"]
    assert fam.position("14") == 5


# ---------------------------------------------------------------------------
# The dialect that fails silently
# ---------------------------------------------------------------------------


def test_the_v1_reader_does_not_raise_on_a_v2_file(tmp_path):
    """The finding that makes exception-driven fallback wrong.

    This is not a hypothetical: the adapter this reader was promoted from
    dispatched by catching ValueError from v1, which on these files never
    comes. It is asserted here so that nobody re-introduces that dispatch.
    """
    d, fs = _signal()
    path = M.write_v2(tmp_path / "v2.DAT", NAMES, d, fs)
    bogus = FamosV1(path)                        # no exception
    assert _famos_plausible(bogus)               # ... but it means nothing
    assert bogus.names == [] or len(bogus.names) != bogus.n_ch


def test_the_dispatcher_picks_v2_anyway(tmp_path):
    d, fs = _signal()
    fam = FamosFile(M.write_v2(tmp_path / "v2.DAT", NAMES, d, fs))
    assert isinstance(fam, FamosV2)


def test_a_plausibility_check_rejects_an_impossible_sample_rate():
    class Fake:
        names, n_ch, fs, n_samples = ["1", "2"], 2, 0.0625, 1000
    assert "not a plausible acquisition rate" in _famos_plausible(Fake())


# ---------------------------------------------------------------------------
# v1 still works, and placeholders are refused
# ---------------------------------------------------------------------------


def test_a_v1_file_still_dispatches_to_v1(tmp_path):
    M.main(str(tmp_path))
    card = sorted(tmp_path.glob("*.DAT"))[0]
    fam = FamosFile(card)
    assert isinstance(fam, FamosV1)
    assert fam.fs == pytest.approx(10_000.0)
    assert len(fam.segment_names) == 15


def test_an_empty_placeholder_is_refused_by_name(tmp_path):
    """Campaign folders carry 0-byte files for cards that were never
    connected. Letting one through produces a card with no channels rather
    than an error, which reads downstream as a card that measured nothing.
    """
    p = tmp_path / "Karte_9.DAT"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="placeholder"):
        FamosFile(p)


def test_an_unreadable_file_reports_both_attempts(tmp_path):
    """"v1 said X, v2 said Y" can be acted on; "incomplete FAMOS header"
    cannot."""
    p = tmp_path / "junk.DAT"
    p.write_bytes(b"not a famos file at all " * 100)
    with pytest.raises(ValueError) as exc:
        FamosFile(p)
    text = str(exc.value)
    assert "FamosV1" in text and "FamosV2" in text
    assert "famos_probe" in text
