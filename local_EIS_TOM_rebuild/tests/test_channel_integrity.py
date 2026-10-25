"""One signal per channel: the RO2612085 Karte_3 failure, as a test.

datago returned the UC2 waveform under all 16 names of a FAMOS-v2 card (it
ignored the per-channel |CP byte offset). The pipeline turned that into a
complete set of plausible-looking spectra with Z = K. Two guards now exist:

  * FamosV2 reads each channel from the column its own |CP offset declares;
  * channel_integrity refuses a card whose segment channels are bit-identical
    to each other or to the reference, and bronze stops on it.
"""

from __future__ import annotations

import numpy as np
import pytest

import make_synth_famos as M
from eis_local import FamosFile, FamosV2, channel_integrity, ChannelIntegrityError

NAMES = ["UC2", "54", "55", "69", "70", "68"]


def _data(n=40_000, fs=25_000.0, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    cols = [0.645 + 0.003 * np.sin(2 * np.pi * 37 * t)]
    cols += [1.0 + 0.1 * k + 0.02 * np.sin(2 * np.pi * 37 * t + 0.1 * k)
             for k in range(1, len(NAMES))]
    d = np.column_stack(cols) + 1e-4 * rng.standard_normal((n, len(NAMES)))
    return d, fs


def test_v2_reads_each_channel_from_its_declared_byte_offset(tmp_path):
    d, fs = _data()
    # |CN lists 68 last but it sits in column 3 of the buffer
    cols = [0, 1, 2, 4, 5, 3]
    fam = FamosFile(M.write_v2(tmp_path / "k3.DAT", NAMES, d, fs,
                               bytes_per_val=4, columns=cols))
    assert isinstance(fam, FamosV2)
    assert fam._column_basis == "|CP byte offset"
    for i, nm in enumerate(NAMES):
        assert np.max(np.abs(fam.channel(nm) - d[:, i])) < 1e-6
        assert fam.position(nm) == cols[i]


def test_a_sound_card_passes(tmp_path):
    d, fs = _data()
    fam = FamosFile(M.write_v2(tmp_path / "ok.DAT", NAMES, d, fs, bytes_per_val=4))
    assert channel_integrity(fam, "UC2") == []


def test_every_channel_carrying_the_reference_is_refused(tmp_path):
    d, fs = _data()
    d[:, 1:] = d[:, :1]                       # what datago delivered
    fam = FamosFile(M.write_v2(tmp_path / "bad.DAT", NAMES, d, fs, bytes_per_val=4))
    bad = channel_integrity(fam, "UC2")
    assert len(bad) == 1 and "identical to the reference UC2" in bad[0]


def test_two_segments_sharing_one_waveform_are_refused(tmp_path):
    d, fs = _data()
    d[:, 4] = d[:, 3]
    fam = FamosFile(M.write_v2(tmp_path / "dup.DAT", NAMES, d, fs, bytes_per_val=4))
    bad = channel_integrity(fam, "UC2")
    assert bad and "69, 70" in bad[0] and "each other" in bad[0]


def test_a_dead_channel_is_a_note_not_an_error(tmp_path):
    d, fs = _data()
    d[:, 2] = 0.5
    fam = FamosFile(M.write_v2(tmp_path / "dead.DAT", NAMES, d, fs, bytes_per_val=4))
    notes = []
    assert channel_integrity(fam, "UC2", notes=notes) == []
    assert notes and "55" in notes[0]


def test_bronze_stops_on_a_duplicated_card(tmp_path):
    import bronze
    from config import DEFAULT
    d, fs = _data()
    d[:, 1:] = d[:, :1]
    fp = M.write_v2(tmp_path / "Leepa_1_Current_450A_Test_01_Karte_3.DAT",
                    NAMES, d, fs, bytes_per_val=4)
    with pytest.raises(ChannelIntegrityError):
        bronze.inventory_channels([fp], DEFAULT)


def test_a_card_four_times_off_the_plate_fails_plausibility():
    """RO2612085 datago run: Karte_3 at |Z| +413 %, j_dc -37 % PASSED before."""
    from card_gain import CardFactor
    from plausibility import card_factor_check, FAIL, PASS
    ok = {f"Karte_{k}": CardFactor(f"Karte_{k}", 15,
                                   {"lt1Hz": d, "1_10Hz": d, "10_100Hz": d,
                                    "gt100Hz": d}, j_dc_pct=-d / 2)
          for k, d in ((1, 6.0), (2, -9.0), (4, 19.0), (5, 4.0))}
    assert card_factor_check(ok).verdict == PASS
    bad = dict(ok)
    bad["Karte_3"] = CardFactor("Karte_3", 15, {"lt1Hz": 272.0, "1_10Hz": 344.0,
                                                "10_100Hz": 481.0, "gt100Hz": 787.0},
                                j_dc_pct=-36.8)
    c = card_factor_check(bad)
    assert c.verdict == FAIL and "Karte_3" in c.detail
