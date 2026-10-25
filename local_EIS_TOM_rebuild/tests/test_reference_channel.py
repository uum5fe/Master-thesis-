"""The shared reference is UC2 on every card, by name, not by argmax.

Only one cell-voltage line is fanned out to the Dewetron cards on this
campaign, on UC2. The pipeline used to take the UC channel with the largest
standard deviation on each card independently, which is a per-file decision
that nothing downstream cross-checks -- and an unconnected UC input, being
noisy, is exactly the kind of channel an argmax on std likes.

These tests write real FAMOS bytes (via make_synth_famos) with a LOUDER UC1
than UC2, so the old rule and the new one give different answers and the
tests can tell them apart.
"""

from __future__ import annotations

import numpy as np
import pytest

import bronze
import make_synth_famos as M
from config import Config
from eis_local import DEFAULT_REF_CHANNEL, FamosFile, pick_reference_channel


FS = 50_000.0
NAMES = ["UC1", "UC2", "1", "2", "3", "Temp_1"]


def _card(path, uc1_amp: float, uc2_amp: float, n: int = 8192):
    """A card whose UC1 is deliberately louder than its shared UC2."""
    t = np.arange(n) / FS
    cols = [0.78 + uc1_amp * np.sin(2 * np.pi * 137 * t),
            0.78 + uc2_amp * np.sin(2 * np.pi * 137 * t + 0.1)]
    cols += [0.13 + 0.01 * np.sin(2 * np.pi * 137 * t + k) for k in range(3)]
    cols += [1.0 + 0.0 * t]
    return M.write_v2(path, NAMES, np.column_stack(cols), FS)


def test_the_reference_is_uc2_even_when_uc1_is_louder(tmp_path):
    """The whole point: amplitude no longer decides which channel is shared."""
    fam = FamosFile(_card(tmp_path / "k1.DAT", uc1_amp=0.05, uc2_amp=0.001))
    assert max(fam.uc_names,
               key=lambda c: float(np.std(fam.channel(c)))) == "UC1"
    assert pick_reference_channel(fam) == "UC2"
    assert DEFAULT_REF_CHANNEL == "UC2"


def test_every_card_gets_the_same_reference_channel(tmp_path):
    """A per-card argmax is free to disagree card-to-card; naming is not.

    Card 1's loudest UC channel is UC1 and card 2's is UC2. Under the old
    rule the alignment would cross-correlate card 1's UC1 against card 2's
    UC2 -- two different signals -- and report the lag as if it meant
    something.
    """
    files = [_card(tmp_path / "Karte_1.DAT", uc1_amp=0.05, uc2_amp=0.001),
             _card(tmp_path / "Karte_2.DAT", uc1_amp=0.001, uc2_amp=0.05)]
    cfg = Config(dat_dir=tmp_path, out_dir=tmp_path, verbose=False)
    _channels, cards = bronze.inventory_channels(files, cfg)

    assert len(cards) == 2
    assert {c.ref_name for c in cards.values()} == {"UC2"}
    assert {c.ref_slot for c in cards.values()} == {NAMES.index("UC2")}


def test_an_explicit_choice_is_honoured(tmp_path):
    """A campaign wired to another line is a config change, not a code change."""
    fam = FamosFile(_card(tmp_path / "k1.DAT", uc1_amp=0.05, uc2_amp=0.001))
    assert pick_reference_channel(fam, "UC1") == "UC1"
    assert pick_reference_channel(fam, "uc2") == "UC2"      # name, not case


def test_empty_setting_restores_the_old_argmax(tmp_path):
    """The escape hatch stays, so an unknown plate can still be processed."""
    fam = FamosFile(_card(tmp_path / "k1.DAT", uc1_amp=0.05, uc2_amp=0.001))
    assert pick_reference_channel(fam, "") == "UC1"


def test_a_card_without_uc2_is_reported_not_silently_substituted(tmp_path):
    """Falling back is allowed; doing it quietly is not.

    A missing UC2 means this card is not wired like the others, and the lag
    measured against it is then a comparison of two different signals. That
    has to reach the log, because everything downstream -- the dwell windows,
    the consensus schedule, ref_slot -- is built on the assumption it did not
    happen.
    """
    t = np.arange(4096) / FS
    names = ["UC1", "1", "2"]
    data = np.column_stack([0.78 + 0.05 * np.sin(2 * np.pi * 137 * t),
                            0.13 + 0.01 * np.sin(2 * np.pi * 137 * t),
                            0.13 + 0.01 * np.cos(2 * np.pi * 137 * t)])
    fam = FamosFile(M.write_v2(tmp_path / "odd.DAT", names, data, FS))

    said = []

    class _Log:
        def warning(self, msg):
            said.append(msg)

    assert pick_reference_channel(fam, "UC2", log=_Log()) == "UC1"
    assert said and "UC2" in said[0]


def test_a_card_with_no_uc_channel_at_all_is_none(tmp_path):
    t = np.arange(4096) / FS
    data = np.column_stack([0.13 + 0.01 * np.sin(2 * np.pi * 137 * t),
                            0.13 + 0.01 * np.cos(2 * np.pi * 137 * t)])
    fam = FamosFile(M.write_v2(tmp_path / "nouc.DAT", ["1", "2"], data, FS))
    assert pick_reference_channel(fam) is None


def test_the_default_config_names_uc2():
    assert Config().ref_channel == "UC2"
    assert Config.from_cli(["--dat", "."]).ref_channel == "UC2"
    assert Config.from_cli(["--dat", ".", "--ref-channel", "UC1"]
                           ).ref_channel == "UC1"


@pytest.mark.parametrize("prefer", ["UC2", "UC1"])
def test_process_card_reads_the_same_channel_the_inventory_recorded(tmp_path,
                                                                    prefer):
    """A and B must be the same signal: `ref_slot` comes from the inventory,
    `A_ref` from process_card, and they are multiplied together."""
    fp = _card(tmp_path / "Karte_1.DAT", uc1_amp=0.05, uc2_amp=0.001)
    cfg = Config(dat_dir=tmp_path, out_dir=tmp_path, verbose=False,
                 ref_channel=prefer)
    _channels, cards = bronze.inventory_channels([fp], cfg)
    fam = FamosFile(fp)
    assert cards[fp.stem].ref_name == pick_reference_channel(
        fam, cfg.ref_channel, bronze.cfg_stride(cfg))
