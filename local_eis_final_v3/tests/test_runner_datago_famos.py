"""FAMOS_FROM_DATAGO: a card is rebuilt from datago's rows, checked, and
used only when it holds one signal per channel and -- when the Volume has the
card -- the Volume's samples; the channel (slot) order comes from a .DAT
header of the same card."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runner import datago
from runner import settings as S
import runner.session as session_mod
from eis_local import FamosFile, famos_start_time

ORDER_K2 = ["UC2", "22", "16", "17", "18", "19", "20", "21", "15",
            "23", "24", "25", "26", "27", "28", "29"]
FS = 25000.0
N = 40000


def _signals(seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    out = {}
    for j, nm in enumerate(ORDER_K2):
        x = 0.01 * np.sin(2 * np.pi * (37 + j) * t) + 1e-4 * rng.normal(size=N)
        out[nm] = (x + 0.3 * (nm != "UC2")).astype("<f4").astype(float)
    return out


def _volume_dat(path, sig, start=datetime(2026, 10, 19, 12, 35, 20)):
    return datago.write_famos_v1(path, ORDER_K2, [sig[n] for n in ORDER_K2],
                                 FS, start)


def _rows(fid, sig, bug=False, swap=None):
    """datago's rows of one card, shuffled (Delta keeps no order)."""
    rng = np.random.default_rng(3)
    names = list(sig)
    if swap:
        a, b = swap
        sig = {**sig, a: sig[b], b: sig[a]}
    parts = []
    for nm in names:
        v = sig["UC2"] if bug else sig[nm]
        parts.append(pd.DataFrame({"file_id": fid, "channel": nm,
                                   "t": np.arange(N) / FS, "v": v}))
    df = pd.concat(parts, ignore_index=True)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


class _Q:
    def __init__(self, df):
        self.df = df

    def toPandas(self):
        return self.df.copy()

    def collect(self):
        return [dict(r) for _, r in self.df.iterrows()]


class _Spark:
    def __init__(self, files, vals):
        self.files, self.vals = files, vals

    def sql(self, q):
        if "metafiles_delta" in q and "WHERE file_id = '" in q:
            fid = q.split("file_id = '")[1].split("'")[0]
            return _Q(self.files[self.files.file_id == fid])
        if "metafiles_delta" in q:
            return _Q(self.files)
        if "values_delta" in q:
            fid = q.split("file_id = '")[1].split("'")[0]
            return _Q(self.vals[self.vals.file_id == fid][["channel", "t", "v"]])
        return _Q(pd.DataFrame({"orderId": ["RO2612085"]}))


def _files(*rows):
    return pd.DataFrame([{"file_id": fid, "raw_file_name": name,
                          "measurement_type": None,
                          "t_begin": pd.Timestamp("2026-10-19 10:35:20"),
                          "t_end": None, "starttime": None, "iacreq": None,
                          "freqinit": None, "freqfinal": None,
                          "ptsperdec": None} for fid, name in rows])


@pytest.fixture(autouse=True)
def _fresh_cache():
    datago._FILES.clear()
    yield
    datago._FILES.clear()


def test_famos_key_reads_condition_test_and_card():
    assert datago.famos_key("RO2612085-01_Current_450A_Test_01_Karte_3.DAT") \
        == ("450A", "01", 3)
    assert datago.famos_key("Leepa_RO2612086_Current_60A_Test_02_Karte_4.DAT") \
        == ("60A", "02", 4)
    assert datago.famos_key("bench.mf4") is None


def test_card_rebuilt_from_datago_equals_the_volume_dat(tmp_path):
    sig = _signals()
    vol = _volume_dat(tmp_path / "vol.DAT", sig)
    files = _files(("k2", "RO2612085-01_Current_450A_Test_01_Karte_2.DAT"))
    spark = _Spark(files, _rows("k2", sig))
    row = next(datago.files(spark, "2612085").itertuples())
    out, note = datago.famos_card(spark, row, tmp_path / "dg.DAT", same_dat=vol)
    assert out is not None, note
    fam, ref = FamosFile(out), FamosFile(vol)
    assert fam.names == ORDER_K2                       # slot order of the .DAT
    assert [fam.position(n) for n in ORDER_K2] == list(range(16))
    assert fam.fs == pytest.approx(FS) and fam.n_samples == N
    for nm in ORDER_K2:
        assert np.array_equal(fam.channel(nm), ref.channel(nm))
    assert famos_start_time(out) == famos_start_time(vol)


def test_datago_v2_bug_uc2_under_every_name_is_rejected(tmp_path):
    sig = _signals()
    vol = _volume_dat(tmp_path / "vol.DAT", sig)
    files = _files(("k3", "RO2612085-01_Current_450A_Test_01_Karte_3.DAT"))
    spark = _Spark(files, _rows("k3", sig, bug=True))
    row = next(datago.files(spark, "2612085").itertuples())
    out, note = datago.famos_card(spark, row, tmp_path / "dg.DAT", same_dat=vol)
    assert out is None and "SAME waveform" in note
    assert not (tmp_path / "dg.DAT").exists()


def test_two_channels_swapped_in_datago_are_caught_by_the_volume(tmp_path):
    sig = _signals()
    vol = _volume_dat(tmp_path / "vol.DAT", sig)
    files = _files(("k2", "RO2612085-01_Current_450A_Test_01_Karte_2.DAT"))
    spark = _Spark(files, _rows("k2", sig, swap=("16", "17")))
    row = next(datago.files(spark, "2612085").itertuples())
    out, note = datago.famos_card(spark, row, tmp_path / "dg.DAT", same_dat=vol)
    assert out is None and "16" in note and "17" in note


def test_card_ingested_twice_is_read_once(tmp_path):
    """RO2611976: datago holds every sample of a card twice (2x the .DAT's
    samples per channel). The repeats are dropped, the card then equals the
    .DAT."""
    sig = _signals()
    vol = _volume_dat(tmp_path / "vol.DAT", sig)
    files = _files(("k2", "RO2611976-01_Current_60A_Test_01_Karte_2.DAT"))
    rows = _rows("k2", sig)
    spark = _Spark(files, pd.concat([rows, rows.iloc[::-1]], ignore_index=True))
    row = next(datago.files(spark, "2611976").itertuples())
    out, note = datago.famos_card(spark, row, tmp_path / "dg.DAT", same_dat=vol)
    assert out is not None, note
    assert f"{16 * N:,} repeated rows dropped" in note
    fam, ref = FamosFile(out), FamosFile(vol)
    assert fam.n_samples == N
    for nm in ORDER_K2:
        assert np.array_equal(fam.channel(nm), ref.channel(nm))


def test_repeats_with_other_values_are_rejected(tmp_path):
    sig = _signals()
    files = _files(("k2", "RO2611976-01_Current_60A_Test_01_Karte_2.DAT"))
    rows = _rows("k2", sig)
    other = rows.copy()
    other["v"] = other["v"] + 1e-3
    spark = _Spark(files, pd.concat([rows, other], ignore_index=True))
    row = next(datago.files(spark, "2611976").itertuples())
    out, note = datago.famos_card(spark, row, tmp_path / "dg.DAT")
    assert out is None and "different values" in note


def test_a_longer_record_in_datago_is_rejected_with_both_durations(tmp_path):
    sig = _signals()
    vol = _volume_dat(tmp_path / "vol.DAT", sig)
    files = _files(("k2", "RO2611976-01_Current_60A_Test_01_Karte_2.DAT"))
    rows = _rows("k2", sig)
    late = rows.copy()
    late["t"] = late["t"] + N / FS
    spark = _Spark(files, pd.concat([rows, late], ignore_index=True))
    row = next(datago.files(spark, "2611976").itertuples())
    out, note = datago.famos_card(spark, row, tmp_path / "dg.DAT", same_dat=vol)
    assert out is None
    assert f"{2 * N} samples per channel (3.2 s" in note and "(1.6 s" in note


def test_slot_order_from_another_header_else_approximate(tmp_path):
    sig = _signals()
    other = _volume_dat(tmp_path / "other_test.DAT", _signals(seed=9))
    files = _files(("k2", "RO2612085-01_Current_450A_Test_01_Karte_2.DAT"))
    spark = _Spark(files, _rows("k2", sig))
    row = next(datago.files(spark, "2612085").itertuples())
    out, note = datago.famos_card(spark, row, tmp_path / "a.DAT",
                                  slot_dat=[tmp_path / "missing.DAT", other])
    assert out is not None and "other_test.DAT" in note
    assert FamosFile(out).names == ORDER_K2
    out2, note2 = datago.famos_card(spark, row, tmp_path / "b.DAT")
    assert out2 is not None and "approximate" in note2
    assert FamosFile(out2).names[:3] == ["UC2", "15", "16"]
    for nm in ORDER_K2:                    # samples right, only slots guessed
        assert np.array_equal(FamosFile(out2).channel(nm), sig[nm])


def _session(tmp_path, spark, monkeypatch, on=True):
    monkeypatch.setattr(S, "FAMOS_FROM_DATAGO", on)
    monkeypatch.setattr(S, "DATAGO_FIRST", True)
    vol = tmp_path / "Famos"
    vol.mkdir()
    R = session_mod.Session({"spark": spark}, famos_root=vol,
                            gamry_root=tmp_path / "Gamry",
                            results_root=tmp_path / "res",
                            config_root=tmp_path / "cfg",
                            scratch=tmp_path / "scratch")
    R.sel = {"leepa": "2612085", "test_id": "01"}
    return R, vol


def test_session_mixes_datago_and_volume_cards(tmp_path, monkeypatch):
    sig2, sig3 = _signals(1), _signals(2)
    files = _files(("k2", "RO2612085-01_Current_450A_Test_01_Karte_2.DAT"),
                   ("k3", "RO2612085-01_Current_450A_Test_01_Karte_3.DAT"),
                   ("k9", "RO2612085-01_Current_150A_Test_01_Karte_2.DAT"))
    vals = pd.concat([_rows("k2", sig2), _rows("k3", sig3, bug=True)])
    spark = _Spark(files, vals)
    R, vol = _session(tmp_path, spark, monkeypatch)
    _volume_dat(vol / "RO2612085-01_Current_450A_Test_01_Karte_3.DAT", sig3)
    _volume_dat(vol / "RO2612085-01_Current_60A_Test_02_Karte_2.DAT", sig2)

    ct = R.conditions_and_tests("2612085")
    assert ct == {"60A": ["02"], "150A": ["01"], "450A": ["01"]}

    d, src = R.famos_dir("450A")
    assert src == {2: "datago (rows)", 3: "Volume"}
    k2 = FamosFile(d / "Leepa_2612085_Current_450A_Test_01_Karte_2.DAT")
    k3 = FamosFile(d / "Leepa_2612085_Current_450A_Test_01_Karte_3.DAT")
    assert k2.names == ORDER_K2              # slot order from the 60 A .DAT
    assert np.array_equal(k3.channel("55" if "55" in k3.names else "16"),
                          sig3["16"])
    import bronze
    from config import DEFAULT
    cfg = DEFAULT.replace(dat_dir=d, leepa="2612085", condition="450A",
                          test_id="01")
    assert len(bronze.discover_files(cfg)) == 2


def test_switch_off_keeps_the_volume(tmp_path, monkeypatch):
    files = _files(("k2", "RO2612085-01_Current_450A_Test_01_Karte_2.DAT"))
    spark = _Spark(files, _rows("k2", _signals()))
    R, vol = _session(tmp_path, spark, monkeypatch, on=False)
    assert R.conditions_and_tests("2612085") == {}
    assert R.famos_dir("450A") == (vol, {})
    monkeypatch.setattr(S, "FAMOS_FROM_DATAGO", True)
    monkeypatch.setattr(S, "DATAGO_FIRST", False)
    assert R.famos_dir("450A") == (vol, {})


# ---------------------------------------------------------------------------
# the original uploaded file (the only datago source of a FAMOS-v2 card)
# ---------------------------------------------------------------------------
V2_NAMES = ["UC2", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63",
            "64", "69", "70", "71", "68"]


def _v2_card(path, seed=5):
    import make_synth_famos as M
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    data = np.column_stack([0.3 * (j > 0) + 0.01 * np.sin(2 * np.pi * (11 + j) * t)
                            + 1e-4 * rng.normal(size=N) for j in range(16)])
    M.write_v2(path, V2_NAMES, data, FS, bytes_per_val=4,
               columns=list(range(16)))
    return path, data


def _v2_session(tmp_path, monkeypatch, record_extra=None, template=""):
    """RO2612086 30 A Karte_3: datago rows hold UC2 under every name, the
    Volume holds nothing; the original upload is somewhere else."""
    up = tmp_path / "landing"
    up.mkdir()
    raw, data = _v2_card(up / "RO2612086-01_Current_30A_Test_01_Karte_3.DAT")
    sig = {nm: data[:, j] for j, nm in enumerate(V2_NAMES)}
    files = _files(("k3", "RO2612086-01_Current_30A_Test_01_Karte_3.DAT"))
    for k, v in (record_extra or {}).items():
        files[k] = v.format(up=up)
    monkeypatch.setattr(S, "DATAGO_RAW_FILE_TEMPLATE", template.format(up=up))
    spark = _Spark(files, _rows("k3", sig, bug=True))
    R, vol = _session(tmp_path, spark, monkeypatch)
    R.sel = {"leepa": "2612086", "test_id": "01"}
    return R, raw, data


def test_v2_card_from_a_path_column_of_the_datago_record(tmp_path, monkeypatch):
    R, raw, data = _v2_session(
        tmp_path, monkeypatch,
        {"storage_path": "{up}/RO2612086-01_Current_30A_Test_01_Karte_3.DAT"})
    d, src = R.famos_dir("30A")
    assert src == {3: "datago (original file)"}
    fam = FamosFile(d / "Leepa_2612086_Current_30A_Test_01_Karte_3.DAT")
    assert type(fam).__name__ == "FamosV2" and fam.names == V2_NAMES
    for j, nm in enumerate(V2_NAMES):           # every channel, not UC2
        assert np.allclose(fam.channel(nm), data[:, j].astype("<f4"))


def test_v2_card_from_a_folder_column(tmp_path, monkeypatch):
    R, raw, data = _v2_session(tmp_path, monkeypatch, {"landing_dir": "{up}/"})
    d, src = R.famos_dir("30A")
    assert src == {3: "datago (original file)"}


def test_v2_card_from_the_template(tmp_path, monkeypatch):
    R, raw, data = _v2_session(tmp_path, monkeypatch,
                               template="{up}/{{raw_file_name}}")
    d, src = R.famos_dir("30A")
    assert src == {3: "datago (original file)"}


def test_original_through_spark_binaryfile(tmp_path, monkeypatch):
    raw, data = _v2_card(tmp_path / "x.DAT")

    class _Reader:
        def format(self, f):
            assert f == "binaryFile"
            return self

        def load(self, uri):
            assert uri == "abfss://landing@acc.dfs.core.windows.net/RO/x.DAT"
            return self

        def select(self, *a):
            return self

        def limit(self, n):
            return self

        def collect(self):
            return [{"content": bytearray(raw.read_bytes())}]

    files = _files(("k3", "RO2612086-01_Current_30A_Test_01_Karte_3.DAT"))
    files["uri"] = "abfss://landing@acc.dfs.core.windows.net/RO/x.DAT"
    spark = _Spark(files, pd.DataFrame())
    spark.read = _Reader()
    row = next(datago.files(spark, "2612086").itertuples())
    out, note = datago.raw_card(spark, row, tmp_path / "out.DAT")
    assert out is not None and "abfss://" in note
    assert FamosFile(out).names == V2_NAMES


def test_without_any_original_the_v2_card_falls_back_and_says_why(
        tmp_path, monkeypatch):
    R, raw, data = _v2_session(tmp_path, monkeypatch)
    said = []
    R.say = said.append
    d, src = R.famos_dir("30A")
    assert src == {}                       # rows rejected, no Volume copy
    text = "\n".join(said)
    assert "no path to the original file" in text and "raw_file_name" in text
    assert "SAME waveform" in text and "left out" in text


def test_space_padded_v2_start_time(tmp_path):
    p = tmp_path / "a.DAT"
    p.write_bytes(b"|CF,2,1,1;|CK,1,3,1,1;|NT,1,22,20,10,2026, 8,24,52.00;" + b"x" * 64)
    assert famos_start_time(p) == datetime(2026, 10, 20, 8, 24, 52)
