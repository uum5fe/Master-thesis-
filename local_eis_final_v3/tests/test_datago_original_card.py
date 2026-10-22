"""RO2612091 45 A Karte_3: datago's rows hold UC2 under every channel.

That is datago's import of this FAMOS version, not the pipeline: the other
15 channels are not in the values table and cannot be rebuilt from it. The
card comes back only from its ORIGINAL .DAT -- so the pipeline now looks for
it harder (the record's bucket / folder, any record column in the template,
the Volume by name in sub-folders) and, when it still cannot, says where the
record points."""

from __future__ import annotations

import pandas as pd

from runner import datago
from runner import settings as S
from eis_local import FamosFile

from test_runner_datago_famos import (_Spark, _files, _rows, _session,
                                      _signals, _volume_dat)

NAME = "RO2612091-01_Current_45A_Test_01_Karte_3.DAT"
REC = {"file_id": "k3", "raw_file_name": NAME, "bucket": "datago-landing",
       "folder": "RO2612091/FAMOS", "source": "upload", "measurement_id": "m1"}


def test_the_template_can_use_every_record_column(monkeypatch):
    monkeypatch.setattr(S, "DATAGO_RAW_FILE_TEMPLATE",
                        "abfss://{bucket}@acct.dfs.core.windows.net/"
                        "{folder}/{raw_file_name}")
    c = datago.raw_file_candidates(REC, "RO2612091")
    assert c[0] == ("DATAGO_RAW_FILE_TEMPLATE",
                    "abfss://datago-landing@acct.dfs.core.windows.net/"
                    f"RO2612091/FAMOS/{NAME}")
    # an unknown placeholder is left empty, not a KeyError
    monkeypatch.setattr(S, "DATAGO_RAW_FILE_TEMPLATE", "/x/{nope}/{raw_file_name}")
    assert datago.raw_file_candidates(REC)[0][1] == f"/x//{NAME}"


def test_bucket_and_folder_are_tried_as_a_path(monkeypatch):
    monkeypatch.setattr(S, "DATAGO_RAW_FILE_TEMPLATE", "")
    uris = [u for _w, u in datago.raw_file_candidates(REC)]
    assert f"s3://datago-landing/RO2612091/FAMOS/{NAME}" in uris
    rec = {**REC, "bucket": "/Volumes/cat/sch/landing"}
    assert (f"/Volumes/cat/sch/landing/RO2612091/FAMOS/{NAME}"
            in [u for _w, u in datago.raw_file_candidates(rec)])


def test_when_nothing_is_found_the_record_says_where_to_look(tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(S, "DATAGO_RAW_FILE_TEMPLATE", "")

    class _Sp:
        def sql(self, q):
            class Q:
                def toPandas(self_):
                    return pd.DataFrame([REC])
            return Q()
    row = type("R", (), {"file_id": "k3", "raw_file_name": NAME})()
    path, note = datago.raw_card(_Sp(), row, tmp_path / "k3.DAT")
    assert path is None
    assert "bucket='datago-landing'" in note and "folder='RO2612091/FAMOS'" in note
    assert "DATAGO_RAW_FILE_TEMPLATE" in note


def test_the_original_in_a_volume_sub_folder_replaces_the_broken_rows(
        tmp_path, monkeypatch):
    sig2, sig3 = _signals(1), _signals(2)
    files = _files(("k2", "RO2612085-01_Current_450A_Test_01_Karte_2.DAT"),
                   ("k3", "RO2612085-01_Current_450A_Test_01_Karte_3.DAT"))
    vals = pd.concat([_rows("k2", sig2), _rows("k3", sig3, bug=True)])
    R, vol = _session(tmp_path, _Spark(files, vals), monkeypatch)
    sub = vol / "RO2612085" / "450A"
    sub.mkdir(parents=True)
    _volume_dat(sub / "RO2612085-01_Current_450A_Test_01_Karte_3.DAT", sig3)
    _volume_dat(vol / "RO2612085-01_Current_60A_Test_02_Karte_2.DAT", sig2)
    d, src = R.famos_dir("450A")
    assert src[3] == "Volume"
    k3 = FamosFile(d / "Leepa_2612085_Current_450A_Test_01_Karte_3.DAT")
    assert (k3.channel("16") != k3.channel("UC2")).any()
