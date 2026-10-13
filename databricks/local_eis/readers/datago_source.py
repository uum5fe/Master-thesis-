"""
datago_source.py
================
FAMOS card recordings from the datago Delta tables, with the same interface
as eis_local.FamosFile. The logic is the runner's "Datago Source" cell, moved
here so the standard pipeline notebook can use it too; the only change is
that the Spark session is passed in instead of taken from the notebook.

    ids = discover_famos_file_ids(spark, "2612030")   # {cond: [file_id x 5]}
    fam = DatagoFamosFile(spark, ids["450A"][0], "Karte_1")

Each card is 16 channels x ~2.5 M samples, i.e. tens of millions of Delta
rows: ~30-60 s per card against ~5 s from the Volumes .DAT files.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

VAL_TBL = "ps_xplatform_dev.rvadvtec_ops.datago_advtec_values_delta"
META_TBL = "ps_xplatform_dev.rvadvtec_ops.datago_advtec_metadata"
GP_TBL = "ps_xplatform_dev.rvadvtec_ops.datago_advtec_generalproperties"
FAMOS_ROOT = Path("/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Famos")
PROTOCOL_ORDER = ["45A", "60A", "150A", "450A"]


def discover_famos_file_ids(spark, leepa: str, famos_root=FAMOS_ROOT) -> dict:
    """{condition: [file_id_1, ..., file_id_5]} for an order (one per card).

    FAMOS files carry no measurement type; they are grouped by the minute
    they started, groups of exactly 5 (one per card) are a condition, and the
    groups are named in time order from the Volumes file names when those
    cover every group, else in the protocol order 45 -> 60 -> 150 -> 450 A.
    """
    df = spark.sql(f"""
        SELECT DISTINCT gp.file_id, gp.measurementBegin
        FROM {META_TBL} m
        JOIN {GP_TBL} gp ON m.measurement_id = gp.measurement_id
        WHERE m.orderId = 'RO{leepa}'
          AND gp.measurement_type IS NULL
        ORDER BY gp.measurementBegin
    """).toPandas()
    if df.empty:
        return {}
    df["minute"] = df["measurementBegin"].dt.floor("min")
    groups = df.groupby("minute")["file_id"].apply(list).to_dict()
    card_groups = {k: v for k, v in groups.items() if len(v) == 5}
    conditions_ordered = []
    try:
        vol_files = sorted(Path(famos_root).glob(f"Leepa_{leepa}_Current_*_Karte_1.DAT"))
        conditions_ordered = [f.name.split("_")[3] for f in vol_files]
    except Exception:                                       # noqa: BLE001
        pass
    if len(conditions_ordered) < len(card_groups):
        conditions_ordered = PROTOCOL_ORDER
    result = {}
    for idx, (_ts, file_ids) in enumerate(sorted(card_groups.items())):
        cond = (conditions_ordered[idx] if idx < len(conditions_ordered)
                else f"Cond-{idx+1}")
        result[cond] = file_ids
    return result


class DatagoFamosFile:
    """FamosFile-compatible reader that pulls one card from datago.

    .path, .fs, .n_ch, .n_samples, .names, .segment_names, .uc_names,
    .temp_names, .channel(name), .position(name)

    The ADC slot order is APPROXIMATED (UC, then segments, then Temp): Delta
    rows carry no header order.
    """

    def __init__(self, spark, file_id: str, card_label: str = "datago",
                 preload: bool = True):
        self.spark = spark
        self.file_id = file_id
        self.path = Path(f"/datago/{card_label}.DAT")
        self._channels = {}
        self._load_metadata()
        if preload:
            self.load_all_channels()

    def _load_metadata(self):
        df = self.spark.sql(f"""
            SELECT channel, COUNT(*) as n_pts,
                   MIN(CAST(time_value AS DOUBLE)) as t_min,
                   MAX(CAST(time_value AS DOUBLE)) as t_max
            FROM {VAL_TBL}
            WHERE file_id = '{self.file_id}'
            GROUP BY channel
            ORDER BY channel
        """).toPandas()
        self.names = df["channel"].tolist()
        self.n_ch = len(self.names)
        if not df.empty:
            row0 = df.iloc[0]
            duration = row0["t_max"] - row0["t_min"]
            self.n_samples = int(row0["n_pts"])
            self.fs = round(self.n_samples / duration) if duration > 0 else 10000
        else:
            self.n_samples, self.fs = 0, 10000
        self.segment_names = [n for n in self.names if n.isdigit()]
        self.uc_names = [n for n in self.names if n.startswith("UC")]
        self.temp_names = [n for n in self.names if n.startswith("Temp")]
        ordered = self.uc_names + self.segment_names + self.temp_names
        self._position_map = {name: i for i, name in enumerate(ordered)}

    def position(self, channel_name: str) -> int:
        return self._position_map.get(channel_name, 0)

    def channel(self, name: str) -> np.ndarray:
        if name not in self._channels:
            df = self.spark.sql(f"""
                SELECT CAST(time_value AS DOUBLE) as t,
                       CAST(value AS DOUBLE) as v
                FROM {VAL_TBL}
                WHERE file_id = '{self.file_id}' AND channel = '{name}'
                ORDER BY t
            """).toPandas()
            self._channels[name] = df["v"].values
        return self._channels[name]

    def load_all_channels(self):
        t0 = time.time()
        df = self.spark.sql(f"""
            SELECT channel, CAST(time_value AS DOUBLE) as t,
                   CAST(value AS DOUBLE) as v
            FROM {VAL_TBL}
            WHERE file_id = '{self.file_id}'
            ORDER BY channel, t
        """).toPandas()
        for ch, grp in df.groupby("channel"):
            self._channels[ch] = grp["v"].values
        print(f"    [{self.path.stem}] {self.n_ch} channels, "
              f"{self.n_samples:,} pts/ch from datago in {time.time()-t0:.1f}s")

    def __getitem__(self, name: str) -> np.ndarray:
        return self.channel(name)
