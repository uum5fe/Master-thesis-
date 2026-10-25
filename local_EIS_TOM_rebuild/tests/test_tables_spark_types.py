"""Every frame tables.frames() makes must be accepted by a real Spark as it
is -- one type per column, names Spark accepts -- and survive a write and a
read back. Plain Spark (no Delta needed); skipped without pyspark."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

from runner import tables  # noqa: E402

from test_tables_results_auto import _run  # noqa: E402

#: a real saved run to check as well (set to a run folder)
REAL_RUN = os.environ.get("LOCAL_EIS_RUN_DIR", "")


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    from pyspark.sql import SparkSession
    wh = tmp_path_factory.mktemp("wh")
    s = (SparkSession.builder.master("local[1]").appName("types")
         .config("spark.sql.warehouse.dir", str(wh))
         .config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "1").getOrCreate())
    yield s
    s.stop()


def _check(spark, run_dir):
    for suffix, pdf in tables.frames(run_dir).items():
        sdf = spark.createDataFrame(pdf)
        name = f"t_{suffix}"
        sdf.write.mode("overwrite").partitionBy("order_id").saveAsTable(name)
        back = spark.table(name).toPandas()
        assert len(back) == len(pdf), suffix
        assert set(back.columns) == set(pdf.columns), suffix


def test_synthetic_run_frames_go_through_spark(spark, tmp_path):
    d = _run(tmp_path, "2612090", "450A_Test01")
    # the awkward columns real files have: text with gaps, flags, mixed
    pd.DataFrame({"segment": [1, 2, 3], "flags": ["a", np.nan, "b c"],
                  "fault": [True, False, True], "R (mΩ·cm²)": [1.0, np.nan, 3.0]}
                 ).to_csv(d / "gold" / "plate_summary.csv", index=False)
    _check(spark, d)


@pytest.mark.skipif(not REAL_RUN, reason="LOCAL_EIS_RUN_DIR not set")
def test_a_real_run_goes_through_spark(spark):
    _check(spark, Path(REAL_RUN))
