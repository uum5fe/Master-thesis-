"""runner/tables.py on a real Delta Lake (local Spark). Skipped unless
pyspark and delta-spark are installed -- on Databricks they always are."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")
delta = pytest.importorskip("delta")

from runner import tables  # noqa: E402
from runner import settings as S  # noqa: E402

from test_tables_results_auto import _run  # noqa: E402


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    from pyspark.sql import SparkSession
    wh = tmp_path_factory.mktemp("warehouse")
    b = (SparkSession.builder.master("local[1]").appName("local_eis_tables")
         .config("spark.sql.warehouse.dir", str(wh))
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog",
                 "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "1"))
    try:
        s = delta.configure_spark_with_delta_pip(b).getOrCreate()
        s.sql("CREATE DATABASE IF NOT EXISTS local_eis_test")
    except Exception as exc:                                # noqa: BLE001
        pytest.skip(f"no local Delta Lake ({type(exc).__name__}: the Delta "
                    f"jars come from Maven Central)")
    yield s
    s.stop()


def test_publish_replace_read_delete_on_delta(spark, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "TABLE_SCHEMA", "local_eis_test")
    q = lambda *a: None                                     # noqa: E731
    a = _run(tmp_path, "2612090", "450A_Test01", digest="d1", rohm=50)
    b = _run(tmp_path, "2612091_gen2", "60A_Test02", digest="d1", rohm=40)
    tables.publish(spark, a, say=q)
    tables.publish(spark, b, say=q)
    n = spark.table(tables.table_name("silver_spectra")).count()
    a2 = _run(tmp_path / "again", "2612090", "450A_Test01", digest="d2", rohm=55)
    tables.publish(spark, a2, say=q)                        # rerun: replaced
    assert spark.table(tables.table_name("silver_spectra")).count() == n
    runs = tables.runs(spark).set_index("run_key")
    assert set(runs.index) == {"2612090/450A_Test01", "2612091_gen2/60A_Test02"}
    assert runs.loc["2612090/450A_Test01", "digest"] == "d2"
    assert runs.loc["2612091_gen2/60A_Test02", "plate"] == "gen2"
    gp = tables.read(spark, "gold_plate", "parameter = 'R_ohmic'")
    assert sorted(gp["median"]) == [40.0, 55.0]
    part = spark.sql(f"DESCRIBE DETAIL {tables.table_name('runs')}").collect()[0]
    assert list(part["partitionColumns"]) == ["order_id"]
    tables.delete_run(spark, "2612090/450A_Test01", say=q)
    assert set(tables.runs(spark)["run_key"]) == {"2612091_gen2/60A_Test02"}
    assert spark.table(tables.table_name("silver_spectra")).where(
        "run_key = '2612090/450A_Test01'").count() == 0
