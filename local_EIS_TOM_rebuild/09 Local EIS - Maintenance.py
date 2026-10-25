# Databricks notebook source
# MAGIC %md
# MAGIC <!-- local_eis:audience-banner -->
# MAGIC > ### ⚠️ Maintenance only — destructive
# MAGIC >
# MAGIC > Removes ONE run from every `local_eis_*` table, and optionally its folder under
# MAGIC > `EIS_Results`. There is no undo. Type the run key twice to confirm.
# MAGIC >
# MAGIC > The last two cells rebuild ALL tables from the saved folders (nothing is evaluated):
# MAGIC > needed once when a table refuses a run because a column type changed
# MAGIC > ("column type differs from the table ... drop it once").

# COMMAND ----------

# MAGIC %md
# MAGIC # Local EIS — Maintenance: remove a run

# COMMAND ----------

# DBTITLE 1,Setup
import sys
from pathlib import Path

try:   # the folder this notebook sits in holds the modules
    _nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    PIPELINE_DIR = '/Workspace' + str(Path(_nb).parent)
except Exception:
    PIPELINE_DIR = str(Path.cwd())
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

import eis_paths   # noqa: F401  (core/, pipeline/, ... on sys.path)
import runner

# COMMAND ----------

# DBTITLE 1,Which run
dbutils.widgets.text("run_key", "", "Run key, e.g. 2612090/450A_Test01")
dbutils.widgets.text("confirm", "", "Type the run key again")
dbutils.widgets.dropdown("delete_folder", "no", ["no", "yes"], "Also delete its folder")
KEY = dbutils.widgets.get("run_key").strip()
_runs = runner.tables.runs(spark)
_hit = _runs[_runs["run_key"] == KEY] if KEY and len(_runs) else _runs.iloc[0:0]
print(_hit.to_string(index=False) if len(_hit) else f"run_key '{KEY}': not in the runs table")

# COMMAND ----------

# DBTITLE 1,Remove (only when confirmed)
import shutil
if not KEY or dbutils.widgets.get("confirm").strip() != KEY:
    print("not confirmed -- nothing removed")
else:
    runner.tables.delete_run(spark, KEY)
    if dbutils.widgets.get("delete_folder") == "yes":
        folder = runner.settings.RESULTS_ROOT / KEY
        shutil.rmtree(folder, ignore_errors=True)
        print(f"  folder removed: {folder}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rebuild all tables from the saved folders
# MAGIC Drops every `local_eis_*` table and publishes every run again: from its folder under
# MAGIC `EIS_Results`, or -- when the folder is gone -- from its copy in `local_eis_run_files`
# MAGIC (read back BEFORE anything is dropped). If a published run is in neither, nothing is
# MAGIC dropped and the run is named. Folders are not touched, nothing is re-evaluated.
# MAGIC Type `REBUILD` to confirm.

# COMMAND ----------

# DBTITLE 1,Rebuild (only when confirmed)
dbutils.widgets.text("rebuild", "", "Type REBUILD to drop and refill all tables")
if dbutils.widgets.get("rebuild").strip() != "REBUILD":
    print("not confirmed -- tables unchanged")
else:
    result = runner.tables.rebuild(spark)
    display(spark.createDataFrame([(k, v) for k, v in result.items()], "run_key string, status string"))