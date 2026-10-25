# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC <!-- local_eis:audience-banner -->
# MAGIC > ### writes results and tables
# MAGIC >
# MAGIC > Evaluates the selected order / conditions / test and **writes**: the run folders
# MAGIC > `EIS_Results/<order>/<condition>_Test<nn>/` (Run mode `rerun` overwrites them) and,
# MAGIC > with `PUBLISH_TABLES`, their rows in the Delta tables `<TABLE_SCHEMA>.local_eis_*`
# MAGIC > (with every file of the run in `local_eis_run_files`). Run mode `cache` restores a
# MAGIC > published run from the tables when its folder is gone (`CACHE_FROM_TABLES`).
# MAGIC >
# MAGIC > To look at results use **02 Local EIS - Results** (read-only). Settings and paths:
# MAGIC > `runner/settings.py`.

# COMMAND ----------

# MAGIC %md
# MAGIC # Local EIS — Process
# MAGIC Pick the order, conditions and test in the widgets, then **Run all**.

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

R = runner.Session(globals(), package_dir=PIPELINE_DIR)
print(f"Local EIS {runner.settings.RUNNER_VERSION} -- modules from {PIPELINE_DIR}")

# COMMAND ----------

# DBTITLE 1,Selection (widgets)
SEL = R.widgets()

# COMMAND ----------

# DBTITLE 1,Evaluate (Run mode: cache = keep saved results, rerun = recompute and overwrite)
RESULTS = R.run()

# COMMAND ----------

# DBTITLE 1,ECM fits (standard and DRT-informed), saved with each run
ECM = R.fit_ecm()

# COMMAND ----------

# DBTITLE 1,Publish to the tables (silver / gold layer)
if runner.settings.PUBLISH_TABLES:
    PUBLISHED = R.publish()
else:
    print("PUBLISH_TABLES = False -- tables not written")

# COMMAND ----------

# DBTITLE 1,Summary
_summary = R.summary()
if len(_summary):
    display(_summary)
else:
    print("no saved runs for this selection")