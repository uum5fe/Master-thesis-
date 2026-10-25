# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC <!-- local_eis:audience-banner -->
# MAGIC > ### 📊 Analysis — read-only
# MAGIC >
# MAGIC > Reads the published runs from the Delta tables `local_eis_*` (`RESULTS_SOURCE =
# MAGIC > "tables"`: every file of a run comes back from `local_eis_run_files`, so the Volume
# MAGIC > folder is not needed); a run that was never published is read from its folder
# MAGIC > `EIS_Results/...`. Evaluates **nothing** and writes no table; it saves the viewer
# MAGIC > pages to `EIS_Results/<order>/plots/` (scratch when the Volume is not writable).
# MAGIC >
# MAGIC > A run that is missing here has not been evaluated yet: use
# MAGIC > **01 Local EIS - Process**, or wait for the **03 Local EIS - Auto** job.

# COMMAND ----------

# MAGIC %md
# MAGIC # Local EIS — Results

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

R = runner.Results(globals(), package_dir=PIPELINE_DIR)
print(f"Local EIS {runner.settings.RUNNER_VERSION} -- results from {R.results_root}")

# COMMAND ----------

# DBTITLE 1,Selection: a saved order, its conditions and test
SEL = R.widgets()

# COMMAND ----------

# MAGIC %md
# MAGIC ## One order

# COMMAND ----------

# DBTITLE 1,Frequency response of the measuring chain
R.show_frequency_response()

# COMMAND ----------

# DBTITLE 1,Nyquist / Bode — every segment with the Gamry sweep of the same test
R.show_nyquist()

# COMMAND ----------

# DBTITLE 1,Median Nyquist — plate median, card medians, aggregate, Gamry
R.show_median_nyquist()

# COMMAND ----------

# DBTITLE 1,Plate heat maps
R.show_heatmaps()

# COMMAND ----------

# DBTITLE 1,MF4 bench log — the whole test, every channel
R.show_mf4()

# COMMAND ----------

# DBTITLE 1,ECM fits — standard ECM and DRT-informed ECM
R.show_ecm()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Across orders (tables)

# COMMAND ----------

# DBTITLE 1,Every published run
RUNS = R.show_runs()

# COMMAND ----------

# DBTITLE 1,Plate parameters against current, every order
R.show_trends()