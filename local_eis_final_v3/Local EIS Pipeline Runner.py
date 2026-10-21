# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Local EIS — R2-D2 segmented plate
# MAGIC Pick the order, conditions and test in the widgets, then **Run all**.
# MAGIC Every plot is saved automatically to `EIS_Results/<order>/plots/`, every result to
# MAGIC `EIS_Results/<order>/<condition>_Test<nn>/`. Settings and paths: `runner/settings.py`.

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
print(f"Local EIS runner {runner.settings.RUNNER_VERSION} -- modules from {PIPELINE_DIR}")

# COMMAND ----------

# DBTITLE 1,Selection (widgets)
SEL = R.widgets()

# COMMAND ----------

# DBTITLE 1,Run the pipeline (Run mode: cache = use saved results, rerun = recompute and overwrite)
RESULTS = R.run()

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
