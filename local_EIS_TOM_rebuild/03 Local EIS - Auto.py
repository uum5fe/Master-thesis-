# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC <!-- local_eis:audience-banner -->
# MAGIC > ### 🛠 Data engineering — writes results and tables (scheduled job)
# MAGIC >
# MAGIC > Finds every order / condition / test with FAMOS data that is not in the runs table
# MAGIC > (or was published with other settings, `AUTO_RERUN_CHANGED`) and evaluates the first
# MAGIC > `AUTO_MAX_RUNS` of them exactly like **01 Process** with `rerun`, then publishes.
# MAGIC > Meant to run as a Databricks Job (e.g. nightly); the rest follow at the next run.
# MAGIC >
# MAGIC > Plate and orientation per order: `AUTO_PLATE` / `AUTO_ORIENTATION` in `runner/settings.py`,
# MAGIC > else the order's file names (Naboo = gen2, Kashyyyk = gen1), else gen1 / standard.

# COMMAND ----------

# MAGIC %md
# MAGIC # Local EIS — Auto (incremental job)

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
print(f"Local EIS {runner.settings.RUNNER_VERSION} -- automatic evaluation")

# COMMAND ----------

# DBTITLE 1,Job parameters
dbutils.widgets.dropdown("dry_run", "yes", ["yes", "no"], "Dry run (only show the plan)")
dbutils.widgets.text("max_runs", str(runner.settings.AUTO_MAX_RUNS), "Runs per execution")
dbutils.widgets.text("orders", "", "Orders (empty = AUTO_ORDERS / all)")
DRY = dbutils.widgets.get("dry_run") == "yes"
MAX_RUNS = int(dbutils.widgets.get("max_runs") or runner.settings.AUTO_MAX_RUNS)
ORDERS = [o.strip() for o in dbutils.widgets.get("orders").split(",") if o.strip()] or None

# COMMAND ----------

# DBTITLE 1,Plan (dry run) or evaluate
if DRY:
    PLAN = runner.auto.plan(R, orders=ORDERS)
else:
    DONE = runner.auto.run(R, max_runs=MAX_RUNS, orders=ORDERS)
