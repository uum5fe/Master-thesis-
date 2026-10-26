# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC <!-- local_eis:audience-banner -->
# MAGIC > ### 🛠 Data engineering — writes the calibration tables
# MAGIC >
# MAGIC > Publishes a plate's calibration folder (`curr.csv`, `temp.csv`, `chain_gain.csv`,
# MAGIC > `Abgleich/`) as a NEW version in `<TABLE_SCHEMA>.local_eis_calibration_*`. Nothing is
# MAGIC > overwritten: earlier runs keep the version they were evaluated with. With
# MAGIC > `CALIBRATION_SOURCE = "tables"` (`runner/settings.py`) every run then takes its
# MAGIC > calibration from these tables -- the version valid at the measurement date, unless
# MAGIC > `CALIBRATION_VERSION` pins one. A folder identical to the newest version is not
# MAGIC > published again.

# COMMAND ----------

# MAGIC %md
# MAGIC # Local EIS — Calibration

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

# DBTITLE 1,Published versions
_v = runner.calibration.versions(spark)
display(_v) if len(_v) else print("no calibration published yet")

# COMMAND ----------

# DBTITLE 1,Which folder (checked, not yet published)
dbutils.widgets.dropdown("plate", "gen1", ["gen1", "gen2"], "Plate (gen1 green/Kashyyyk, gen2 blue/Naboo)")
dbutils.widgets.text("folder", "", "Calibration folder (empty = EIS_config/<plate> on the Volume)")
dbutils.widgets.text("valid_from", "", "Valid from YYYY-MM-DD (empty: first version 2000-01-01, later today)")
dbutils.widgets.text("note", "", "Note (e.g. Abgleich delivery 10/2026)")
dbutils.widgets.dropdown("publish", "no", ["no", "yes"], "Publish as a new version")
PLATE = dbutils.widgets.get("plate")
FOLDER = dbutils.widgets.get("folder").strip() or str(runner.settings.EIS_CONFIG_ROOT / PLATE)
_bad = runner.calibration.check_folder(FOLDER, PLATE)
print(f"{PLATE}: {FOLDER}\n  " + ("; ".join(_bad) if _bad else "curr.csv and temp.csv OK"))

# COMMAND ----------

# DBTITLE 1,Publish (only with publish = yes)
if dbutils.widgets.get("publish") != "yes":
    print("publish = no -- nothing written")
else:
    VERSION = runner.calibration.publish(
        spark, PLATE, FOLDER, valid_from=dbutils.widgets.get("valid_from").strip() or None,
        note=dbutils.widgets.get("note"))
    display(runner.calibration.versions(spark, PLATE))
