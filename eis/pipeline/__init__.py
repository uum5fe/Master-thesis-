"""The three-tier pipeline.

======================================  ====================================
module                                  responsibility
======================================  ====================================
:mod:`eis.pipeline.config`              every parameter that changes a result
:mod:`eis.pipeline.utils`               logging, provenance, alignment, tables
:mod:`eis.pipeline.bronze`              files -> aligned, scaled signals
:mod:`eis.pipeline.silver`              signals -> spectra with uncertainty
:mod:`eis.pipeline.gold`                spectra -> parameters, maps, report
:mod:`eis.pipeline.main`                the order they run in, and the CLI
======================================  ====================================

The names re-exported here are the pipeline's public surface::

    from eis.pipeline import load_config, run_measurement
"""

from eis.pipeline.bronze import (
    BronzeCondition, CardInventory, CardSync, SegmentSignal, SyncReport,
    inventory_card, measure_sync, run_bronze,
)
from eis.pipeline.config import PipelineConfig, load_config
from eis.pipeline.gold import (
    ConditionResult, map_label, run_gold, scalar_map, summarise, write_outputs,
    write_report,
)
from eis.pipeline.silver import SegmentSpectrum, SilverCondition, run_silver

__all__ = [
    # configuration
    "PipelineConfig", "load_config",
    # bronze
    "BronzeCondition", "CardInventory", "CardSync", "SegmentSignal",
    "SyncReport", "inventory_card", "measure_sync", "run_bronze",
    # silver
    "SegmentSpectrum", "SilverCondition", "run_silver",
    # gold
    "ConditionResult", "run_gold", "scalar_map", "map_label", "summarise",
    "write_outputs", "write_report",
    # orchestration
    "run_condition", "run_measurement",
]


def __getattr__(name):
    """Resolve the orchestration entry points on first use.

    Importing :mod:`eis.pipeline.main` eagerly here would put it in
    ``sys.modules`` before ``python -m eis.pipeline.main`` gets to run it, which
    makes the interpreter warn about executing a module it has already
    imported.  Deferring the import keeps both spellings clean.
    """
    if name in ("run_condition", "run_measurement"):
        from eis.pipeline import main

        return getattr(main, name)
    raise AttributeError(f"module 'eis.pipeline' has no attribute {name!r}")
