"""Locally-resolved EIS analysis pipeline for segmented PEM fuel cells.

Processes multi-card imc FAMOS recordings of 80 segment shunt voltages plus a
global cell voltage into per-segment local impedance spectra, with explicit,
measured inter-card synchronisation and an explicit, measured high-frequency
correction chain.

Layout::

    eis.io          FAMOS reader
    eis.sync        skew, drift, resampling
    eis.calibrate   shunt and temperature calibration
    eis.spectra     gated Welch, multi-resolution Welch, synchronous DFT
    eis.hf          the high-frequency accuracy chain
    eis.validate    Kramers-Kronig, stationarity, plate identity
    eis.model       equivalent-circuit fitting
    eis.viz         static figures
    eis.dashboard   the interactive all-segment plate view
    eis.pipeline    bronze / silver / gold, and the main script

Entry point: :func:`eis.pipeline.run_measurement`.
"""

from typing import Any

__version__ = "2.0.0"

__all__ = ["PipelineConfig", "load_config", "run_measurement", "__version__"]


def __getattr__(name: str) -> Any:
    """Expose the pipeline's headline names without importing it eagerly.

    ``import eis`` should not drag in scipy, pandas and matplotlib; a script
    that only wants :func:`eis.io.famos.parse_famos_header` pays for nothing
    else.  The names below still resolve on first use.
    """
    if name in ("PipelineConfig", "load_config"):
        from eis.pipeline import config

        return getattr(config, name)
    if name == "run_measurement":
        from eis.pipeline.main import run_measurement

        return run_measurement
    raise AttributeError(f"module 'eis' has no attribute {name!r}")
