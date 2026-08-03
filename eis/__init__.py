"""Locally-resolved EIS analysis pipeline for segmented PEM fuel cells.

Processes multi-card imc FAMOS recordings of 80 segment shunt voltages plus a
global cell voltage into per-segment local impedance spectra, with explicit,
measured inter-card synchronisation.

Entry point: :func:`eis.pipeline.run_measurement`.
"""

__version__ = "1.0.0"

from eis.config import PipelineConfig, load_config

__all__ = ["PipelineConfig", "load_config", "__version__"]
