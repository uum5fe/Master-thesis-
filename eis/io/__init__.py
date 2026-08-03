"""Raw data readers."""

from eis.io.famos import FamosFile, classify_channel, discover_files, open_famos

__all__ = ["FamosFile", "open_famos", "discover_files", "classify_channel"]
