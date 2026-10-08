"""Make the pipeline's folders importable for every test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eis_paths  # noqa: E402,F401
