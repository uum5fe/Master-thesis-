"""
runner
======
What the "Local EIS Pipeline Runner" notebook does, as importable code:

    settings.py   every fixed setting and path (edit here, not in the notebook)
    datago.py     the order's Gamry sweeps and bench log from datago
    session.py    widgets, run / saved results, the show_* plots
    views.py      the figures, built from a finished run folder
"""

from .session import Session, condition_key  # noqa: F401
from . import settings  # noqa: F401
