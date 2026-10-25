"""
runner
======
What the Local EIS notebooks do, as importable code:

    settings.py   every fixed setting and path (edit here, not in a notebook)
    datago.py     the order's FAMOS cards, Gamry sweeps and bench log from datago
    session.py    widgets, run / saved results, the show_* plots   (01 Process)
    results.py    the same plots from saved runs only, and across orders
                  from the tables -- never evaluates               (02 Results)
    tables.py     the silver / gold layer as Delta tables, one row set per run
    auto.py       the incremental job: evaluate what is not published yet
                                                                   (03 Auto)
    views.py      the figures, built from a finished run folder
"""

from .session import Session, condition_key  # noqa: F401
from .results import Results  # noqa: F401
from . import settings  # noqa: F401
from . import tables  # noqa: F401
from . import auto  # noqa: F401
