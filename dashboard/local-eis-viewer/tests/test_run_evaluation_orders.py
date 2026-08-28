"""Selecting one order id out of a campaign folder holding several.

The FAMOS root is now the parent that holds one folder per order, so the
order id is the thing that picks a measurement -- and the same order is
written three ways on disk (folder `2612025_27_08`, cards
`RO2612025-01_Current_...`, catalogue key `2612025`). Whichever one someone
copies has to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_evaluation as RE


@pytest.mark.parametrize("written", [
    "2612025",            # the catalogue key
    "RO2612025",          # the card prefix
    "RO2612025-01",       # the card prefix with the station suffix
    "ro2612025-01",       # ... typed in lower case
    "2612025_27_08",      # the folder name
])
def test_every_spelling_of_one_order_selects_it(written):
    assert RE.normalise_order_id(written) == "2612025"


def test_the_station_suffix_is_dropped_not_absorbed():
    """`-01` is a station, not part of the number.

    Stripping non-digits before removing it welds them together --
    "RO2612025-01" becomes "261202501", which matches nothing and looks like
    a path problem rather than a filter one.
    """
    assert RE.normalise_order_id("RO2612025-01") != "261202501"
    assert RE.normalise_order_id("RO2612025-01") == "2612025"


def test_different_orders_stay_different():
    """Tolerance must not collapse two orders into one."""
    ids = {RE.normalise_order_id(x)
           for x in ("RO2611976-01", "RO2612025-01", "RO2611959-01")}
    assert ids == {"2611976", "2612025", "2611959"}
