"""The 72-segment plate as it is physically wired, and how to draw it.

Separate from `app.plates.registry`, which answers "which plate spec is this
run on"; this package answers "where is segment 34 on the copper, and what
shape is it".  The pad matrix is the harness numbering transcribed unchanged,
and a segment is the SET OF PADS carrying its number -- so its area is exact
and its boundary, staircases included, is walked from the pad edges rather
than approximated by a bounding box.

    from app.plates.geometry import PLATE, plate_figure, write_html

`plate_model.self_check()` asserts the 72 segments tile all 900 pads exactly
once and that the areas sum to 304.92 cm2.  A pad matrix with a typo draws a
heat map that looks entirely plausible and is wrong, so that check runs in the
test suite.
"""

from .plate_model import (PLATE, Plate, Segment, draw_plate, synthetic_fields,
                          to_pad_matrix)
from .plate_plotly import plate_figure
from .plate_viewer import Field, write_html

__all__ = ["PLATE", "Plate", "Segment", "draw_plate", "synthetic_fields",
           "to_pad_matrix", "plate_figure", "Field", "write_html"]
