"""Chart colour, in one place, by the job each colour does.

Three jobs appear in this dashboard and they do not share a palette:

* **sequential** -- a plate map of one parameter's magnitude.  One hue, light
  to dark.  Never a rainbow: a rainbow ramp invents boundaries where the data
  has none, which on a 72-segment plate map reads as structure that is not
  there.
* **categorical** -- identity, e.g. three conditions on one Nyquist.  A fixed
  hue order, assigned by slot and never cycled; capped at three because a
  scatter compares every pair, and past three slots the pairs stop being
  separable under colour-vision deficiency.
* **status** -- measured / inferred / band-limited / bad.  Reserved colours,
  never reused as a series, and always shipped with a marker symbol and a text
  label so the state never rests on hue alone.
"""

from __future__ import annotations

#: Chart surfaces, light and dark.
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
TEXT_PRIMARY = {"light": "#0b0b0b", "dark": "#ffffff"}
TEXT_SECONDARY = {"light": "#52514e", "dark": "#c3c2b7"}
GRID = {"light": "#e6e5e1", "dark": "#333331"}

#: Categorical slots, in assignment order.  Scatter/overlay forms use at most
#: the first three (the all-pairs-validated subset).
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"],
}
SCATTER_SLOTS = 3

#: Sequential blue, 100 -> 700, for continuous magnitude.
SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                   "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                   "#184f95", "#104281", "#0d366b"]
#: A second sequential context on the same screen takes the next hue.
SEQUENTIAL_ORANGE = ["#fce3d6", "#f8c3a8", "#f4a37c", "#ef8353", "#eb6834",
                     "#d4551f", "#b04617", "#8a3612", "#63270d"]

#: Reserved state colours.  Never a series.
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

#: How a segment's class is drawn.  Symbol and label carry the meaning; the
#: colour only reinforces it.
SEGMENT_STATE = {
    "measured":     {"color": STATUS["good"],     "symbol": "square",
                     "label": "measured"},
    "inferred":     {"color": STATUS["warning"],  "symbol": "square-open",
                     "label": "inferred from neighbours"},
    # Filled symbols only.  The "-thin" and "-open" plotly symbols are drawn
    # from marker.line rather than marker.color, so with the surface-coloured
    # ring every other state uses they come out invisible.
    "band_limited": {"color": STATUS["serious"],  "symbol": "diamond",
                     "label": "band-limited (not measurable)"},
    "bad":          {"color": STATUS["critical"], "symbol": "x",
                     "label": "hardware-bad"},
}


def plotly_colorscale(steps: list[str] | None = None) -> list[list]:
    """A plotly colorscale from an ordered list of hex steps."""
    steps = steps or SEQUENTIAL_BLUE
    n = len(steps) - 1
    return [[i / n, c] for i, c in enumerate(steps)]


def layout(mode: str = "light", **kw) -> dict:
    """Recessive axes, transparent-to-surface background, readable ink."""
    m = "dark" if mode == "dark" else "light"
    base = dict(
        paper_bgcolor=SURFACE[m], plot_bgcolor=SURFACE[m],
        font=dict(color=TEXT_PRIMARY[m], size=13),
        xaxis=dict(gridcolor=GRID[m], zerolinecolor=GRID[m],
                   linecolor=GRID[m], ticks="outside", ticklen=4,
                   tickcolor=GRID[m]),
        yaxis=dict(gridcolor=GRID[m], zerolinecolor=GRID[m],
                   linecolor=GRID[m], ticks="outside", ticklen=4,
                   tickcolor=GRID[m]),
        margin=dict(l=60, r=24, t=48, b=52),
        hoverlabel=dict(bgcolor=SURFACE[m], font_size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    base.update(kw)
    return base
