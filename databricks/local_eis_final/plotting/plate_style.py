#!/usr/bin/env python3
"""
plate_style.py
==============
One look for every plate heat map: the colour scale, the text sizes, which
way round the plate is drawn, and where the gas and coolant ports are.

WHY ONE MODULE
--------------
The pipeline draws the plate in six places (gold's map_*.png and map_*.html,
plate_figure, plate_maps, r2d2_geometry.plot_map and the notebook cells), and
each had grown its own colour ramp -- viridis for the HFR, inferno for R_ct,
magma for R_mt, RdYlBu_r in gold -- and gold even drew the plate upside down
(pad row 1 at the bottom) while every other map put it at the top. A reader
comparing two maps could not trust that one colour meant one value, or that
the top left of one map was the top left of the other. Everything that
decides how a map LOOKS now lives here, and every drawing reads it.

COLOUR
------
Plotly's "Jet" scale, stop for stop (plotly.js colorscale definition):
dark blue -> blue -> cyan -> yellow -> red -> dark red. It is the scale of
the bench's reference intensity plots (temperature and current density per
step), so a local-EIS map and a bench map read the same way.

ORIENTATION
-----------
r2d2_geometry's pad map is in PLATE coordinates: x = 0 at pad column 1,
y = 0 at pad row 1, y pointing down. The gases and the coolant are fixed to
those coordinates (config.py, "WHICH WAY THE GASES ACTUALLY GO"):

    x = 0     (pad column 1)  : H2 in (bottom), air out (top), coolant out
    x = 252   (pad column 45) : air in (bottom), H2 out (top), coolant in

(the coolant end is config.COOLANT_INLET_END).

config.PLATE_VIEW_MIRRORED draws the plate seen from the other side, i.e.
mirrored left <-> right. EVERYTHING mirrors together -- segments, numbers,
ports, arrows, coolant, sensors -- so the air inlet stays next to the
segments it really feeds. Moving only the port labels would put the air
inlet beside the wrong segments and every gradient on the map would be read
backwards.

TEXT
----
The VALUE is what a reader came for, so it is the large bold text in every
segment; the segment NUMBER is small, above it.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------

#: plotly.js "Jet": (position, colour). Same stops as the reference plots.
JET_STOPS: tuple[tuple[float, str], ...] = (
    (0.000, "rgb(0,0,131)"),
    (0.125, "rgb(0,60,170)"),
    (0.375, "rgb(5,255,255)"),
    (0.625, "rgb(255,255,0)"),
    (0.875, "rgb(250,0,0)"),
    (1.000, "rgb(128,0,0)"),
)

#: name every drawing uses for the heat-map ramp
HEATMAP_CMAP = "plotly_jet"

#: MATLAB's "parula" (the ramp of the bench's interpolated MATLAB maps:
#: dark blue -> blue -> teal -> green -> yellow), as nine stops. Used by the
#: interpolated maps (plate_figure.draw_flow_plate(render="interpolated")).
PARULA_STOPS: tuple[str, ...] = (
    "#352a87", "#0f5cdd", "#1481d6", "#06a4ca", "#2eb7a4",
    "#87bf77", "#d1bb59", "#fec832", "#f9fb0e")

#: interpolated maps: ramp, and the strength of the soft gloss (0 = flat)
INTERP_CMAP = "parula"
INTERP_GLOSS = 0.35
#: outline of the "segment measured" squares on interpolated maps
MEASURED_MARK = "#c2185b"

#: fill of a segment with no value
MISSING_FILL = (0.83, 0.83, 0.84, 1.0)


def _rgb01(spec: str) -> tuple[float, float, float]:
    r, g, b = (int(x) for x in spec[spec.index("(") + 1:-1].split(","))
    return r / 255.0, g / 255.0, b / 255.0


def mpl_cmap(name: str | None = None):
    """Matplotlib colormap for the heat maps (plotly Jet unless overridden).

    `name` lets a caller still ask for another matplotlib ramp explicitly;
    None, "jet" and HEATMAP_CMAP all give the reference scale.
    """
    from matplotlib.colors import LinearSegmentedColormap
    if name in (None, "", "jet", "Jet", HEATMAP_CMAP):
        return LinearSegmentedColormap.from_list(
            HEATMAP_CMAP, [(p, _rgb01(c)) for p, c in JET_STOPS], N=256)
    if name in ("parula", "Parula"):
        return LinearSegmentedColormap.from_list("parula", PARULA_STOPS,
                                                 N=256)
    import matplotlib.pyplot as plt
    return plt.get_cmap(name)


def plotly_colorscale(name: str | None = None) -> list[list]:
    """The same scale as a Plotly colorscale (Jet; "parula" on request)."""
    if name in ("parula", "Parula"):
        n = len(PARULA_STOPS) - 1
        return [[i / n, c] for i, c in enumerate(PARULA_STOPS)]
    return [[p, c] for p, c in JET_STOPS]


def ink(rgba) -> str:
    """Text colour that stays legible on a fill (WCAG relative luminance)."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
           for c in rgba[:3]]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#141414" if lum > 0.42 else "#ffffff"


# ---------------------------------------------------------------------------
# text sizes (points): the value large and bold, the number small
# ---------------------------------------------------------------------------

VALUE_FONT = 9.4
VALUE_FONT_NARROW = 6.4
NUMBER_FONT = 5.8
NUMBER_FONT_NARROW = 4.6
#: offsets from the label point, in plate mm (y down): number above, value below
NUMBER_DY_MM = -2.6
VALUE_DY_MM = 1.5


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------

def mirrored() -> bool:
    """True when the plate is drawn mirrored (read at call time, so a
    notebook can flip config.PLATE_VIEW_MIRRORED and redraw)."""
    try:
        import config
        return bool(getattr(config, "PLATE_VIEW_MIRRORED", False))
    except Exception:                                       # noqa: BLE001
        return False


def xlim(lo: float, hi: float) -> tuple[float, float]:
    """Matplotlib x limits: reversed when mirrored, so every patch, line and
    text position mirrors with the axis and nothing has to be recomputed."""
    return (hi, lo) if mirrored() else (lo, hi)


def outward_ha(physical_side: str) -> str:
    """Horizontal alignment for text set just OUTSIDE the x = 0 ("x0") or
    x = W ("xW") edge, so it grows away from the plate in either view."""
    left_on_screen = (physical_side == "x0") != mirrored()
    return "right" if left_on_screen else "left"


def screen_side(physical_side: str) -> str:
    """'left' or 'right': where the x = 0 / x = W edge lands on screen."""
    return "left" if (physical_side == "x0") != mirrored() else "right"


# ---------------------------------------------------------------------------
# ports: what enters and leaves at each end, in plate coordinates
# ---------------------------------------------------------------------------

ANODE_COLOUR = "#a5341f"
CATHODE_COLOUR = "#2c455d"
COOLANT_COLOUR = "#0e7c86"


def coolant_inlet_end() -> str:
    """'x0' or 'xW': the plate end the coolant enters at (config)."""
    try:
        import config
        end = getattr(config, "COOLANT_INLET_END", "xW")
    except Exception:                                       # noqa: BLE001
        end = "xW"
    return end if end in ("x0", "xW") else "xW"


def flow_arrangement() -> str:
    try:
        import config
        return getattr(config, "FLOW_ARRANGEMENT", "unknown")
    except Exception:                                       # noqa: BLE001
        return "unknown"


def end_streams() -> dict[str, list[tuple[str, str, str]]]:
    """Per plate end ('x0', 'xW'): (label, role, colour) of what enters or
    leaves there, gas ports first, coolant last."""
    co = flow_arrangement() == "co"
    x0 = [("H₂ IN", "anode inlet", ANODE_COLOUR)]
    xw = [("H₂ OUT", "anode outlet", ANODE_COLOUR)]
    if co:
        x0.append(("AIR IN", "cathode inlet", CATHODE_COLOUR))
        xw.append(("AIR OUT", "cathode outlet", CATHODE_COLOUR))
    else:
        x0.append(("AIR OUT", "cathode outlet", CATHODE_COLOUR))
        xw.append(("AIR IN", "cathode inlet", CATHODE_COLOUR))
    cin = coolant_inlet_end()
    (x0 if cin == "x0" else xw).append(("COOLANT IN", "coolant inlet",
                                        COOLANT_COLOUR))
    (xw if cin == "x0" else x0).append(("COOLANT OUT", "coolant outlet",
                                        COOLANT_COLOUR))
    return {"x0": x0, "xW": xw}


def port_slot(label: str) -> str:
    """Where a port sits in its end's column: the coolant in the middle, the
    gas outlet above it and the gas inlet below it."""
    if label.startswith("COOLANT"):
        return "mid"
    return "top" if label.endswith("OUT") else "bottom"


def port_kind(label: str) -> str:
    """'anode', 'cathode' or 'coolant'."""
    return ("coolant" if label.startswith("COOLANT")
            else "anode" if label.startswith("H") else "cathode")


def _arrow(stream_from: str) -> str:
    """'left to right' / 'right to left' for a stream entering at an end."""
    return ("left to right" if screen_side(stream_from) == "left"
            else "right to left")


def flow_note() -> str:
    """One line describing the flows AS DRAWN (it changes with mirroring)."""
    arr = flow_arrangement()
    h2 = _arrow("x0")
    air = _arrow("x0" if arr == "co" else "xW")
    cool = _arrow(coolant_inlet_end())
    head = {"counter": "counter-flow", "co": "co-flow"}.get(
        arr, "flow arrangement not recorded")
    view = "  [mirrored view]" if mirrored() else ""
    return f"{head}: H2 {h2}, air {air}, coolant {cool}{view}"


def annotate_ends(ax, width_mm: float, height_mm: float,
                  gap_mm: float = 3.0, fontsize: float = 8.5) -> None:
    """Write what enters and leaves at each end beside a plain plate map.

    For the maps without drawn ports (plate_maps, r2d2_geometry.plot_map).
    Positions are in plate mm, so the labels follow the plate when the axis
    is mirrored; the caller must leave room in its x limits.
    """
    for end, x in (("x0", -gap_mm), ("xW", width_mm + gap_mm)):
        ha = outward_ha(end)
        for lab, role, col in end_streams()[end]:
            # gas outlet above the coolant, gas inlet below it
            y = height_mm * {"top": 0.12, "mid": 0.5,
                             "bottom": 0.88}[port_slot(lab)]
            ax.text(x, y, f"{lab}\n{role}", ha=ha, va="center",
                    fontsize=fontsize, fontweight="bold", color=col,
                    linespacing=1.15, zorder=7)
