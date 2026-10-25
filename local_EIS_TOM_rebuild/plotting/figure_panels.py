#!/usr/bin/env python3
"""
figure_panels.py
================
Show a multi-panel Plotly figure as separate, full-width figures, one below
the other.

WHY
---
Every spectrum cell in the runner builds its figure with `make_subplots`:
Nyquist, |Z| and phase side by side in one 1500 px row, or a 2 x 3 grid of
segments. Side by side, each panel gets a third of the width -- the Nyquist
arc is squeezed into a tall narrow box and the legend steals the rest. Rather
than rewrite eleven cells, each keeps building the figure it always built and
hands it to `split_subplots`, which returns one figure per panel carrying:

  * that panel's traces (legend re-enabled, one entry per legend group, so a
    segment can be toggled in the Bode panel too, not only in the Nyquist);
  * that panel's axis settings (log type, titles, ranges, scaleanchor);
  * shapes and annotations anchored to that panel (the 2 % / 5 % guides);
  * the main title, prefixed with the panel's own subplot title;
  * any dropdown (updatemenus) with its visibility masks cut down to the
    traces that panel holds, so the "Seg N" selector still works per panel.

A figure that was not built with subplots comes back unchanged, as a list of
one, so a caller never has to know which kind it was handed.
"""

from __future__ import annotations

import copy
import re

try:                                            # plotly is optional for the
    import plotly.graph_objects as go           # pipeline, required here
except ImportError:                             # pragma: no cover
    go = None

_AXIS_RE = re.compile(r"^xaxis(\d*)$")

#: layout keys that belong to one panel or to the page size, rebuilt per panel
_PER_PANEL_KEYS = ("annotations", "shapes", "updatemenus", "sliders",
                   "images", "width", "height", "title", "grid")


def _panels(layout: dict) -> list[tuple[str, str, str, str]]:
    """(xref, yref, xkey, ykey) for every x axis anchored to a y axis."""
    out = []
    for key in sorted((k for k in layout if _AXIS_RE.match(k)),
                      key=lambda k: int(_AXIS_RE.match(k).group(1) or 1)):
        num = _AXIS_RE.match(key).group(1)
        anchor = (layout.get(key) or {}).get("anchor")
        if not anchor or anchor == "free" or not anchor.startswith("y"):
            continue
        ykey = "yaxis" + anchor[1:]
        if ykey not in layout:
            continue
        out.append(("x" + num, anchor, key, ykey))
    return out


def _clean_axis(ax: dict, own_ref: str, new_ref: str) -> dict:
    ax = copy.deepcopy(ax)
    for k in ("domain", "anchor", "matches", "overlaying", "position"):
        ax.pop(k, None)
    if "scaleanchor" in ax:
        if ax["scaleanchor"] == own_ref:
            ax["scaleanchor"] = new_ref
        else:
            ax.pop("scaleanchor")
    return ax


def _remap_ref(ref, xref, yref):
    """'x3' -> 'x', 'x3 domain' -> 'x domain'; None when it is another panel's."""
    if ref is None:
        return None
    for own, new in ((xref, "x"), (yref, "y")):
        if ref == own:
            return new
        if ref == f"{own} domain":
            return f"{new} domain"
    return None


def _title_text(layout: dict) -> str:
    t = layout.get("title")
    if isinstance(t, dict):
        return t.get("text") or ""
    return t or ""


def _with_panel(title: str, sub: str) -> str:
    if not sub:
        return title
    if not title:
        return f"<b>{sub}</b>"
    head, _, rest = title.partition("<br>")
    return f"<b>{sub}</b> · {head}" + (f"<br>{rest}" if rest else "")


def split_subplots(fig, panel_height: int = 560, panel_width: int | None = 1150,
                   skip_empty: bool = True) -> list:
    """One figure per subplot panel, in row-major order. See module docstring."""
    if go is None:
        return [fig]
    full = fig.to_plotly_json()
    layout = full.get("layout", {}) or {}
    data = full.get("data", []) or []
    panels = _panels(layout)
    if len(panels) <= 1:
        return [fig]

    title = _title_text(layout)
    annotations = layout.get("annotations", []) or []
    shapes = layout.get("shapes", []) or []
    menus = layout.get("updatemenus", []) or []

    base = {k: copy.deepcopy(v) for k, v in layout.items()
            if k not in _PER_PANEL_KEYS and not k.startswith(("xaxis", "yaxis"))}
    margin = dict(base.get("margin") or {})
    margin["t"] = max(int(margin.get("t", 0) or 0), 100)
    base["margin"] = margin

    # Legend groups that had an entry anywhere in the original figure. A
    # group whose entry lived only on the Nyquist panel gets one on every
    # panel; a trace deliberately kept out of the legend stays out.
    def _key(tr):
        return tr.get("legendgroup") or tr.get("name")

    legend_keys = {_key(t) for t in data
                   if t.get("showlegend", True) is not False and _key(t)}

    used_title_idx: set[int] = set()
    out = []
    for p_idx, (xref, yref, xkey, ykey) in enumerate(panels):
        idx = [i for i, t in enumerate(data)
               if (t.get("xaxis") or "x") == xref
               and (t.get("yaxis") or "y") == yref]
        if skip_empty and not idx:
            continue

        seen: set = set()
        traces = []
        for i in idx:
            tr = copy.deepcopy(data[i])
            tr["xaxis"], tr["yaxis"] = "x", "y"
            k = _key(tr)
            if k in legend_keys and k not in seen:
                tr["showlegend"] = True
                seen.add(k)
            else:
                tr["showlegend"] = False
            traces.append(tr)

        xdom = (layout.get(xkey) or {}).get("domain", [0, 1])
        ydom = (layout.get(ykey) or {}).get("domain", [0, 1])
        sub, keep_ann = "", []
        for a_idx, a in enumerate(annotations):
            ax_ref, ay_ref = a.get("xref", "x"), a.get("yref", "y")
            if ax_ref == "paper" and ay_ref == "paper":
                x, y = a.get("x"), a.get("y")
                is_title = (x is not None and y is not None
                            and abs(x - (xdom[0] + xdom[1]) / 2) < 1e-3
                            and abs(y - ydom[1]) < 1e-3)
                if is_title and a_idx not in used_title_idx:
                    sub = a.get("text", "") or ""
                    used_title_idx.add(a_idx)
                continue
            nx, ny = _remap_ref(ax_ref, xref, yref), _remap_ref(ay_ref, xref, yref)
            if nx and ny:
                a = copy.deepcopy(a)
                a["xref"], a["yref"] = nx, ny
                keep_ann.append(a)

        keep_shapes = []
        for s in shapes:
            nx = _remap_ref(s.get("xref", "x"), xref, yref)
            ny = _remap_ref(s.get("yref", "y"), xref, yref)
            if nx and ny:
                s = copy.deepcopy(s)
                s["xref"], s["yref"] = nx, ny
                keep_shapes.append(s)

        panel_title = _with_panel(title, sub)
        keep_menus = []
        for m in menus:
            m = copy.deepcopy(m)
            for b in m.get("buttons", []) or []:
                args = b.get("args") or []
                if args and isinstance(args[0], dict):
                    for k, v in list(args[0].items()):
                        if isinstance(v, list) and len(v) == len(data):
                            args[0][k] = [v[i] for i in idx]
                if len(args) > 1 and isinstance(args[1], dict):
                    t = args[1].get("title.text")
                    if isinstance(t, str):
                        args[1]["title.text"] = _with_panel(t, sub)
            keep_menus.append(m)

        lay = copy.deepcopy(base)
        lay["xaxis"] = _clean_axis(layout.get(xkey) or {}, xref, "x")
        lay["yaxis"] = _clean_axis(layout.get(ykey) or {}, xref, "x")
        lay["annotations"] = keep_ann
        lay["shapes"] = keep_shapes
        if keep_menus:
            lay["updatemenus"] = keep_menus
        title_obj = copy.deepcopy(layout.get("title")) \
            if isinstance(layout.get("title"), dict) else {}
        title_obj["text"] = panel_title
        lay["title"] = title_obj
        lay["height"] = panel_height
        if panel_width:
            lay["width"] = panel_width
        out.append(go.Figure({"data": traces, "layout": lay}))
    return out or [fig]
