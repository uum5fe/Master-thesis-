#!/usr/bin/env python3
"""
viewers.py
==========
ONE plot on screen, chosen from drop-downs, instead of a stack of figures.

Every result cell in the runner builds a figure per condition (and per
parameter, per panel ...). Stacked one below the other they fill the
notebook and the one you want is never the one on screen. `selector_html`
takes all of them, keyed by the choices that lead to them, and returns one
HTML block with a drop-down per choice (Condition, Parameter, View, Plot ...)
and a single plot area. Switching is instant -- the figures are all in the
page, nothing is re-run -- and the page also works when saved as a file.

    html = viewers.selector_html({("45A", "Nyquist"): fig1,
                                  ("45A", "|Z| Bode"): fig2, ...},
                                 dims=["Condition", "Plot"])
    displayHTML(html)

An item can be a Plotly figure or a piece of HTML (a table, an <img>), so
tables and PNGs go through the same selector.

Helpers for the runner cells:
    panels(fig)            a multi-panel figure -> [(panel name, figure)]
    by_condition(figs)     {condition: multi-panel figure} -> selector HTML
    plate_maps(...)        {condition: plate_summary.csv} -> heat-map viewer
                           with Condition / Parameter / View (2D spatial
                           interpolated, or values per segment)
"""

from __future__ import annotations

import html as _html
import json
import uuid
from pathlib import Path

import numpy as np

#: the heat-map parameters, in the order of the drop-down:
#: (column in plate_summary.csv, label, unit, decimals)
PLATE_FIELDS = (
    ("R_ohmic", "HFR (Rs)", "mΩ·cm²", 1),
    ("ReZ_1kHz", "Re Z at 1 kHz", "mΩ·cm²", 1),
    ("R_ct", "R_ct (charge transfer)", "mΩ·cm²", 1),
    ("R_mt", "R_mt (mass transport)", "mΩ·cm²", 1),
    ("R_pol", "R_pol (total polarisation)", "mΩ·cm²", 1),
    ("j_dc", "Current density", "A/cm²", 3),
    ("Z_mag_100Hz", "|Z| at 100 Hz", "mΩ·cm²", 1),
    ("phase_100Hz", "Phase at 100 Hz", "°", 1),
    ("T_degC", "Temperature", "°C", 2),
    ("chain_tau_us", "Current-chain lag τ (removed)", "µs", 0),
)
#: joins the choices into one key (a character no label contains: "|Z|"
#: does, so "|" cannot be used)
SEP = "\u241f"
VIEW_2D = "2D spatial (interpolated)"
VIEW_SEG = "Values per segment"


def _plotlyjs_url() -> str:
    try:
        from plotly.offline import get_plotlyjs_version
        return f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    except Exception:                                       # noqa: BLE001
        return "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _payload(item) -> dict:
    """A figure as {"fig": json} or HTML as {"html": str}."""
    if isinstance(item, str):
        return {"html": item}
    if hasattr(item, "to_json"):
        return {"fig": json.loads(item.to_json())}
    raise TypeError(f"cannot show {type(item).__name__}")


def selector_html(items: dict, dims: list[str], title: str = "",
                  height: int | None = None, note: str = "") -> str:
    """One drop-down per entry of `dims`, one plot area.

    items   {(choice for dims[0], choice for dims[1], ...): figure | html}
    dims    names of the drop-downs, e.g. ["Condition", "Parameter", "View"]

    A combination that is not in `items` shows a short message rather than
    a stale figure. The options of each drop-down keep the order in which
    they first appear in `items`.
    """
    items = {tuple(str(x) for x in (k if isinstance(k, tuple) else (k,))): v
             for k, v in items.items() if v is not None}
    if not items:
        return "<p><i>nothing to show</i></p>"
    n = len(dims)
    opts = [list(dict.fromkeys(k[i] for k in items)) for i in range(n)]
    data = {SEP.join(k): _payload(v) for k, v in items.items()}
    # Pieces repeated in many figures (the segment outline, the inlet /
    # outlet labels) are stored once and referenced, which keeps the page
    # small enough for a notebook cell.
    shared, index = [], {}
    seen: dict[str, int] = {}
    for v in data.values():
        if "fig" in v:
            for t in v["fig"].get("data", []) + [
                    v["fig"].get("layout", {}).get("annotations", [])]:
                k = json.dumps(t, sort_keys=True)
                if len(k) >= 2000:
                    seen[k] = seen.get(k, 0) + 1

    def _ref(obj):
        key = json.dumps(obj, sort_keys=True)
        if seen.get(key, 0) < 2:
            return obj
        if key not in index:
            index[key] = len(shared)
            shared.append(obj)
        return {"__ref__": index[key]}

    for v in data.values():
        if "fig" in v:
            v["fig"]["data"] = [_ref(t) for t in v["fig"].get("data", [])]
            lay = v["fig"].get("layout", {})
            if "annotations" in lay:
                lay["annotations"] = _ref(lay["annotations"])
    uid = "sel" + uuid.uuid4().hex[:8]
    ctl = "".join(
        f'<label style="margin-right:18px;font:13px sans-serif">'
        f'<b>{_html.escape(d)}</b> <select id="{uid}_{i}" '
        f'style="font:13px sans-serif;padding:3px 6px;min-width:120px">'
        + "".join(f"<option>{_html.escape(o)}</option>" for o in opts[i])
        + "</select></label>" for i, d in enumerate(dims))
    head = (f'<div style="font:600 15px sans-serif;margin:4px 0 8px">'
            f'{_html.escape(title)}</div>' if title else "")
    foot = (f'<div style="font:11px sans-serif;color:#666;margin-top:4px">'
            f'{_html.escape(note)}</div>' if note else "")
    hpx = f"min-height:{int(height)}px;" if height else ""
    return f"""
{head}<div style="margin-bottom:8px">{ctl}</div>
<div id="{uid}_plot" style="{hpx}"></div>
<div id="{uid}_html"></div>{foot}
<script src="{_plotlyjs_url()}"></script>
<script>
(function() {{
  var DATA = {json.dumps(data, separators=(",", ":"))};
  var SHARED = {json.dumps(shared, separators=(",", ":"))};
  function deref(x) {{
    return (x && x.__ref__ !== undefined) ? SHARED[x.__ref__] : x;
  }}
  var N = {n};
  function sel(i) {{ return document.getElementById("{uid}_" + i); }}
  function draw() {{
    var key = [];
    for (var i = 0; i < N; i++) key.push(sel(i).value);
    var it = DATA[key.join({json.dumps(SEP)})];
    var plot = document.getElementById("{uid}_plot");
    var box = document.getElementById("{uid}_html");
    if (!it) {{
      Plotly.purge(plot); plot.innerHTML = "";
      box.innerHTML = "<p style='font:13px sans-serif;color:#a33'>" +
        "Not available for this combination.</p>";
      return;
    }}
    if (it.html !== undefined) {{
      Plotly.purge(plot); plot.innerHTML = ""; box.innerHTML = it.html;
    }} else {{
      box.innerHTML = "";
      var lay = Object.assign({{}}, it.fig.layout);
      if (lay.annotations) lay.annotations = deref(lay.annotations);
      Plotly.react(plot, it.fig.data.map(deref), lay, {{responsive: true}});
    }}
  }}
  for (var i = 0; i < N; i++) sel(i).addEventListener("change", draw);
  draw();
}})();
</script>"""


# ---------------------------------------------------------------------------
# helpers for the runner cells
# ---------------------------------------------------------------------------

def panels(fig, panel_height: int = 560, panel_width: int | None = 1150
           ) -> list[tuple[str, object]]:
    """A (multi-panel) figure as [(panel name, single-panel figure)].

    Panel names are the subplot titles; a figure without subplots is one
    panel called "Plot"."""
    import figure_panels
    figs = figure_panels.split_subplots(fig, panel_height=panel_height,
                                        panel_width=panel_width)
    names = []
    for ann in (fig.layout.annotations or []):
        if getattr(ann, "xref", "") == "paper" and getattr(ann, "text", ""):
            names.append(ann.text)
    if len(figs) == 1:
        return [("Plot", figs[0])]
    if len(names) < len(figs):
        names = [f"Panel {i + 1}" for i in range(len(figs))]
    return list(zip(names[:len(figs)], figs))


def by_condition(figs: dict, title: str = "", note: str = "",
                 panel_dim: str = "Plot", **kw) -> str:
    """{condition: figure or html} -> selector; a multi-panel figure becomes
    a second drop-down (`panel_dim`) over its panels."""
    items, multi = {}, False
    for cond, fig in figs.items():
        if fig is None:
            continue
        if isinstance(fig, str):
            items[(cond, "Table")] = fig
            continue
        ps = panels(fig, **kw)
        multi |= len(ps) > 1
        for name, f in ps:
            items[(cond, name)] = f
    if not multi:
        items = {(k[0],): v for k, v in items.items()}
        return selector_html(items, ["Condition"], title=title, note=note)
    return selector_html(items, ["Condition", panel_dim], title=title,
                         note=note)


def img_html(path, width: str = "100%") -> str:
    """A PNG as an inline <img>, for selectors over saved figures."""
    import base64
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="max-width:{width}">'


def table_html(df, decimals: int = 4) -> str:
    return df.round(decimals).to_html(border=0, classes="dataframe")


def plate_fields(summary_csv, fields=PLATE_FIELDS) -> list:
    """plate_plotly.Field per parameter present in a plate_summary.csv."""
    import pandas as pd
    import plate_plotly
    df = pd.read_csv(summary_csv)
    cls = ({str(int(r["segment"])): str(r["class"]) for _, r in df.iterrows()}
           if "class" in df.columns else {})
    out = []
    for col, label, unit, dec in fields:
        if col not in df.columns:
            continue
        vals = {str(int(r["segment"])): float(r[col]) for _, r in df.iterrows()
                if pd.notna(r[col]) and np.isfinite(float(r[col]))}
        if vals:
            out.append(plate_plotly.Field(col, label, unit, vals, cls,
                                          decimals=dec))
    return out


def plate_maps(summaries: dict, title: str = "", fields=PLATE_FIELDS,
               views=(VIEW_2D, VIEW_SEG), height: int = 680,
               width: int | None = 1300) -> str:
    """{condition: plate_summary.csv or list[Field]} -> one heat-map viewer
    with Condition / Parameter / View drop-downs."""
    import plate_plotly
    items = {}
    for cond, src in summaries.items():
        flds = src if isinstance(src, list) else plate_fields(src, fields)
        head = f"{title} / {cond}" if title else str(cond)
        for fd in flds:
            for view in views:
                if view == VIEW_2D:
                    fig = plate_plotly.interpolated_plate(
                        fd, title=head, height=height, width=width)
                else:
                    fig = plate_plotly.interactive_plate(
                        [fd], title=head, height=height, width=width)
                items[(cond, fd.label, view)] = fig
    return selector_html(items, ["Condition", "Parameter", "View"],
                         title=f"Plate maps — {title}" if title else "")
