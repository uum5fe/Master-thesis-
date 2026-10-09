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


def _slim(obj, sig: int = 3):
    """Plotly writes numeric arrays as binary blocks ({"dtype", "bdata"}),
    which do not compress. Turn them into plain numbers rounded to `sig`
    significant digits (far below what a plot can show) so gzip can work."""
    import base64
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            a = np.frombuffer(base64.b64decode(obj["bdata"]),
                              dtype=np.dtype(obj["dtype"]))
            if "shape" in obj:
                shp = obj["shape"]
                shp = ([int(x) for x in shp.split(",")] if isinstance(shp, str)
                       else list(shp))
                a = a.reshape(shp)
            if a.dtype.kind == "f":
                return [None if not np.isfinite(x) else float(f"{x:.{sig}g}")
                        for x in a.ravel()] if a.ndim == 1 else [
                    [None if not np.isfinite(x) else float(f"{x:.{sig}g}")
                     for x in row] for row in a]
            return a.tolist()
        return {k: _slim(v, sig) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_slim(v, sig) for v in obj]
    if isinstance(obj, float):
        return float(f"{obj:.{max(sig, 6)}g}")
    return obj


def _payload(item) -> dict:
    """A figure as {"fig": json} or HTML as {"html": str}."""
    if isinstance(item, str):
        return {"html": item}
    if hasattr(item, "to_json"):
        return {"fig": _slim(json.loads(item.to_json()))}
    raise TypeError(f"cannot show {type(item).__name__}")


def _pack(obj) -> str:
    """JSON -> gzip -> base64: the payload of a page. Plot data compresses
    5-10x, and Databricks drops an HTML output that is too large."""
    import base64
    import gzip
    raw = json.dumps(obj, separators=(",", ":")).encode()
    return base64.b64encode(gzip.compress(raw, 9)).decode()


def _shell(dims: list[str], opts: list[list[str]], payload: str,
           render_js: str, title: str = "", note: str = "",
           height: int | None = None) -> str:
    """Drop-downs + one plot area; the browser unpacks the payload and calls
    render(key, P, plot, box) on every change. `render_js` defines render."""
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
  var B64 = "{payload}";
  var N = {len(dims)}, SEP = {json.dumps(SEP)};
  {render_js}
  function sel(i) {{ return document.getElementById("{uid}_" + i); }}
  var plot = document.getElementById("{uid}_plot");
  var box = document.getElementById("{uid}_html");
  async function unpack() {{
    var bin = Uint8Array.from(atob(B64), function(c) {{ return c.charCodeAt(0); }});
    var s = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(s).text());
  }}
  unpack().then(function(P) {{
    function draw() {{
      var key = [];
      for (var i = 0; i < N; i++) key.push(sel(i).value);
      if (!render(key.join(SEP), P, plot, box)) {{
        Plotly.purge(plot); plot.innerHTML = "";
        box.innerHTML = "<p style='font:13px sans-serif;color:#a33'>" +
          "Not available for this combination.</p>";
      }}
    }}
    for (var i = 0; i < N; i++) sel(i).addEventListener("change", draw);
    draw();
  }}).catch(function(e) {{
    box.innerHTML = "<p style='font:13px sans-serif;color:#a33'>Could not " +
      "unpack the plots (" + e + "). A current Chrome, Edge or Firefox is needed.</p>";
  }});
}})();
</script>"""


_FIG_RENDER = """
  function render(key, P, plot, box) {
    var it = P.data[key];
    if (!it) return false;
    function deref(x) { return (x && x.__ref__ !== undefined) ? P.shared[x.__ref__] : x; }
    if (it.html !== undefined) {
      Plotly.purge(plot); plot.innerHTML = ""; box.innerHTML = it.html; return true;
    }
    box.innerHTML = "";
    var lay = Object.assign({}, it.fig.layout);
    if (lay.annotations) lay.annotations = deref(lay.annotations);
    if (lay.template) lay.template = deref(lay.template);
    Plotly.react(plot, it.fig.data.map(deref), lay, {responsive: true});
    return true;
  }
"""


def selector_html(items: dict, dims: list[str], title: str = "",
                  height: int | None = None, note: str = "") -> str:
    """One drop-down per entry of `dims`, one plot area.

    items   {(choice for dims[0], choice for dims[1], ...): figure | html}
    dims    names of the drop-downs, e.g. ["Condition", "Plot"]

    A combination that is not in `items` shows a short message rather than
    a stale figure. The options of each drop-down keep the order in which
    they first appear in `items`. Pieces repeated in several figures (the
    plotly style template, a segment outline, the inlet / outlet labels)
    are stored once, and the whole payload is gzip-compressed.
    """
    items = {tuple(str(x) for x in (k if isinstance(k, tuple) else (k,))): v
             for k, v in items.items() if v is not None}
    if not items:
        return "<p><i>nothing to show</i></p>"
    n = len(dims)
    opts = [list(dict.fromkeys(k[i] for k in items)) for i in range(n)]
    data = {SEP.join(k): _payload(v) for k, v in items.items()}

    def _parts(v):
        lay = v["fig"].get("layout", {})
        return (list(v["fig"].get("data", []))
                + [lay.get("annotations", []), lay.get("template", {})])

    shared, index, seen = [], {}, {}
    for v in data.values():
        if "fig" in v:
            for t in _parts(v):
                k = json.dumps(t, sort_keys=True)
                if len(k) >= 1000:
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
            for k in ("annotations", "template"):
                if k in lay:
                    lay[k] = _ref(lay[k])
    return _shell(dims, opts, _pack({"data": data, "shared": shared}),
                  _FIG_RENDER, title=title, note=note, height=height)


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


def _plate_geometry(p, gloss: float) -> dict:
    """Everything every map of the plate shares, sent once: the pad owner
    grid, the segment outlines, label points, centres, the inlet / outlet
    labels, the two colour scales and the gloss overlay."""
    import base64
    import io

    import plate_figure
    import plate_plotly
    import plate_style as style
    import r2d2_geometry as geom
    from matplotlib.image import imsave

    W, H = geom.PLATE_W_MM, geom.PLATE_H_MM
    nr, nc = geom.N_ROWS, geom.N_COLS
    owner = np.zeros((nr, nc), int)
    for n, seg in p.segments.items():
        for c, r in seg.pads:
            owner[r - 1, c - 1] = int(n)
    bx, by = plate_plotly._boundary_xy(p)
    lab, cen = {}, {}
    for n, seg in p.segments.items():
        lx, ly, narrow = plate_plotly._label_xy(p, n, seg)
        lab[n] = [round(lx, 2), round(ly, 2), int(narrow)]
        # where the map draws the segment: its label pad, the red number
        # box of the plate drawing
        cx, cy = plate_figure.node_xy(p, n)
        cen[n] = [round(cx, 3), round(cy, 3)]
    # the gloss: a faint diagonal sheen, white with an alpha ramp
    gx, gy = np.meshgrid(np.linspace(0, W, 127), np.linspace(0, H, 61))
    u = gx / W * 0.6 + (1 - gy / H) * 0.4
    sheen = np.clip(1.0 - np.abs(u - 0.62) / 0.38, 0, 1) ** 2
    rgba = np.ones(gx.shape + (4,))
    rgba[..., 3] = np.clip(0.30 * gloss * sheen, 0, 1)
    buf = io.BytesIO()
    imsave(buf, (rgba * 255).astype(np.uint8), format="png")
    xr = [-48.0, W + 48.0]
    return {
        "W": W, "H": H, "pw": geom.PAD_W_MM, "ph": geom.PAD_H_MM,
        "owner": owner.tolist(),
        "bx": [None if v is None else round(v, 2) for v in bx],
        "by": [None if v is None else round(v, 2) for v in by],
        "lab": lab, "cen": cen,
        "ann": plate_plotly._end_annotations(W, H),
        "xr": xr[::-1] if style.mirrored() else xr,
        "ndy": style.NUMBER_DY_MM, "vdy": style.VALUE_DY_MM,
        "jet": style.plotly_colorscale(),
        "parula": style.plotly_colorscale(style.INTERP_CMAP),
        "mark": style.MEASURED_MARK,
        "sheen": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
    }


def _plate_item(p, fd, head: str, step: float = 4.0) -> dict:
    """One condition x parameter: the values, and the interpolated field on
    a 4 mm grid quantised to 0..250 and stored as differences along each row
    (about 2 kB once compressed; the browser smooths it, so the grid does
    not show -- a segment is 11..28 mm wide)."""
    import plate_figure
    import plate_plotly
    import plate_style as style
    import r2d2_geometry as geom

    vals = plate_plotly._clean(fd.values)
    vmin, vmax, note = plate_plotly._limits(fd, vals)
    classes = {str(k): str(v) for k, v in (fd.classes or {}).items()}
    it = {"label": fd.label, "unit": fd.unit, "dec": fd.decimals,
          "vmin": vmin, "vmax": vmax,
          "v": {k: round(v, 6) for k, v in vals.items()},
          "est": [k for k in vals if classes.get(k, "measured")
                  not in ("", "measured")],
          "t2d": f"<b>{head} — {fd.label}</b><br><sup>" + " | ".join(
              x for x in (plate_plotly._subtitle(fd), note,
                          "□ = segment at its drawing position (label pad); "
                          "rest: 2D linear interpolation", style.flow_note()) if x) + "</sup>",
          "tseg": f"<b>{head} — {fd.label}</b><br><sup>" + " | ".join(
              x for x in (plate_plotly._subtitle(fd), note,
                          style.flow_note()) if x) + "</sup>"}
    got = plate_figure.interpolate_field(p, vals, geom.PLATE_W_MM,
                                         geom.PLATE_H_MM, step=step)
    if got is not None:
        gx, gy, zi = got
        span = (vmax - vmin) or 1.0
        q = np.clip(np.round((zi - vmin) / span * 250), 0, 250).astype(np.uint8)
        d = q.astype(int)
        d[:, 1:] = np.diff(d, axis=1)
        it["g"] = d.ravel().tolist()
        it["gs"] = [int(q.shape[0]), int(q.shape[1]), step]
    return it


_PLATE_RENDER = r"""
  function hex2rgb(c) {
    if (c[0] === "#") return [parseInt(c.substr(1,2),16), parseInt(c.substr(3,2),16), parseInt(c.substr(5,2),16)];
    return c.slice(c.indexOf("(") + 1, -1).split(",").map(Number);
  }
  function colorAt(sc, t) {
    t = Math.min(1, Math.max(0, t));
    for (var i = 1; i < sc.length; i++) {
      if (t <= sc[i][0]) {
        var a = hex2rgb(sc[i-1][1]), b = hex2rgb(sc[i][1]);
        var f = (t - sc[i-1][0]) / ((sc[i][0] - sc[i-1][0]) || 1);
        return [0,1,2].map(function(k) { return a[k] + f * (b[k] - a[k]); });
      }
    }
    return hex2rgb(sc[sc.length-1][1]);
  }
  function ink(rgb) {
    var lin = rgb.map(function(c) { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); });
    return (0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]) > 0.42 ? "#141414" : "#ffffff";
  }
  function fmt(v, d) { return v.toFixed(d); }
  function cbar(it) {
    return {title: {text: it.label + (it.unit ? "<br>[" + it.unit + "]" : ""), side: "top"},
            len: 0.8, thickness: 18, tickfont: {size: 11}};
  }
  function layout(G, title, extra) {
    var L = {title: {text: title, x: 0.01, xanchor: "left"},
      xaxis: {range: G.xr, visible: false, constrain: "domain"},
      yaxis: {range: [G.H + 4, -4], visible: false, scaleanchor: "x", scaleratio: 1},
      plot_bgcolor: "#f2f2f3", paper_bgcolor: "white", height: 640,
      margin: {l: 10, r: 10, t: 95, b: 10}, annotations: G.ann,
      hoverlabel: {bgcolor: "white", font: {size: 13}}};
    return Object.assign(L, extra || {});
  }
  function segView(G, it) {
    var u = it.unit ? " " + it.unit : "";
    var z = G.owner.map(function(row) { return row.map(function(n) {
      var v = it.v[String(n)]; return v === undefined ? null : v; }); });
    var xc = [], yc = [];
    for (var c = 0; c < G.owner[0].length; c++) xc.push((c + 0.5) * G.pw);
    for (var r = 0; r < G.owner.length; r++) yc.push((r + 0.5) * G.ph);
    var heat = {type: "heatmap", x: xc, y: yc, z: z, customdata: G.owner,
      colorscale: G.jet, zmin: it.vmin, zmax: it.vmax, hoverongaps: false,
      hovertemplate: "<b>segment %{customdata}</b><br>" + it.label + " = %{z:." + it.dec + "f}" + u + "<extra></extra>",
      colorbar: cbar(it)};
    var nx = [], ny = [], nt = [], nco = [], vx = [], vy = [], vt = [], vco = [], vsz = [];
    var span = (it.vmax - it.vmin) || 1;
    Object.keys(G.lab).forEach(function(n) {
      var L = G.lab[n], v = it.v[n];
      var rgb = v === undefined ? [212, 212, 214] : colorAt(G.jet, (v - it.vmin) / span);
      var col = ink(rgb);
      nx.push(L[0]); ny.push(L[1] + (v === undefined ? 0 : G.ndy)); nt.push(n); nco.push(col);
      if (v !== undefined) {
        vx.push(L[0]); vy.push(L[1] + G.vdy); vco.push(col); vsz.push(L[2] ? 10 : 13);
        vt.push("<b>" + fmt(v, it.dec) + (it.est.indexOf(n) >= 0 ? "*" : "") + "</b>");
      }
    });
    var data = [heat,
      {type: "scatter", mode: "text", x: nx, y: ny, text: nt, hoverinfo: "skip",
       textfont: {size: 8, color: nco}, showlegend: false},
      {type: "scatter", mode: "text", x: vx, y: vy, text: vt, hoverinfo: "skip",
       textfont: {size: vsz, color: vco}, showlegend: false},
      {type: "scatter", mode: "lines", x: G.bx, y: G.by, hoverinfo: "skip",
       line: {color: "rgba(20,20,22,0.65)", width: 1}, showlegend: false}];
    return [data, layout(G, it.tseg)];
  }
  function mapView(G, it) {
    var u = it.unit ? " " + it.unit : "";
    var data = [];
    if (it.g) {
      var g = it.g, nr = it.gs[0], nc = it.gs[1];
      var span = it.vmax - it.vmin, z = [], x = [], y = [];
      // cell centres spread over the active area, so the field ends at its edge
      for (var c = 0; c < nc; c++) x.push((c + 0.5) * G.W / nc);
      for (var r = 0; r < nr; r++) {
        y.push((r + 0.5) * G.H / nr);
        var row = [], acc = 0;                  // undo the row differences
        for (var c = 0; c < nc; c++) {
          acc += g[r * nc + c];
          row.push(it.vmin + acc / 250 * span);
        }
        z.push(row);
      }
      data.push({type: "heatmap", x: x, y: y, z: z, zsmooth: "best",
        colorscale: G.parula, zmin: it.vmin, zmax: it.vmax, colorbar: cbar(it),
        hovertemplate: it.label + " ≈ %{z:." + it.dec + "f}" + u + "<extra>interpolated</extra>"});
    }
    // every measured segment = its label pad (pw x ph mm), outlined at the
    // place of the drawing's red number box, its number inside
    var sx = [], sy = [], stx = [], sno = [], sco = [];
    var shapes = [{type: "rect", x0: 0, y0: 0, x1: G.W, y1: G.H, line: {color: "#5d5d60", width: 1.5}}];
    Object.keys(it.v).forEach(function(n) {
      var C = G.cen[n]; if (!C) return;
      var est = it.est.indexOf(n) >= 0, col = est ? "rgba(194,24,91,0.55)" : G.mark;
      sx.push(C[0]); sy.push(C[1]); sno.push(n); sco.push(col);
      stx.push("<b>segment " + n + "</b><br>" + it.label + " = " + fmt(it.v[n], it.dec) + u + "<br>" + (est ? "rebuilt" : "measured"));
      shapes.push({type: "rect", x0: C[0] - G.pw / 2, x1: C[0] + G.pw / 2,
        y0: C[1] - G.ph / 2, y1: C[1] + G.ph / 2, layer: "above",
        line: {color: col, width: 1.6, dash: est ? "dash" : "solid"}});
    });
    data.push({type: "scatter", mode: "text", x: sx, y: sy, text: sno, hovertext: stx, hoverinfo: "text",
      textfont: {size: 9, color: sco}, showlegend: false});
    var extra = {
      shapes: shapes,
      images: [{source: G.sheen, xref: "x", yref: "y", x: 0, y: 0, sizex: G.W, sizey: G.H,
                xanchor: "left", yanchor: "top", sizing: "stretch", layer: "above", opacity: 1}]};
    return [data, layout(G, it.t2d, extra)];
  }
  function render(key, P, plot, box) {
    var k = key.split(SEP), it = P.items[k[0] + SEP + k[1]];
    if (!it) return false;
    box.innerHTML = "";
    var fig = (k[2] === P.v2d) ? mapView(P.G, it) : segView(P.G, it);
    Plotly.react(plot, fig[0], fig[1], {responsive: true});
    return true;
  }
"""


def plate_maps(summaries: dict, title: str = "", fields=PLATE_FIELDS,
               views=(VIEW_2D, VIEW_SEG), gloss: float | None = None) -> str:
    """{condition: plate_summary.csv or list[Field]} -> one heat-map viewer
    with Condition / Parameter / View drop-downs.

    Only the numbers travel: 72 values and a one-byte-per-1.5-mm field per
    condition and parameter, plus the plate geometry once. The browser
    builds the map that is selected, so four conditions x ten parameters
    stay a few hundred kB -- a page of finished figures was megabytes and
    Databricks dropped it."""
    import plate_style as style
    import r2d2_geometry as geom
    p = geom.ACTIVE_PLATE
    items, conds, labels = {}, [], []
    for cond, src in summaries.items():
        flds = src if isinstance(src, list) else plate_fields(src, fields)
        head = f"{title} / {cond}" if title else str(cond)
        for fd in flds:
            items[f"{cond}{SEP}{fd.label}"] = _plate_item(p, fd, head)
            conds.append(str(cond))
            labels.append(fd.label)
    if not items:
        return "<p><i>no plate maps to show</i></p>"
    P = {"G": _plate_geometry(p, style.INTERP_GLOSS if gloss is None
                              else gloss),
         "items": items, "v2d": VIEW_2D}
    opts = [list(dict.fromkeys(conds)), list(dict.fromkeys(labels)),
            list(views)]
    return _shell(["Condition", "Parameter", "View"], opts, _pack(P),
                  _PLATE_RENDER,
                  title=f"Plate maps — {title}" if title else "")


def save_page(fragment: str, path, title: str = "") -> Path:
    """A viewer (what displayHTML shows) as a standalone .html file: the same
    drop-downs and plots, opened in any browser. Plotly comes from the CDN."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_html.escape(title or path.stem)}</title></head>"
        "<body style='margin:16px;background:#fff'>"
        f"{fragment}</body></html>", encoding="utf-8")
    return path


def unpack_payload(html: str):
    """The selector's gzip+base64 payload, decoded (tests only)."""
    import base64
    import gzip
    import json as _json
    import re as _re
    b64 = _re.search(r'var B64 = "([^"]+)"', html).group(1)
    return _json.loads(gzip.decompress(base64.b64decode(b64)))
