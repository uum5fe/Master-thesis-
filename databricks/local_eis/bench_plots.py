#!/usr/bin/env python3
"""
bench_plots.py
==============
The test-bench (MF4) parameters of an order as ONE interactive Plotly figure:

  * a dropdown picks what is drawn -- one parameter ("Temperature · Inlet
    anode (T_Si_A)"), or every parameter of a group that shares a unit
    ("Temperature — all [°C]"); mixed units never share an axis;
  * the legend toggles: click an entry to hide / show it, double-click to
    show it alone;
  * hovering shows the value of every visible trace at the cursor's time
    (unified hover, with a spike line), and the range slider under the plot
    zooms the time axis.

The runner cell used to draw one fixed matplotlib panel per channel, group by
group, which could neither be switched, toggled nor read at the cursor.

Reading needs `asammdf`; building the figure needs only plotly, so the figure
is testable without a bench file:

    import bench_plots
    chans = bench_plots.read_mf4("RO2611976-01.mf4")
    fig = bench_plots.bench_figure(chans, title="2611976")
    fig.show()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The bench channels the runner always plotted: group -> (name, label, unit)
CHANNEL_GROUPS: dict[str, list[tuple[str, str, str]]] = {
    "Electrical": [
        ("U_S",       "Stack voltage",    "V"),
        ("I_S",       "Stack current",    "A"),
        ("P_S_el",    "Electrical power", "W"),
        ("R_S_HFR",   "HFR",              "mΩ"),
        ("U_CVM001",  "Cell voltage 1",   "V"),
        ("U_CVM002",  "Cell voltage 2",   "V"),
    ],
    "Temperature": [
        ("T_Si_A",    "Inlet anode",      "°C"),
        ("T_Si_C",    "Inlet cathode",    "°C"),
        ("T_So_A",    "Outlet anode",     "°C"),
        ("T_So_C",    "Outlet cathode",   "°C"),
        ("T_Si_CL",   "Coolant inlet",    "°C"),
        ("T_So_CL",   "Coolant outlet",   "°C"),
        ("T_Ti_CL",   "CL pre-heater",    "°C"),
        ("T_To_CL",   "CL post-heater",   "°C"),
    ],
    "Flow Rates": [
        ("FN_Si_H2_A",    "H₂ anode",       "Nl/min"),
        ("FN_Si_Air_C",   "Air cathode",    "Nl/min"),
        ("FN_Si_N2_A",    "N₂ anode",       "Nl/min"),
        ("FN_Si_CL",      "Coolant",        "l/min"),
    ],
    "Pressure": [
        ("p_Si_A",    "Inlet anode",      "bara"),
        ("p_Si_C",    "Inlet cathode",    "bara"),
        ("p_So_A",    "Outlet anode",     "bara"),
        ("p_So_C",    "Outlet cathode",   "bara"),
        ("p_Ti_CL",   "Coolant inlet",    "bara"),
        ("p_To_CL",   "Coolant outlet",   "bara"),
    ],
    "Humidity": [
        ("RH_Si_A_gas",  "RH anode",          "%"),
        ("RH_Si_C_gas",  "RH cathode",        "%"),
        ("DPT_Si_A",     "Dew point anode",   "°C"),
        ("DPT_Si_C",     "Dew point cathode", "°C"),
    ],
}

#: group of channels that are in the file but not in CHANNEL_GROUPS
OTHER_GROUP = "Other channels"

#: |value| at or above this is a sentinel (R_S_HFR writes -1e12), not data
SENTINEL_ABS = 1e10


@dataclass
class Channel:
    name: str
    label: str
    unit: str
    group: str
    t_s: np.ndarray            # seconds from the start of the file
    v: np.ndarray

    @property
    def stats(self) -> dict:
        if self.v.size == 0:
            return dict(mean=np.nan, min=np.nan, max=np.nan)
        return dict(mean=float(np.mean(self.v)), min=float(np.min(self.v)),
                    max=float(np.max(self.v)))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def decimate(t: np.ndarray, v: np.ndarray, max_points: int = 3000):
    """At most ~max_points samples, keeping each bucket's min AND max so a
    spike survives the thinning (a plain stride would drop it)."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    if max_points <= 0 or t.size <= max_points:
        return t, v
    n_b = max(1, max_points // 2)
    edges = np.linspace(0, t.size, n_b + 1).astype(int)
    keep = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = v[a:b]
        i0, i1 = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        keep.extend(sorted({i0, i1}))
    keep = np.asarray(keep, int)
    return t[keep], v[keep]


def _clean(t, v):
    t = np.asarray(t)
    v = np.asarray(v)
    if v.dtype.kind not in "fiub" or v.ndim != 1 or t.shape != v.shape:
        return None
    t = t.astype(float)
    v = v.astype(float)
    ok = np.isfinite(t) & np.isfinite(v) & (np.abs(v) < SENTINEL_ABS)
    return t[ok], v[ok]


def channels_from_mdf(mdf, groups: dict | None = None,
                      include_others: bool = True,
                      max_points: int = 3000) -> list[Channel]:
    """Every known channel present in an open asammdf.MDF, then (optionally)
    every other numeric channel of the file under OTHER_GROUP."""
    groups = CHANNEL_GROUPS if groups is None else groups
    where: dict[str, int] = {}
    for gi, grp in enumerate(mdf.groups):
        for ch in grp.channels:
            if ch.name not in ("t", "Time", "time"):
                where[ch.name] = gi
    out: list[Channel] = []
    known = set()
    for gname, chans in groups.items():
        for name, label, unit in chans:
            known.add(name)
            if name not in where:
                continue
            try:
                sig = mdf.get(name, group=where[name])
            except Exception:                               # noqa: BLE001
                continue
            tv = _clean(sig.timestamps, sig.samples)
            if tv is None or tv[0].size < 2:
                continue
            t, v = decimate(*tv, max_points=max_points)
            out.append(Channel(name, label, unit, gname, t, v))
    if include_others:
        for name in sorted(where, key=str.lower):
            if name in known:
                continue
            try:
                sig = mdf.get(name, group=where[name])
            except Exception:                               # noqa: BLE001
                continue
            tv = _clean(sig.timestamps, sig.samples)
            if tv is None or tv[0].size < 2:
                continue
            unit = str(getattr(sig, "unit", "") or "")
            t, v = decimate(*tv, max_points=max_points)
            out.append(Channel(name, name, unit, OTHER_GROUP, t, v))
    return out


def read_mf4(path, include_others: bool = True,
             max_points: int = 3000) -> list[Channel]:
    """Read an MF4 bench log. Requires `asammdf`."""
    from asammdf import MDF
    mdf = MDF(str(path))
    try:
        return channels_from_mdf(mdf, include_others=include_others,
                                 max_points=max_points)
    finally:
        mdf.close()


def mf4_start_time(path):
    """The recording's start time (datetime) or None."""
    try:
        from asammdf import MDF
        mdf = MDF(str(path))
        try:
            return mdf.header.start_time
        finally:
            mdf.close()
    except Exception:                                       # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    a = abs(x)
    return f"{x:.4g}" if a < 1e-2 or a >= 1e4 else f"{x:.2f}"


def bench_figure(channels: list[Channel], title: str = "",
                 start: str | None = "Temperature",
                 reference_lines: list[tuple[str, str, float]] | None = None,
                 height: int = 640, width: int | None = None):
    """One figure, every channel a trace; a dropdown chooses what is shown.

    start            group name (its first unit) or channel name shown first
    reference_lines  (label, unit, value): constant lines drawn with every
                     view in that unit, e.g. the FAMOS plate sensors in °C
    """
    import plotly.graph_objects as go
    import plotly.express as px

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    fig = go.Figure()
    meta = []            # per trace: (kind, group, unit, name)

    t_max = max((float(c.t_s[-1]) for c in channels if c.t_s.size), default=1.0)
    for i, c in enumerate(channels):
        # Scatter, not Scattergl: WebGL traces are not drawn in the range
        # slider. Only the visible traces are rendered, so this stays light.
        fig.add_trace(go.Scatter(
            x=c.t_s / 60.0, y=c.v, mode="lines", name=f"{c.label} ({c.name})",
            line=dict(width=1.4, color=palette[i % len(palette)]),
            legendgroup=c.group, legendgrouptitle=dict(text=c.group),
            hovertemplate=(f"{c.label}: %{{y:.4g}} {c.unit}<extra></extra>"),
            visible=False))
        meta.append(("ch", c.group, c.unit, c.name))
    for j, (lab, unit, val) in enumerate(reference_lines or []):
        fig.add_trace(go.Scatter(
            x=[0.0, t_max / 60.0], y=[val, val], mode="lines", name=lab,
            line=dict(width=1.2, dash="dot",
                      color=palette[(len(channels) + j) % len(palette)]),
            legendgroup="reference", legendgrouptitle=dict(text="reference"),
            hovertemplate=f"{lab}: {val:.2f} {unit}<extra></extra>",
            visible=False))
        meta.append(("ref", "reference", unit, lab))

    def mask(pred):
        return [bool(pred(k, m)) for k, m in enumerate(meta)]

    buttons, first = [], None
    # groups: one entry per (group, unit), so mixed units never share an axis
    seen = []
    for c in channels:
        key = (c.group, c.unit)
        if key not in seen:
            seen.append(key)
    for g, u in seen:
        n = sum(1 for c in channels if (c.group, c.unit) == (g, u))
        if n < 2:
            continue
        lab = f"{g} — all [{u}] ({n})" if u else f"{g} — all ({n})"
        vis = mask(lambda k, m, g=g, u=u: (m[0] == "ch" and m[1] == g
                                           and m[2] == u)
                   or (m[0] == "ref" and m[2] == u))
        buttons.append(dict(label=lab, method="update", args=[
            {"visible": vis},
            {"title.text": f"<b>{title} — {g}</b>" if title else f"<b>{g}</b>",
             "yaxis.title.text": f"{g} [{u}]" if u else g}]))
        if first is None and start == g:
            first = len(buttons) - 1
    # single channels
    for i, c in enumerate(channels):
        st = c.stats
        lab = f"{c.group} · {c.label}" + (f" ({c.name})" if c.label != c.name
                                          else "")
        vis = mask(lambda k, m, i=i, u=c.unit: k == i
                   or (m[0] == "ref" and m[2] == u))
        sub = (f"mean {_fmt(st['mean'])} · min {_fmt(st['min'])} · "
               f"max {_fmt(st['max'])} {c.unit}")
        head = f"{title} — {c.label} ({c.name})" if title else c.label
        buttons.append(dict(label=lab, method="update", args=[
            {"visible": vis},
            {"title.text": f"<b>{head}</b><br><sup>{sub}</sup>",
             "yaxis.title.text": f"{c.label} [{c.unit}]" if c.unit
             else c.label}]))
        if first is None and start == c.name:
            first = len(buttons) - 1
    if first is None:
        first = 0
    if buttons:
        for k, v in zip(range(len(fig.data)), buttons[first]["args"][0]["visible"]):
            fig.data[k].visible = v

    fig.update_layout(
        title=dict(text=buttons[first]["args"][1]["title.text"] if buttons
                   else title, x=0.01, xanchor="left", y=0.97),
        height=height, width=width, template="plotly_white",
        hovermode="x unified", hoverlabel=dict(font_size=12),
        margin=dict(t=120, r=20, l=70, b=40),
        legend=dict(title=dict(text="click: hide/show · double-click: only this"),
                    groupclick="toggleitem", itemclick="toggle",
                    itemdoubleclick="toggleothers"),
        xaxis=dict(title="Time [min]", hoverformat=".2f", showspikes=True, spikemode="across",
                   spikesnap="cursor", spikethickness=1, spikedash="dot",
                   rangeslider=dict(visible=True, thickness=0.07)),
        yaxis=dict(title=(buttons[first]["args"][1]["yaxis.title.text"]
                          if buttons else ""), showspikes=True,
                   spikethickness=1, spikedash="dot"),
        updatemenus=[dict(
            buttons=buttons, direction="down", showactive=True, active=first,
            x=1.0, xanchor="right", y=1.17, yanchor="top",
            bgcolor="white", bordercolor="#999", font=dict(size=12))]
        if buttons else [])
    if buttons:
        fig.add_annotation(text="parameter:", x=1.0, xref="paper", y=1.215,
                           yref="paper", showarrow=False, xanchor="right",
                           font=dict(size=11, color="#5d5d60"))
    return fig
