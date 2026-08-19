"""Shared UI pieces and the small amount of styling the app needs."""

from __future__ import annotations

import pandas as pd
from dash import dash_table, dcc, html

GRAPH_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

COLOURS = {
    "bg": "#f7f8fa", "panel": "#ffffff", "line": "#e3e6ea",
    "text": "#1f2328", "muted": "#5b6470", "accent": "#1f6feb",
    "warn": "#b7791f", "bad": "#c0392b", "good": "#207d4a",
}

PANEL = {
    "background": COLOURS["panel"], "border": f"1px solid {COLOURS['line']}",
    "borderRadius": "8px", "padding": "14px 16px", "marginBottom": "12px",
}


def panel(children, style: dict | None = None) -> html.Div:
    return html.Div(children, style={**PANEL, **(style or {})})


def field(label: str, control, hint: str = "", label_id: str = "",
          field_id: str = "") -> html.Div:
    """A labelled control, with an optional one-line explanation under it.

    ``label_id`` and ``field_id`` let a callback reword or hide the field: what
    a picker is *called* depends on what it is picking, and a control that
    cannot apply should not be shown at all.
    """
    kids = [html.Label(label, id=label_id or f"label-{id(control)}",
                       style={"fontSize": "12px", "fontWeight": 600,
                              "color": COLOURS["muted"],
                              "display": "block", "marginBottom": "4px"}),
            control]
    if hint:
        kids.append(html.Div(hint, style={"fontSize": "11px",
                                          "color": COLOURS["muted"],
                                          "marginTop": "3px"}))
    return html.Div(kids, id=field_id or f"field-{id(control)}",
                    style={"marginBottom": "12px"})


def stat(label: str, value: str, tone: str = "text") -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": COLOURS["muted"],
                               "textTransform": "uppercase",
                               "letterSpacing": "0.04em"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": 650,
                               "color": COLOURS.get(tone, COLOURS["text"]),
                               "marginTop": "2px"}),
    ], style={"padding": "10px 14px", "background": COLOURS["panel"],
              "border": f"1px solid {COLOURS['line']}", "borderRadius": "8px",
              "minWidth": "150px", "flex": "1 1 150px"})


def stat_row(items) -> html.Div:
    return html.Div([stat(*i) for i in items],
                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                           "marginBottom": "12px"})


def note(text: str, tone: str = "muted") -> html.Div:
    return html.Div(text, style={"fontSize": "12px", "color": COLOURS.get(tone),
                                 "marginBottom": "6px", "lineHeight": "1.5"})


def warnings_block(messages, title: str = "Warnings") -> html.Div:
    if not messages:
        return html.Div()
    return panel([
        html.Div(title, style={"fontWeight": 650, "marginBottom": "6px",
                               "color": COLOURS["warn"]}),
        html.Ul([html.Li(m, style={"fontSize": "12px", "marginBottom": "3px"})
                 for m in messages], style={"margin": 0, "paddingLeft": "18px"}),
    ], {"borderLeft": f"3px solid {COLOURS['warn']}"})


def table(frame: pd.DataFrame, table_id: str = "", max_rows: int = 200,
          height: str = "360px") -> html.Div:
    """A compact, sortable, exportable data table."""
    if frame is None or frame.empty:
        return note("nothing to show")
    frame = frame.head(max_rows)
    return dash_table.DataTable(
        id=table_id or f"table-{id(frame)}",
        data=frame.to_dict("records"),
        columns=[{"name": str(c), "id": str(c),
                  "type": "numeric" if pd.api.types.is_numeric_dtype(frame[c]) else "text",
                  "format": {"specifier": ".5g"} if pd.api.types.is_numeric_dtype(frame[c]) else None}
                 for c in frame.columns],
        sort_action="native", filter_action="native", page_size=25,
        export_format="csv", export_headers="display",
        style_table={"overflowX": "auto", "maxHeight": height, "overflowY": "auto"},
        style_cell={"fontFamily": "ui-monospace, SFMono-Regular, Menlo, monospace",
                    "fontSize": "11px", "padding": "5px 8px", "textAlign": "right"},
        style_header={"fontWeight": 650, "background": COLOURS["bg"],
                      "borderBottom": f"1px solid {COLOURS['line']}"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fbfcfd"},
        ],
    )


def graph(graph_id: str, height: str = "auto") -> dcc.Graph:
    return dcc.Graph(id=graph_id, config=GRAPH_CONFIG,
                     style={"height": height} if height != "auto" else {})


def kv_table(pairs) -> html.Table:
    return html.Table([
        html.Tbody([
            html.Tr([
                html.Td(str(k), style={"padding": "3px 12px 3px 0", "fontSize": "12px",
                                       "color": COLOURS["muted"],
                                       "whiteSpace": "nowrap", "verticalAlign": "top"}),
                html.Td(str(v), style={"padding": "3px 0", "fontSize": "12px",
                                       "fontFamily": "ui-monospace, Menlo, monospace",
                                       "wordBreak": "break-all"}),
            ]) for k, v in pairs
        ])
    ], style={"width": "100%", "borderCollapse": "collapse"})
