"""One plot per figure.

The spectrum cells build Nyquist, |Z| and phase side by side (or a grid of
segments) and hand the figure to split_subplots, which must return one figure
per panel without losing what made each panel what it was: its traces, its
log axes, its guide lines, its legend entries and its dropdown.

The Nyquist cell itself is also executed here, against a small results tree,
so a change that breaks the cell breaks a test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

go = pytest.importorskip("plotly.graph_objects")
from plotly.subplots import make_subplots                   # noqa: E402

import figure_panels                                        # noqa: E402


RUNNER = Path(__file__).with_name("Local EIS Pipeline Runner.py")


def _three_panel(n_seg=3):
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Nyquist", "|Z|(f) Bode", "Phase(f)"])
    for s in range(n_seg):
        g = f"Seg {s}"
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name=g, legendgroup=g),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name=g, legendgroup=g,
                                 showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[1, 2], y=[5, 6], name=g, legendgroup=g,
                                 showlegend=False), row=1, col=3)
    fig.update_xaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.add_hline(y=2, row=1, col=3)
    fig.update_layout(title="<b>Local EIS — 150A</b><br><span>note</span>",
                      width=1500, height=550)
    return fig


def test_one_figure_per_panel_in_order() -> None:
    figs = figure_panels.split_subplots(_three_panel())
    assert len(figs) == 3
    heads = [f.layout.title.text for f in figs]
    assert heads[0].startswith("<b>Nyquist</b>")
    assert heads[1].startswith("<b>|Z|(f) Bode</b>")
    assert heads[2].startswith("<b>Phase(f)</b>")
    # the main title and its note travel with every panel
    assert all("Local EIS — 150A" in h and "note" in h for h in heads)


def test_each_panel_keeps_only_its_own_traces() -> None:
    figs = figure_panels.split_subplots(_three_panel())
    assert [len(f.data) for f in figs] == [3, 3, 3]
    assert list(figs[1].data[0].y) == [3, 4]
    assert list(figs[2].data[0].y) == [5, 6]
    for f in figs:
        assert all(t.xaxis in (None, "x") and t.yaxis in (None, "y")
                   for t in f.data)


def test_axis_types_and_guides_follow_their_panel() -> None:
    figs = figure_panels.split_subplots(_three_panel())
    assert figs[1].layout.xaxis.type == "log"
    assert figs[1].layout.yaxis.type == "log"
    assert figs[0].layout.xaxis.type != "log"
    assert len(figs[2].layout.shapes) == 1 and len(figs[0].layout.shapes) == 0
    assert figs[2].layout.shapes[0].xref == "x domain"


def test_every_panel_gets_a_legend_to_toggle_segments() -> None:
    """In the side-by-side figure only the Nyquist traces carried the legend;
    a Bode panel on its own would have none."""
    figs = figure_panels.split_subplots(_three_panel())
    for f in figs:
        assert [t.showlegend for t in f.data] == [True, True, True]


def test_a_trace_kept_out_of_the_legend_stays_out() -> None:
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=[1], y=[1], name="fit", showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[1], y=[1], name="data"), row=1, col=2)
    a, b = figure_panels.split_subplots(fig)
    assert a.data[0].showlegend is False
    assert b.data[0].showlegend is True


def test_dropdown_masks_are_cut_to_the_panel() -> None:
    fig = _three_panel(n_seg=2)
    vis = [True, False, True, False, True, False]
    fig.update_layout(updatemenus=[dict(buttons=[dict(
        label="Seg 0", method="update",
        args=[{"visible": vis}, {"title.text": "Seg 0 only"}])])])
    figs = figure_panels.split_subplots(fig)
    for f in figs:
        args = f.layout.updatemenus[0].buttons[0].args
        assert len(args[0]["visible"]) == len(f.data) == 2
        assert args[1]["title.text"].endswith("Seg 0 only")


def test_empty_grid_cells_are_skipped() -> None:
    fig = make_subplots(rows=2, cols=2)
    for r, c in ((1, 1), (1, 2), (2, 1)):
        fig.add_trace(go.Scatter(x=[1], y=[1]), row=r, col=c)
    assert len(figure_panels.split_subplots(fig)) == 3


def test_a_single_plot_comes_back_untouched() -> None:
    fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
    fig.update_layout(width=900, height=450)
    out = figure_panels.split_subplots(fig)
    assert len(out) == 1 and out[0] is fig


# ---------------------------------------------------------------------------
# the Nyquist cell, executed
# ---------------------------------------------------------------------------


def _cell(marker: str) -> str:
    for chunk in RUNNER.read_text().split("\n# COMMAND ----------\n"):
        if marker in chunk:
            return chunk
    raise AssertionError(f"cell {marker!r} not found")


def _setup_show_fig(ns: dict) -> None:
    """Lift PLOTS_ONE_BY_ONE .. show_fig out of the setup cell."""
    src = _cell("LOCAL EIS PIPELINE RUNNER")
    start = src.index("PLOTS_ONE_BY_ONE = True")
    end = src.index("from config import Config, DEFAULT")
    exec(src[start:end], ns)


def test_the_nyquist_cell_draws_three_separate_plots(tmp_path,
                                                    monkeypatch) -> None:
    pytest.importorskip("IPython")
    f = np.geomspace(0.2, 1500.0, 20)
    rows = []
    for seg in (1, 2, 3):
        z = 0.060 + 0.1 / (1 + 1j * 2 * np.pi * f * 2e-3)
        for fi, zi in zip(f, z * 1000):
            rows.append(dict(segment=seg, freq_hz=fi, z_re_mohm_cm2=zi.real,
                             z_im_mohm_cm2=zi.imag))
    (tmp_path / "silver").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "silver" / "spectra_clean.csv",
                              index=False)

    shown = []
    ns = {"figure_panels": figure_panels, "displayHTML": lambda h: None,
          "PIPELINE_RESULTS": {"150A": {"out_dir": tmp_path}},
          "LEEPA": "2612030", "pd": pd, "np": np, "Path": Path,
          "spectra_csv": lambda d: Path(d) / "silver" / "spectra_clean.csv",
          "maps_dir": lambda d: Path(d)}
    _setup_show_fig(ns)
    monkeypatch.setattr(go.Figure, "show",
                        lambda self, *a, **k: shown.append(self))
    exec(_cell("INTERACTIVE NYQUIST + BODE PER CONDITION"), ns)
    assert len(shown) == 3
    assert shown[0].layout.title.text.startswith("<b>Nyquist</b>")
    assert shown[1].layout.xaxis.type == "log"
    assert all(len(fig.data) == 3 for fig in shown)
