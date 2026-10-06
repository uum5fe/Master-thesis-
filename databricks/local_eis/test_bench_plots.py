"""The MF4 bench parameters: one figure, a dropdown per parameter, legend
toggling and unified hover -- built from a stand-in for asammdf.MDF."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("plotly")

import bench_plots as bp                                    # noqa: E402


class _FakeMDF:
    """The slice of asammdf.MDF that channels_from_mdf uses."""

    def __init__(self, signals: dict):
        self._sig = signals
        names = list(signals)
        self.groups = [SimpleNamespace(channels=[SimpleNamespace(name="t")]
                                       + [SimpleNamespace(name=n)
                                          for n in names])]

    def get(self, name, group=None):
        t, v, unit = self._sig[name]
        return SimpleNamespace(timestamps=t, samples=v, unit=unit)


def _mdf():
    t = np.linspace(0.0, 600.0, 601)
    hfr = np.full_like(t, 80.0)
    hfr[::50] = -1e12                       # the bench's "no value" sentinel
    return _FakeMDF({
        "T_Si_CL": (t, 60 + t / 600, "°C"),
        "T_So_CL": (t, 62 + t / 600, "°C"),
        "U_S": (t, np.full_like(t, 0.7), "V"),
        "I_S": (t, np.full_like(t, 45.0), "A"),
        "R_S_HFR": (t, hfr, "mΩ"),
        "Status_text": (t, np.array(["ok"] * t.size), ""),   # not numeric
        "X_extra": (t, np.sin(t), "bar"),
    })


def test_known_channels_are_grouped_and_others_kept():
    chans = bp.channels_from_mdf(_mdf())
    by = {c.name: c for c in chans}
    assert by["T_Si_CL"].group == "Temperature"
    assert by["T_Si_CL"].label == "Coolant inlet"
    assert by["X_extra"].group == bp.OTHER_GROUP and by["X_extra"].unit == "bar"
    assert "Status_text" not in by
    assert not np.any(np.abs(by["R_S_HFR"].v) >= bp.SENTINEL_ABS)
    only = bp.channels_from_mdf(_mdf(), include_others=False)
    assert "X_extra" not in {c.name for c in only}


def test_decimation_keeps_a_spike():
    t = np.arange(100_000, dtype=float)
    v = np.zeros_like(t)
    v[51_234] = 9.0
    td, vd = bp.decimate(t, v, max_points=500)
    assert td.size <= 502 and vd.max() == 9.0


def test_dropdown_has_groups_by_unit_and_single_parameters():
    chans = bp.channels_from_mdf(_mdf())
    fig = bp.bench_figure(chans, title="2611976",
                          reference_lines=[("FAMOS temp1", "°C", 58.7)])
    labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert "Temperature — all [°C] (2)" in labels
    # V and A never share an axis: no "Electrical — all" entry for 1 + 1
    assert not any(lab.startswith("Electrical — all") for lab in labels)
    assert "Temperature · Coolant inlet (T_Si_CL)" in labels
    assert "Other channels · X_extra" in labels

    names = [t.name for t in fig.data]
    btn = fig.layout.updatemenus[0].buttons[
        labels.index("Temperature · Coolant inlet (T_Si_CL)")]
    vis = dict(zip(names, btn.args[0]["visible"]))
    assert vis["Coolant inlet (T_Si_CL)"] and vis["FAMOS temp1"]
    assert not vis["Coolant outlet (T_So_CL)"] and not vis["Stack voltage (U_S)"]
    assert "mean" in btn.args[1]["title.text"]

    # starts on the temperature group, both coolant traces + the reference
    shown = {t.name for t in fig.data if t.visible}
    assert shown == {"Coolant inlet (T_Si_CL)", "Coolant outlet (T_So_CL)",
                     "FAMOS temp1"}


def test_hover_shows_values_and_legend_toggles():
    fig = bp.bench_figure(bp.channels_from_mdf(_mdf()))
    assert fig.layout.hovermode == "x unified"
    assert fig.layout.legend.itemclick == "toggle"
    assert fig.layout.legend.itemdoubleclick == "toggleothers"
    assert fig.layout.xaxis.rangeslider.visible
    assert all("%{y" in t.hovertemplate for t in fig.data)
