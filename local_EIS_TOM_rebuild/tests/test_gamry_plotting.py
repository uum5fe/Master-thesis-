"""The Gamry is drawn exactly as recorded: every point of the .DTA at the
frequency the file gives, in the segment Nyquist / Bode / phase views (black)
and in the aggregate comparison, which shows those two curves and nothing
else."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import make_synth_sweep as synth          # noqa: E402
import nyquist                            # noqa: E402
import viewers                            # noqa: E402

AREA = nyquist.A_CELL_CM2


def _dta(tmp_path):
    """A Gamry sweep with awkward frequencies, running to 30 kHz."""
    f = np.array([30046.88, 2390.625, 946.5144, 9.450605, 0.4783163, 0.1897773])
    z = np.array([synth.true_z((0.055, 0.065, 0.09), fi) / AREA for fi in f])
    p = tmp_path / "V00_000_HFR_10x_CurrVal_450.DTA"
    synth.write_dta(p, f, np.arange(len(f)), z_cell=z)
    return p, f, z


def _segments(f):
    rows = []
    for s in ("1", "2", "40"):
        for fi in f:
            z = 1e3 * AREA * synth.true_z((0.05, 0.06, 0.08), fi) / AREA
            rows.append(dict(segment=s, freq_hz=fi, z_re_mohm_cm2=z.real,
                             z_im_mohm_cm2=z.imag))
    return pd.DataFrame(rows)


def test_every_gamry_point_is_read_at_its_own_frequency(tmp_path):
    p, f, z = _dta(tmp_path)
    g = nyquist.gamry_frame(p)
    assert len(g) == len(f)                       # nothing cut to a band
    # the synthetic file stores 6 significant digits (make_synth_sweep)
    assert np.allclose(np.sort(g.freq_hz), np.sort(f), rtol=1e-5)
    zg = (g.z_re_mohm_cm2 + 1j * g.z_im_mohm_cm2).to_numpy()
    want = (z * AREA * 1e3)[np.argsort(f)]
    assert np.allclose(zg, want, rtol=1e-5)       # nothing removed or scaled


def test_segment_views_carry_the_gamry_in_black(tmp_path):
    p, f, _z = _dta(tmp_path)
    fig = nyquist.figure(_segments(f[1:]), title="t", gamry=p,
                         gamry_name="Gamry 450A")
    g = [t for t in fig.data if t.legendgroup == "gamry"]
    assert len(g) == 3                            # Nyquist, |Z|, phase
    assert all(len(t.x) == len(f) for t in g)     # the 30 kHz point too
    assert all(t.line.color == "black" for t in g)
    assert "customdata:.5g" in g[0].hovertemplate  # 0.47832 Hz, not 0.5


def test_the_comparison_is_the_aggregate_and_the_gamry_only(tmp_path):
    p, f, _z = _dta(tmp_path)
    agg = _segments(f[1:])
    agg = agg[agg.segment == "1"].drop(columns="segment")
    fig = nyquist.comparison_figure(agg, p, title="450A")
    assert {t.legendgroup for t in fig.data} == {"agg", "gamry"}
    assert [len(t.x) for t in fig.data if t.legendgroup == "gamry"] == [len(f)] * 3
    png = nyquist.comparison_png(agg, p, tmp_path / "c.png")
    assert png.exists()


def test_viewer_declares_utf8_for_plotly(tmp_path):
    """Without it the minus of negative ticks read as 'â^'' in Databricks."""
    p, f, _z = _dta(tmp_path)
    html = viewers.by_condition({"450A": nyquist.comparison_figure(None, p)})
    assert 'charset="utf-8"></script>' in html
