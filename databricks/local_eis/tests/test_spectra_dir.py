"""The bench tool's spectra folders (Spectrum<n>_<T>degC_<I>A/z<n>.csv) end to
end: read, matched to the runner's conditions, evaluated in the FAMOS output
format, plotted coloured along the air path."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import csv_source
import r2d2_geometry as geom

F = np.geomspace(0.2, 10078.13, 30)[::-1]


def _seg_z(seg: int, f=F, hf_loop=False):
    """mOhm*cm2: R_s + one RC arc, R_s graded along x like a drying inlet."""
    g = geom.SEGMENTS[str(seg)]
    rs = 45.0 + 10.0 * g.cx_mm / geom.PLATE_W_MM
    z = rs + 120.0 / (1 + 1j * 2 * np.pi * f * 1e-3)
    if hf_loop:
        # the chain's loop above ~2 kHz: Z'' positive, Z' falling to ~0
        w = 1.0 / (1.0 + (2000.0 / f) ** 4)
        z = z * (1 - w) + w * (1j * 30.0)
    return z


def _write_folder(root: Path, name: str, hf_loop=False) -> Path:
    d = root / name
    d.mkdir(parents=True)
    for s in geom.SEGMENTS:
        g = geom.SEGMENTS[s]
        z = _seg_z(int(s), hf_loop=hf_loop)
        lines = [f"x= {g.cx_mm:.6f} cm", f"y= {g.cy_mm:.6f} cm",
                 "T= 65.000000 degC"]
        lines += [f"f= {fi:.6f} Hz\t z= {zi.real:.6f}{zi.imag:+.6f}i mOhmcm2"
                  for fi, zi in zip(F, z)]
        (d / f"z{s}.csv").write_text("\n".join(lines) + "\n")
    hfr = [f"s{s}: x= 1cm y= 1cm:\thfr_int= {45 + 10 * geom.SEGMENTS[s].cx_mm / geom.PLATE_W_MM:.6f} mOhmcm2"
           f"\t hfr_1kHz= 60.0 mOhmcm2\t hfr_300Hz= 80.0 mOhmcm2"
           for s in geom.SEGMENTS]
    (d / "hfr.csv").write_text("\n".join(hfr) + "\n")
    return d


@pytest.fixture
def campaign(tmp_path):
    geom.use_plate("gen1")
    root = tmp_path / "csv_files"
    for n in ("Spectrum2_65degC_449A", "Spectrum6_65degC_150A",
              "Spectrum9_65degC_45A"):
        _write_folder(root, n)
    return root


def test_folders_are_found_and_labelled_by_setpoint(campaign):
    camps = csv_source.campaigns(campaign)
    assert list(camps) == ["."]
    cmap = csv_source.condition_map(camps["."])
    assert set(cmap) == {"45A", "150A", "450A"}
    assert cmap["450A"].name == "Spectrum2_65degC_449A"
    assert csv_source.nominal_label(449) == "450A"
    assert csv_source.nominal_label(97) == "97A"


def test_a_result_folder_is_not_mistaken_for_a_sweep(campaign, tmp_path):
    res = tmp_path / "csv_files" / "results" / "450A"
    (res / "gold").mkdir(parents=True)
    for k in range(4):
        (res / f"t{k}.csv").write_text("a,b\n1,2\n")
    assert set(csv_source.campaigns(campaign)) == {"."}


def test_the_reader_keeps_the_tools_numbers(campaign):
    m = csv_source.read(campaign / "Spectrum2_65degC_449A")
    assert m.dialect == "spectra_dir" and m.kind == "frequency"
    assert len(m.segments) == len(geom.SEGMENTS)
    assert m.meta["numbering_ok"]
    assert m.meta["operating_point"]["current_A"] == 449


def test_capacitive_band_drops_the_hf_loop():
    import csv_pipeline
    z = _seg_z(1, hf_loop=True) / 1000.0
    k = csv_pipeline.capacitive_band(F, z)
    assert k.any() and np.all(z.imag[k] < 0)
    assert F[k].max() < 2600


def test_a_csv_run_writes_the_famos_summary(campaign, tmp_path):
    import csv_pipeline
    from config import DEFAULT
    out = tmp_path / "out"
    cfg = DEFAULT.replace(source_format="csv", plate="gen1",
                          csv_path=campaign / "Spectrum2_65degC_449A",
                          out_dir=out, condition="450A", write_png=True,
                          verbose=False)
    csv_pipeline.run_csv(cfg, stop_after="gold")
    ps = pd.read_csv(out / "gold" / "plate_summary.csv")
    assert len(ps) == len(geom.SEGMENTS)
    # mOhm*cm2 like FAMOS, and the HF intercept matches the tool's hfr_int
    assert 40 < ps["R_ohmic"].median() < 60
    assert np.allclose(ps["R_ohmic"], ps["hfr_tool"], atol=1.5)
    assert (out / "gold" / "nyquist.html").is_file()
    assert (out / "gold" / "nyquist.png").is_file()
    assert (out / "gold" / "plate_R_ohmic_interp.png").is_file()


def test_nyquist_colours_run_from_air_inlet_to_outlet():
    import nyquist
    geom.use_plate("gen1")
    pos = nyquist.flow_position(geom.SEGMENTS)
    x_in = nyquist.air_inlet_x()
    near = min(geom.SEGMENTS, key=lambda s: abs(geom.SEGMENTS[s].cx_mm - x_in))
    far = max(geom.SEGMENTS, key=lambda s: abs(geom.SEGMENTS[s].cx_mm - x_in))
    assert pos[near] < 0.1 and pos[far] > 0.9


def test_a_viewer_is_saved_as_a_standalone_page(tmp_path):
    import plotly.graph_objects as go
    import viewers
    html = viewers.by_condition({"45A": go.Figure(go.Scatter(x=[1], y=[2]))})
    p = viewers.save_page(html, tmp_path / "v" / "nyquist.html", "t")
    text = p.read_text()
    assert text.startswith("<!doctype html>") and "<select" in text


def test_mirror_x_reassigns_segments_left_right(campaign):
    geom.use_plate("gen1")
    m = csv_source.read(campaign / "Spectrum2_65degC_449A")
    mp = csv_source.mirror_map_x()
    assert sorted(mp.values(), key=int) == sorted(geom.SEGMENTS, key=int)
    before = {s: v[1].copy() for s, v in m.spectra.items()}
    csv_source.mirror_x(m)
    s = "1"
    assert np.allclose(m.spectra[mp[s]][1], before[s])
    assert m.meta["mirrored_x"]
