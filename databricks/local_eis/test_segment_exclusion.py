"""Leaving a segment out has to mean leaving it out.

`exclude_segments` makes bronze skip a segment before its channel is read, so
it has no spectrum. That is only half of "left out": gold's spatial completion
returns a value for EVERY segment, so an excluded one came back with a
Gaussian-process value taken from its neighbours, labelled "inferred", and
drawn on every heat map. Asking for segment 33 to be left out and getting a
33-shaped guess in its place is the opposite of the request -- and it is
indistinguishable, in the output, from a segment whose fit genuinely failed.
"""

from __future__ import annotations

import numpy as np
import pytest

import gold
import r2d2_geometry as geom
from config import DEFAULT
from silver import SilverRun, SilverSpectrum


def _spectrum(seg: str, r_ohmic: float) -> SilverSpectrum:
    f = np.geomspace(1.0, 1000.0, 20)
    z = r_ohmic + 0.3 / (1 + 1j * 2 * np.pi * f * 2e-3)
    return SilverSpectrum(
        segment=seg, card="Karte_1", freq=f, Z_corr=z, Z_model=z,
        Z_model_sd=np.full(f.size, 1e-3), sigma_rel=np.full(f.size, 0.02),
        R_ohmic=r_ohmic, R_ohmic_sd=1e-3, R_ohmic_drt=r_ohmic,
        hf_arc_open=0.0, hf_closure=1.0, R_pol=0.3, L=2e-7, dt_applied=0.0,
        tau_peak=2e-3, gamma=np.zeros(4), tau_grid=np.geomspace(1e-4, 1e-1, 4),
        kk_res_max=0.01, kk_mu=0.6, zhit_dev=0.01, zhit_drift=0.0,
        hf_phase_slope=0.0, T_degC=60.0, K=1.0, K_imputed=False, u_dc=0.77,
        area_cm2=float(geom.SEGMENTS[seg].area_cm2), n_used=20, n_dropped=0,
        snr_med_db=30.0, thd_med=0.01, tier="A")


@pytest.fixture()
def silver_run():
    """Every segment measured, so anything absent later is absent on purpose."""
    segs = sorted(geom.SEGMENTS, key=int)
    rng = np.random.default_rng(0)
    spectra = {s: _spectrum(s, 0.060 + 0.004 * rng.standard_normal())
               for s in segs}
    f = np.geomspace(1.0, 1000.0, 20)
    return SilverRun(spectra=spectra, skew={}, dc_closure={},
                     cell_freq=f, Z_cell=np.full(f.size, 0.06 + 0j))


def _record(run, seg):
    return run.records[seg]


def test_without_an_exclusion_every_segment_is_measured(silver_run) -> None:
    """The control: the fixture must not be missing anything by accident."""
    g = gold.run(silver_run, DEFAULT.replace(verbose=False), log=None)
    assert _record(g, "33").cls == "measured"


def test_an_excluded_segment_is_not_measured(silver_run) -> None:
    cfg = DEFAULT.replace(verbose=False, exclude_segments=frozenset({"33"}))
    # bronze would never have produced it; silver therefore has no spectrum
    del silver_run.spectra["33"]
    g = gold.run(silver_run, cfg, log=None)
    assert _record(g, "33").cls == "excluded"


def test_an_excluded_segment_gets_no_inferred_value(silver_run) -> None:
    """The defect this file exists for.

    With spatial completion ON, the GP returns a value for every segment.
    Segment 33 sits surrounded by measured neighbours, so it would be inferred
    confidently -- a number with no measurement behind it, on a segment the
    operator asked to leave out.
    """
    cfg = DEFAULT.replace(verbose=False, spatial_enable=True,
                          infer_missing_segments=True,
                          exclude_segments=frozenset({"33"}))
    del silver_run.spectra["33"]
    g = gold.run(silver_run, cfg, log=None)

    rec = _record(g, "33")
    assert rec.cls == "excluded"
    assert not rec.values, f"excluded segment carries values: {rec.values}"
    assert "excluded by configuration" in " ".join(rec.flags)


def test_an_unmeasured_segment_is_still_inferred(silver_run) -> None:
    """Exclusion must not switch off inference for everything else.

    A segment that simply has no channel is a different fact from one that was
    excluded, and it is exactly what the spatial field is for.
    """
    cfg = DEFAULT.replace(verbose=False, spatial_enable=True,
                          infer_missing_segments=True,
                          exclude_segments=frozenset({"33"}))
    del silver_run.spectra["33"]
    del silver_run.spectra["34"]
    g = gold.run(silver_run, cfg, log=None)

    assert _record(g, "34").cls == "inferred"
    assert _record(g, "34").values.get("R_ohmic") is not None
    assert not _record(g, "33").values


def test_the_two_reasons_for_absence_read_differently(silver_run) -> None:
    """A reader must be able to tell "left out" from "stopped working"."""
    cfg = DEFAULT.replace(verbose=False, exclude_segments=frozenset({"33"}))
    del silver_run.spectra["33"]
    del silver_run.spectra["34"]
    g = gold.run(silver_run, cfg, log=None)

    excluded_flag = " ".join(_record(g, "33").flags)
    missing_flag = " ".join(_record(g, "34").flags)
    assert "excluded" in excluded_flag and "fit failed" not in excluded_flag
    assert excluded_flag != missing_flag


def test_the_summary_counts_and_names_the_exclusions(silver_run) -> None:
    cfg = DEFAULT.replace(verbose=False, exclude_segments=frozenset({"33"}))
    del silver_run.spectra["33"]
    g = gold.run(silver_run, cfg, log=None)

    assert g.stats["n_excluded"] == 1
    assert g.stats["excluded_segments"] == ["33"]
    assert "33" not in g.measured()


def test_excluding_several_segments_works(silver_run) -> None:
    cfg = DEFAULT.replace(verbose=False, spatial_enable=True,
                          infer_missing_segments=True,
                          exclude_segments=frozenset({"33", "59"}))
    for s in ("33", "59"):
        del silver_run.spectra[s]
    g = gold.run(silver_run, cfg, log=None)
    for s in ("33", "59"):
        assert _record(g, s).cls == "excluded"
        assert not _record(g, s).values


def test_bronze_reports_exclusion_separately_from_absence(silver_run) -> None:
    """The two kinds of missing must not read the same in bronze either.

    "not measured" invites a hunt for a wiring fault. "excluded on purpose"
    does not, and from a list of segment numbers a reader cannot tell which
    is which.
    """
    import bronze
    run = bronze.BronzeRun(
        schedule=[], channels={}, spectra={}, cards={}, grid={},
        config_digest="x", input_digest="y", n_files=1,
        excluded=frozenset({"33"}))
    assert run.summary()["excluded_segments"] == ["33"]
    assert "33" in run.segments_missing()      # absent, and said to be


# ---------------------------------------------------------------------------
# saying it: the four ways in, and one of them was silently wrong
# ---------------------------------------------------------------------------


def test_the_environment_variable_is_a_list_not_a_string(monkeypatch) -> None:
    """A string is one value, not a sequence of characters.

    EIS_EXCLUDE_SEGMENTS="33,59" was iterated character by character into
    {'3', '5', '9', ','}: it excluded segments 3, 5 and 9 and kept 33 and 59.
    Every element of that set is a plausible segment number, so no stage
    downstream had any way to notice.
    """
    from config import Config

    monkeypatch.setenv("EIS_EXCLUDE_SEGMENTS", "33,59")
    assert sorted(Config.from_env().exclude_segments) == ["33", "59"]


@pytest.mark.parametrize("raw, want", [
    ("33", ["33"]),
    ("33,59", ["33", "59"]),
    ("33, 59", ["33", "59"]),
    ("33;59", ["33", "59"]),
    ("", []),
])
def test_every_spelling_of_the_list_parses(raw, want, monkeypatch) -> None:
    from config import Config

    monkeypatch.setenv("EIS_EXCLUDE_SEGMENTS", raw)
    assert sorted(Config.from_env().exclude_segments) == want


def test_the_command_line_agrees_with_the_environment() -> None:
    from config import Config

    cli = Config.from_cli(["--dat", ".", "--exclude", "33, 59"])
    assert sorted(cli.exclude_segments) == ["33", "59"]


def test_a_list_from_json_still_works() -> None:
    from config import Config

    assert sorted(Config._coerce({"exclude_segments": ["33", "59"]}
                                 ).exclude_segments) == ["33", "59"]
