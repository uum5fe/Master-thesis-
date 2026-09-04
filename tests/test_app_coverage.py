"""The Coverage tab: why a spectrum stops, and why a segment is missing."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.views import coverage


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture()
def run(tmp_path, monkeypatch):
    import pandas as pd
    from app.data.sources import Catalog
    from app.settings import Settings

    folder = tmp_path / "results" / "2611976" / "45A"
    (folder / "gold").mkdir(parents=True)
    pd.DataFrame({"segment": ["1"], "R_ohmic": [0.06]}).to_csv(
        folder / "gold" / "plate_summary.csv", index=False)

    _write(folder / "segment_reach.csv", [
        {"segment": "1", "n_points": "40", "n_kept": "26", "f_min_hz": "0.2",
         "f_max_hz": "400", "n_above_f_max": "14", "blocked_by": "snr",
         "explanation": "SNR below the gate for this point",
         "verdict": "kept: 26 of 40 points"},
        {"segment": "33", "n_points": "0", "n_kept": "0", "f_min_hz": "",
         "f_max_hz": "", "n_above_f_max": "0", "blocked_by": "no_channel",
         "explanation": "no ADC channel ... never recorded, so there is "
                        "nothing to evaluate",
         "verdict": "not wired"},
        {"segment": "57", "n_points": "40", "n_kept": "2", "f_min_hz": "0.2",
         "f_max_hz": "0.4", "n_above_f_max": "38", "blocked_by": "snr",
         "explanation": "SNR below the gate for this point",
         "verdict": "dropped: only 2 point(s) survived, "
                    "cfg.min_points_per_spectrum is 8"},
    ])
    _write(folder / "point_rejections.csv",
           [{"segment": "1", "freq_hz": f"{f}", "kept": "1", "reason": "",
             "snr_db": "40", "cycles": "8", "sigma_rel": "0.01",
             "segment_verdict": "kept"} for f in (0.2, 2, 20, 200)]
           + [{"segment": "1", "freq_hz": f"{f}", "kept": "0", "reason": "snr",
               "snr_db": "2", "cycles": "8", "sigma_rel": "0.9",
               "segment_verdict": "kept"} for f in (800, 1600, 3200)])

    catalog = Catalog(Settings(results_roots=[str(tmp_path / "results")],
                               famos_roots=[], csv_roots=[])).refresh(
        kinds=("results",))
    monkeypatch.setattr(coverage.store, "current_catalog", lambda: catalog)
    return {"kind": "results", "measurement_id": "2611976",
            "condition": "45A", "plate": "gen1_r2d2_72"}


def test_the_tab_says_which_gate_limits_the_bandwidth(run):
    status, fig, reasons, vs_f, table = coverage.render(run)
    assert table is not None
    assert fig.data, "the coverage map drew nothing"
    # the reason bar names SNR, and the frequency box plot separates kept
    labels = list(reasons.data[0].y)
    assert "SNR below the gate" in labels
    assert any(t.name == "kept" for t in vs_f.data)


def test_an_unwired_segment_is_distinguished_from_a_rejected_one(run):
    """Both end as "no spectrum". Telling them apart is the whole point."""
    status, _fig, _r, _v, _t = coverage.render(run)
    text = str(status)
    assert "33" in text and "no ADC channel" in text
    assert "57" in text and "too few points survive" in text
    # and they are not conflated
    assert text.index("33") != text.index("57")


def test_a_run_without_a_coverage_report_says_how_to_get_one(tmp_path,
                                                             monkeypatch):
    import pandas as pd
    from app.data.sources import Catalog
    from app.settings import Settings

    folder = tmp_path / "results" / "2611976" / "45A" / "gold"
    folder.mkdir(parents=True)
    pd.DataFrame({"segment": ["1"], "R_ohmic": [0.06]}).to_csv(
        folder / "plate_summary.csv", index=False)
    catalog = Catalog(Settings(results_roots=[str(tmp_path / "results")],
                               famos_roots=[], csv_roots=[])).refresh(
        kinds=("results",))
    monkeypatch.setattr(coverage.store, "current_catalog", lambda: catalog)

    status, fig, _r, _v, table = coverage.render(
        {"kind": "results", "measurement_id": "2611976", "condition": "45A"})
    assert table is None and not fig.data
    assert "segment_reach.csv" in str(status)


def test_an_empty_selection_does_not_raise():
    assert coverage.render({})[4] is None
    assert coverage.render(None)[4] is None
