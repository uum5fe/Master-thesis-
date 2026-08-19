#!/usr/bin/env python3
"""Command-line driver for the `eis/` pipeline.

NOT the script for the Local EIS evaluation and dashboard. There are three
entry points at the top of this project and only one of them is this file:

    run_dashboard.py    start the dashboard
    run_evaluation.py   raw FAMOS .DAT -> gold/silver results, using local_eis/
    run_pipeline.py     this file: the separate `eis/` pipeline

`eis/` is a different implementation with a different output shape - measured
inter-card skew and clock drift, tables rather than a gold layer. It takes its
paths as arguments rather than from `.env`, and it does not compute the DRT
process split, the inferred segments or the fault labels. If what you want is
the evaluation that ran on Databricks, use `run_evaluation.py`.

Examples
--------
Process every condition found for one measurement::

    python run_pipeline.py --raw-dir /data/Famos --measurement-id 2611976 \
        --shunt-csv cal/curr.csv --temp-csv cal/temp.csv --out out/2611976

Use a designed multisine (leakage-free synchronous analysis)::

    python run_pipeline.py --config config/leepa.yaml \
        --base-frequency 1.0 --tones 2,3,5,8,13,21,34,55,89,144,233,377,610

Report the synchronisation only, without computing spectra::

    python run_pipeline.py --raw-dir /data/Famos --sync-only

Generate a self-contained demonstration on synthetic data with known truth::

    python run_pipeline.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Locally-resolved EIS: synchronised current density and "
                    "cell voltage -> per-segment impedance spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="YAML configuration file")
    p.add_argument("--raw-dir", help="directory holding the FAMOS .DAT files")
    p.add_argument("--measurement-id", default="", help="measurement / order ID")
    p.add_argument("--conditions", help="comma-separated list; default is all found")
    p.add_argument("--out", dest="output_dir", help="output directory")

    cal = p.add_argument_group("calibration")
    cal.add_argument("--shunt-csv", help="per-segment via-shunt c0;c1 file")
    cal.add_argument("--temp-csv", help="per-sensor temperature c0;c1 file")
    cal.add_argument("--segment-area", type=float, help="segment area [cm^2]")
    cal.add_argument("--cell-area", type=float, help="active cell area [cm^2]")

    sp = p.add_argument_group("spectral estimation")
    sp.add_argument("--method", choices=["auto", "welch", "synchronous"])
    sp.add_argument("--nperseg", type=int, help="Welch window length")
    sp.add_argument("--estimator", choices=["hv", "h1", "h2"])
    sp.add_argument("--f-min", type=float)
    sp.add_argument("--f-max", type=float)
    sp.add_argument("--coherence", type=float, help="gamma^2 quality gate")
    sp.add_argument("--base-frequency", type=float,
                    help="multisine base frequency [Hz]; enables synchronous DFT")
    sp.add_argument("--tones", help="comma-separated excitation tones [Hz]")
    sp.add_argument("--no-notch", action="store_true", help="disable mains notches")
    sp.add_argument("--no-gate", action="store_true", help="disable window gating")

    sy = p.add_argument_group("synchronisation")
    sy.add_argument("--uc-strategy", choices=["auto", "same_card", "reference"])
    sy.add_argument("--sync-tolerance-ns", type=float,
                    help="residual skew budget after alignment")
    sy.add_argument("--sync-only", action="store_true",
                    help="measure and report timing, then stop")

    md = p.add_argument_group("validation and modelling")
    md.add_argument("--no-kk", action="store_true", help="skip Kramers-Kronig")
    md.add_argument("--no-ecm", action="store_true", help="skip circuit fitting")
    md.add_argument("--models", help="comma-separated ECM ladder")

    p.add_argument("--plots", action="store_true", help="write figures alongside tables")
    p.add_argument("--demo", action="store_true",
                   help="run on generated synthetic data with known ground truth")
    p.add_argument("--quiet", action="store_true")
    return p


def apply_overrides(cfg, args) -> None:
    if args.measurement_id:
        cfg.measurement_id = args.measurement_id
    if args.raw_dir:
        cfg.raw_dir = args.raw_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.conditions:
        cfg.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    if args.shunt_csv:
        cfg.calibration.shunt_csv = args.shunt_csv
    if args.temp_csv:
        cfg.calibration.temperature_csv = args.temp_csv
    if args.segment_area:
        cfg.geometry.segment_area_cm2 = args.segment_area
    if args.cell_area:
        cfg.geometry.cell_area_cm2 = args.cell_area

    if args.method:
        cfg.spectral.method = args.method
    if args.nperseg:
        cfg.spectral.nperseg = args.nperseg
    if args.estimator:
        cfg.spectral.estimator = args.estimator
    if args.f_min is not None:
        cfg.spectral.f_min_hz = args.f_min
    if args.f_max is not None:
        cfg.spectral.f_max_hz = args.f_max
    if args.coherence is not None:
        cfg.spectral.min_coherence = args.coherence
    if args.base_frequency:
        cfg.spectral.base_frequency_hz = args.base_frequency
    if args.tones:
        cfg.spectral.excitation_tones_hz = [
            float(t) for t in args.tones.split(",") if t.strip()
        ]
    if args.no_notch:
        cfg.spectral.apply_notch = False
    if args.no_gate:
        cfg.spectral.gate_windows = False

    if args.uc_strategy:
        cfg.sync.uc_strategy = args.uc_strategy
    if args.sync_tolerance_ns:
        cfg.sync.residual_tolerance_s = args.sync_tolerance_ns * 1e-9

    if args.no_kk:
        cfg.validation.run_lin_kk = False
    if args.no_ecm:
        cfg.model.run_ecm = False
    if args.models:
        cfg.model.models = [m.strip() for m in args.models.split(",") if m.strip()]


def make_demo(cfg, verbose: bool = True):
    """Generate a synthetic measurement with imposed timing faults."""
    from tests.synthetic import simulate_measurement

    raw = Path(cfg.output_dir) / "demo_raw"
    raw.mkdir(parents=True, exist_ok=True)
    truth = simulate_measurement(
        raw, measurement_id="DEMO", condition="150A",
        n_cards=4, segments_per_card=5, duration_s=40.0,
        card_delays_s={2: 1.7e-4, 3: -5.2e-4, 4: 3.1e-3},   # 1.7 / -5.2 / 31 samples
        card_drift_ppm={4: 6.0},                            # free-running clock
        channel_delays_s={2: {"3": 50e-6, "4": 50e-6}},     # intra-card skew
        dead_uc_cards=(3,),                                 # a failed UC channel
    )
    cfg.measurement_id = "DEMO"
    cfg.raw_dir = str(raw)
    cfg.conditions = ["150A"]
    cfg.calibration.shunt_csv = str(raw / "curr.csv")
    cfg.calibration.temperature_csv = str(raw / "temp.csv")
    cfg.spectral.base_frequency_hz = 1.0
    cfg.spectral.excitation_tones_hz = list(truth.tones_hz)
    cfg.acquisition.channel_delay_table = {2: {"3": 50e-6, "4": 50e-6}}
    cfg.acquisition.channel_delay_table_version = "demo-known-truth"
    if verbose:
        print(
            "Synthetic demonstration\n"
            "  imposed: card 2 +170 us, card 3 -520 us, card 4 +3.1 ms\n"
            "           card 4 clock +6 ppm, card 2 channels 3/4 skewed 50 us\n"
            "           card 3 cell-voltage channel dead\n"
        )
    return truth


def report_truth(results, truth) -> None:
    """Compare recovered impedances against the values that were put in."""
    print(f"\n{'=' * 72}\n  ACCURACY AGAINST KNOWN TRUTH\n{'=' * 72}")
    print(f"  {'card':>4}  {'segs':>4}  {'median |dZ|/|Z|':>16}  {'worst':>8}")
    for result in results.values():
        per_card: dict[int, list[float]] = {}
        for seg, r in result.segments.items():
            Z_true = truth.Z_at(seg, r.spectrum.f)
            error = np.abs(r.spectrum.Z - Z_true) / np.abs(Z_true)
            per_card.setdefault(r.card, []).append(float(np.median(error)))
        for card, errors in sorted(per_card.items()):
            print(f"  {card:>4}  {len(errors):>4}  {np.median(errors):>15.4%}  "
                  f"{max(errors):>7.3%}")
        if result.rejected:
            print(f"  rejected: {len(result.rejected)} segments")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from eis.config import load_config
    from eis.pipeline import run_measurement, write_outputs

    cfg = load_config(args.config)
    apply_overrides(cfg, args)

    truth = None
    if args.demo:
        cfg.output_dir = cfg.output_dir or "./out/demo"
        truth = make_demo(cfg, verbose=not args.quiet)
    if not cfg.raw_dir or cfg.raw_dir == ".":
        print("error: --raw-dir is required (or use --demo)\n", file=sys.stderr)
        print("  This is run_pipeline.py, the CLI for the separate `eis/`\n"
              "  pipeline. It takes its paths as arguments, not from .env.\n",
              file=sys.stderr)
        print("  For the Local EIS evaluation - the bronze/silver/gold pipeline\n"
              "  that ran on Databricks, reading its paths from .env - you want:\n"
              "\n"
              "      python run_evaluation.py --list\n"
              "      python run_evaluation.py --all\n"
              "\n"
              "  and then:\n"
              "\n"
              "      python run_dashboard.py --open\n",
              file=sys.stderr)
        return 2
    cfg.output_dir = cfg.output_dir or "./out"

    if args.sync_only:
        from eis.io.famos import discover_files
        from eis.pipeline import inventory_card, measure_sync

        found = discover_files(
            cfg.raw_dir, cfg.acquisition.discovery_regex, cfg.measurement_id or None
        )
        for condition, files in sorted(found.items()):
            if cfg.conditions and condition not in cfg.conditions:
                continue
            print(f"\n=== {condition} ===")
            inventories = {c: inventory_card(p, c) for c, p in sorted(files.items())}
            report, _ = measure_sync(inventories, cfg)
            frame = report.to_frame()
            print(frame.to_string(index=False))
            print(f"reference card {report.reference_card}; "
                  f"closure residuals [ns]: "
                  f"{ {k: round(v * 1e9, 1) for k, v in report.closure_s.items()} }")
            print(f"overall: {'PASS' if report.passed else 'FAIL'}")
            for note in report.notes:
                print(f"  ! {note}")
        return 0

    results = run_measurement(cfg, write=True, verbose=not args.quiet)

    if args.plots:
        from eis.viz import write_report_figures

        for condition, result in results.items():
            paths = write_report_figures(
                result, cfg, Path(cfg.output_dir) / "figures" / condition
            )
            if not args.quiet:
                for path in paths:
                    print(f"  figure: {path}")

    if truth is not None:
        report_truth(results, truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
