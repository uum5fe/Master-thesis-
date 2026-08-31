#!/usr/bin/env python3
"""Main script of the pipeline: orchestration and the command line.

    FAMOS .DAT (N cards)
      -> bronze : measured skew, drift, alignment, physical scaling
      -> silver : spectra, uncertainty, the high-frequency chain, verdicts
      -> gold   : circuit models, maps, tables, the interactive plate view

Every tier is importable on its own; this module is the only place that knows
the order they run in, which is what keeps the tiers free of each other.

Examples
--------
Process every condition found for one measurement::

    python -m eis.pipeline.main --raw-dir /data/Famos --measurement-id 2611976 \
        --shunt-csv cal/curr.csv --temp-csv cal/temp.csv --out out/2611976

Use a designed multisine (leakage-free synchronous analysis)::

    python -m eis.pipeline.main --config config/example.yaml \
        --base-frequency 1.0 --tones 2,3,5,8,13,21,34,55,89,144,233,377,610

Report the synchronisation only, without computing spectra::

    python -m eis.pipeline.main --raw-dir /data/Famos --sync-only

Self-contained demonstration on synthetic data with known ground truth::

    python -m eis.pipeline.main --demo --plots
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from eis import __version__
from eis.calibrate import load_shunt_calibration, load_temperature_calibration
from eis.io.famos import discover_files
from eis.pipeline.bronze import run_bronze
from eis.pipeline.config import PipelineConfig, load_config
from eis.pipeline.gold import (
    ConditionResult, run_gold, summarise, write_outputs, write_report,
)
from eis.pipeline.silver import run_silver
from eis.pipeline.utils import banner, make_logger, provenance


# ---------------------------------------------------------------------------
# One operating point
# ---------------------------------------------------------------------------

def run_condition(
    cfg: PipelineConfig,
    condition: str,
    card_files: dict[int, str],
    shunt=None,
    temp_cal=None,
    reference_spectrum: tuple[np.ndarray, np.ndarray] | None = None,
    verbose: bool = True,
) -> ConditionResult:
    """Bronze, then silver, then gold, for a single condition."""
    started = time.time()
    log = make_logger(verbose)
    log(banner(f"{cfg.measurement_id} / {condition}"))

    bronze = run_bronze(cfg, condition, card_files, shunt, temp_cal, log=log)
    silver = run_silver(bronze, cfg, reference_spectrum, log=log)
    result = run_gold(
        silver, cfg, bronze.temperature, provenance(cfg, __version__), log=log
    )
    result.elapsed_s = time.time() - started
    log(f"  done in {result.elapsed_s:.1f} s")
    log(summarise(result, cfg))
    return result


# ---------------------------------------------------------------------------
# A whole measurement
# ---------------------------------------------------------------------------

def run_measurement(
    cfg: PipelineConfig,
    reference_spectra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    write: bool = True,
    verbose: bool = True,
) -> dict[str, ConditionResult]:
    """Discover, process and persist every requested condition."""
    log = make_logger(verbose)

    discovered = discover_files(
        cfg.raw_dir, cfg.acquisition.discovery_regex, cfg.measurement_id or None
    )
    if not discovered:
        raise FileNotFoundError(
            f"no files under {cfg.raw_dir} match {cfg.acquisition.discovery_regex!r}"
        )
    conditions = cfg.conditions or sorted(discovered)
    log(f"discovered conditions: {sorted(discovered)}  ->  processing {conditions}")

    shunt = (
        load_shunt_calibration(cfg.calibration.shunt_csv)
        if cfg.calibration.shunt_csv else None
    )
    temp_cal = (
        load_temperature_calibration(cfg.calibration.temperature_csv)
        if cfg.calibration.temperature_csv else None
    )
    if shunt is None:
        log("WARNING: no shunt calibration configured - impedances will be in "
            "shunt volts per amp, not physical ohms, and every segment is "
            "marked no_calibration")

    results: dict[str, ConditionResult] = {}
    for condition in conditions:
        if condition not in discovered:
            log(f"  skipping {condition}: no files")
            continue
        results[condition] = run_condition(
            cfg, condition, discovered[condition], shunt, temp_cal,
            (reference_spectra or {}).get(condition), verbose=verbose,
        )

    if write and results:
        write_outputs(cfg, results, log=log)
    return results


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eis.pipeline.main",
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
    sp.add_argument("--method",
                    choices=["auto", "welch", "multiresolution", "synchronous"])
    sp.add_argument("--nperseg", type=int, help="longest analysis window")
    sp.add_argument("--estimator", choices=["hv", "h1", "h2"])
    sp.add_argument("--f-min", type=float)
    sp.add_argument("--f-max", type=float)
    sp.add_argument("--coherence", type=float, help="gamma^2 quality gate")
    sp.add_argument("--base-frequency", type=float,
                    help="multisine base frequency [Hz]; enables synchronous DFT")
    sp.add_argument("--tones", help="comma-separated excitation tones [Hz]")
    sp.add_argument("--no-notch", action="store_true", help="disable mains notches")
    sp.add_argument("--no-gate", action="store_true", help="disable window gating")

    hf = p.add_argument_group("high-frequency accuracy")
    hf.add_argument("--shunt-inductance-nh", type=float,
                    help="via-shunt loop inductance [nH]; drives the complex "
                         "shunt response, the largest HF correction there is")
    hf.add_argument("--aa-order", type=int, help="anti-alias filter order")
    hf.add_argument("--aa-corner", type=float, help="anti-alias corner [Hz]")
    hf.add_argument("--aa-mismatch", type=float,
                    help="fractional corner mismatch between the segment and "
                         "cell-voltage channels")
    hf.add_argument("--crosstalk-alpha", type=float,
                    help="in-plane current fraction shared with neighbours")
    hf.add_argument("--no-common-mode", action="store_true",
                    help="skip the pooled identification of the shared delay")
    hf.add_argument("--no-hf", action="store_true",
                    help="disable the whole high-frequency correction chain")

    sy = p.add_argument_group("synchronisation")
    sy.add_argument("--uc-strategy", choices=["auto", "same_card", "reference"])
    sy.add_argument("--sync-tolerance-ns", type=float,
                    help="residual skew budget after alignment")
    sy.add_argument("--sync-only", action="store_true",
                    help="measure and report timing, then stop")

    md = p.add_argument_group("validation and modelling")
    md.add_argument("--no-kk", action="store_true", help="skip Kramers-Kronig")
    md.add_argument("--no-stationarity", action="store_true",
                    help="skip the split-half stationarity test")
    md.add_argument("--no-ecm", action="store_true", help="skip circuit fitting")
    md.add_argument("--models", help="comma-separated ECM ladder")

    rp = p.add_argument_group("reporting")
    rp.add_argument("--plots", action="store_true",
                    help="write figures and the interactive plate view")
    rp.add_argument("--no-dashboard", action="store_true",
                    help="write the static figures but not the HTML view")

    p.add_argument("--demo", action="store_true",
                   help="run on generated synthetic data with known ground truth")
    p.add_argument("--quiet", action="store_true")
    return p


def apply_overrides(cfg: PipelineConfig, args) -> None:
    """Map command-line flags onto the configuration object."""
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

    if args.shunt_inductance_nh is not None:
        cfg.hf.shunt_inductance_nh = args.shunt_inductance_nh
    if args.aa_order is not None:
        cfg.hf.aa_order = args.aa_order
    if args.aa_corner is not None:
        cfg.hf.aa_corner_hz = args.aa_corner
    if args.aa_mismatch is not None:
        cfg.hf.aa_corner_mismatch = args.aa_mismatch
    if args.crosstalk_alpha is not None:
        cfg.hf.crosstalk_alpha = args.crosstalk_alpha
    if args.no_common_mode:
        cfg.hf.identify_common_mode = False
    if args.no_hf:
        cfg.hf.enabled = False

    if args.uc_strategy:
        cfg.sync.uc_strategy = args.uc_strategy
    if args.sync_tolerance_ns:
        cfg.sync.residual_tolerance_s = args.sync_tolerance_ns * 1e-9

    if args.no_kk:
        cfg.validation.run_lin_kk = False
    if args.no_stationarity:
        cfg.validation.run_stationarity = False
    if args.no_ecm:
        cfg.model.run_ecm = False
    if args.models:
        cfg.model.models = [m.strip() for m in args.models.split(",") if m.strip()]

    cfg.report.write_figures = bool(args.plots)
    cfg.report.write_dashboard = bool(args.plots) and not args.no_dashboard


# ---------------------------------------------------------------------------
# Demonstration on known truth
# ---------------------------------------------------------------------------

def make_demo(cfg: PipelineConfig, verbose: bool = True):
    """Generate a synthetic measurement with imposed instrument faults."""
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
        shunt_inductance_nh=0.6,                            # the HF error
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
    cfg.hf.shunt_inductance_nh = 0.6
    if verbose:
        print(
            "Synthetic demonstration\n"
            "  imposed: card 2 +170 us, card 3 -520 us, card 4 +3.1 ms\n"
            "           card 4 clock +6 ppm, card 2 channels 3/4 skewed 50 us\n"
            "           card 3 cell-voltage channel dead\n"
            "           via-shunt loop inductance 0.6 nH on every segment\n"
        )
    return truth


def report_truth(results: dict[str, ConditionResult], truth) -> None:
    """Compare recovered impedances against the values that were put in."""
    print(banner("ACCURACY AGAINST KNOWN TRUTH"))
    print(f"  {'card':>4}  {'segs':>4}  {'median |dZ|/|Z|':>16}  {'worst':>8}")
    for result in results.values():
        per_card: dict[int, list[float]] = {}
        for seg, r in result.segments.items():
            if not r.active or len(r.spectrum.f) == 0:
                continue
            Z_true = truth.Z_at(seg, r.spectrum.f)
            error = np.abs(r.spectrum.Z - Z_true) / np.abs(Z_true)
            per_card.setdefault(r.card, []).append(float(np.median(error)))
        for card, errors in sorted(per_card.items()):
            print(f"  {card:>4}  {len(errors):>4}  {np.median(errors):>15.4%}  "
                  f"{max(errors):>7.3%}")
        inactive = result.rejected
        if inactive:
            print(f"  inactive: {len(inactive)} segments")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    cfg = load_config(args.config)
    apply_overrides(cfg, args)

    truth = None
    if args.demo:
        cfg.output_dir = cfg.output_dir or "./out/demo"
        truth = make_demo(cfg, verbose=not args.quiet)
    if not cfg.raw_dir or cfg.raw_dir == ".":
        print("error: --raw-dir is required (or use --demo)", file=sys.stderr)
        return 2
    cfg.output_dir = cfg.output_dir or "./out"

    if args.sync_only:
        return _sync_only(cfg)

    results = run_measurement(cfg, write=True, verbose=not args.quiet)

    if cfg.report.write_figures or cfg.report.write_dashboard:
        write_report(cfg, results, log=make_logger(not args.quiet))

    if truth is not None:
        report_truth(results, truth)
    return 0


def _sync_only(cfg: PipelineConfig) -> int:
    """Measure and report the timing without computing any spectrum."""
    from eis.pipeline.bronze import inventory_card, measure_sync

    found = discover_files(
        cfg.raw_dir, cfg.acquisition.discovery_regex, cfg.measurement_id or None
    )
    for condition, files in sorted(found.items()):
        if cfg.conditions and condition not in cfg.conditions:
            continue
        print(f"\n=== {condition} ===")
        inventories = {c: inventory_card(p, c) for c, p in sorted(files.items())}
        report, _ = measure_sync(inventories, cfg)
        print(report.to_frame().to_string(index=False))
        print(f"reference card {report.reference_card}; closure residuals [ns]: "
              f"{ {k: round(v * 1e9, 1) for k, v in report.closure_s.items()} }")
        print(f"overall: {'PASS' if report.passed else 'FAIL'}")
        for note in report.notes:
            print(f"  ! {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
