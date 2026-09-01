#!/usr/bin/env python3
"""
main.py  --  the pipeline
=========================

    python main.py --dat /Volumes/.../Famos --curr-cal curr.csv \
        --temp-cal temp.csv --leepa 2611976 --condition 450A \
        --out ./results --current 450

    python main.py --self-test          # no data needed
    python main.py --print-config

FLOW
----
    bronze   FAMOS -> consensus schedule -> raw phasors        (no physics)
    silver   de-skew -> uncertainty -> model -> validate       (per segment)
    gold     process split -> spatial field -> maps            (per plate)

Each layer writes its own directory and can be re-run from the one before it,
which matters because bronze is the slow part: re-reading tens of gigabytes of
.DAT to try a different skew model is what made the old pipeline painful to
improve.

WHAT CHANGED, AND WHAT THE EVIDENCE FOR IT IS
---------------------------------------------
Every claim below is checked by `validate_hf.py` against synthetic spectra
with known truth.  Run it; it takes a couple of minutes.

  1. The acquisition skew is per-SEGMENT and mostly known in advance.
     A multiplexed converter samples channel slot p at p/(n_ch*fs).  The
     legacy per-card single delay cannot represent that at all: on the
     synthetic card it left the segment-to-segment skew completely
     untouched, 23.4 us, which is 29.5 degrees of phase error at 3.5 kHz and
     is what fanned the high-frequency arcs apart.  The structural model
     leaves 0.4 us, or 0.5 degrees.

  2. Once that is fixed, the elaborate estimator stops paying.
     R_ohmic is now a Cramer-Rao-weighted mean of Re Z over the top third of
     a decade.  Against a Boukamp lin-KK R0 and a GP-DRT posterior on the
     same data it was five times more accurate when the arc closes inside
     the band, and no worse when it does not.  This is a negative result and
     it is reported as one.

  3. What the band cannot reach is reported, not invented.
     If a relaxation sits above f_max, R_ohmic absorbs it and EVERY method
     tested carried the same bias -- checked down to 0.2 % noise, where the
     model bias got worse rather than better because it was then fitting its
     own prior.  So the residual imaginary part at f_max is turned into a
     bound on the hidden resistance and added to the uncertainty.  The fix
     for this is a higher sampling rate, not a cleverer inversion.

  4. No segment is ever deleted.
     Quality tiers replace the old drop-on-any-gate behaviour, missing
     calibration rows are imputed from the plate median and flagged, and
     unmeasurable segments are filled from a Gaussian-process field and drawn
     hatched.  A hole in a heat map reads to the eye as a cold spot, which is
     worse than an honest "inferred" marker.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import utils
from config import Config, DEFAULT


# ---------------------------------------------------------------------------


#: --f-max not given at all. Distinct from None, which is the real value
#: meaning "derive the ceiling from fs" and which the user may ask for.
_F_MAX_UNSET = object()


def _f_max_arg(value: str):
    """--f-max: a number, or 'auto' for 0.45*fs."""
    text = str(value).strip().lower()
    return None if text in ("auto", "none", "fs", "") else float(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("hardware / source")
    g.add_argument("--plate", choices=["gen1", "gen2"], default=None,
                   help="gen1 = green/Kashyyyk, gen2 = blue/Naboo "
                        "(default: gen1)")
    g.add_argument("--source", choices=["famos", "csv"], default=None,
                   help="measurement file format (default: famos)")
    g.add_argument("--csv", help="CSV measurement file or folder, with --source csv")
    g.add_argument("--csv-dialect", default=None,
                   help="force a CSV layout instead of auto-detecting; "
                        "see csv_source.DIALECTS")

    g = p.add_argument_group("input")
    g.add_argument("--dat", help="directory of FAMOS .DAT recordings")
    g.add_argument("--curr-cal", help="per-segment Abgleich, 72 rows 'c0;c1'")
    g.add_argument("--temp-cal", help="per-sensor Abgleich, 4 rows 'c0;c1'")
    g.add_argument("--areas", help="optional per-segment area override CSV")
    g.add_argument("--equal-areas", action="store_true",
                   help="treat every segment as A_cell/72 = 4.235 cm2")
    g.add_argument("--gain", help="optional ex-situ chain response")
    g.add_argument("--gamry", dest="gamry_dir",
                   help="folder of whole-cell Gamry .DTA sweeps; the 72 "
                        "segments in parallel are compared against them")
    g.add_argument("--bench-log", dest="bench_log",
                   help="ASAM MDF4 bench log, to report the operating point "
                        "at each reference sweep (default: found in --gamry)")
    g.add_argument("--exclude", dest="exclude_segments", default=None,
                   help="comma-separated segments to skip entirely, for a "
                        "DEMONSTRATED hardware fault. Empty by default: an "
                        "excluded segment is never measured, so the exclusion "
                        "can never be disproved")
    g.add_argument("--leepa", default="", help="order id, e.g. 2611976")
    g.add_argument("--condition", default="ALL", help="e.g. 450A")
    g.add_argument("--current", type=float, default=None,
                   help="load setpoint in A; only ever used as a cross-check")

    g = p.add_argument_group("output")
    g.add_argument("--out", default="./results")
    g.add_argument("--no-png", action="store_true")
    g.add_argument("--no-html", action="store_true")
    g.add_argument("-q", "--quiet", action="store_true")

    g = p.add_argument_group("band")
    g.add_argument("--f-min", type=float, default=None)
    g.add_argument("--f-max", type=_f_max_arg, default=_F_MAX_UNSET,
                   help="detection ceiling in Hz; 'auto' (the default) is "
                        "0.45*fs, so recording faster actually searches "
                        "higher")
    g.add_argument("--ppd", type=int, default=None)
    g.add_argument("--no-require-timebase", dest="require_timebase",
                   action="store_false", default=None,
                   help="evaluate even when the cards are not one consistent "
                        "measurement. The result is then diagnostic only: "
                        "windows that mean different instants on different "
                        "cards produce a confident plate, not a worse one")
    g.add_argument("--excitation", choices=["auto", "stepped", "multisine"],
                   default=None,
                   help="excitation type; 'auto' (the default) counts the "
                        "tones per dwell and decides")

    g = p.add_argument_group("algorithm")
    g.add_argument("--skew", choices=["structural", "per_card", "none"],
                   default=None,
                   help="acquisition-skew model (default: structural)")
    g.add_argument("--phasor", choices=["joint7", "independent"], default=None,
                   help="sine-fit estimator (default: joint7)")
    g.add_argument("--preset", choices=["default", "permissive", "strict"],
                   default=None,
                   help="gate preset; 'permissive' keeps almost everything, "
                        "for comparing against a coherence-threshold pipeline")
    g.add_argument("--no-drt", action="store_true",
                   help="skip the DRT; loses the process split and tau maps")
    g.add_argument("--no-spatial", action="store_true",
                   help="do not infer unmeasured segments")
    g.add_argument("--keep-L", action="store_true",
                   help="leave the series inductance in the output curves")

    g = p.add_argument_group("other")
    g.add_argument("--config", help="JSON config file; CLI flags win over it")
    g.add_argument("--print-config", action="store_true")
    g.add_argument("--self-test", action="store_true",
                   help="run the synthetic checks; needs no measurement data")
    g.add_argument("--stop-after", choices=["bronze", "silver", "gold"],
                   default="gold")
    return p


def config_from_args(a) -> Config:
    cfg = Config.from_json(a.config) if a.config else DEFAULT
    kw = {}
    if a.plate:
        kw["plate"] = a.plate
    if a.source:
        kw["source_format"] = a.source
    if a.csv:
        kw["csv_path"] = Path(a.csv)
        kw.setdefault("source_format", "csv")
    if a.csv_dialect:
        kw["csv_dialect"] = a.csv_dialect
    if a.dat:
        kw["dat_dir"] = Path(a.dat)
    if a.out:
        kw["out_dir"] = Path(a.out)
    if a.curr_cal:
        kw["curr_cal"] = Path(a.curr_cal)
    if a.temp_cal:
        kw["temp_cal"] = Path(a.temp_cal)
    if a.areas:
        kw["areas_file"] = Path(a.areas)
    if a.equal_areas:
        kw["equal_areas"] = True
    if a.gain:
        kw["gain_file"] = Path(a.gain)
    if a.gamry_dir:
        kw["gamry_dir"] = Path(a.gamry_dir)
    if a.bench_log:
        kw["bench_log"] = Path(a.bench_log)
    if a.exclude_segments is not None:
        kw["exclude_segments"] = frozenset(
            x.strip() for x in a.exclude_segments.split(",") if x.strip())
    if a.leepa:
        kw["leepa"] = a.leepa
    if a.condition:
        kw["condition"] = a.condition
    if a.current is not None:
        kw["i_setpoint_a"] = a.current
    if a.f_min is not None:
        kw["f_min_hz"] = a.f_min
    # "auto" arrives as the sentinel below rather than as None, because None
    # is itself a meaningful value for f_max_hz (derive it from fs) and an
    # absent flag must leave the config's own default alone.
    if a.f_max is not _F_MAX_UNSET:
        kw["f_max_hz"] = a.f_max
    if a.ppd is not None:
        kw["ppd"] = a.ppd
    if getattr(a, "excitation", None):
        kw["excitation"] = a.excitation
    if getattr(a, "require_timebase", None) is False:
        kw["require_timebase"] = False
    if a.skew:
        kw["skew_model"] = a.skew
    if a.phasor:
        kw["phasor_method"] = a.phasor
    if a.no_drt:
        kw["drt_enable"] = False
    if a.no_spatial:
        kw["infer_missing_segments"] = False
    if a.keep_L:
        kw["remove_inductance"] = False
    if a.no_png:
        kw["write_png"] = False
    if a.no_html:
        kw["write_html"] = False
    if a.quiet:
        kw["verbose"] = False
    cfg = cfg.replace(**kw)
    if a.preset:
        cfg = cfg.preset(a.preset)
    return cfg


# ---------------------------------------------------------------------------


def geom_area(cfg: Config) -> float:
    """The cell area the comparison must use.

    Read from the geometry actually in force rather than the module constant,
    so an area override or a different plate generation reaches the whole-cell
    comparison too.
    """
    import r2d2_geometry as geom
    return float(sum(geom.areas().values()))


def run_pipeline(cfg: Config, stop_after: str = "gold") -> dict:
    """bronze -> silver -> gold, with a manifest tying the three together.

    With `cfg.source_format == "csv"` the FAMOS stages are bypassed entirely
    and csv_pipeline.run_csv() is used instead -- see the note on
    Config.source_format for why the two are separate pipelines rather than
    two readers feeding one.
    """
    import bronze
    import silver
    import gold
    import r2d2_geometry as geom

    # SELECT THE PLATE BEFORE ANY MODULE READS geom.SEGMENTS.  Every stage
    # takes areas and centroids from the module-level binding, so this has to
    # happen first, once, and be recorded in the manifest.
    plate = geom.use_plate(cfg.plate)

    if cfg.source_format == "csv":
        import csv_pipeline
        return csv_pipeline.run_csv(cfg, stop_after=stop_after)

    log = utils.get_logger(cfg.verbose)
    t0 = time.time()
    utils.banner("R2-D2 LOCAL EIS PIPELINE", log)
    log.info(f"  plate: {plate.title}   source: FAMOS")
    log.info(f"  {cfg.leepa or '-'} / {cfg.condition}   "
             f"phasor={cfg.phasor_method}  skew={cfg.skew_model}  "
             f"drt={'on' if cfg.drt_enable else 'off'}  "
             f"spatial={'on' if cfg.spatial_enable else 'off'}")
    log.info(f"  in  : {cfg.dat_dir}")
    log.info(f"  out : {cfg.out_dir}")

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(Path(cfg.out_dir) / "config_used.json")

    manifest = {"config": cfg.to_dict(), "plate": plate.key,
                "plate_drawing": plate.drawing, "stages": {}}

    br = bronze.run(cfg, log)
    bronze.save(br, cfg, log)
    manifest["stages"]["bronze"] = br.summary()
    if stop_after == "bronze":
        return manifest

    sr = silver.run(br, cfg, log)
    silver.save(sr, cfg, log)
    manifest["stages"]["silver"] = {
        "n_segments": len(sr.spectra), "tiers": sr.tiers(),
        "dc_closure": sr.dc_closure,
        "skew": {k: {"slot_us": v.k_slot * 1e6,
                     "slot_nominal_us": v.k_nominal * 1e6,
                     "dt0_us": v.dt0 * 1e6, "applied": v.applied,
                     "note": v.note} for k, v in sr.skew.items()},
    }
    if stop_after == "silver":
        return manifest

    gr = gold.run(sr, cfg, log)
    gold.save(gr, sr, cfg, log)
    manifest["stages"]["gold"] = gr.stats

    # ---- the whole-cell cross-check ---------------------------------------
    # Last, because it needs the aggregate silver has just written, and
    # non-fatal, because a missing or unreadable reference must not throw away
    # a run that is otherwise complete.
    reference_asr = None
    reference_hfr = float("nan")
    if cfg.gamry_dir:
        utils.banner("WHOLE-CELL REFERENCE  --  local aggregate vs Gamry", log)
        try:
            import gamry_compare
            comps = gamry_compare.run(
                cfg.out_dir, cfg.gamry_dir, geom_area(cfg),
                out_dir=cfg.out_dir, bench_path=cfg.bench_log,
                chain_applied=cfg.gain_file is not None,
                only=cfg.condition, order_id=cfg.leepa, log=log)
            manifest["stages"]["gamry"] = [c.summary() for c in comps]
            if comps and comps[0].freq.size:
                # the reference arm of the plausibility aggregate check, in
                # the same ohm.cm2 the local side already uses
                reference_asr = (comps[0].freq, comps[0].Z_ref)
                # ... and its R_s, for the parallel-sum closure. The measured
                # intercept when the sweep reached it, otherwise the
                # extrapolation, which is what the local parallel sum should
                # be compared against when neither curve crosses the axis.
                import math
                reference_hfr = (comps[0].hfr_ref
                                 if math.isfinite(comps[0].hfr_ref)
                                 else comps[0].hfr_ref_fit)
            if cfg.write_png and comps:
                gamry_compare.plot(
                    comps, Path(cfg.out_dir) / "gamry_comparison.png")
        except Exception as exc:                            # noqa: BLE001
            log.warning(f"  whole-cell comparison skipped: {exc}")

    # ---- does the finished plate hold together as physics? ----------------
    # After gold, because it judges the assembled map rather than any single
    # point, and non-fatal for the same reason the cross-check is: a
    # diagnostic that can end a run is a liability, not a safeguard.
    try:
        import plausibility
        rep = plausibility.check_run(
            sr, cfg, plate_key=cfg.plate, reference=reference_asr,
            reference_hfr=reference_hfr)
        plausibility.report(rep, log)
        plausibility.save(rep, cfg.out_dir)
        manifest["stages"]["plausibility"] = rep.rows()
    except Exception as exc:                                # noqa: BLE001
        log.warning(f"  plausibility checks skipped: {exc}")

    dt = time.time() - t0
    utils.banner("DONE", log)
    log.info(f"  {gr.stats['n_measured']}/{gr.stats['n_total']} segments "
             f"measured, {gr.stats['n_inferred']} inferred, "
             f"{gr.stats['n_bad']} hardware-bad")
    if "R_ohmic" in gr.stats:
        r = gr.stats["R_ohmic"]
        log.info(f"  R_ohmic {1000*r['mean']:.1f} +/- {1000*r['sd']:.1f} "
                 f"mOhm cm2, spread {r['spread']:.2f}x")
    log.info(f"  {dt:.1f} s -> {cfg.out_dir}")

    manifest["elapsed_s"] = dt
    utils.write_json(Path(cfg.out_dir) / "run_manifest.json", manifest)
    return manifest


def self_test() -> int:
    """Everything that can be checked without measurement data."""
    log = utils.get_logger(True)
    fails = 0

    import utils as u
    fails += u._selftest()

    import validate_hf
    fails += validate_hf.main()

    utils.banner("import and geometry checks", log)
    import bronze, silver, gold           # noqa: F401
    import r2d2_geometry as geom
    for key in geom.PLATES:
        geom.use_plate(key)
        chk = geom.self_check(verbose=False)
        ok = len(geom.SEGMENTS) == 72 and not chk["problems"]
        log.info(f"  geometry {key}: {len(geom.SEGMENTS)} segments, "
                 f"areas {chk['area_min_cm2']:.3f}"
                 f"..{chk['area_max_cm2']:.3f} cm2, "
                 f"sum {chk['area_sum_cm2']:.2f} cm2   "
                 f"{'PASS' if ok else 'FAIL: ' + '; '.join(chk['problems'])}")
        fails += not ok
    geom.use_plate(geom.DEFAULT_PLATE)

    import csv_source
    n = csv_source._selftest(log)
    log.info(f"  csv_source: {'PASS' if not n else str(n) + ' FAILURES'}")
    fails += n

    import gamry_compare
    n = gamry_compare._selftest(log)
    log.info(f"  gamry_compare: {'PASS' if not n else str(n) + ' FAILURES'}")
    fails += n

    import plate_conditions
    n = plate_conditions._selftest(log)
    log.info(f"  plate_conditions: {'PASS' if not n else str(n) + ' FAILURES'}")
    fails += n
    log.info(f"\n  {'ALL PASS' if not fails else str(fails) + ' FAILURES'}")
    return int(fails)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)

    if a.self_test:
        return self_test()

    cfg = config_from_args(a)
    if a.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0
    if cfg.source_format == "csv":
        if not cfg.csv_path:
            print("error: --csv is required with --source csv.", file=sys.stderr)
            return 2
    else:
        if not a.dat:
            return self_test()
        if not a.curr_cal:
            print("error: --curr-cal is required. It is the only absolute scale "
                  "in the chain once the potentiostat is gone.", file=sys.stderr)
            return 2

    run_pipeline(cfg, stop_after=a.stop_after)  # Ensure '--curr-cal' argument is provided during execution.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
