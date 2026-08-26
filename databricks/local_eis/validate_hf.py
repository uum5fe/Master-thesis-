#!/usr/bin/env python3
"""
validate_hf.py
==============
Does the new high-frequency chain actually work?

Every claim the pipeline makes about accuracy is tested here against
synthetic spectra whose truth is known, WITH the acquisition skew present --
because a comparison without skew flatters the naive estimator and tests
nothing that matters.

The synthetic plate mimics the real one: several segments on one card, each
at a different multiplexer slot, so the skew differs from segment to segment
exactly as the hardware makes it differ.

Run:  python validate_hf.py
"""

from __future__ import annotations

import numpy as np

import silver
import utils
from config import DEFAULT


# ---------------------------------------------------------------------------


def synth_card(rng, n_seg=13, n_ch=16, fs=10_000.0, f_max=3500.0,
               noise=0.04, ref_slot=0, dt0_true=-18e-6, r_hf=0.012,
               tau_hf=3.5e-5):
    """One card: n_seg segments, each at its own conversion slot.

    The spectrum carries TWO relaxations, and the fast one is the point.  A
    PEMFC's high-frequency arc -- contact resistance and the proton transport
    in the cathode catalyst layer -- sits at a few kHz, so with the band
    stopping at 0.35*fs the arc has NOT closed at f_max.  `Re Z` at the top
    point is then the intercept PLUS whatever is left of that arc, and no
    amount of averaging fixes a bias that is in the data by construction.

    A synthetic whose only relaxation is at 8 Hz closes long before 3.5 kHz
    and makes the naive estimator look unbeatable.  It is the wrong test, and
    running it was what first suggested the model was not earning its place.
    Set r_hf = 0 to recover that easier case.
    """
    f = np.logspace(np.log10(0.5), np.log10(f_max), 45)
    w = 2 * np.pi * f
    k_true = 1.0 / (n_ch * fs)          # nominal slot time
    out = []
    for i in range(n_seg):
        slot = i + 1
        Rs = 0.060 + 0.006 * rng.standard_normal()
        Rct = 0.30 + 0.03 * rng.standard_normal()
        tau = 0.02 * (1 + 0.1 * rng.standard_normal())
        R_hf = r_hf * (1 + 0.15 * rng.standard_normal())
        t_hf = tau_hf * (1 + 0.15 * rng.standard_normal())
        L = 2e-7
        Z = (Rs + 1j * w * L + Rct / (1 + 1j * w * tau) ** 0.85
             + R_hf / (1 + 1j * w * t_hf))
        dt = dt0_true + k_true * (slot - ref_slot)       # the real skew
        Z = Z * np.exp(-1j * w * dt)
        Z = Z * (1 + noise * (rng.standard_normal(len(f))
                              + 1j * rng.standard_normal(len(f))))
        out.append(dict(freq=f, Z=Z, Rs=Rs, slot=slot, slot_delta=slot - ref_slot,
                        dt_true=dt, sigma=np.full(len(f), noise)))
    return out, k_true, dt0_true


def _prep(items):
    """Pack the synthetic card the way silver.optimise_skew expects it."""
    return [(it["freq"], it["Z"], it["sigma"], it["slot"], 0) for it in items]


def _cost(items, dt0, k, M=None):
    return silver.skew_cost(_prep(items), dt0, k, M)


def fit_structural(items, k_nom):
    """Calls silver's REAL optimiser, so this tests shipped code."""
    dt0, k, base, best = silver.optimise_skew(_prep(items), k_nom, DEFAULT)
    return dt0, k, (base - best) / base


def fit_per_card(items):
    """Legacy: one free delay for the whole card."""
    prep = _prep(items)
    best, bc = 0.0, np.inf
    for dt in np.linspace(-200e-6, 200e-6, 161):
        c = silver.skew_cost(prep, dt, 0.0, 12, "slot")
        if c < bc:
            bc, best = c, dt
    return best


def report(name, est, true):
    e = np.asarray(est) - np.asarray(true)
    print(f"  {name:34s} bias {np.nanmean(e):+7.2f}  "
          f"rms {np.sqrt(np.nanmean(e ** 2)):6.2f}  "
          f"spread {np.nanmax(est) / np.nanmin(est):5.2f}x")


def main() -> int:
    rng = np.random.default_rng(11)
    utils.banner("high-frequency chain: synthetic validation")

    items, k_true, dt0_true = synth_card(rng)
    k_nom = k_true

    # ---------------- 1. skew model ------------------------------------
    utils.section("1. acquisition skew")
    dt0_s, k_s, gain = fit_structural(items, k_nom)
    dt_card = fit_per_card(items)
    print(f"  truth       : dt0 = {dt0_true*1e6:+.2f} us, "
          f"slot = {k_true*1e6:.3f} us")
    print(f"  structural  : dt0 = {dt0_s*1e6:+.2f} us, "
          f"slot = {k_s*1e6:.3f} us  (ratio {k_s/k_nom:.2f}), "
          f"cost -{100*gain:.0f} %")
    print(f"  per-card    : dt  = {dt_card*1e6:+.2f} us  (one value for all)")

    # Residual skew, split into the part that matters and the part that does
    # not.  A COMMON residual rotates every segment equally and shifts the
    # whole plate; a DIFFERENTIAL residual differs from segment to segment and
    # is what paints false structure onto the map.  Judging a skew model by
    # the total conflates the two and rewards the wrong thing.
    truth = np.array([it["dt_true"] for it in items])
    slot = np.array([it["slot_delta"] for it in items], float)
    models = {
        "no correction":     np.zeros(len(items)),
        "per-card (legacy)": np.full(len(items), dt_card),
        "nominal mux order": k_nom * slot,
        "structural (new)":  dt0_s + k_s * slot,
    }
    f_top = items[0]["freq"].max()
    print(f"\n  residual skew after correction, and the phase it leaves at "
          f"{f_top:.0f} Hz:")
    print(f"    {'model':20s} {'differential':>14s}  {'common':>10s}")
    diffs = {}
    for nm, est in models.items():
        r = truth - est
        r_diff = r - r.mean()                 # segment-to-segment part
        sd = float(np.std(r_diff))
        diffs[nm] = sd
        print(f"    {nm:20s} {sd*1e6:8.2f} us "
              f"({np.degrees(2*np.pi*f_top*sd):5.1f} deg) "
              f"{r.mean()*1e6:+8.2f} us")

    ok_skew = (diffs["structural (new)"] < diffs["per-card (legacy)"]
               and diffs["nominal mux order"] < diffs["per-card (legacy)"])

    # ---------------- 2. Rs estimators ---------------------------------
    utils.section("2. R_ohmic, with the skew present")
    cfg = DEFAULT
    true, naive, naive_dsk, bayes, sd = [], [], [], [], []
    for it in items:
        f, Z = it["freq"], it["Z"]
        true.append(it["Rs"] * 1000)
        naive.append(np.real(Z[np.argmax(f)]) * 1000)
        Zc = utils.apply_delay(f, Z, k_nom * it["slot_delta"])
        naive_dsk.append(np.real(Zc[np.argmax(f)]) * 1000)
        r = silver.bayesian_drt(f, Zc, it["sigma"], cfg)
        Rt, Rt_sd, _ = silver.r_ohmic_topband(f, Zc, it["sigma"])
        arc = r.get("hf_arc_open", 0.0) if r.get("ok") else 0.0
        bayes.append(Rt * 1000)
        sd.append(np.sqrt(Rt_sd ** 2 + arc ** 2) * 1000)

    true = np.array(true)
    print(f"  true spread across the card: {true.max()/true.min():.2f}x")
    report("Re Z at f_max, no de-skew", naive, true)
    report("Re Z at f_max, mux de-skew", naive_dsk, true)
    report("R_ohmic top-band, mux de-skew", bayes, true)

    bayes, sd = np.array(bayes), np.array(sd)
    cov = np.mean(np.abs(bayes - true) <= 2 * sd)
    print(f"  median posterior sd {np.nanmedian(sd):.2f} mOhm cm2, "
          f"95 % coverage {100*cov:.0f} %")

    rms_naive = np.sqrt(np.nanmean((np.array(naive) - true) ** 2))
    rms_bayes = np.sqrt(np.nanmean((bayes - true) ** 2))
    ok_rs = rms_bayes < rms_naive
    ok_cov = 0.80 <= cov <= 1.0

    # ---------------- 3. verdict ---------------------------------------
    utils.section("verdict")
    for nm, ok in (("structural skew beats per-card", ok_skew),
                   ("shipped R_ohmic beats uncorrected", ok_rs),
                   ("posterior interval is honest", ok_cov)):
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}")
    fails = sum(not o for o in (ok_skew, ok_rs, ok_cov))
    print(f"\n  {'ALL PASS' if not fails else f'{fails} FAILURES'}")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
