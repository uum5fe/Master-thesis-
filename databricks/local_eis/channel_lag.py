#!/usr/bin/env python3
"""
channel_lag.py
==============
Every segment's current-measurement chain, checked against the plate on every
run: does it see the same high-frequency phase as the others, and if it lags,
by how much?  Removes the lag before R_ohmic is read.

WHY THIS IS A PIPELINE STAGE
----------------------------
R_ohmic is read as Re Z at the top of the kept band. On 2612030 that band
ends near 1.2 kHz, and some segment chains (shunt L/R, traces, amplifier)
lag the cell voltage by a first-order time constant of up to ~110 us -- 37 deg
there, and a Re Z ~20 mOhm*cm2 low. Neighbouring segments 14 and 19 read 66.5
and 44.0 mOhm*cm2 for that reason alone. The lag was the same at 45 A and
60 A to within a few us (correlation 1.00 over all segments), i.e. it belongs
to the channel. A per-campaign table would cover one plate and one wiring; a
stage that measures it on every run covers whatever is measured next.

WHAT IT DOES
------------
  1. Reference: the complex plate median per frequency (real and imaginary
     parts separately), from every segment's usable points.
  2. Per segment, over f_lo..f_max: fit  arg(Z_seg / Z_ref) = -atan(w tau).
  3. Accept tau only if the data look like that model:
       * at least `channel_lag_min_points` points,
       * phase residual <= `channel_lag_max_resid_deg` (a misaligned card or a
         noise-dominated top band does NOT look like a first-order lag, and is
         reported rather than "corrected"),
       * |tau| <= `channel_lag_max_us`;
     and correct only if |tau| >= `channel_lag_min_us`.
  4. Correct:  Z <- Z * (1 + j w tau), i.e. divide by H = 1/(1 + j w tau).

cfg.channel_lag = "off" | "report" | "correct" (default "correct").

WHAT IT CANNOT DO
-----------------
tau is relative to the plate median, so a lag that EVERY channel shares stays
in; that is the ex-situ chain response's job (cfg.gain_file, applied in
bronze before this stage, so this then sees only what the file did not
remove). A genuine local high-frequency difference moves the phase by a
degree or two at most and survives the correction to within ~1 mOhm*cm2
(see test_channel_lag_stage.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("off", "report", "correct")

#: statuses written per segment
CORRECTED = "corrected"
SMALL = "small"
REPORT_ONLY = "report_only"
NOT_FIRST_ORDER = "not_first_order"
TOO_LARGE = "too_large"
TOO_FEW = "too_few_points"

#: statuses that mean the high-frequency phase of the channel is not to be
#: trusted (the segment is demoted and the plausibility report says so)
UNRELIABLE = (NOT_FIRST_ORDER, TOO_LARGE)


@dataclass
class ChannelLag:
    segment: str
    tau_s: float = float("nan")      # fitted lag; nan if not estimable
    applied_s: float = 0.0           # what was removed from Z
    n_points: int = 0
    resid_deg: float = float("nan")  # rms phase residual of the fit
    dphi_top_deg: float = float("nan")   # phase vs plate at the top point
    f_top_hz: float = float("nan")
    status: str = ""

    def row(self, card: str = "") -> dict:
        def r(x, k=3):
            return round(float(x), k) if np.isfinite(x) else ""
        return {"segment": self.segment, "card": card,
                "tau_us": r(1e6 * self.tau_s, 2),
                "applied_us": r(1e6 * self.applied_s, 2),
                "resid_deg": r(self.resid_deg, 2),
                "dphi_top_deg": r(self.dphi_top_deg, 2),
                "f_top_hz": r(self.f_top_hz, 1),
                "n_points": self.n_points, "status": self.status}


def _opt(cfg, name: str, default):
    return getattr(cfg, name, default) if cfg is not None else default


def mode(cfg) -> str:
    m = str(_opt(cfg, "channel_lag", "correct") or "off").lower()
    return m if m in MODES else "correct"


def _key(f: float) -> float:
    return float(f"{float(f):.6g}")


def correction(freq, tau_s: float) -> np.ndarray:
    """The factor that removes a first-order lag: Z_true = Z_meas * this."""
    return 1.0 + 1j * 2 * np.pi * np.asarray(freq, float) * float(tau_s)


def plate_reference(items: dict) -> dict[float, complex]:
    """Complex plate median per frequency, from the usable points only.

    A frequency enters only if at least max(5, 30 %) of the segments carry a
    usable point there: a median of three is not a plate reference.
    """
    by_f: dict[float, list[complex]] = {}
    for freq, Z, usable in items.values():
        for f, z, u in zip(freq, Z, usable):
            if u and np.isfinite(z):
                by_f.setdefault(_key(f), []).append(complex(z))
    need = max(5, int(np.ceil(0.3 * len(items))))
    return {f: complex(np.median(np.real(v)), np.median(np.imag(v)))
            for f, v in by_f.items() if len(v) >= need}


def fit_tau(freq: np.ndarray, phase: np.ndarray, tau_max: float
            ) -> tuple[float, float]:
    """(tau, rms residual in deg) for phase = -atan(w tau), |tau| <= tau_max."""
    w = 2 * np.pi * np.asarray(freq, float)
    ph = np.asarray(phase, float)

    def cost(t):
        return float(np.sum((ph + np.arctan(w * t)) ** 2))

    try:
        from scipy.optimize import minimize_scalar
        r = minimize_scalar(cost, bounds=(-tau_max, tau_max), method="bounded",
                            options={"xatol": 1e-8})
        tau = float(r.x)
    except Exception:                                       # noqa: BLE001
        grid = np.linspace(-tau_max, tau_max, 4001)
        tau = float(grid[np.argmin([cost(t) for t in grid])])
    resid = np.degrees(np.sqrt(cost(tau) / max(len(ph), 1)))
    return tau, float(resid)


def estimate(items: dict, cfg=None, log=None) -> dict[str, ChannelLag]:
    """Lag per segment.  `items`: segment -> (freq, Z, usable mask)."""
    m = mode(cfg)
    if m == "off" or not items:
        return {}
    f_lo = float(_opt(cfg, "channel_lag_f_lo_hz", 50.0))
    f_hi = float(_opt(cfg, "f_max_hz", np.inf))
    n_min = int(_opt(cfg, "channel_lag_min_points", 4))
    t_min = 1e-6 * float(_opt(cfg, "channel_lag_min_us", 5.0))
    t_max = 1e-6 * float(_opt(cfg, "channel_lag_max_us", 250.0))
    r_max = float(_opt(cfg, "channel_lag_max_resid_deg", 10.0))

    band = {}
    for seg, (freq, Z, usable) in items.items():
        freq = np.asarray(freq, float)
        Z = np.asarray(Z, complex)
        u = (np.asarray(usable, bool) & np.isfinite(freq) & np.isfinite(Z)
             & (freq >= f_lo) & (freq <= f_hi))
        band[seg] = (freq, Z, u)
    ref = plate_reference(band)

    out: dict[str, ChannelLag] = {}
    for seg, (freq, Z, u) in band.items():
        lag = ChannelLag(segment=str(seg))
        idx = [i for i in np.flatnonzero(u) if _key(freq[i]) in ref]
        lag.n_points = len(idx)
        if len(idx) < n_min:
            lag.status = TOO_FEW
            out[str(seg)] = lag
            continue
        f = freq[idx]
        r = np.array([ref[_key(x)] for x in f])
        ph = np.angle(Z[idx] / r)
        # 1.2 x the limit as the search bound, so "at the limit" is visible
        tau, resid = fit_tau(f, ph, 1.2 * t_max)
        top = int(np.argmax(f))
        lag.tau_s, lag.resid_deg = tau, resid
        lag.f_top_hz = float(f[top])
        lag.dphi_top_deg = float(np.degrees(ph[top]))
        if resid > r_max:
            lag.status = NOT_FIRST_ORDER
        elif abs(tau) > t_max:
            lag.status = TOO_LARGE
        elif abs(tau) < t_min:
            lag.status = SMALL
        elif m == "correct":
            lag.status, lag.applied_s = CORRECTED, tau
        else:
            lag.status = REPORT_ONLY
        out[str(seg)] = lag

    if log is not None:
        _log_summary(out, m, f_lo, log)
    return out


def _log_summary(out: dict[str, ChannelLag], m: str, f_lo: float, log) -> None:
    taus = np.array([1e6 * g.tau_s for g in out.values()
                     if np.isfinite(g.tau_s)])
    count = {}
    for g in out.values():
        count[g.status] = count.get(g.status, 0) + 1
    if taus.size:
        log.info(f"  plate-median reference from {f_lo:g} Hz; tau over "
                 f"{taus.size} segments: median {np.median(taus):+.1f} us, "
                 f"range {taus.min():+.0f}..{taus.max():+.0f} us")
    log.info("  " + ", ".join(f"{k}: {v}" for k, v in sorted(count.items())))
    big = sorted((g for g in out.values() if g.status == CORRECTED),
                 key=lambda g: -abs(g.applied_s))[:6]
    if big:
        log.info("  largest removed: " + ", ".join(
            f"{g.segment} {1e6 * g.applied_s:+.0f} us" for g in big))
    bad = [g.segment for g in out.values() if g.status in UNRELIABLE]
    if bad:
        log.warning(f"  phase NOT consistent with a first-order lag on "
                    f"{len(bad)} segment(s): {', '.join(sorted(bad, key=int))}"
                    f" -- left uncorrected and demoted; a card that is not "
                    f"time-aligned looks exactly like this")
    if m == "report" and big == [] and taus.size:
        log.info("  mode 'report': nothing removed")


def card_summary(lags: dict[str, ChannelLag], card_of: dict[str, str]
                 ) -> dict[str, dict]:
    """Per card: how many segments, how many unreliable, median tau."""
    out: dict[str, dict] = {}
    for s, g in lags.items():
        c = card_of.get(s, "")
        d = out.setdefault(c, {"n": 0, "unreliable": 0, "taus": []})
        d["n"] += 1
        d["unreliable"] += int(g.status in UNRELIABLE)
        if np.isfinite(g.tau_s):
            d["taus"].append(1e6 * g.tau_s)
    for d in out.values():
        d["median_tau_us"] = float(np.median(d["taus"])) if d["taus"] else np.nan
        del d["taus"]
    return out
