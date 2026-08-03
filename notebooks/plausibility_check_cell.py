# Databricks notebook source
# DBTITLE 1,Plausibility Check - Per-Segment Verdicts + Parallel Impedance vs Gamry
# ══════════════════════════════════════════════════════════════════════════════
# PLAUSIBILITY CHECK - Per-Segment Verdicts + Parallel Impedance vs Gamry
#
# Each segment goes through independent checks collecting OK / WARN / FAIL.
# Nothing here fits an ECM - this is the sanity gate you run *before* trusting
# any impedance result.
#
# Reads from the notebook globals: TONE_ONLY_ALL, REJECTED_ALL, GAMRY_DATA,
# CONDITIONS, LEEPA, SEG_AREA_CM2, CELL_AREA_CM2, ALL_SEG_COORDS, SEG_TO_CARD,
# F_MIN, F_MAX.
#
# Checks
# ==================  ========================================================
# rejected_upstream   segment never reached the results dict at all
# rs_sign             Rs must be positive
# rs_magnitude        Rs outside a physically plausible band [mOhm*cm2]
# arc_physical        the arc must be capacitive below its own -Z'' peak
# frequency_coverage  too few tones -> unreliable spectrum
# hf_lf_consistency   Z_LF/Rs far from what the rest of the plate shows
# coherence           median gamma^2 too low to trust the spectrum
# kramers_kronig      KK residual from process_condition
# sign_flipped        this card's sign was decided by a majority vote
# uniformity          Rs/median outside the range a healthy cell shows
# outlier             robust median/MAD outlier against the population
# neighbour           segment far from its spatial neighbours
# card_bias           a step at a card boundary after removing the plate trend
# parallel_sum        sum(1/Z_seg), area-scaled, vs the Gamry cell measurement
# ==================  ========================================================
#
# Verdicts are split: plate-level findings (card bias, parallel sum) are
# reported once at plate level instead of being stamped onto every segment,
# which would turn whole cards red and bury the individually faulty ones.
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
from dataclasses import dataclass, field as dc_field

OK, WARN, FAIL = 'OK', 'WARN', 'FAIL'
_RANK = {OK: 0, WARN: 1, FAIL: 2, 'SKIPPED': 0}

# Segments with a confirmed hardware fault.  They are *checked anyway* and
# merely annotated: excluding them up front means the report can never tell you
# whether the exclusion is still justified, or that another one has gone bad.
KNOWN_FAULT_SEGMENTS = {
    33: 'hardware error during data acquisition (confirmed)',
    59: 'hardware error during data acquisition (confirmed)',
}
# Segments to drop from the parallel sum only (a faulty admittance would
# corrupt the whole-plate number even though the segment itself is reported).
EXCLUDE_FROM_PARALLEL = set(KNOWN_FAULT_SEGMENTS)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
@dataclass
class Limits:
    """Every threshold the verdicts depend on."""
    # Rs area-specific resistance [mOhm*cm2]
    rs_asr_min: float = 1.0
    rs_asr_max: float = 500.0
    rs_asr_warn_min: float = 5.0
    rs_asr_warn_max: float = 300.0

    # uniformity: Rs/median ratio (median, not mean - the mean is dragged by
    # exactly the segments this check is hunting)
    uniformity_warn: tuple = (0.40, 2.00)
    uniformity_fail: tuple = (0.20, 3.50)

    # robust outlier score (|Rs - median| / (1.4826 * MAD))
    outlier_warn: float = 3.5
    outlier_fail: float = 6.0
    # floor on the robust scale as a fraction of the median, so a very uniform
    # plate does not score every segment hundreds of "sigma" from the middle
    outlier_mad_floor: float = 0.01

    # spatial neighbour deviation (relative)
    neighbour_warn: float = 0.40
    neighbour_fail: float = 0.80
    neighbour_radius_cm: float = 4.0

    # card step after removing the plate-wide spatial trend (relative)
    card_bias_warn: float = 0.10
    card_bias_fail: float = 0.25

    # minimum number of frequency points
    n_pts_warn: int = 8
    n_pts_fail: int = 4

    # Z_LF/Rs relative to the population median of the same ratio
    hf_lf_ratio_warn: float = 2.5
    hf_lf_ratio_fail: float = 5.0

    # arc physicality: fraction of -Z'' > 0 *below the -Z'' peak*
    arc_positive_frac_warn: float = 0.75
    arc_positive_frac_fail: float = 0.40

    # coherence and Kramers-Kronig, when the results carry them
    coherence_warn: float = 0.90
    coherence_fail: float = 0.70
    kk_warn: float = 5.0
    kk_fail: float = 15.0

    # Rs/Rct split independence: observed corr(Rs, Rs+Rct) divided by the
    # sigma_Rs/sigma_total it would have to equal if the two were independent.
    # 1.0 is a clean split; near 0 means only the dividing point is moving.
    split_independence_warn: float = 0.60
    split_independence_fail: float = 0.35

    # parallel sum vs Gamry (relative deviation)
    par_sum_warn: float = 0.15
    par_sum_fail: float = 0.35


# ---------------------------------------------------------------------------
# Per-segment record
# ---------------------------------------------------------------------------
@dataclass
class SegmentCheck:
    segment: int
    card: int
    condition: str
    n_pts: int = 0

    Rs_mOhm: float = np.nan
    Rs_asr: float = np.nan          # mOhm*cm2
    Rs_is_measured: bool = False    # arc actually crossed Im(Z) = 0
    Rct_asr: float = np.nan
    Z_lf_real_asr: float = np.nan
    arc_pos_frac: float = np.nan
    hf_lf_ratio: float = np.nan
    median_coh: float = np.nan
    kk_pct: float = np.nan
    flipped: bool = False

    uniformity: float = np.nan
    outlier_score: float = np.nan
    neighbour_rel: float = np.nan

    verdict: str = OK
    flags: list = dc_field(default_factory=list)

    def raise_to(self, level: str, message: str) -> None:
        self.flags.append(f"{level}:{message}")
        if _RANK[level] > _RANK[self.verdict]:
            self.verdict = level


# ---------------------------------------------------------------------------
# Spectrum helpers
# ---------------------------------------------------------------------------
def rs_from_spectrum(f, Z, n_hf=6):
    """Rs [Ohm] and whether it is a measurement or only an upper bound.

    Rs is where the arc crosses Im(Z) = 0.  Reading Re(Z) at the top tone
    instead over-reads it: at a few kHz the arc has not collapsed yet, which on
    this hardware is about +8%.  When the arc does cross in band the crossing is
    interpolated and Rs is a measurement; when it does not, the smallest
    measured Re(Z) is the honest upper bound and recovering Rs itself needs a
    model.  The returned flag says which case you are in.
    """
    order = np.argsort(f)
    f = np.asarray(f, float)[order]
    Z = np.asarray(Z, complex)[order]
    if len(f) < 3:
        return np.nan, False

    take = int(min(max(n_hf, 3), len(f)))
    re_hf, im_hf = Z.real[-take:], Z.imag[-take:]
    crossings = np.where(np.diff(np.sign(im_hf)) != 0)[0]
    if crossings.size:
        i = int(crossings[-1])
        y0, y1 = im_hf[i], im_hf[i + 1]
        if y1 != y0:
            t = -y0 / (y1 - y0)
            return float(re_hf[i] + t * (re_hf[i + 1] - re_hf[i])), True
    return float(np.min(Z.real)), False


def arc_positive_fraction(f, Z):
    """Fraction of -Z'' > 0 *below* the -Z'' peak.

    Restricted to the capacitive side on purpose: above the peak a real cell
    legitimately turns inductive from the cabling, and counting those points as
    faults penalises physically correct spectra.  Below the peak, points on the
    wrong side of the real axis mean a sign error or an uncorrected delay.
    """
    order = np.argsort(f)
    Z = np.asarray(Z, complex)[order]
    neg_im = -Z.imag
    if len(neg_im) < 3:
        return np.nan
    peak = int(np.argmax(neg_im))
    below = neg_im[:peak + 1]
    if below.size == 0:
        return 0.0
    return float(np.mean(below > 0))


# ---------------------------------------------------------------------------
# Check engine
# ---------------------------------------------------------------------------
def run_checks(results, cond, limits, rejected=None, seg_area_cm2=None,
               seg_to_card=None, seg_coords=None):
    """Run all plausibility checks on one condition. Returns (checks, summary)."""
    seg_to_card = seg_to_card or {}
    seg_coords = seg_coords or {}
    checks, plate_flags = {}, []

    for seg_str, d in results.items():
        seg = int(seg_str)
        card = d.get('card', seg_to_card.get(seg, -1))
        f = np.asarray(d['f'], float)
        Z = np.asarray(d['Z'], complex)
        c = SegmentCheck(segment=seg, card=card, condition=cond, n_pts=len(f))

        rs_ohm, measured = rs_from_spectrum(f, Z)
        c.Rs_mOhm = rs_ohm * 1000
        c.Rs_asr = c.Rs_mOhm * seg_area_cm2
        c.Rs_is_measured = measured

        order = np.argsort(f)
        n_lf = max(3, len(f) // 5)
        c.Z_lf_real_asr = float(np.median(Z[order][:n_lf].real)) * 1000 * seg_area_cm2
        c.Rct_asr = c.Z_lf_real_asr - c.Rs_asr
        c.arc_pos_frac = arc_positive_fraction(f, Z)
        if c.Rs_asr > 0:
            c.hf_lf_ratio = c.Z_lf_real_asr / c.Rs_asr

        # process_condition writes 'coh'; the Gold-table loader writes
        # 'coherence' - accept either rather than silently skipping the check.
        coh = d.get('coh', d.get('coherence'))
        if coh is not None and np.size(coh):
            c.median_coh = float(np.median(np.asarray(coh, float)))
        if d.get('kk_pct') is not None:
            c.kk_pct = float(d['kk_pct'])
        c.flipped = bool(d.get('flipped', False))

        checks[seg] = c

    # Segments the pipeline threw away never appear in `results`, so without
    # this a condition that lost 30 segments upstream still prints
    # "every segment passed".
    for seg, reason in (rejected or {}).items():
        seg = int(seg)
        if seg in checks:
            continue
        c = SegmentCheck(segment=seg, card=seg_to_card.get(seg, -1),
                         condition=cond, n_pts=0)
        c.raise_to(FAIL, f"rejected upstream: {reason}")
        checks[seg] = c

    if not checks:
        return checks, {'note': 'no segments at all'}

    rs_values = np.array([c.Rs_asr for c in checks.values() if np.isfinite(c.Rs_asr)])
    if rs_values.size == 0:
        return checks, {'note': 'no finite Rs values'}

    median = float(np.median(rs_values))
    mad = float(np.median(np.abs(rs_values - median))) * 1.4826
    mad_eff = max(mad, limits.outlier_mad_floor * abs(median))
    ratios = np.array([c.hf_lf_ratio for c in checks.values()
                       if np.isfinite(c.hf_lf_ratio)])
    ratio_median = float(np.median(ratios)) if ratios.size else np.nan

    summary = {
        'n_segments': len(rs_values),
        'n_records': len(checks),
        'Rs_asr_median': median,
        'Rs_asr_mean': float(np.mean(rs_values)),
        'Rs_asr_std': float(np.std(rs_values)),
        'Rs_asr_mad': mad,
        'Rs_asr_min': float(rs_values.min()),
        'Rs_asr_max': float(rs_values.max()),
        'hf_lf_ratio_median': ratio_median,
        'n_rs_measured': int(sum(c.Rs_is_measured for c in checks.values())),
    }

    # --- 1. per-segment, independent of the population --------------------
    for c in checks.values():
        if c.n_pts == 0:
            continue                     # already failed as rejected upstream

        if np.isfinite(c.Rs_mOhm) and c.Rs_mOhm < 0:
            c.raise_to(FAIL, f"negative Rs={c.Rs_asr:.1f} mOhm*cm2 (phase cal error?)")
        if not np.isfinite(c.Rs_asr):
            c.raise_to(FAIL, "no finite Rs (calibration or channel missing)")
            continue
        if not (limits.rs_asr_min <= c.Rs_asr <= limits.rs_asr_max):
            c.raise_to(FAIL, f"Rs={c.Rs_asr:.1f} mOhm*cm2 outside "
                             f"[{limits.rs_asr_min}, {limits.rs_asr_max}]")
        elif not (limits.rs_asr_warn_min <= c.Rs_asr <= limits.rs_asr_warn_max):
            c.raise_to(WARN, f"Rs={c.Rs_asr:.1f} mOhm*cm2 near plausible edge")

        if c.n_pts < limits.n_pts_fail:
            c.raise_to(FAIL, f"only {c.n_pts} frequency points")
        elif c.n_pts < limits.n_pts_warn:
            c.raise_to(WARN, f"only {c.n_pts} frequency points (spectrum sparse)")

        if np.isfinite(c.arc_pos_frac):
            if c.arc_pos_frac < limits.arc_positive_frac_fail:
                c.raise_to(FAIL, f"only {c.arc_pos_frac:.0%} of -Z'' > 0 below the "
                                 f"arc peak (no capacitive arc / sign error)")
            elif c.arc_pos_frac < limits.arc_positive_frac_warn:
                c.raise_to(WARN, f"{c.arc_pos_frac:.0%} of -Z'' > 0 below the arc "
                                 f"peak (distorted arc)")

        # Judged against what the rest of the plate shows, not an absolute
        # number: Rct/Rs is legitimately 10-20 at low current, so a fixed
        # threshold only tells you which operating point you are at.
        if np.isfinite(c.hf_lf_ratio) and np.isfinite(ratio_median) and ratio_median > 0:
            rel = c.hf_lf_ratio / ratio_median
            if rel > limits.hf_lf_ratio_fail or rel < 1.0 / limits.hf_lf_ratio_fail:
                c.raise_to(FAIL, f"Z_LF/Rs = {c.hf_lf_ratio:.1f} is {rel:.1f}x the "
                                 f"plate median ({ratio_median:.1f})")
            elif rel > limits.hf_lf_ratio_warn or rel < 1.0 / limits.hf_lf_ratio_warn:
                c.raise_to(WARN, f"Z_LF/Rs = {c.hf_lf_ratio:.1f} vs plate median "
                                 f"{ratio_median:.1f}")

        if np.isfinite(c.median_coh):
            if c.median_coh < limits.coherence_fail:
                c.raise_to(FAIL, f"median coherence {c.median_coh:.2f}")
            elif c.median_coh < limits.coherence_warn:
                c.raise_to(WARN, f"median coherence {c.median_coh:.2f}")

        if np.isfinite(c.kk_pct):
            if c.kk_pct > limits.kk_fail:
                c.raise_to(FAIL, f"Kramers-Kronig residual {c.kk_pct:.1f}%")
            elif c.kk_pct > limits.kk_warn:
                c.raise_to(WARN, f"Kramers-Kronig residual {c.kk_pct:.1f}%")

        if c.flipped:
            c.raise_to(WARN, "sign of this card was flipped by majority vote - "
                             "confirm the wiring polarity instead of inferring it")

        if not c.Rs_is_measured:
            c.flags.append("INFO:arc does not cross Im(Z)=0 in band, so Rs is an "
                           "upper bound from min(Re Z), not a measured intercept")

    # --- 2. population-relative -------------------------------------------
    for c in checks.values():
        if not np.isfinite(c.Rs_asr):
            continue
        c.uniformity = c.Rs_asr / median if median > 0 else np.nan
        c.outlier_score = abs(c.Rs_asr - median) / mad_eff if mad_eff > 0 else 0.0

        lo_f, hi_f = limits.uniformity_fail
        lo_w, hi_w = limits.uniformity_warn
        if not (lo_f <= c.uniformity <= hi_f):
            c.raise_to(FAIL, f"Rs/median = {c.uniformity:.2f}, outside [{lo_f}, {hi_f}]")
        elif not (lo_w <= c.uniformity <= hi_w):
            c.raise_to(WARN, f"Rs/median = {c.uniformity:.2f}")

        if c.outlier_score > limits.outlier_fail:
            c.raise_to(FAIL, f"robust outlier: {c.outlier_score:.1f} MAD from median")
        elif c.outlier_score > limits.outlier_warn:
            c.raise_to(WARN, f"{c.outlier_score:.1f} MAD from median")

    # --- 3. spatial neighbours --------------------------------------------
    if seg_coords:
        for c in checks.values():
            if c.segment not in seg_coords or not np.isfinite(c.Rs_asr):
                continue
            x, y = seg_coords[c.segment][:2]
            neighbours = [
                o.Rs_asr for o in checks.values()
                if o.segment != c.segment and o.segment in seg_coords
                and np.isfinite(o.Rs_asr)
                and np.hypot(seg_coords[o.segment][0] - x,
                             seg_coords[o.segment][1] - y) <= limits.neighbour_radius_cm
            ]
            if len(neighbours) < 2:
                continue
            local = float(np.median(neighbours))
            if abs(local) < 1e-12:
                continue
            c.neighbour_rel = abs(c.Rs_asr - local) / abs(local)
            if c.neighbour_rel > limits.neighbour_fail:
                c.raise_to(FAIL, f"{c.neighbour_rel:.0%} from neighbours "
                                 f"(local median {local:.1f} mOhm*cm2)")
            elif c.neighbour_rel > limits.neighbour_warn:
                c.raise_to(WARN, f"{c.neighbour_rel:.0%} from neighbours")
    else:
        summary['neighbour_check'] = 'skipped: no segment coordinates'

    # --- 4. per-card bias, after removing the plate-wide trend -------------
    # Cards map to contiguous regions of the plate, and Rs really does vary
    # across a cell.  Comparing a card's raw mean against the plate mean
    # therefore flags a genuine gradient as a fault.  What distinguishes a
    # calibration or timing error is a *step* at the card boundary, so the test
    # runs on the residual left after a smooth spatial trend is removed.
    keys, positions, values = [], [], []
    for c in checks.values():
        if not np.isfinite(c.Rs_asr):
            continue
        if seg_coords and c.segment in seg_coords:
            x, y = seg_coords[c.segment][:2]
        else:
            x, y = float(c.segment), 0.0
        keys.append(c.segment); positions.append((x, y)); values.append(c.Rs_asr)

    detrended, trend_removed = {}, False
    if len(values) >= 6:
        A = np.column_stack([np.ones(len(positions)),
                             [p[0] for p in positions],
                             [p[1] for p in positions]])
        try:
            coef, *_ = np.linalg.lstsq(A, np.asarray(values), rcond=None)
            detrended = dict(zip(keys, np.asarray(values) - A @ coef))
            trend_removed = True
        except np.linalg.LinAlgError:
            pass
    if not trend_removed:
        detrended = {k: v - median for k, v in zip(keys, values)}
    summary['card_bias_detrended'] = trend_removed

    card_medians, card_verdicts = {}, {}
    for card in sorted({c.card for c in checks.values()}):
        vals = [c.Rs_asr for c in checks.values()
                if c.card == card and np.isfinite(c.Rs_asr)]
        res = [detrended[c.segment] for c in checks.values()
               if c.card == card and c.segment in detrended]
        if not vals or not res:
            continue
        card_medians[card] = float(np.median(vals))
        step = float(np.median(res))
        deviation = abs(step) / abs(median) if median else np.nan
        if deviation > limits.card_bias_fail:
            level, note = FAIL, 'a card-level timing or calibration error, not physics'
        elif deviation > limits.card_bias_warn:
            level, note = WARN, 'check this card delay table and calibration'
        else:
            card_verdicts[card] = OK
            continue
        card_verdicts[card] = level
        plate_flags.append((level, f"card {card} steps {deviation:.0%} of the plate "
                                   f"median Rs away from the spatial trend ({note})"))
        # Context for its segments, never proof that any one of them is broken.
        for c in checks.values():
            if c.card == card:
                c.raise_to(WARN, f"card {card} steps {deviation:.0%} off the plate trend")
    summary['card_medians'] = card_medians
    summary['card_verdicts'] = card_verdicts

    # --- 4b. is the Rs/Rct split stable? ----------------------------------
    # Rs and Rct are physically independent, so if they were both measured
    # cleanly, corr(Rs, Rs+Rct) would come out at exactly sigma_Rs/sigma_total.
    # Coming out far below that means Rct moves to cancel Rs: the *total* low
    # frequency impedance is stable and only the point where it gets divided is
    # wandering.  That is a high-frequency problem - a residual delay, or the
    # top tones being too noisy to place the intercept - and it invalidates
    # every per-segment Rs verdict below, which is why it belongs at plate
    # level rather than as 50 individual flags.
    pairs = [(c.Rs_asr, c.Rct_asr) for c in checks.values()
             if np.isfinite(c.Rs_asr) and np.isfinite(c.Rct_asr)]
    if len(pairs) >= 8:
        rs_arr = np.array([p[0] for p in pairs])
        tot_arr = rs_arr + np.array([p[1] for p in pairs])
        if rs_arr.std() > 0 and tot_arr.std() > 0:
            expected = rs_arr.std() / tot_arr.std()
            observed = float(np.corrcoef(rs_arr, tot_arr)[0, 1])
            ratio = observed / expected if expected > 0 else np.nan
            summary['split_independence'] = ratio
            summary['rs_span'] = float(rs_arr.max() / max(rs_arr.min(), 1e-9))
            summary['total_span'] = float(
                np.percentile(tot_arr, 90) / max(np.percentile(tot_arr, 10), 1e-9))
            if np.isfinite(ratio) and ratio < limits.split_independence_fail:
                plate_flags.append((FAIL, (
                    f"the Rs/Rct split is not independent (corr(Rs,Rs+Rct)="
                    f"{observed:.2f} against {expected:.2f} expected, ratio "
                    f"{ratio:.2f}): Rs spans {summary['rs_span']:.1f}x while the "
                    f"total spans only {summary['total_span']:.1f}x. The total is "
                    f"stable and the intercept is wandering - a high-frequency "
                    f"problem, not 50 different membranes. Treat the per-segment "
                    f"Rs verdicts below as unreliable until it is resolved.")))
            elif np.isfinite(ratio) and ratio < limits.split_independence_warn:
                plate_flags.append((WARN, (
                    f"the Rs/Rct split looks only partly independent (ratio "
                    f"{ratio:.2f}); check the high-frequency end before quoting "
                    f"per-segment Rs")))

    # --- 4c. is the card mapping actually populated? -----------------------
    distinct = {c.card for c in checks.values()}
    if len(distinct) < 2:
        plate_flags.append((WARN, (
            f"every segment reports card {sorted(distinct)[0]}, so the card-bias "
            f"check could not run. SEG_TO_CARD is usually keyed by the FAMOS "
            f"channel *name* (a string) while the lookup uses int(segment) - "
            f"rebuild it as {{int(seg): card}} to enable the check.")))

    # --- 5. annotate the known hardware faults ----------------------------
    for seg, reason in KNOWN_FAULT_SEGMENTS.items():
        c = checks.get(seg)
        if c is None:
            continue
        if c.verdict == OK:
            c.raise_to(WARN, f"listed as a known hardware fault ({reason}) but "
                             f"passes every check - is the exclusion still justified?")
        else:
            c.flags.append(f"INFO:known hardware fault - {reason}")

    summary['plate_flags'] = plate_flags
    summary['plate_verdict'] = max([lv for lv, _ in plate_flags] or [OK],
                                   key=lambda v: _RANK[v])
    return checks, summary


# ---------------------------------------------------------------------------
# Parallel impedance
# ---------------------------------------------------------------------------
def parallel_impedance(results, seg_area_cm2, cell_area_cm2, exclude=frozenset()):
    """Recombine the segments as parallel conductances and scale to the cell.

    Two things the obvious implementation gets wrong:

    * Interpolating onto one segment's grid and summing with ``np.nansum``
      makes a segment that has no data at a frequency contribute *zero
      conductance* - i.e. it is silently treated as an open circuit while still
      being counted.  Here the sum runs on the exact intersection of the tone
      sets, so every segment contributes at every frequency.
    * Scaling by a segment *count* is only right when all segments have the
      same area.  Scaling by area is right either way::

          Z_cell(f) = Z_par(f) * A_measured / A_cell
    """
    usable = {int(s): d for s, d in results.items()
              if int(s) not in exclude and np.size(d['f']) >= 3}
    if len(usable) < 3:
        return None

    def key(x):
        return np.round(np.asarray(x, float), 6)

    common = None
    for d in usable.values():
        ks = set(key(d['f']).tolist())
        common = ks if common is None else (common & ks)
    if not common or len(common) < 3:
        return None
    f_common = np.array(sorted(common), float)
    lookup_keys = np.round(f_common, 6).tolist()

    Y = np.zeros(len(f_common), dtype=complex)
    for d in usable.values():
        table = dict(zip(key(d['f']).tolist(), np.asarray(d['Z'], complex)))
        Zk = np.array([table[v] for v in lookup_keys])
        Y += 1.0 / Zk

    Z_par = 1.0 / Y
    good = np.isfinite(Z_par.real) & np.isfinite(Z_par.imag)
    f_par, Z_par = f_common[good], Z_par[good]
    if len(f_par) < 3:
        return None

    n_segs = len(usable)
    area_measured = n_segs * seg_area_cm2
    scale = area_measured / cell_area_cm2          # -> whole-cell impedance
    Z_cell = Z_par * scale

    rs_ohm, measured = rs_from_spectrum(f_par, Z_cell)
    n_dropped = max(len(set().union(*[set(key(d['f']).tolist())
                                      for d in usable.values()])) - len(f_common), 0)
    return {
        'f': f_par, 'Z_par': Z_par, 'Z_cell': Z_cell,
        'Rs_cell_mOhm': rs_ohm * 1000, 'Rs_is_measured': measured,
        'n_segs': n_segs, 'area_measured_cm2': area_measured,
        'area_scale': scale, 'n_freq_common': len(f_par),
        'n_freq_dropped': n_dropped,
    }


def gamry_rs_mohm(gamry, f_min, f_max, f_ref_hz=None):
    """Gamry ohmic resistance [mOhm], read the same way as the plate's."""
    if not gamry or gamry.get('n_pts', 0) <= 5:
        return np.nan, False
    f = np.asarray(gamry['freq'], float)
    Z = np.asarray(gamry['zreal'], float) + 1j * np.asarray(gamry['zimag'], float)
    # Compare like with like: restrict Gamry to the band the plate actually
    # covers, otherwise Gamry's higher top frequency gives it a less biased
    # intercept than the plate's and the ratio inherits the difference.
    hi = f_max if f_ref_hz is None else min(f_max, f_ref_hz)
    m = (f >= f_min) & (f <= hi)
    if m.sum() < 3:
        return np.nan, False
    rs_ohm, measured = rs_from_spectrum(f[m], Z[m])
    return rs_ohm * 1000, measured


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_segment_report(checks, summary, cond, leepa):
    line = '=' * 104
    print(f"\n{line}")
    print(f"  SEGMENT PLAUSIBILITY - Leepa {leepa} / {cond}")
    print(line)

    if 'n_segments' not in summary:
        print(f"  {summary.get('note', 'nothing to report')}")
        return

    print(f"  records            : {summary['n_records']} "
          f"({summary['n_segments']} with a finite Rs)")
    print(f"  Rs ASR median      : {summary['Rs_asr_median']:.1f} mOhm*cm2")
    print(f"  Rs ASR mean+/-std  : {summary['Rs_asr_mean']:.1f} +/- "
          f"{summary['Rs_asr_std']:.1f}  (robust sigma {summary['Rs_asr_mad']:.1f})")
    print(f"  Rs ASR range       : {summary['Rs_asr_min']:.1f} .. "
          f"{summary['Rs_asr_max']:.1f} mOhm*cm2")
    print(f"  Rs from a measured intercept: {summary['n_rs_measured']}/"
          f"{summary['n_records']} segments"
          + ("" if summary['n_rs_measured'] else
             "  <- the arc never crosses Im=0 in band, so every Rs here is an "
             "upper bound"))

    cm, cv = summary.get('card_medians', {}), summary.get('card_verdicts', {})
    if cm:
        basis = ('step away from the fitted spatial trend'
                 if summary.get('card_bias_detrended') else 'offset from the median')
        print(f"\n  per-card median Rs [mOhm*cm2]  (judged on the {basis}):")
        for card, value in sorted(cm.items()):
            dev = value / summary['Rs_asr_median'] - 1.0
            print(f"    card {card}: {value:.1f}  ({dev:+.0%} vs plate median)  "
                  f"[{cv.get(card, OK)}]")

    if 'neighbour_check' in summary:
        print(f"\n  note: {summary['neighbour_check']}")

    pf = summary.get('plate_flags', [])
    print(f"\n  PLATE-LEVEL VERDICT: {summary.get('plate_verdict', OK)}")
    if pf:
        for level, msg in pf:
            print(f"    [{level}] {msg}")
    else:
        print("    no plate-wide problem detected")

    verdicts = [c.verdict for c in checks.values()]
    print(f"\n  SEGMENT VERDICTS: OK={verdicts.count(OK)}  WARN={verdicts.count(WARN)}"
          f"  FAIL={verdicts.count(FAIL)}")

    print(f"\n{'-' * 104}")
    print(f"  {'seg':>4} {'card':>4} {'Rs[mOhm*cm2]':>13} {'Rct[mOhm*cm2]':>14} "
          f"{'Rs/med':>7} {'MAD':>6} {'nbr%':>6} {'arc+':>5} {'coh':>5} {'pts':>4} "
          f"{'verdict':>7}")
    print(f"{'-' * 104}")
    for c in sorted(checks.values(), key=lambda x: x.segment):
        nbr = f"{c.neighbour_rel:.0%}" if np.isfinite(c.neighbour_rel) else '-'
        arc = f"{c.arc_pos_frac:.0%}" if np.isfinite(c.arc_pos_frac) else '-'
        coh = f"{c.median_coh:.2f}" if np.isfinite(c.median_coh) else '-'
        print(f"  {c.segment:>4} {c.card:>4} {c.Rs_asr:>13.1f} {c.Rct_asr:>14.1f} "
              f"{c.uniformity:>7.2f} {c.outlier_score:>6.1f} {nbr:>6} {arc:>5} "
              f"{coh:>5} {c.n_pts:>4} {c.verdict:>7}")

    flagged = [c for c in checks.values() if c.verdict != OK]
    if flagged:
        print(f"\n{'-' * 104}")
        print("  FLAGGED SEGMENTS")
        print(f"{'-' * 104}")
        for c in sorted(flagged, key=lambda x: x.segment):
            print(f"  segment {c.segment:>3} (card {c.card}) [{c.verdict}]")
            for flag in c.flags:
                print(f"      - {flag}")
    else:
        print("\n  every segment passed.")


def run_plausibility(tone_only_all, gamry_data, conditions, leepa,
                     seg_area_cm2, cell_area_cm2, seg_coords, seg_to_card,
                     f_min, f_max, rejected_all=None, limits=None, verbose=True):
    """Drive every condition and return (all_checks, all_summaries, parallel, overall)."""
    limits = limits or Limits()
    rejected_all = rejected_all or {}
    all_checks, all_summaries, parallel = {}, {}, {}

    n_expected = cell_area_cm2 / seg_area_cm2
    if verbose:
        print('=' * 70)
        print("  PLAUSIBILITY CHECK: Per-Segment + Parallel Impedance vs Gamry")
        print(f"  Leepa {leepa} | conditions: {list(conditions)}")
        print('=' * 70)
        print(f"  Rs band [{limits.rs_asr_min}, {limits.rs_asr_max}] mOhm*cm2, "
              f"outlier > {limits.outlier_warn} MAD, neighbour > "
              f"{limits.neighbour_warn:.0%}")
        print(f"  Known hardware faults (checked anyway, annotated): "
              f"{sorted(KNOWN_FAULT_SEGMENTS)}")
        print(f"  Cell area {cell_area_cm2:.2f} cm2 = {n_expected:.1f} segments "
              f"of {seg_area_cm2} cm2  <- the parallel sum scales by area, not count")

    for cond in conditions:
        results = tone_only_all.get(cond, {})
        if not results and not rejected_all.get(cond):
            if verbose:
                print(f"\n  {cond}: no data available.")
            continue
        checks, summary = run_checks(
            results, cond, limits, rejected=rejected_all.get(cond),
            seg_area_cm2=seg_area_cm2, seg_to_card=seg_to_card,
            seg_coords=seg_coords)
        all_checks[cond] = checks
        all_summaries[cond] = summary
        if verbose:
            print_segment_report(checks, summary, cond, leepa)

        par = parallel_impedance(results, seg_area_cm2, cell_area_cm2,
                                 exclude=EXCLUDE_FROM_PARALLEL)
        if par is not None:
            parallel[cond] = par

    # --- parallel sum vs Gamry ------------------------------------------
    if verbose:
        print(f"\n\n{'=' * 104}")
        print("  PARALLEL SUM: Z_cell(f) = [1/sum(1/Z_seg)] * A_measured/A_cell  "
              "vs Gamry")
        print("  (the strongest single test of the impedance calibration chain)")
        print('=' * 104)
        print(f"\n  {'Cond':<7} {'N':<5} {'f_common':<9} {'A_scale':<9} "
              f"{'Rs_cell[mOhm]':<14} {'Gamry[mOhm]':<13} {'Ratio':<8} {'Verdict'}")
        print(f"  {'-' * 88}")

    par_worst = OK
    for cond in conditions:
        p = parallel.get(cond)
        if p is None:
            if verbose:
                print(f"  {cond:<7} - no parallel data -")
            continue
        f_ref = float(np.max(p['f']))
        rs_g, _ = gamry_rs_mohm(gamry_data.get(cond), f_min, f_max, f_ref_hz=f_ref)
        if np.isfinite(rs_g) and rs_g > 0:
            ratio = p['Rs_cell_mOhm'] / rs_g
            deviation = abs(ratio - 1.0)
            v = (FAIL if deviation > limits.par_sum_fail else
                 WARN if deviation > limits.par_sum_warn else OK)
            if _RANK[v] > _RANK[par_worst]:
                par_worst = v
            p['gamry_mOhm'], p['ratio'], p['verdict'] = rs_g, ratio, v
            if verbose:
                print(f"  {cond:<7} {p['n_segs']:<5} {p['n_freq_common']:<9} "
                      f"{p['area_scale']:<9.4f} {p['Rs_cell_mOhm']:<14.4f} "
                      f"{rs_g:<13.4f} {ratio:<8.3f} {v}")
        else:
            p['verdict'] = 'SKIPPED'
            if verbose:
                print(f"  {cond:<7} {p['n_segs']:<5} {p['n_freq_common']:<9} "
                      f"{p['area_scale']:<9.4f} {p['Rs_cell_mOhm']:<14.4f} "
                      f"{'- n/a -':<13} {'-':<8} SKIPPED")

    if verbose:
        any_bound = any(not p.get('Rs_is_measured', True) for p in parallel.values())
        if any_bound:
            print("\n  ! the recombined arc does not cross Im(Z)=0 in band, so both "
                  "sides of this\n    comparison are upper bounds on Rs, not measured "
                  "intercepts. The ratio is\n    still meaningful; the absolute "
                  "values are not.")
        dropped = sum(p['n_freq_dropped'] for p in parallel.values())
        if dropped:
            print(f"  ! {dropped} frequencies were dropped because not every segment "
                  f"covered them")

    all_verdicts = [c.verdict for ch in all_checks.values() for c in ch.values()]
    plate_verdicts = [s.get('plate_verdict', OK) for s in all_summaries.values()]
    overall = max(all_verdicts + plate_verdicts + [par_worst],
                  key=lambda v: _RANK[v]) if all_verdicts else par_worst

    if verbose:
        n_ok = all_verdicts.count(OK)
        n_warn = all_verdicts.count(WARN)
        n_fail = all_verdicts.count(FAIL)
        print(f"\n{'=' * 104}")
        print(f"  OVERALL: {overall}   (segments: OK={n_ok} WARN={n_warn} "
              f"FAIL={n_fail} | plate: {max(plate_verdicts or [OK], key=lambda v: _RANK[v])}"
              f" | parallel sum: {par_worst})")
        if overall == FAIL:
            print("  A single FAIL anywhere means this measurement needs "
                  "investigation before downstream analysis.")
        print('=' * 104)

    return all_checks, all_summaries, parallel, overall


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_parallel_vs_gamry(parallel, gamry_data, conditions, leepa, f_min, f_max,
                           overall):
    """Nyquist + Bode of the *area-scaled* plate impedance against Gamry.

    The scaled Z_cell is plotted, not the raw parallel sum: the table compares
    the scaled value, and showing the unscaled one here would display a
    systematic offset that is not physically there.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['<b>Nyquist: Z_cell (segments, area-scaled) vs Gamry</b>',
                        '<b>Bode |Z|: Z_cell vs Gamry</b>'],
        horizontal_spacing=0.10)
    colours = {'45A': '#1f77b4', '60A': '#ff7f0e', '150A': '#2ca02c', '450A': '#d62728'}

    for cond in conditions:
        p = parallel.get(cond)
        gd = gamry_data.get(cond)
        clr = colours.get(cond, 'gray')

        if p is not None:
            Zm = p['Z_cell'] * 1000
            fig.add_trace(go.Scatter(
                x=Zm.real, y=-Zm.imag, mode='lines+markers',
                name=f"{cond} plate ({p['n_segs']} segs x{p['area_scale']:.3f})",
                line=dict(color=clr, width=2.5), marker=dict(size=5, color=clr),
                legendgroup=cond, customdata=p['f'],
                hovertemplate=f'<b>{cond} plate</b><br>f=%{{customdata:.0f}} Hz<br>'
                              'Re=%{x:.3f}<br>-Im=%{y:.3f} mOhm<extra></extra>',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=p['f'], y=np.abs(Zm), mode='lines',
                line=dict(color=clr, width=2.5),
                legendgroup=cond, showlegend=False,
            ), row=1, col=2)

        if gd and gd.get('n_pts', 0) > 5:
            gf = np.asarray(gd['freq'], float)
            m = (gf >= f_min) & (gf <= f_max)
            gre = np.asarray(gd['zreal'], float)[m] * 1000
            gim = np.asarray(gd['zimag'], float)[m] * 1000
            fig.add_trace(go.Scatter(
                x=gre, y=-gim, mode='lines+markers', name=f'{cond} Gamry',
                line=dict(color=clr, width=2, dash='dash'),
                marker=dict(size=7, symbol='diamond', color=clr,
                            line=dict(width=1.5, color='white')),
                legendgroup=cond, customdata=gf[m],
                hovertemplate=f'<b>{cond} Gamry</b><br>f=%{{customdata:.0f}} Hz<br>'
                              'Re=%{x:.3f}<br>-Im=%{y:.3f} mOhm<extra></extra>',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=gf[m], y=np.sqrt(gre**2 + gim**2), mode='lines+markers',
                line=dict(color=clr, width=2, dash='dash'),
                marker=dict(size=5, symbol='diamond', color=clr),
                legendgroup=cond, showlegend=False,
            ), row=1, col=2)

    fig.update_xaxes(title_text="Z' [mOhm]", showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mOhm]", showgrid=True, row=1, col=1)
    fig.update_xaxes(title_text='f [Hz]', type='log', showgrid=True, row=1, col=2)
    fig.update_yaxes(title_text='|Z| [mOhm]', type='log', showgrid=True, row=1, col=2)
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5, row=1, col=1)
    fig.update_layout(
        title=dict(text=(f'<b>Plausibility: Leepa {leepa} -- {overall}</b><br>'
                         f'<sup>Z_cell = [1/sum(1/Z_seg)] x A_measured/A_cell, '
                         f'vs Gamry Ref 3000 -- mOhm</sup>'),
                   font=dict(size=13), x=0.01, xanchor='left'),
        height=500, width=1200, paper_bgcolor='white', plot_bgcolor='white',
        legend=dict(x=1.02, y=1.0, font=dict(size=10)),
        margin=dict(t=80, b=50, r=230))
    return fig


# ---------------------------------------------------------------------------
# DRIVER - runs only when the notebook globals are present
# ---------------------------------------------------------------------------
_REQUIRED = ('TONE_ONLY_ALL', 'CONDITIONS', 'LEEPA', 'SEG_AREA_CM2',
             'CELL_AREA_CM2', 'ALL_SEG_COORDS', 'SEG_TO_CARD', 'F_MIN', 'F_MAX')

if all(name in dir() for name in _REQUIRED):
    ALL_CHECKS, ALL_SUMMARIES, Z_PARALLEL, OVERALL = run_plausibility(
        tone_only_all=TONE_ONLY_ALL,
        gamry_data=(GAMRY_DATA if 'GAMRY_DATA' in dir() else {}),
        conditions=CONDITIONS,
        leepa=LEEPA,
        seg_area_cm2=SEG_AREA_CM2,
        cell_area_cm2=CELL_AREA_CM2,
        seg_coords=ALL_SEG_COORDS,
        seg_to_card=SEG_TO_CARD,
        f_min=F_MIN,
        f_max=F_MAX,
        rejected_all=(REJECTED_ALL if 'REJECTED_ALL' in dir() else {}),
    )

    _fig = plot_parallel_vs_gamry(
        Z_PARALLEL, (GAMRY_DATA if 'GAMRY_DATA' in dir() else {}),
        CONDITIONS, LEEPA, F_MIN, F_MAX, OVERALL)
    displayHTML(_fig.to_html(
        full_html=False, include_plotlyjs='cdn',
        config={'scrollZoom': True, 'displayModeBar': True,
                'toImageButtonOptions': {'format': 'png', 'scale': 3,
                                         'filename': f'plausibility_{LEEPA}'}}))
