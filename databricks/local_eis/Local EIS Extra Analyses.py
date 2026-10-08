# Databricks notebook source
# MAGIC %md
# MAGIC # Local EIS — extra analyses
# MAGIC Optional diagnostics, moved out of the main runner so it shows only the
# MAGIC results: plate numbering check, Gamry vs aggregate ASR decomposition,
# MAGIC Lin-KK, DRT, the DRT-informed ECM and its comparison with the pipeline fit.
# MAGIC
# MAGIC The first cell runs the main runner (cached results load in seconds), so
# MAGIC every variable it defines is available here.

# COMMAND ----------

_EIS_NO_RESTART = True     # the runner must not restart Python under %run

# COMMAND ----------

# MAGIC %run "./Local EIS Pipeline Runner"

# COMMAND ----------

# DBTITLE 1,Plate map: confirm the numbering before trusting any heat map
# ═══════════════════════════════════════════════════════════════════════════════
# Draw the selected plate.  Compare it against the coordinate drawing once,
# at the start of a campaign — this is the cheapest way to catch the one
# mistake that produces a plausible-looking wrong answer: a gen2 recording
# evaluated with the gen1 map.
#
# gen1 (green / Kashyyyk): segments 37..72 are six full-height edge strips.
# gen2 (blue  / Naboo)   : the edge segments are re-cut and interleaved into
#                          the wide strips at the top and bottom of the plate,
#                          so 49, 51, 55 and 57 sit along the TOP edge and
#                          52, 54, 58 and 60 along the bottom.
# Segments 1..36 keep their positions on both, but NOT their areas: the
# strips that gained a top/bottom edge segment lost pad rows to it.
# ═══════════════════════════════════════════════════════════════════════════════
import os, tempfile
from IPython.display import display, Image as IPImage
 
_map_png = Path(tempfile.gettempdir()) / f'plate_{PLATE}_{os.getuid()}.png'
geom.plot_map(_map_png)
print(f"  {_plate.title} — reconstructed from {_plate.drawing}")
_chk = geom.self_check(verbose=False)
print(f"  {_chk['n_segments']} segments, {_chk['pads_covered_once']}/900 pads "
      f"covered exactly once, area sum {_chk['area_sum_cm2']:.2f} cm², "
      f"areas {_chk['area_min_cm2']:.3f}–{_chk['area_max_cm2']:.3f} cm²")
if _chk['problems']:
    print('  PROBLEMS: ' + '; '.join(_chk['problems']))
display(IPImage(filename=str(_map_png)))
 
# Where a gen1 segment number lands on the gen2 plate, for orientation only —
# the two plates have no one-to-one segment correspondence, the edge segments
# were re-cut rather than renamed.
if PLATE == 'gen2':
    _ren = geom.renumbering('gen1', 'gen2')
    print("\n  gen1 segment -> the gen2 segment covering its centre "
          "(orientation only, NOT a data conversion):")
    print('  ' + ', '.join(f'{k}->{v}' for k, v in
                           sorted(_ren.items(), key=lambda kv: int(kv[0]))
                           if k != v))

# COMMAND ----------

# ── Gamry vs local-aggregate: is the ASR the same quantity? ──────────────────
import numpy as np, pandas as pd
from pathlib import Path
import gamry_dta
from config import A_CELL_CM2
 
# THREE HARD-CODED VALUES USED TO LIVE HERE, AND EACH ONE WAS A TRAP.
#   LEEPA = '2612030'      -- and it OVERWROTE the notebook's LEEPA global, so
#                             every later cell silently changed plate.
#   /tmp/eis_1003          -- another user's scratch directory. On any other
#                             cluster login this is simply someone else's run.
#   SNR_TAG = 'snr_-20.0'  -- a cache folder from one particular sweep, read
#                             no matter which SNR the widget said.
# All three now come from the widgets, through the same resolver the heat maps
# and the overlay use, so this table describes the run that is selected.
_ASR_LEEPA = str(_widget('leepa_id', 'LEEPA', 'leepa'))

# The filenames used to be written out here, V26_092 and all -- one campaign's
# sweeps, pasted into a cell that runs for whichever order is selected. Pick a
# different order and this table still read RO2612030's references.
# They are discovered now, from the build token that order resolves to.
# GAMRY_DIR is deliberately NOT reassigned: the run cell reads that global to
# build its Config, and a display cell that overwrites it changes the run.
_ASR_GAMRY_DIR = gamry_root_for(_ASR_LEEPA) or Path(
    '/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Gamry')
_ASR_FILES, _ASR_BUILD = gamry_files(_ASR_LEEPA)

DTA = {}
for _f in _ASR_FILES:
    _m = re.search(r'CurrVal_(\d+(?:[.,]\d+)?)', _f.stem)
    if _m:
        DTA.setdefault(_m.group(1).replace(',', '.') + 'A', _f.name)
# The setpoint is in the key, so it does not need a second table that can
# disagree with the first.
SETPOINT = {k: float(k[:-1]) for k in DTA}

print(f"  order {_ASR_LEEPA} -> build {_ASR_BUILD or 'UNKNOWN'} -> "
      f"{len(DTA)} sweep(s) in {_ASR_GAMRY_DIR}")
for _k in sorted(DTA, key=lambda c: SETPOINT[c]):
    print(f"    {_k:>6}: {DTA[_k]}")
if not DTA:
    print("    none found -- check GAMRY_SEARCH_ROOTS in the setup cell")

_ASR_PROV = {}


def find_silver(cond):
    """The silver directory for this condition, or None. See result_dir()."""
    d, prov = result_dir(_ASR_LEEPA, cond)
    _ASR_PROV[cond] = prov
    if d is None:
        return None
    sv = Path(d) / 'silver'
    return sv if (sv / 'cell_aggregate.csv').exists() else None
 
def decompose(f, Zg, Zl):
    """Zl = a*Zg + R + jwL  with a, R, L real — separates the three causes."""
    w = 2*np.pi*f; wt = 1.0/np.abs(Zg)
    A = np.vstack([np.column_stack([Zg.real, np.ones_like(f), np.zeros_like(f)])*wt[:,None],
                   np.column_stack([Zg.imag, np.zeros_like(f), w            ])*wt[:,None]])
    y = np.concatenate([Zl.real*wt, Zl.imag*wt])
    p,*_ = np.linalg.lstsq(A, y, rcond=None)
    res = Zl - (p[0]*Zg + p[1] + 1j*w*p[2])
    return p, 100*np.median(np.abs(res)/np.abs(Zl))
 
print(f"A_CELL_CM2 = {A_CELL_CM2:.2f} cm2   (plate footprint)\n")
hdr = (f"{'cond':>5} {'A_used':>7} {'n_f':>4} {'band [Hz]':>17} {'scale a':>8} "
       f"{'implied A':>10} {'R_ser mΩcm²':>11} {'L_ser nH':>11} {'resid':>7} {'I_err':>7}")
print(hdr); print('-'*len(hdr))
 
for cond, fn in DTA.items():
    sv = find_silver(cond)
    if sv is None:
        print(f"{cond:>5}   no cell_aggregate.csv "
              f"({_ASR_PROV.get(cond, 'not found')})")
        continue
    dta = _ASR_GAMRY_DIR/fn
    if not dta.exists():
        print(f"{cond:>5}   missing {dta}"); continue
 
    loc = pd.read_csv(sv/'cell_aggregate.csv').sort_values('freq_hz')
    f_l = loc.freq_hz.values
    Z_l = (loc.z_re_mohm_cm2.values + 1j*loc.z_im_mohm_cm2.values)/1e3      # ohm.cm2
 
    sw  = gamry_dta.read_dta(dta).sorted()          # field is .Z, in OHM
    f_g, Z_g = np.asarray(sw.freq,float), np.asarray(sw.Z,complex)
 
    lo, hi = max(f_l.min(), f_g.min()), min(f_l.max(), f_g.max())
    m = (f_l >= lo) & (f_l <= hi)
    if m.sum() < 8:
        print(f"{cond:>5}   only {m.sum()} overlapping points"); continue
    f, Zl = f_l[m], Z_l[m]
    Zg = (np.interp(np.log(f), np.log(f_g), Z_g.real)
          + 1j*np.interp(np.log(f), np.log(f_g), Z_g.imag)) * A_CELL_CM2    # <- area under test
 
    p, resid = decompose(f, Zg, Zl)
    seg = pd.read_csv(sv/'segments_summary.csv')
    A_used = float(seg.area_cm2.sum())
    I_full = float((seg.j_dc_A_cm2*seg.area_cm2).sum()) * A_CELL_CM2 / A_used
    print(f"{cond:>5} {A_used:7.1f} {len(f):4d} {lo:7.2f}-{hi:8.1f} {p[0]:8.4f} "
          f"{p[0]*A_CELL_CM2:10.1f} {1e3*p[1]:+9.2f} {1e9*p[2]:+11.4g} "
          f"{resid:6.1f}% {100*(I_full/SETPOINT[cond]-1):+6.1f}%")
    
    
COMMON = (0.30, 700.0)        # a band all four conditions cover
BANDS  = [(0.2,1),(1,10),(10,100),(100,1000),(1000,5000)]
 
print(f"{'cond':>5} | " + "  ".join(f"{a:g}-{b:g}Hz" for a,b in BANDS)
      + "   ||   a(common)   R_ser   resid")
for cond, fn in DTA.items():
    sv = find_silver(cond)
    if sv is None: print(f"{cond:>5}  (no data)"); continue
    loc = pd.read_csv(sv/'cell_aggregate.csv').sort_values('freq_hz')
    f_l = loc.freq_hz.values
    Z_l = (loc.z_re_mohm_cm2.values + 1j*loc.z_im_mohm_cm2.values)/1e3
    sw = gamry_dta.read_dta(_ASR_GAMRY_DIR/fn).sorted()
    f_g, Z_g = np.asarray(sw.freq,float), np.asarray(sw.Z,complex)
    lo, hi = max(f_l.min(), f_g.min()), min(f_l.max(), f_g.max())
    m = (f_l>=lo)&(f_l<=hi); f, Zl = f_l[m], Z_l[m]
    Zg = (np.interp(np.log(f), np.log(f_g), Z_g.real)
          + 1j*np.interp(np.log(f), np.log(f_g), Z_g.imag)) * A_CELL_CM2
    r = np.abs(Zl)/np.abs(Zg)
    cells = [f"{np.median(r[(f>=a)&(f<b)]):9.3f}" if ((f>=a)&(f<b)).sum() else f"{'--':>9}"
             for a,b in BANDS]
    k = (f>=COMMON[0])&(f<=COMMON[1])
    if k.sum() >= 8:
        p, res = decompose(f[k], Zg[k], Zl[k])
        tail = f"  ||  {p[0]:8.4f}  {1e3*p[1]:+7.2f}  {res:4.1f}%"
    else:
        tail = f"  ||  only {k.sum()} pts in the common band"
    print(f"{cond:>5} | " + " ".join(cells) + tail)

# COMMAND ----------

# DBTITLE 1,Lin-KK: Linear Kramers-Kronig Validation
# ═══════════════════════════════════════════════════════════════════════════════
# LINEAR KRAMERS-KRONIG VALIDATION (Boukamp 1995 / Schoenleber 2014)
#
# Gate test: is this spectrum causal, linear, time-invariant?  A spectrum that
# fails KK cannot be interpreted by an ECM or DRT.  Uses a Voigt ladder with
# FIXED time constants so the fit is LINEAR in R_k; the residual measures ONLY
# the data's departure from KK compliance, never the model flexibility.
#
# The M (number of RC elements) is chosen by the Schoenleber over-fitting
# criterion: mu = 1 - sum(R_k<0)/sum(R_k>0) drops below 0.85 once over-fitting.
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
 
# ─── Lin-KK Implementation ───
def _voigt_basis(freq, taus):
    """Complex design matrix for Voigt ladder with fixed time constants."""
    w = 2 * np.pi * freq[:, None]
    return 1.0 / (1.0 + 1j * w * taus[None, :])
 
 
def lin_kk(freq, Z, M=None, add_inductance=True, c_crit=0.85):
    """Linear KK test. Returns dict with residuals and fitted model.
    
    Parameters
    ----------
    freq : Hz, ascending
    Z    : complex impedance (any unit)
    M    : number of RC elements (None = auto via Schoenleber criterion)
    add_inductance : include series jwL term (needed for fuel-cell data)
    c_crit : mu threshold for over-fitting detection (default 0.85)
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    n = len(freq)
 
    def _fit(M):
        taus = np.logspace(np.log10(1 / (2 * np.pi * freq.max())),
                           np.log10(1 / (2 * np.pi * freq.min())), M)
        A = _voigt_basis(freq, taus)
        cols = [A.real, np.ones((n, 1))]
        colsi = [A.imag, np.zeros((n, 1))]
        if add_inductance:
            cols.append(np.zeros((n, 1)))
            colsi.append((2 * np.pi * freq)[:, None])
        X = np.vstack([np.hstack(cols), np.hstack(colsi)])
        y = np.concatenate([Z.real, Z.imag])
        wgt = np.concatenate([1.0 / np.abs(Z), 1.0 / np.abs(Z)])
        p, *_ = np.linalg.lstsq(X * wgt[:, None], y * wgt, rcond=None)
        Zf = X @ p
        Zfit = Zf[:n] + 1j * Zf[n:]
        Rk = p[:M]
        pos = Rk[Rk > 0].sum()
        neg = -Rk[Rk < 0].sum()
        mu = 1.0 - neg / pos if pos > 0 else 0.0
        return p, Zfit, taus, mu
 
    if M is None:
        M_grid = list(range(3, min(n - 2, 60)))
        mus = []
        fits = {}
        for m in M_grid:
            fits[m] = _fit(m)
            mus.append(fits[m][3])
        mus = np.asarray(mus)
        # Find peak mu then first drop below c_crit
        start = max(0, int(np.argmax(np.where(np.arange(len(mus)) >= 2, mus, -np.inf))))
        chosen = None
        for i in range(start + 1, len(M_grid)):
            if mus[i] < c_crit:
                chosen = M_grid[i]
                break
        M = chosen if chosen else M_grid[min(start, len(M_grid) - 1)]
 
    p, Zfit, taus, mu = _fit(M)
    res_re = (Z.real - Zfit.real) / np.abs(Z)
    res_im = (Z.imag - Zfit.imag) / np.abs(Z)
    return {
        "M": M, "mu": mu, "taus": taus, "params": p, "Z_fit": Zfit,
        "res_re": res_re, "res_im": res_im,
        "res_re_pct": 100 * res_re, "res_im_pct": 100 * res_im,
        "chi2": float(np.sum(res_re**2 + res_im**2)),
        "max_abs_res_pct": float(100 * np.max(np.abs(res_re + 1j * res_im))),
        "rms_res_pct": float(100 * np.sqrt(np.mean(res_re**2 + res_im**2))),
    }
 
 
# ─── Run Lin-KK on silver spectra from pipeline results ───
KK_RESULTS = {}  # {cond: {seg: kk_dict}}
 
for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    spectra_path = spectra_csv(pr['out_dir'])
    if not spectra_path.exists():
        print(f"  {cond}: no spectra_clean.csv")
        continue
 
    df = pd.read_csv(spectra_path)
    segments = sorted(df['segment'].unique())
    KK_RESULTS[cond] = {}
 
    for seg in segments:
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values
        zi = sd['z_im_mohm_cm2'].values
        Z = zr + 1j * zi
 
        # Basic quality filter
        keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0)
        if keep.sum() < 10:
            continue
        f, Z = f[keep], Z[keep]
 
        kk = lin_kk(f, Z)
        KK_RESULTS[cond][seg] = kk
 
    n_pass = sum(1 for v in KK_RESULTS[cond].values() if v['rms_res_pct'] < 2.0)
    print(f"  {cond}: {len(KK_RESULTS[cond])} segments tested, "
          f"{n_pass} pass (<2% RMS residual)")
 
# ─── Plot: KK diagnostic (Plotly interactive) ───
for cond, kk_cond in KK_RESULTS.items():
    if not kk_cond:
        continue
 
    all_segs = sorted(kk_cond.keys(), key=lambda x: int(x))
    step = max(1, len(all_segs) // 6)
    show_segs = all_segs[::step][:6]
    n_show = len(show_segs)
    n_cols = min(n_show, 3)
    n_row_groups = (n_show + n_cols - 1) // n_cols
    n_rows = 2 * n_row_groups
 
    titles = []
    for i, seg in enumerate(show_segs):
        kk = kk_cond[seg]
        titles.append(f'Seg {seg} Nyquist (M={kk["M"]}, μ={kk["mu"]:.2f})')
    for i, seg in enumerate(show_segs):
        titles.append(f'Seg {seg} Residuals (RMS={kk_cond[seg]["rms_res_pct"]:.2f}%)')
    # Interleave: row1=nyquist, row2=residual for each row group
    ordered_titles = []
    for rg in range(n_row_groups):
        for c in range(n_cols):
            idx = rg * n_cols + c
            ordered_titles.append(titles[idx] if idx < n_show else '')
        for c in range(n_cols):
            idx = rg * n_cols + c
            ordered_titles.append(titles[n_show + idx] if idx < n_show else '')
 
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=ordered_titles,
                        vertical_spacing=0.08, horizontal_spacing=0.06)
 
    spectra_path = spectra_csv(PIPELINE_RESULTS[cond]['out_dir'])
    df = pd.read_csv(spectra_path)
    _colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
 
    for idx, seg in enumerate(show_segs):
        rg = idx // n_cols
        col = idx % n_cols + 1
        row_nyq = rg * 2 + 1
        row_res = rg * 2 + 2
        kk = kk_cond[seg]
 
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values
        zi = sd['z_im_mohm_cm2'].values
        keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0)
        Z_data = (zr + 1j * zi)[keep]
        Zf = kk['Z_fit']
 
        # Nyquist
        fig.add_trace(go.Scatter(x=Z_data.real, y=-Z_data.imag, mode='markers',
            marker=dict(size=4, color=_colors[idx % 6], opacity=0.6),
            name=f'Data s{seg}', showlegend=(idx == 0),
            hovertemplate="Z'=%{x:.1f}<br>-Z''=%{y:.1f}<extra>Data</extra>"),
            row=row_nyq, col=col)
        fig.add_trace(go.Scatter(x=Zf.real, y=-Zf.imag, mode='lines',
            line=dict(color='red', width=2),
            name=f'KK fit', showlegend=(idx == 0),
            hovertemplate="Z'=%{x:.1f}<br>-Z''=%{y:.1f}<extra>KK fit</extra>"),
            row=row_nyq, col=col)
 
        # Residuals
        f_plot = f[keep]
        fig.add_trace(go.Scatter(x=f_plot, y=kk['res_re_pct'], mode='markers',
            marker=dict(size=3, color='#1f77b4'), name='ΔRe', showlegend=(idx == 0)),
            row=row_res, col=col)
        fig.add_trace(go.Scatter(x=f_plot, y=kk['res_im_pct'], mode='markers',
            marker=dict(size=3, color='#d62728'), name='ΔIm', showlegend=(idx == 0)),
            row=row_res, col=col)
        # ±2% threshold lines
        fig.add_hline(y=2, line=dict(color='grey', dash='dash', width=0.8), row=row_res, col=col)
        fig.add_hline(y=-2, line=dict(color='grey', dash='dash', width=0.8), row=row_res, col=col)
        fig.add_hline(y=0, line=dict(color='black', width=0.5), row=row_res, col=col)
        fig.update_xaxes(type='log', row=row_res, col=col)
        fig.update_yaxes(range=[-8, 8], row=row_res, col=col)
 
    fig.update_layout(title=f'<b>Lin-KK Validation — {cond} (Leepa {LEEPA})</b>',
                      height=450 * n_row_groups, width=1400,
                      paper_bgcolor='white', plot_bgcolor='white',
                      showlegend=True, hovermode='closest')
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    show_fig(fig, html=True)   # one plot per figure, stacked

# COMMAND ----------

# DBTITLE 1,DRT: Distribution of Relaxation Times (Tikhonov)
# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION OF RELAXATION TIMES — Tikhonov Ridge Regression
#
# Deconvolves gamma(ln tau) from the impedance spectrum:
#   Z(f) = R_inf + jωL + ∫ gamma(ln τ) / (1 + jωτ) d(ln τ)
#
# The inversion is ill-posed (small data errors blow up in gamma), so a
# smoothness penalty λ·||γ||² is added.  Lambda is a genuine trade-off:
# too small → oscillations; too large → real peaks merge.
#
# WHY DRT: tells you HOW MANY processes exist and at WHAT timescales,
# before choosing an ECM model order.  Avoids the failure mode of copying
# a circuit from a paper without verification.
#
# References:
#   Wan, Saccoccio, Chen, Ciucci, Electrochim. Acta 184, 483 (2015)
#   Ivers-Tiffee & Weber, J. Ceram. Soc. Japan 125, 193 (2017)
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls
from pathlib import Path
 
 
# ─── DRT Implementation ───
def _voigt_basis(freq, taus):
    w = 2 * np.pi * freq[:, None]
    return 1.0 / (1.0 + 1j * w * taus[None, :])
 
 
def drt_tikhonov(freq, Z, lam=1e-3, n_tau_per_decade=10, extend_decades=0.0,
                 use="imag", nonneg=True, fit_L=True, fit_Rinf=False):
    """Deconvolve gamma(ln tau) from impedance spectrum.
    
    Parameters
    ----------
    freq : Hz
    Z    : complex impedance
    lam  : regularization parameter (smoothness penalty)
    use  : 'imag' (default, least affected by unmodelled Rs), 'real', 'both'
    nonneg : enforce gamma >= 0 (physical for passive systems)
    fit_L  : include series inductance jωL
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    w = 2 * np.pi * freq
 
    lo = np.log10(1 / (2 * np.pi * freq.max())) - extend_decades
    hi = np.log10(1 / (2 * np.pi * freq.min())) + extend_decades
    n_tau = int(np.ceil((hi - lo) * n_tau_per_decade)) + 1
    taus = np.logspace(lo, hi, n_tau)
    ln_tau = np.log(taus)
 
    A = _voigt_basis(freq, taus)
    d_ln_tau = np.mean(np.diff(ln_tau))
    A = A * d_ln_tau  # quadrature weight
 
    blocks_r, blocks_i = [A.real], [A.imag]
    extra = 0
    if fit_Rinf:
        blocks_r.append(np.ones((len(freq), 1)))
        blocks_i.append(np.zeros((len(freq), 1)))
        extra += 1
    if fit_L:
        blocks_r.append(np.zeros((len(freq), 1)))
        blocks_i.append(w[:, None])
        extra += 1
    Xr, Xi = np.hstack(blocks_r), np.hstack(blocks_i)
 
    if use == "imag":
        X, y = Xi, Z.imag
    elif use == "real":
        X, y = Xr, Z.real
    else:
        X, y = np.vstack([Xr, Xi]), np.concatenate([Z.real, Z.imag])
 
    # First-difference smoothness penalty on gamma only
    npar = X.shape[1]
    D = np.zeros((n_tau - 1, npar))
    for k in range(n_tau - 1):
        D[k, k], D[k, k + 1] = -1.0, 1.0
 
    # Scale penalty against data term
    s_dat = np.linalg.norm(X, 2)
    s_pen = np.linalg.norm(D, 2)
    scale = s_dat / max(s_pen, 1e-30)
    Xa = np.vstack([X, np.sqrt(lam) * scale * D])
    ya = np.concatenate([y, np.zeros(n_tau - 1)])
 
    if nonneg:
        if fit_L:
            Xa_ext = np.hstack([Xa, -Xa[:, -1:]])
            sol, _ = nnls(Xa_ext, ya)
            p = np.concatenate([sol[:npar - 1], [sol[npar - 1] - sol[npar]]])
        else:
            p, _ = nnls(Xa, ya)
    else:
        p, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
 
    gamma = p[:n_tau]
    out = {"tau": taus, "gamma": gamma, "lam": lam,
           "R_pol": float(np.sum(gamma) * d_ln_tau),
           "Z_fit": (Xr @ p) + 1j * (Xi @ p)}
 
    if not fit_Rinf:
        out["R_inf"] = float(np.median(Z.real - (Xr @ p)))
    j = n_tau
    if fit_Rinf:
        out["R_inf"] = float(p[j]); j += 1
    if fit_L:
        out["L"] = float(p[j])
    return out
 
 
def drt_peaks(res, min_rel_height=0.05):
    """Find peak timescales and their areas (= resistance of each process)."""
    g, tau = res["gamma"], res["tau"]
    if not np.any(g > 0):
        return []
    thr = min_rel_height * g.max()
    peaks = []
    d_ln = np.mean(np.diff(np.log(tau)))
    for k in range(1, len(g) - 1):
        if g[k] > thr and g[k] >= g[k - 1] and g[k] > g[k + 1]:
            lo_idx = k
            while lo_idx > 0 and g[lo_idx - 1] < g[lo_idx]:
                lo_idx -= 1
            hi_idx = k
            while hi_idx < len(g) - 1 and g[hi_idx + 1] < g[hi_idx]:
                hi_idx += 1
            peaks.append({"tau": float(tau[k]),
                          "f_peak": float(1 / (2 * np.pi * tau[k])),
                          "gamma": float(g[k]),
                          "R": float(np.sum(g[lo_idx:hi_idx + 1]) * d_ln)})
    return sorted(peaks, key=lambda p: -p["f_peak"])
 
 
# ─── Run DRT on silver spectra ───
DRT_RESULTS = {}  # {cond: {seg: drt_dict}}
 
for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    spectra_path = spectra_csv(pr['out_dir'])
    if not spectra_path.exists():
        continue
 
    df = pd.read_csv(spectra_path)
    segments = sorted(df['segment'].unique())
    DRT_RESULTS[cond] = {}
 
    for seg in segments:
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values
        zi = sd['z_im_mohm_cm2'].values
        Z = zr + 1j * zi
 
        keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0)
        if keep.sum() < 10:
            continue
        f, Z = f[keep], Z[keep]
 
        # Run DRT with multiple lambda to show sensitivity
        drt = drt_tikhonov(f, Z, lam=1e-3)
        peaks = drt_peaks(drt)
        drt['peaks'] = peaks
        DRT_RESULTS[cond][seg] = drt
 
    # Summary
    n_segs = len(DRT_RESULTS[cond])
    n_2peak = sum(1 for v in DRT_RESULTS[cond].values() if len(v.get('peaks', [])) >= 2)
    print(f"  {cond}: {n_segs} segments, {n_2peak} show ≥2 DRT peaks")
 
# ─── Plot: DRT gamma(tau) (Plotly interactive) ───
_drt_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
 
for cond, drt_cond in DRT_RESULTS.items():
    if not drt_cond:
        continue
 
    all_segs = sorted(drt_cond.keys(), key=lambda x: int(x))
    step = max(1, len(all_segs) // 6)
    show_segs = all_segs[::step][:6]
    n_show = len(show_segs)
    n_cols = min(n_show, 3)
    n_rows = (n_show + n_cols - 1) // n_cols
 
    titles = [f'Seg {s} — R_pol={drt_cond[s]["R_pol"]:.1f}, {len(drt_cond[s].get("peaks",[]))} peaks'
              for s in show_segs]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.12)
 
    for idx, seg in enumerate(show_segs):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        drt = drt_cond[seg]
        tau_ms = drt['tau'] * 1e3
        gamma = drt['gamma']
        peaks = drt.get('peaks', [])
 
        # Filled area + line
        fig.add_trace(go.Scatter(x=tau_ms, y=gamma, mode='lines',
            fill='tozeroy', fillcolor=f'rgba(70,130,180,0.25)',
            line=dict(color='steelblue', width=2), name=f'Seg {seg}',
            showlegend=(idx == 0),
            hovertemplate='τ=%{x:.3f} ms<br>γ=%{y:.2f}<extra></extra>'),
            row=r, col=c)
 
        # Peak markers
        for pk in peaks:
            fig.add_trace(go.Scatter(x=[pk['tau']*1e3], y=[pk['gamma']],
                mode='markers+text', marker=dict(size=8, color='red', symbol='diamond'),
                text=[f"{pk['f_peak']:.0f} Hz"], textposition='top center',
                textfont=dict(size=9, color='red'), showlegend=False),
                row=r, col=c)
 
        fig.update_xaxes(type='log', title_text='τ [ms]', row=r, col=c)
        fig.update_yaxes(title_text='γ(ln τ) [mΩ·cm²]', row=r, col=c)
 
    fig.update_layout(title=f'<b>DRT — Distribution of Relaxation Times — {cond} (Leepa {LEEPA})</b>',
                      height=350 * n_rows, width=1400,
                      paper_bgcolor='white', plot_bgcolor='white', hovermode='closest')
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    show_fig(fig, html=True)   # one plot per figure, stacked
 
# ─── Lambda sensitivity plot (Plotly interactive) ───
for cond, drt_cond in DRT_RESULTS.items():
    if not drt_cond:
        continue
    demo_seg = None
    for seg in sorted(drt_cond.keys(), key=lambda x: int(x)):
        if len(drt_cond[seg].get('peaks', [])) >= 2:
            demo_seg = seg
            break
    if demo_seg is None:
        demo_seg = sorted(drt_cond.keys(), key=lambda x: int(x))[0]
 
    spectra_path = spectra_csv(PIPELINE_RESULTS[cond]['out_dir'])
    df = pd.read_csv(spectra_path)
    sd = df[df['segment'] == demo_seg].sort_values('freq_hz')
    f = sd['freq_hz'].values
    zr = sd['z_re_mohm_cm2'].values
    zi = sd['z_im_mohm_cm2'].values
    Z = (zr + 1j * zi)
    keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0)
    f, Z = f[keep], Z[keep]
 
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    fig = go.Figure()
    for lam in lambdas:
        d = drt_tikhonov(f, Z, lam=lam)
        fig.add_trace(go.Scatter(x=d['tau'] * 1e3, y=d['gamma'], mode='lines',
            name=f'λ={lam:.0e}', hovertemplate='τ=%{x:.3f} ms<br>γ=%{y:.2f}<extra></extra>'))
 
    fig.update_xaxes(type='log', title_text='τ [ms]')
    fig.update_yaxes(title_text='γ(ln τ) [mΩ·cm²]')
    fig.update_layout(title=f'<b>DRT λ Sensitivity — {cond}, Seg {demo_seg} (Leepa {LEEPA})</b>',
                      height=450, width=900,
                      paper_bgcolor='white', plot_bgcolor='white',
                      hovermode='x unified')
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    show_fig(fig, html=True)   # one plot per figure, stacked
    break  # only show for first condition

# COMMAND ----------

# DBTITLE 1,ECM Fit: L + Rs + (Rct1||CPE1) + (Rct2||CPE2) — DRT-informed
 
# ═══════════════════════════════════════════════════════════════════════════════
# ECM FIT: DRT-INFORMED EQUIVALENT CIRCUIT MODEL  (method B; ecm_drt.py)
#
# Model: Z(f) = jωL + Rs + fast arc [+ slow arc], each arc R/(1+(jωτ)^n);
#        the fast arc may be a catalyst-layer transmission line (TLM).
# Starting values from the DRT peaks (Metrohm AN-EIS-007; Liu & Ciucci 2020).
#
# Revised for the thesis (all in ecm_drt.fit_drt_ecm):
#   1. arcs sorted by τ after the fit -> reported as FAST and SLOW arc
#   2. weighted by the measurement uncertainty σ = σ_rel·|Z|; χ²ν is a χ² only
#      then. Without σ the figure of merit is the RMS relative residual.
#   3. one or two arcs, chosen by AICc (not forced to two)
#   4. "clean" = converged, nothing on a bound, every arc ≥ 1 % of R_pol, every
#      τ inside 1/(2π f_max) .. 1/(2π f_min) of the fitted band
#   5. fitted only up to ECM_B_FMAX_HZ (1 kHz) with Rs held at the pipeline's
#      R_ohmic; a fast arc with n < 0.7 is flagged and the TLM element tried
#   6. points on odd mains harmonics (50/150/250 Hz) dropped and listed
#   7. summary = medians (IQR) over ALL fitted segments, clean count beside it
# ═══════════════════════════════════════════════════════════════════════════════
import importlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import ecm_drt
importlib.reload(ecm_drt)

ECM_B_FMAX_HZ = 1000.0       # point 5: top band left out
ECM_B_FIX_RS = True          # point 5: Rs = pipeline R_ohmic (silver)
ECM_B_FAST = "auto"          # "zarc" | "tlm" | "auto" (TLM when n_fast < 0.7)

ECM_FIT_RESULTS = {}  # {cond: {seg: fit dict}}

for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    spectra_path = spectra_csv(pr['out_dir'])
    if spectra_path is None or not Path(spectra_path).exists():
        continue
    df = pd.read_csv(spectra_path)
    _ss = Path(pr['out_dir']) / 'silver' / 'segments_summary.csv'
    _rs = (pd.read_csv(_ss).set_index('segment')['R_ohmic_mohm_cm2'].to_dict()
           if (ECM_B_FIX_RS and _ss.exists()) else {})
    ECM_FIT_RESULTS[cond] = {}
    _errors = {}
    for seg in sorted(df['segment'].unique(), key=int):
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        Z = sd['z_re_mohm_cm2'].values + 1j * sd['z_im_mohm_cm2'].values
        sig = sd['sigma_rel'].values if 'sigma_rel' in sd.columns else None
        drt_res = DRT_RESULTS.get(cond, {}).get(seg) if 'DRT_RESULTS' in globals() else None
        peaks = drt_peaks(drt_res) if drt_res is not None else None
        rs = _rs.get(seg)
        r = ecm_drt.fit_drt_ecm(
            f, Z, sig, peaks=peaks,
            rs_fixed=(float(rs) if rs is not None and np.isfinite(rs) else None),
            f_max_fit_hz=ECM_B_FMAX_HZ, fast_element=ECM_B_FAST)
        if r.get('ok'):
            ECM_FIT_RESULTS[cond][int(seg)] = r
        else:
            _errors[int(seg)] = r.get('reason')
    S = ecm_drt.summarize(ECM_FIT_RESULTS[cond])
    _drop = next(iter(ECM_FIT_RESULTS[cond].values()), {}).get('dropped_mains_hz', [])
    print(f"  {cond}: {S['n_fitted']} fitted ({S['n_two_arc']} two-arc, "
          f"{S['n_tlm']} TLM fast arc), {S['n_clean']} clean | "
          f"Rs median {S['Rs'][0]:.1f} mΩ·cm²"
          f"{' (fixed from pipeline)' if _rs else ''} | mains points dropped: "
          f"{', '.join(f'{x:.1f} Hz' for x in _drop) or 'none'}")
    if _errors:
        print(f"      not fitted: {_errors}")

# ─── Plot: Nyquist overlay + residuals (6 segments per condition) ───
for cond, ecm_cond in ECM_FIT_RESULTS.items():
    if not ecm_cond:
        continue
    all_segs = sorted(ecm_cond, key=int)
    step = max(1, len(all_segs) // 6)
    show_segs = all_segs[::step][:6]
    n_cols = min(len(show_segs), 3)
    n_rows = (len(show_segs) + n_cols - 1) // n_cols

    def _tag(e):
        q = (f"χ²ν={e['chi2_nu']:.3g}" if e['weight'] == 'sigma'
             else f"rms rel. resid={100 * e['rms_rel_resid']:.1f}%")
        return ('✓ ' if e['clean'] else '⚠ ') + q
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[f"Seg {s} {_tag(ecm_cond[s])}" for s in show_segs],
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    for idx, seg in enumerate(show_segs):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        e = ecm_cond[seg]
        fs = np.logspace(np.log10(e['freq'].min()), np.log10(e['freq'].max()), 300)
        Zs = ecm_drt.model_curve(e, fs)
        fig.add_trace(go.Scatter(x=e['Z_meas'].real, y=-e['Z_meas'].imag, mode='markers',
            marker=dict(size=4, color='steelblue', opacity=0.6), name='Data',
            showlegend=(idx == 0)), row=r, col=c)
        fig.add_trace(go.Scatter(x=Zs.real, y=-Zs.imag, mode='lines',
            line=dict(color='red', width=2.5), name='ECM (method B)',
            showlegend=(idx == 0)), row=r, col=c)
        _xref = 'x domain' if idx == 0 else f'x{idx+1} domain'
        _yref = 'y domain' if idx == 0 else f'y{idx+1} domain'
        fig.add_annotation(
            text=(f"Rs={e['Rs']:.1f} R_fast={e['R_fast']:.1f} R_slow={e['R_slow']:.1f}"
                  + (" (TLM)" if e['fast_element'] == 'tlm' else "")),
            xref=_xref, yref=_yref, x=0.98, y=0.95, showarrow=False,
            font=dict(size=9), xanchor='right', yanchor='top')
        fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=r, col=c)
        fig.update_yaxes(title_text="-Z'' [mΩ·cm²]",
                         scaleanchor=('x' if idx == 0 else f'x{idx+1}'), row=r, col=c)
    fig.update_layout(title=f'<b>ECM Fit (method B, ≤ {ECM_B_FMAX_HZ:.0f} Hz) — {cond} (Leepa {LEEPA})</b>',
                      height=400 * n_rows, width=1400, paper_bgcolor='white',
                      plot_bgcolor='white', hovermode='closest')
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    show_fig(fig, html=True)

    fig2 = make_subplots(rows=n_rows, cols=n_cols,
                         subplot_titles=[f'Seg {s} — max {ecm_cond[s]["res_pct"].max():.1f}%' for s in show_segs],
                         horizontal_spacing=0.06, vertical_spacing=0.12)
    for idx, seg in enumerate(show_segs):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        e = ecm_cond[seg]
        fig2.add_trace(go.Scatter(x=e['freq'], y=e['res_pct'], mode='lines+markers',
            marker=dict(size=3, color='steelblue'), line=dict(width=1, color='steelblue'),
            showlegend=False), row=r, col=c)
        fig2.add_hline(y=2, line=dict(color='green', dash='dash', width=1), row=r, col=c)
        fig2.add_hline(y=5, line=dict(color='orange', dash='dash', width=1), row=r, col=c)
        fig2.update_xaxes(type='log', title_text='f [Hz]', row=r, col=c)
        fig2.update_yaxes(title_text='|Residual| [%]', rangemode='tozero', row=r, col=c)
    fig2.update_layout(title=f'<b>ECM Residuals |ΔZ|/|Z| (method B) — {cond} (Leepa {LEEPA})</b>',
                       height=350 * n_rows, width=1400, paper_bgcolor='white',
                       plot_bgcolor='white', hovermode='x unified')
    show_fig(fig2, html=True)

# ─── Parameter table: every fitted segment; medians over ALL of them ───
for cond, ecm_cond in ECM_FIT_RESULTS.items():
    if not ecm_cond:
        continue
    rows = [{'segment': s, 'clean': e['clean'], 'n_arcs': e['n_arcs'],
             'fast_element': e['fast_element'], 'Rs': e['Rs'],
             'R_fast': e['R_fast'], 'tau_fast_ms': 1e3 * e['tau_fast'], 'n_fast': e['n_fast'],
             'R_slow': e['R_slow'], 'tau_slow_ms': 1e3 * e['tau_slow'], 'n_slow': e['n_slow'],
             'R_pol': e['R_pol'], 'rms_rel_resid_pct': 100 * e['rms_rel_resid'],
             'chi2_nu': e['chi2_nu'], 'flags': '; '.join(e['flags']),
             'notes': '; '.join(e['notes'])}
            for s, e in sorted(ecm_cond.items())]
    df_params = pd.DataFrame(rows)
    S = ecm_drt.summarize(ecm_cond)
    print(f"\n  ECM parameters (method B) — {cond}: medians over all "
          f"{S['n_fitted']} fitted segments ({S['n_clean']} clean)")
    print(f"  {'─'*70}")
    for k, unit, sc in (('Rs', 'mΩ·cm²', 1), ('R_fast', 'mΩ·cm²', 1),
                        ('tau_fast', 'ms', 1e3), ('n_fast', '', 1),
                        ('R_slow', 'mΩ·cm²', 1), ('tau_slow', 'ms', 1e3),
                        ('n_slow', '', 1), ('R_pol', 'mΩ·cm²', 1),
                        ('rms_rel_resid', '%', 100), ('chi2_nu', '', 1)):
        m, q1, q3, n = S[k]
        print(f"  {k:14s} {sc*m:9.3g}  (IQR {sc*q1:.3g} – {sc*q3:.3g}) {unit}  n={n}")
    display(df_params.round(4))

# COMMAND ----------

import re
import numpy as np, pandas as pd
from pathlib import Path
import gamry_dta
from config import A_CELL_CM2

# Second copy of the same hard-coded table, removed for the same reason: it
# named one campaign's files and one folder, whichever order was selected.
# _ASR_GAMRY_DIR, DTA and SETPOINT come from the cell above, which resolves
# them from the order through its build token.
_ASR_GAMRY_DIR = globals().get('_ASR_GAMRY_DIR') or (
    gamry_root_for(_widget('leepa_id', 'LEEPA', 'leepa'))
    or Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Gamry'))
if not globals().get('DTA'):
    _files, _build = gamry_files(_widget('leepa_id', 'LEEPA', 'leepa'))
    DTA = {}
    for _f in _files:
        _m = re.search(r'CurrVal_(\d+(?:[.,]\d+)?)', _f.stem)
        if _m:
            DTA.setdefault(_m.group(1).replace(',', '.') + 'A', _f.name)
    SETPOINT = {k: float(k[:-1]) for k in DTA}
MIN_COV  = 0.80        # area fraction a frequency must have to be trusted

def gamry_series_L(f, Z, top_decade=1.0):
    k = f >= f.max()/10**top_decade
    if k.sum() < 4: return 0.0
    w = 2*np.pi*f[k]
    return float(np.sum(w*Z.imag[k])/np.sum(w*w))

def hfr_band(f, Z, frac=0.5):
    """HF side of the arc: from where -Z'' falls to `frac` of its peak, to f_max.
    This is the region that determines the ohmic intercept."""
    y = -Z.imag; k = int(np.argmax(y)); thr = frac*y[k]; j = k
    while j < len(f)-1 and y[j+1] > thr: j += 1
    return f[j], f[-1]

def r_ohmic_topband(f, Z, decades=1/3):
    """Mean Re Z over the top third of a decade — the estimator silver uses,
    applied identically to both sides."""
    k = f >= f.max()*10**(-decades)
    if k.sum() < 2: k = f >= f.max()*10**(-1.0)
    return float(np.mean(Z.real[k])), int(k.sum())

def interp_log(f_new, f, Z):
    o = np.argsort(f)
    return (np.interp(np.log(f_new), np.log(f[o]), Z[o].real)
            + 1j*np.interp(np.log(f_new), np.log(f[o]), Z[o].imag))

for cond, fn in DTA.items():
    sv = find_silver(cond)                       # from the earlier script
    if sv is None or not (_ASR_GAMRY_DIR/fn).exists():
        print(f"{cond}: missing data"); continue

    ca  = pd.read_csv(sv/'cell_aggregate.csv').sort_values('freq_hz')
    seg = pd.read_csv(sv/'segments_summary.csv')
    sp  = pd.read_csv(sv/'spectra_clean.csv')
    area  = seg.set_index('segment').area_cm2
    A_tot = float(area.sum())

    # --- per-frequency area coverage --------------------------------------
    covmap = {}
    for f0, g in sp.groupby('freq_hz'):
        Z = g.z_re_mohm_cm2.values + 1j*g.z_im_mohm_cm2.values
        ok = np.isfinite(Z) & (Z != 0)
        covmap[round(float(f0), 6)] = g.segment.map(area).values[ok].sum()/A_tot

    f_l = ca.freq_hz.values
    Z_l = (ca.z_re_mohm_cm2.values + 1j*ca.z_im_mohm_cm2.values)/1e3
    cov = np.array([covmap.get(round(float(x), 6), np.nan) for x in f_l])

    # --- like-for-like corrections ----------------------------------------
    I_full = float((seg.j_dc_A_cm2*seg.area_cm2).sum())*A_CELL_CM2/A_tot
    I_err  = I_full/SETPOINT[cond] - 1.0
    Z_l    = Z_l * (1.0 + I_err)

    sw  = gamry_dta.read_dta(_ASR_GAMRY_DIR/fn).sorted()
    f_g = np.asarray(sw.freq, float); Z_g = np.asarray(sw.Z, complex)*A_CELL_CM2
    Z_g = Z_g - 1j*2*np.pi*f_g*gamry_series_L(f_g, Z_g)

    lo, hi = max(f_l.min(), f_g.min()), min(f_l.max(), f_g.max())
    m = (f_l >= lo) & (f_l <= hi)
    f, Zl, cv = f_l[m], Z_l[m], cov[m]
    Zg = interp_log(f, f_g, Z_g)

    a, b = hfr_band(f, Zg)
    hk   = (f >= a) & (f <= b)
    hk_c = hk & (cv >= MIN_COV)

    print(f"\n════ {cond}  (I_err {100*I_err:+.1f}% applied, band {lo:.2f}-{hi:.0f} Hz) ════")
    for nm, k in [("all frequencies", np.ones_like(f, bool)),
                  ("LF   < 1 Hz",  f < 1),
                  ("mid  1-100 Hz", (f>=1)&(f<100)),
                  (f"HFR  {a:.0f}-{b:.0f} Hz", hk),
                  (f"HFR, coverage>={100*MIN_COV:.0f}%", hk_c)]:
        if k.sum() == 0: continue
        d  = 100*(np.abs(Zl[k])-np.abs(Zg[k]))/np.abs(Zg[k])
        dr = 100*(Zl[k].real-Zg[k].real)/Zg[k].real
        print(f"   {nm:26s} n={k.sum():3d}  |Z| dev {np.median(d):+6.1f}% "
              f"(p95 {np.percentile(np.abs(d),95):5.1f}%)   Re Z dev {np.median(dr):+6.1f}%")

    RsA,nA = r_ohmic_topband(f[hk_c] if hk_c.sum()>3 else f, Zl[hk_c] if hk_c.sum()>3 else Zl)
    RsB,_  = r_ohmic_topband(f[hk_c] if hk_c.sum()>3 else f, Zg[hk_c] if hk_c.sum()>3 else Zg)
    print(f"   R_s  local {1e3*RsA:6.1f}  vs Gamry {1e3*RsB:6.1f} mohm.cm2  "
          f"-> {100*(RsA/RsB-1):+.1f}%   (from {nA} points)")

    print(f"   frequencies below {100*MIN_COV:.0f}% coverage in the HFR band: "
          f"{int((hk & (cv < MIN_COV)).sum())} of {int(hk.sum())}")

# COMMAND ----------

# DBTITLE 1,ECM method comparison per condition (A: AICc pipeline fit vs B: DRT-informed) — saved
# ═══════════════════════════════════════════════════════════════════════════════
# BOTH ECM METHODS, SIDE BY SIDE, ONE SET OF FIGURES PER CONDITION
#
#   A  csv_pipeline.choose_n_arcs  (cell "ECM Fitting"): full band, Rs free,
#      σ-weighted, 1 or 2 arcs by AICc  -> ECM_ALL
#   B  ecm_drt.fit_drt_ecm         (cell "ECM Fit ... DRT-informed"):
#      ≤ 1 kHz, Rs = pipeline R_ohmic, mains harmonics dropped, σ-weighted,
#      1 or 2 arcs by AICc, optional TLM fast arc  -> ECM_FIT_RESULTS
#
# Both are reported in the same terms: arcs sorted by τ (FAST / SLOW),
# R_pol = sum of the arcs, medians over ALL fitted segments.
#
# Saved per condition to <run>/ecm_comparison/ (and copied under
# ECM_COMPARE_DIR if set):
#   ecm_compare_interactive.html  ALL segments, both methods: click a segment
#                             in the legend to toggle it, or pick one / method A
#                             only / method B only / the aggregate in the dropdown
#   ecm_compare_params_interactive.html  A vs B per segment, hover = segment
#   ecm_compare_nyquist.png   6 representative segments + the aggregate
#   ecm_compare_params.png    A vs B per segment for Rs, R_fast, R_slow, R_pol
#   ecm_compare_segments.csv  every segment, both methods
#   ecm_compare_summary.csv   medians / IQR / counts, both methods
# Run the two ECM cells first.
# ═══════════════════════════════════════════════════════════════════════════════
import importlib, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display
import ecm_drt
importlib.reload(ecm_drt)

ECM_COMPARE_DIR = ''          # e.g. '/Volumes/.../EIS_Results/ecm_comparison' (optional copy)
_KEYS = ('Rs', 'R_fast', 'R_slow', 'R_pol')
_COL_A, _COL_B = '#E07B00', '#1F5FA8'


def _row_A(r):
    """Method A result (params in Ohm*cm2) -> fast/slow in mOhm*cm2."""
    p = {k: (1000.0 * v if k.startswith('R') or k == 'L' else v)
         for k, v in r['params'].items()}
    out = ecm_drt.arcs_from_params(p, r['n_arcs'])
    out.update(n_arcs=r['n_arcs'], chi2_nu=r.get('chi2_nu', np.nan),
               verdict=r.get('verdict', ''),
               rms_rel_resid=float(np.sqrt(np.mean((r['res_pct'] / 100) ** 2))))
    return out


def _row_B(e):
    return {k: e.get(k, np.nan) for k in
            ('Rs', 'R_fast', 'tau_fast', 'n_fast', 'R_slow', 'tau_slow', 'n_slow',
             'R_pol', 'n_arcs', 'chi2_nu', 'rms_rel_resid', 'clean', 'fast_element')}


_A_ALL = globals().get('ECM_ALL', {}) or {}
_B_ALL = globals().get('ECM_FIT_RESULTS', {}) or {}
if not _A_ALL or not _B_ALL:
    print('  run both ECM cells first (ECM_ALL and ECM_FIT_RESULTS are needed)')

for cond in [c for c in PIPELINE_RESULTS if c in _A_ALL and c in _B_ALL]:
    out_dir = Path(PIPELINE_RESULTS[cond]['out_dir']) / 'ecm_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    A, B = _A_ALL[cond]['segments'], _B_ALL[cond]
    segs = sorted(set(A) | set(B), key=int)

    # ── per-segment table, both methods ──
    rows = []
    for s in segs:
        ra = _row_A(A[s]) if s in A else {}
        rb = _row_B(B[s]) if s in B else {}
        row = {'segment': int(s)}
        row.update({f'A_{k}': v for k, v in ra.items()})
        row.update({f'B_{k}': v for k, v in rb.items()})
        rows.append(row)
    dseg = pd.DataFrame(rows)
    dseg.to_csv(out_dir / 'ecm_compare_segments.csv', index=False)

    # ── summary: medians over ALL fitted segments (point 7) ──
    srows = []
    for k in ('Rs', 'R_fast', 'tau_fast', 'n_fast', 'R_slow', 'tau_slow', 'R_pol',
              'rms_rel_resid', 'chi2_nu'):
        for m in ('A', 'B'):
            col = f'{m}_{k}'
            v = pd.to_numeric(dseg.get(col, pd.Series(dtype=float)), errors='coerce').dropna()
            srows.append({'parameter': k, 'method': m, 'n': len(v),
                          'median': v.median() if len(v) else np.nan,
                          'q25': v.quantile(.25) if len(v) else np.nan,
                          'q75': v.quantile(.75) if len(v) else np.nan})
    dsum = (pd.DataFrame(srows).pivot(index='parameter', columns='method',
                                      values=['median', 'q25', 'q75', 'n']))
    dsum.to_csv(out_dir / 'ecm_compare_summary.csv')
    n2A = int((dseg.get('A_n_arcs') == 2).sum()) if 'A_n_arcs' in dseg else 0
    n2B = int((dseg.get('B_n_arcs') == 2).sum()) if 'B_n_arcs' in dseg else 0
    ncl = int(dseg.get('B_clean', pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    print(f"\n═══ {cond}: A {len(A)} fitted ({n2A} two-arc) | B {len(B)} fitted "
          f"({n2B} two-arc, {ncl} clean) ═══")
    display(dsum['median'].round(4))

    # ── figure 1: Nyquist, 6 segments spread over R_pol + the aggregate ──
    common = [s for s in segs if s in A and s in B]
    order = sorted(common, key=lambda s: B[s]['R_pol'])
    pick = [order[int(round(i))] for i in np.linspace(0, len(order) - 1, 6)] if order else []
    pick = list(dict.fromkeys(pick))
    agg_A = _A_ALL[cond].get('aggregate')
    agg_B = None
    if agg_A is not None:
        # Rs of the aggregate = the segments' pipeline R_ohmic in parallel,
        # area-weighted (A_cell / sum A_s/Rs_s) -- held fixed as for segments
        _rs_agg = None
        _ssp = Path(PIPELINE_RESULTS[cond]['out_dir']) / 'silver' / 'segments_summary.csv'
        if _ssp.exists():
            _ss = pd.read_csv(_ssp)
            _k = (_ss.R_ohmic_mohm_cm2 > 0) & _ss.area_cm2.notna()
            if _k.any():
                _rs_agg = float(_ss.area_cm2[_k].sum()
                                / (_ss.area_cm2[_k] / _ss.R_ohmic_mohm_cm2[_k]).sum())
        agg_B = ecm_drt.fit_drt_ecm(agg_A['freq'], agg_A['Z_meas'],
                                    rs_fixed=_rs_agg,
                                    f_max_fit_hz=globals().get('ECM_B_FMAX_HZ', 1000.0))
        agg_B = agg_B if agg_B.get('ok') else None
    # ── interactive figure: every segment, both methods, toggle + select ──
    # Same controls as the "ECM Visualization" cell:
    #   legend   click a segment to hide/show it (data + fit A + fit B +
    #            both residual traces move together); double-click isolates it
    #   dropdown All segments, both methods | method A only | method B only |
    #            aggregate | one segment at a time (title shows both fits)
    # Data = markers, method A = solid line, method B = dashed line.
    # Left: Nyquist. Right: |ΔZ|/|Z| residual against frequency.
    # Saved as ecm_compare_interactive.html next to the PNGs.
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    try:
        import plotly.express as _px
        _SEQ = _px.colors.sequential.Viridis
    except Exception:                                     # noqa: BLE001
        _SEQ = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']

    def _clr(v, lo, hi):
        if not np.isfinite(v) or hi <= lo:
            return _SEQ[len(_SEQ) // 2]
        t = min(max((v - lo) / (hi - lo), 0.0), 1.0)
        return _SEQ[int(round(t * (len(_SEQ) - 1)))]

    def _b_curve(rb):
        fs = np.logspace(np.log10(rb['freq'].min()), np.log10(rb['freq'].max()), 300)
        return ecm_drt.model_curve(rb, fs)

    def _desc_A(ra):
        a = ecm_drt.arcs_from_params(
            {k: (1000 * v if k.startswith('R') else v) for k, v in ra['params'].items()},
            ra['n_arcs'])
        return (f"A: Rs={a['Rs']:.1f}, R_fast={a['R_fast']:.1f} (τ {1e3*a['tau_fast']:.3g} ms, "
                f"n {a['n_fast']:.2f}), R_slow={a['R_slow']:.1f}, R_pol={a['R_pol']:.1f}, "
                f"{ra['n_arcs']} arc, χ²ν={ra['chi2_nu']:.3g}")

    def _desc_B(rb):
        if rb is None:
            return 'B: no fit'
        return (f"B: Rs={rb['Rs']:.1f}{' (fixed)' if rb['Rs_fixed'] else ''}, "
                f"R_fast={rb['R_fast']:.1f} (τ {1e3*rb['tau_fast']:.3g} ms, n {rb['n_fast']:.2f}"
                f"{', TLM' if rb['fast_element'] == 'tlm' else ''}), "
                f"R_slow={rb['R_slow']:.1f}, R_pol={rb['R_pol']:.1f}, {rb['n_arcs']} arc"
                + (f", χ²ν={rb['chi2_nu']:.3g}" if rb['weight'] == 'sigma'
                   else f", rms {100*rb['rms_rel_resid']:.1f} %")
                + ('' if rb['clean'] else f" — flagged: {'; '.join(rb['flags'])}"))

    rp = [B[s]['R_pol'] for s in common]
    lo_c, hi_c = (np.percentile(rp, [5, 95]) if rp else (0, 1))
    figi = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44],
                         subplot_titles=['Nyquist — data, method A (solid), method B (dashed)',
                                         'Residual |ΔZ|/|Z|  (A solid, B dashed)'],
                         horizontal_spacing=0.09)
    owner = []          # 'seg:<n>:<data|A|B>' | 'agg:<data|A|B>'

    def _add(o, tr, col):
        figi.add_trace(tr, row=1, col=col)
        owner.append(o)

    for s in common:
        ra, rb = A[s], B[s]
        c = _clr(rb['R_pol'], lo_c, hi_c)
        g = f'seg{s}'
        _add(f'seg:{s}:data', go.Scatter(
            x=ra['Z_meas'].real, y=-ra['Z_meas'].imag, mode='markers',
            marker=dict(size=5, color=c, opacity=0.75, line=dict(width=0.4, color='#333')),
            name=f'Seg {s}', legendgroup=g, showlegend=True,
            hovertemplate=f"<b>Seg {s}</b><br>Z'=%{{x:.1f}}<br>-Z''=%{{y:.1f}} mΩ·cm²<extra>data</extra>"), 1)
        _add(f'seg:{s}:A', go.Scatter(
            x=ra['Z_fit_smooth'].real, y=-ra['Z_fit_smooth'].imag, mode='lines',
            line=dict(color=c, width=1.6), name=f'Seg {s} A', legendgroup=g,
            showlegend=False, hovertemplate=f"Seg {s}<br>{_desc_A(ra)}<extra>method A</extra>"), 1)
        Zb = _b_curve(rb)
        _add(f'seg:{s}:B', go.Scatter(
            x=Zb.real, y=-Zb.imag, mode='lines', line=dict(color=c, width=1.8, dash='dash'),
            name=f'Seg {s} B', legendgroup=g, showlegend=False,
            hovertemplate=f"Seg {s}<br>{_desc_B(rb)}<extra>method B</extra>"), 1)
        _add(f'seg:{s}:A', go.Scatter(
            x=ra['freq'], y=ra['res_pct'], mode='lines+markers',
            marker=dict(size=3, color=c), line=dict(width=1, color=c),
            name=f'Seg {s} res A', legendgroup=g, showlegend=False,
            hovertemplate=f'Seg {s} A<br>f=%{{x:.3g}} Hz<br>|res|=%{{y:.2f}} %<extra></extra>'), 2)
        _add(f'seg:{s}:B', go.Scatter(
            x=rb['freq'], y=rb['res_pct'], mode='lines+markers',
            marker=dict(size=3, color=c, symbol='x'), line=dict(width=1, color=c, dash='dash'),
            name=f'Seg {s} res B', legendgroup=g, showlegend=False,
            hovertemplate=f'Seg {s} B<br>f=%{{x:.3g}} Hz<br>|res|=%{{y:.2f}} %<extra></extra>'), 2)

    if agg_A is not None:
        _add('agg:data', go.Scatter(
            x=agg_A['Z_meas'].real, y=-agg_A['Z_meas'].imag, mode='markers',
            marker=dict(size=8, color='#c0392b', symbol='circle-open', line=dict(width=2)),
            name='Aggregate (whole cell)', legendgroup='agg',
            hovertemplate="<b>Aggregate</b><br>Z'=%{x:.1f}<br>-Z''=%{y:.1f} mΩ·cm²<extra></extra>"), 1)
        _add('agg:A', go.Scatter(
            x=agg_A['Z_fit_smooth'].real, y=-agg_A['Z_fit_smooth'].imag, mode='lines',
            line=dict(color=_COL_A, width=3), name='Aggregate — method A', legendgroup='agg',
            hovertemplate=f"{_desc_A(agg_A)}<extra>aggregate A</extra>"), 1)
        _add('agg:A', go.Scatter(
            x=agg_A['freq'], y=agg_A['res_pct'], mode='lines+markers',
            marker=dict(size=5, color=_COL_A), line=dict(width=2, color=_COL_A),
            name='Aggregate res A', legendgroup='agg', showlegend=False), 2)
        if agg_B is not None:
            Zb = _b_curve(agg_B)
            _add('agg:B', go.Scatter(
                x=Zb.real, y=-Zb.imag, mode='lines', line=dict(color=_COL_B, width=3, dash='dash'),
                name='Aggregate — method B', legendgroup='agg',
                hovertemplate=f"{_desc_B(agg_B)}<extra>aggregate B</extra>"), 1)
            _add('agg:B', go.Scatter(
                x=agg_B['freq'], y=agg_B['res_pct'], mode='lines+markers',
                marker=dict(size=5, color=_COL_B, symbol='x'),
                line=dict(width=2, color=_COL_B, dash='dash'),
                name='Aggregate res B', legendgroup='agg', showlegend=False), 2)

    owner = np.array(owner)

    def _mask(pred):
        return [bool(pred(o)) for o in owner]

    def _ttl(t, sub=''):
        return {'title.text': f'<b>{t} — Leepa {LEEPA}, {cond}</b>'
                              + (f'<br><sup>{sub}</sup>' if sub else '')}

    btn = [
        dict(label=f'All segments — A + B ({len(common)})', method='update',
             args=[{'visible': _mask(lambda o: o.startswith('seg:'))},
                   _ttl(f'ECM comparison — all {len(common)} segments, method A (solid) vs B (dashed)')]),
        dict(label='All segments — method A only', method='update',
             args=[{'visible': _mask(lambda o: o.startswith('seg:') and not o.endswith(':B'))},
                   _ttl('ECM method A (pipeline AICc fit, full band) — all segments')]),
        dict(label='All segments — method B only', method='update',
             args=[{'visible': _mask(lambda o: o.startswith('seg:') and not o.endswith(':A'))},
                   _ttl('ECM method B (DRT-informed, ≤1 kHz, Rs fixed) — all segments')]),
        dict(label='All segments + aggregate', method='update',
             args=[{'visible': _mask(lambda o: True)},
                   _ttl('ECM comparison — all segments + whole-cell aggregate')]),
    ]
    if agg_A is not None:
        btn.append(dict(label='Aggregate only', method='update',
                        args=[{'visible': _mask(lambda o: o.startswith('agg'))},
                              _ttl('ECM comparison — whole-cell aggregate',
                                   _desc_A(agg_A) + '<br>' + _desc_B(agg_B))]))
    for s in common:
        btn.append(dict(label=f'Seg {s}', method='update',
                        args=[{'visible': _mask(lambda o, s=s: o.startswith(f'seg:{s}:'))},
                              _ttl(f'ECM comparison — segment {s}',
                                   _desc_A(A[s]) + '<br>' + _desc_B(B[s]))]))

    figi.add_hline(y=2, line=dict(color='green', dash='dash', width=1), row=1, col=2)
    figi.add_hline(y=5, line=dict(color='orange', dash='dash', width=1), row=1, col=2)
    figi.update_xaxes(title_text="Z' [mΩ·cm²]", showgrid=True, gridcolor='#eee', row=1, col=1)
    figi.update_yaxes(title_text="-Z'' [mΩ·cm²]", showgrid=True, gridcolor='#eee',
                      scaleanchor='x', scaleratio=1, row=1, col=1)
    figi.update_xaxes(title_text='f [Hz]', type='log', showgrid=True, gridcolor='#eee', row=1, col=2)
    figi.update_yaxes(title_text='|ΔZ|/|Z| [%]', rangemode='tozero', showgrid=True,
                      gridcolor='#eee', row=1, col=2)
    figi.update_layout(
        title=dict(text=f'<b>ECM comparison — all {len(common)} segments, method A (solid) '
                        f'vs B (dashed) — Leepa {LEEPA}, {cond}</b>'),
        height=700, width=1500, paper_bgcolor='white', plot_bgcolor='white',
        hovermode='closest', margin=dict(t=140, r=250),
        legend=dict(font=dict(size=9), y=0.5, yanchor='middle', tracegroupgap=0,
                    title=dict(text='click = toggle<br>double-click = isolate')),
        # dropdown on the right, so it never covers the title
        updatemenus=[dict(buttons=btn, direction='down', showactive=True, x=1.0,
                          xanchor='right', y=1.2, yanchor='top', bgcolor='white',
                          bordercolor='#999')])
    for i, o in enumerate(owner):
        figi.data[i].visible = o.startswith('seg:')
    figi.write_html(str(out_dir / 'ecm_compare_interactive.html'), include_plotlyjs='cdn')
    show_fig(figi, html=True)

    # ── interactive parameter comparison: hover names the segment ──
    figp = make_subplots(rows=1, cols=4, subplot_titles=[
        f'{k}  (median A {pd.to_numeric(dseg.get(f"A_{k}"), errors="coerce").median():.1f} | '
        f'B {pd.to_numeric(dseg.get(f"B_{k}"), errors="coerce").median():.1f})' for k in _KEYS],
        horizontal_spacing=0.06)
    _cl = dseg.get('B_clean', pd.Series(False, index=dseg.index)).astype(str).str.lower().eq('true')
    for j, k in enumerate(_KEYS, start=1):
        a = pd.to_numeric(dseg.get(f'A_{k}'), errors='coerce')
        b = pd.to_numeric(dseg.get(f'B_{k}'), errors='coerce')
        ok = a.notna() & b.notna()
        for sel, nm, sym in ((ok & _cl, 'B clean', 'circle'), (ok & ~_cl, 'B flagged', 'circle-open')):
            figp.add_trace(go.Scatter(
                x=a[sel], y=b[sel], mode='markers', name=nm, legendgroup=nm,
                showlegend=(j == 1), marker=dict(size=8, color=_COL_B, symbol=sym),
                text=dseg.segment[sel],
                hovertemplate=f'Seg %{{text}}<br>{k} A=%{{x:.1f}}<br>{k} B=%{{y:.1f}} mΩ·cm²<extra></extra>'),
                row=1, col=j)
        if ok.any():
            lo_, hi_ = float(min(a[ok].min(), b[ok].min())), float(max(a[ok].max(), b[ok].max()))
            figp.add_trace(go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode='lines',
                                      line=dict(color='#999', width=1), name='1:1',
                                      legendgroup='1:1', showlegend=(j == 1)), row=1, col=j)
        figp.update_xaxes(title_text=f'{k}, method A [mΩ·cm²]', row=1, col=j,
                          showgrid=True, gridcolor='#eee')
        figp.update_yaxes(title_text=f'{k}, method B [mΩ·cm²]', row=1, col=j,
                          showgrid=True, gridcolor='#eee')
    figp.update_layout(title=f'<b>ECM parameters per segment, A vs B — Leepa {LEEPA}, {cond}</b>',
                       height=480, width=1600, paper_bgcolor='white', plot_bgcolor='white')
    figp.write_html(str(out_dir / 'ecm_compare_params_interactive.html'), include_plotlyjs='cdn')
    show_fig(figp, html=True)

    panels = [(f'Segment {s}', A[s], B[s]) for s in pick]
    if agg_A is not None:
        panels.append(('Aggregate (whole plate)', agg_A, agg_B))
    nc = 4
    nr = int(np.ceil(len(panels) / nc)) or 1
    fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.6 * nr))
    axs = np.atleast_1d(axs).ravel()
    for ax, (ttl, ra, rb) in zip(axs, panels):
        Zm = ra['Z_meas']
        ax.plot(Zm.real, -Zm.imag, 'o', ms=3.5, color='0.45', label='data')
        ax.plot(ra['Z_fit_smooth'].real, -ra['Z_fit_smooth'].imag, '-', lw=2,
                color=_COL_A, label=f"A: {ra['n_arcs']} arc(s), full band")
        if rb is not None:
            fs = np.logspace(np.log10(rb['freq'].min()), np.log10(rb['freq'].max()), 300)
            Zb = ecm_drt.model_curve(rb, fs)
            ax.plot(Zb.real, -Zb.imag, '--', lw=2, color=_COL_B,
                    label=f"B: {rb['n_arcs']} arc(s){' TLM' if rb['fast_element']=='tlm' else ''}, ≤{rb['f_fit_hz'][1]:.0f} Hz")
        a_row = ecm_drt.arcs_from_params(
            {k: (1000 * v if k.startswith('R') else v) for k, v in ra['params'].items()},
            ra['n_arcs'])
        txt = (f"A: Rs {a_row['Rs']:.1f}  R_pol {a_row['R_pol']:.1f}"
               + (f"\nB: Rs {rb['Rs']:.1f}  R_pol {rb['R_pol']:.1f}" if rb else ''))
        ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.5, family='monospace',
                bbox=dict(fc='white', ec='0.8', alpha=0.85))
        ax.set_title(ttl, fontsize=11, fontweight='bold')
        ax.set_xlabel("Z' [mΩ·cm²]"); ax.set_ylabel("-Z'' [mΩ·cm²]")
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, color='0.9')
        ax.legend(fontsize=7.5, loc='upper left')
    for ax in axs[len(panels):]:
        ax.axis('off')
    fig.suptitle(f'ECM method comparison — Leepa {LEEPA}, {cond}', fontsize=15,
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'ecm_compare_nyquist.png', dpi=170, bbox_inches='tight')
    display(fig); plt.close(fig)

    # ── figure 2: parameters, A vs B for every segment ──
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, k in zip(axs, _KEYS):
        a = pd.to_numeric(dseg.get(f'A_{k}'), errors='coerce')
        b = pd.to_numeric(dseg.get(f'B_{k}'), errors='coerce')
        cl = dseg.get('B_clean', pd.Series(False, index=dseg.index)).fillna(False).astype(bool)
        ok = a.notna() & b.notna()
        ax.scatter(a[ok & cl], b[ok & cl], s=22, color=_COL_B, label='B clean')
        ax.scatter(a[ok & ~cl], b[ok & ~cl], s=22, facecolor='none',
                   edgecolor=_COL_B, label='B flagged')
        if ok.any():
            lo = float(min(a[ok].min(), b[ok].min()))
            hi = float(max(a[ok].max(), b[ok].max()))
            pad = 0.05 * (hi - lo or 1)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '-', color='0.6',
                    lw=1, label='1:1')
            ax.set_title(f'{k}   median A {a[ok].median():.1f} | B {b[ok].median():.1f}',
                         fontsize=10.5, fontweight='bold')
        ax.set_xlabel(f'{k}, method A [mΩ·cm²]')
        ax.set_ylabel(f'{k}, method B [mΩ·cm²]')
        ax.grid(True, color='0.9')
    axs[0].legend(fontsize=8)
    fig.suptitle(f'ECM parameters per segment, A vs B — Leepa {LEEPA}, {cond}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'ecm_compare_params.png', dpi=170, bbox_inches='tight')
    display(fig); plt.close(fig)

    if ECM_COMPARE_DIR:
        dst = Path(ECM_COMPARE_DIR) / str(LEEPA) / cond
        dst.mkdir(parents=True, exist_ok=True)
        for f in out_dir.iterdir():
            shutil.copy2(f, dst / f.name)
        print(f'  copied to {dst}')
    print(f'  saved: {out_dir}')
