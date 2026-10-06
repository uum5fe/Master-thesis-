#!/usr/bin/env python3
"""
polcurve.py
===========
Polarisation curves -- cell voltage against current density -- from the two
data sources this pipeline already reads, drawn in the bench's comparison
style (bold title, "CVM1 - Voltage [mV]" over "Current density [A/cm²]",
boxed legend, the voltage printed at the labelled current densities).

    FAMOS (local EIS)   one point per load condition. Voltage = DC level of
                        the cell-voltage reference channel (UC2) on every
                        card; current = the plate's measured segment
                        currents. Because the plate is segmented, the same
                        voltage also gives a LOCAL curve per flow band (air
                        outlet / middle / air inlet) -- the part a
                        whole-cell curve cannot show.
    Gamry (whole cell)  one point per .dta sweep. Voltage = the Vdc column
                        of the ZCURVE table; current = the set point in the
                        file name ("..._CurrVal_150.dta"), because the
                        sweeps run with IDCREQ = 0 (the bench holds the DC,
                        the potentiostat adds only the AC) so Idc is ~0.
                        HFR from the same sweep, for an iR-free curve.

A curve from any other source (a bench export, a reference cell) can be
overlaid with `load_curve_csv`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from config import A_CELL_CM2
except Exception:                                           # noqa: BLE001
    A_CELL_CM2 = 304.92

# Excel's default series colours, as in the bench comparison plots.
PC_COLORS = ["#00B0F0", "#92D050", "#FFC000", "#7030A0", "#FF0000",
             "#0070C0", "#00B050", "#C00000"]

# Flow bands of the gen1 plate along x (mm): air leaves on the left, enters
# on the right (counter-flow, see config.FLOW_DESCRIPTION).
FLOW_BANDS = (("air outlet band", 0.0, 84.0),
              ("middle band", 84.0, 168.0),
              ("air inlet band", 168.0, 252.0))


@dataclass
class PolCurve:
    label: str
    j: np.ndarray                     # A/cm2
    v_mV: np.ndarray                  # cell voltage, mV
    dashed: bool = False
    marker: str = "^"
    color: str | None = None
    extra: dict = field(default_factory=dict)   # e.g. hfr_mohm_cm2, conds
    show_labels: bool = True                    # print V at `label_at`

    def sorted(self) -> "PolCurve":
        o = np.argsort(self.j)
        ex = {k: (np.asarray(v)[o] if np.ndim(v) == 1 and len(v) == len(o) else v)
              for k, v in self.extra.items()}
        return PolCurve(self.label, np.asarray(self.j)[o],
                        np.asarray(self.v_mV)[o], self.dashed, self.marker,
                        self.color, ex, self.show_labels)

    def ir_free(self) -> "PolCurve":
        """V + j*HFR (HFR in mOhm*cm2 -> mV = A/cm2 * mOhm*cm2)."""
        hfr = np.asarray(self.extra.get("hfr_mohm_cm2", []), float)
        if hfr.size != np.size(self.j):
            raise ValueError(f"{self.label}: no HFR per point")
        return PolCurve(self.label + " (iR-free)", self.j,
                        np.asarray(self.v_mV) + np.asarray(self.j) * hfr,
                        True, self.marker, self.color)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_polcurves(curves: list[PolCurve], title: str = "Comparison Polcurves",
                   ylabel: str = "CVM1 - Voltage [mV]",
                   xlabel: str = "Current density [A/cm²]",
                   xlim=None, ylim=(500, 1000), label_at=(0.1, 1.5),
                   label_tol: float = 0.12, figsize=(11, 7.6)):
    """The bench comparison style. Returns the figure.

    label_at   current densities whose voltage is printed next to the curve
               (the nearest measured point within `label_tol` A/cm2)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Arial", "Liberation Sans", "Helvetica")
                   if f in have), "DejaVu Sans")
    bold = dict(fontweight="bold", family=family)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    jmax = max((float(np.nanmax(c.j)) for c in curves if np.size(c.j)),
               default=2.0)
    if xlim is None:
        xlim = (0.0, max(0.5, np.ceil((jmax + 0.1) / 0.25) * 0.25))
    if ylim is None:
        vs = np.concatenate([np.asarray(c.v_mV, float) for c in curves])
        vs = vs[np.isfinite(vs)]
        ylim = (np.floor(vs.min() / 50) * 50 - 50, np.ceil(vs.max() / 50) * 50 + 50)

    for i, c in enumerate(curves):
        c = c.sorted()
        col = c.color or PC_COLORS[i % len(PC_COLORS)]
        ax.plot(c.j, c.v_mV, ls="--" if c.dashed else "-", lw=2.4,
                marker=c.marker, ms=8, color=col, label=c.label, zorder=3)
        for target in (label_at or ()) if c.show_labels else ():
            if not np.size(c.j):
                continue
            k = int(np.argmin(np.abs(c.j - target)))
            if abs(c.j[k] - target) <= label_tol:
                ax.annotate(f"{c.v_mV[k]:.0f}", (c.j[k], c.v_mV[k]),
                            xytext=(10, -8 - 16 * (i % 3)),
                            textcoords="offset points", color=col,
                            fontsize=17, zorder=4, **bold)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1e-9, 0.25))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1e-9, 50))
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.grid(True, color="#d9d9d9", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_color("#bfbfbf")
    ax.tick_params(labelsize=19, width=1.4, length=5)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
        lab.set_family(family)
    ax.set_xlabel(xlabel, fontsize=28, **bold)
    ax.set_ylabel(ylabel, fontsize=28, **bold)
    ax.set_title(title, fontsize=30, pad=18, **bold)
    leg = ax.legend(loc="upper right", fontsize=14, frameon=True,
                    fancybox=False, edgecolor="black", framealpha=1)
    leg.get_frame().set_linewidth(1.6)
    for t in leg.get_texts():
        t.set_fontweight("bold")
        t.set_family(family)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gamry
# ---------------------------------------------------------------------------
_CURRVAL_RE = re.compile(r"CurrVal[_-]?(\d+(?:[.,]\d+)?)", re.I)


def read_gamry_dc(path) -> dict:
    """Vdc, Idc and HFR of one Gamry EIS sweep (.dta)."""
    path = Path(path)
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("ZCURVE")),
                 None)
    if start is None:
        raise ValueError(f"{path.name}: no ZCURVE table")
    head = [h.strip().lower() for h in lines[start + 1].split("\t")]
    col = {h: i for i, h in enumerate(head) if h}
    vdc, idc = [], []
    for ln in lines[start + 3:]:
        c = ln.split("\t")
        if len(c) < 6 or not c[1].strip().lstrip("+-").isdigit():
            continue
        try:
            if "vdc" in col:
                vdc.append(float(c[col["vdc"]].replace(",", ".")))
            if "idc" in col:
                idc.append(float(c[col["idc"]].replace(",", ".")))
        except (ValueError, IndexError):
            continue
    m = _CURRVAL_RE.search(path.stem)
    setpoint = float(m.group(1).replace(",", ".")) if m else np.nan
    hfr, hfr_from = np.nan, ""
    try:
        import gamry_compare
        sw = gamry_compare.read_cell_sweep(path)
        hfr = float(sw.hfr_ohm())
        hfr_from = "measured" if np.isfinite(hfr) else ""
        if not np.isfinite(hfr):
            # still capacitive at the top of the sweep: extrapolated intercept
            hfr = float(gamry_compare._hf_extrapolated(sw.freq, sw.Z_ohm))
            hfr_from = "extrapolated" if np.isfinite(hfr) else ""
    except Exception:                                       # noqa: BLE001
        pass
    return {"file": path.name, "setpoint_A": setpoint,
            "vdc_V": float(np.median(vdc)) if vdc else np.nan,
            "vdc_sd_mV": float(1e3 * np.std(vdc)) if len(vdc) > 1 else np.nan,
            "idc_A": float(np.median(idc)) if idc else np.nan,
            "hfr_ohm": hfr, "hfr_from": hfr_from}


def gamry_polcurve(dta_files, label: str = "Gamry (whole cell)",
                   area_cm2: float = A_CELL_CM2, n_cells: int = 1,
                   color: str | None = None):
    """One point per sweep -> (PolCurve, table rows).

    Current: the measured Idc when it carries the load (>= 30 % of the set
    point), else the set point from the file name (IDCREQ = 0 sweeps).
    """
    rows = []
    for f in dta_files:
        try:
            r = read_gamry_dc(f)
        except Exception as exc:                            # noqa: BLE001
            rows.append({"file": Path(f).name, "error": str(exc)})
            continue
        sp, idc = r["setpoint_A"], r["idc_A"]
        use_idc = np.isfinite(idc) and np.isfinite(sp) and abs(idc) >= 0.3 * sp
        i_a = abs(idc) if use_idc else sp
        r["current_A"], r["current_from"] = i_a, ("Idc" if use_idc else "file name")
        r["j_A_cm2"] = i_a / area_cm2 if np.isfinite(i_a) else np.nan
        r["v_cell_mV"] = 1e3 * abs(r["vdc_V"]) / n_cells
        r["hfr_mohm_cm2"] = 1e3 * r["hfr_ohm"] * area_cm2
        rows.append(r)
    ok = [r for r in rows if "error" not in r and np.isfinite(r["j_A_cm2"])
          and np.isfinite(r["v_cell_mV"])]
    curve = PolCurve(label, np.array([r["j_A_cm2"] for r in ok]),
                     np.array([r["v_cell_mV"] for r in ok]), False, "D", color,
                     {"hfr_mohm_cm2": np.array([r["hfr_mohm_cm2"] for r in ok])})
    return curve.sorted(), rows


# ---------------------------------------------------------------------------
# FAMOS (local EIS)
# ---------------------------------------------------------------------------
def famos_cell_voltage(files, ref_channel: str = "UC2", stride: int = 25) -> dict:
    """DC level of the cell-voltage channel on every card -> median, spread."""
    import eis_local
    per_card = {}
    for fp in files:
        fam = eis_local.FamosFile(Path(fp))
        name = ref_channel if ref_channel in fam.uc_names else (
            fam.uc_names[0] if fam.uc_names else None)
        if name is None:
            continue
        x = fam.channel(name)[::max(1, stride)]
        per_card[Path(fp).stem] = (name, float(np.nanmean(x)))
    v = np.array([u for _, u in per_card.values()], float)
    return {"v_V": float(np.median(v)) if v.size else np.nan,
            "spread_mV": float(1e3 * np.ptp(v)) if v.size > 1 else np.nan,
            "per_card": per_card}


def famos_current_density(out_dir) -> dict:
    """Plate and flow-band current density from a run's gold outputs."""
    out_dir = Path(out_dir)
    gold = out_dir / "gold" if (out_dir / "gold").exists() else out_dir
    res = {"plate": np.nan}
    man = gold / "gold_manifest.json"
    if man.exists():
        dc = json.loads(man.read_text()).get("dc_closure", {})
        if dc.get("area_measured_cm2"):
            res["plate"] = dc["I_measured_A"] / dc["area_measured_cm2"]
    summ = gold / "plate_summary.csv"
    if summ.exists():
        import csv
        rows = [r for r in csv.DictReader(open(summ, newline="", encoding="utf-8"))
                if r.get("class", "measured") == "measured" and r.get("j_dc")]
        for name, x0, x1 in FLOW_BANDS:
            sel = [r for r in rows if x0 <= float(r["cx_mm"]) < x1 + 1e-9]
            a = sum(float(r["area_cm2"]) for r in sel)
            if a > 0:
                res[name] = sum(float(r["j_dc"]) * float(r["area_cm2"])
                                for r in sel) / a
        if not np.isfinite(res["plate"]) and rows:
            a = sum(float(r["area_cm2"]) for r in rows)
            res["plate"] = sum(float(r["j_dc"]) * float(r["area_cm2"])
                               for r in rows) / a
    return res


def famos_polcurves(runs: dict, ref_channel: str = "UC2",
                    label: str = "FAMOS (local EIS)", bands: bool = True,
                    v_override: dict | None = None):
    """runs: condition -> (list of FAMOS files, run out_dir).

    Returns ([plate curve, band curves...], table rows). `v_override`
    (condition -> cell voltage in V, e.g. from the bench log) replaces the
    FAMOS channel where the reference channel does not carry the DC level.
    """
    rows = []
    for cond, (files, out_dir) in runs.items():
        v = famos_cell_voltage(files, ref_channel) if files else {"v_V": np.nan}
        if v_override and cond in v_override:
            v = {**v, "v_V": float(v_override[cond]), "source": "override"}
        j = famos_current_density(out_dir) if out_dir else {"plate": np.nan}
        rows.append({"condition": cond, "v_cell_mV": 1e3 * v["v_V"],
                     "card_spread_mV": v.get("spread_mV", np.nan),
                     "j_plate_A_cm2": j.get("plate", np.nan),
                     "j_outlet_A_cm2": j.get("air outlet band", np.nan),
                     "j_middle_A_cm2": j.get("middle band", np.nan),
                     "j_inlet_A_cm2": j.get("air inlet band", np.nan),
                     "n_cards": len(v.get("per_card", {}))})
    ok = [r for r in rows if np.isfinite(r["v_cell_mV"])]
    curves = [PolCurve(label + " — plate", np.array([r["j_plate_A_cm2"] for r in ok]),
                       np.array([r["v_cell_mV"] for r in ok]), False, "D",
                       PC_COLORS[0], {"conds": [r["condition"] for r in ok]})]
    if bands:
        keys = [("air outlet band", "j_outlet_A_cm2", PC_COLORS[2], "s"),
                ("middle band", "j_middle_A_cm2", "#A6A6A6", "o"),
                ("air inlet band", "j_inlet_A_cm2", PC_COLORS[1], "^")]
        for name, key, colr, mk in keys:
            jj = np.array([r.get(key, np.nan) for r in ok], float)
            if np.isfinite(jj).any():
                curves.append(PolCurve(f"{name}", jj,
                                       np.array([r["v_cell_mV"] for r in ok]),
                                       True, mk, colr, show_labels=False))
    return [c.sorted() for c in curves], rows


# ---------------------------------------------------------------------------
# any other source
# ---------------------------------------------------------------------------
def load_curve_csv(path, label: str | None = None, j_col: str | None = None,
                   v_col: str | None = None, dashed: bool = True) -> PolCurve:
    """A curve from a CSV (e.g. a bench export or a reference cell).

    Columns are found by name: current density ("j", "A/cm", "current
    density") and voltage ("V", "mV", "voltage", "CVM"); volts are converted
    to mV when all values are below 2.
    """
    import pandas as pd
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}
    j_col = j_col or next((cols[c] for c in cols if "a/cm" in c or c in ("j", "j_a_cm2")
                           or "current density" in c), df.columns[0])
    v_col = v_col or next((cols[c] for c in cols if "volt" in c or "cvm" in c
                           or c in ("v", "u", "mv", "v_mv")), df.columns[1])
    j = pd.to_numeric(df[j_col], errors="coerce").to_numpy(float)
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(float)
    if np.nanmax(np.abs(v)) < 2.0:
        v = v * 1e3
    m = np.isfinite(j) & np.isfinite(v)
    return PolCurve(label or Path(path).stem, j[m], v[m], dashed, "^").sorted()
