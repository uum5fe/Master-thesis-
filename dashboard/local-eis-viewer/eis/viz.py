"""Figures: spectra with error bars, plate maps, and synchronisation diagnostics.

Matplotlib only, so the figures render anywhere the pipeline runs (including a
headless cluster).  The uncertainty computed in :mod:`eis.spectra` is shown
rather than hidden: error bars on the spectra, and a hatch over segments whose
fit is too poorly determined to be read as a number.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def default_segment_grid(
    segments: list[int], plate_w_cm: float, plate_h_cm: float,
    n_cols: int | None = None, n_rows: int | None = None,
) -> dict[int, tuple[float, float, float, float]]:
    """A plain row-major grid, for when no real geometry is configured."""
    n = len(segments)
    if not n:
        return {}
    if n_cols is None or n_rows is None:
        n_cols = int(np.ceil(np.sqrt(n * plate_w_cm / max(plate_h_cm, 1e-9))))
        n_cols = max(1, min(n_cols, n))
        n_rows = int(np.ceil(n / n_cols))
    w, h = plate_w_cm / n_cols, plate_h_cm / n_rows
    coords = {}
    for i, seg in enumerate(sorted(segments)):
        r, c = divmod(i, n_cols)
        coords[seg] = (
            (c + 0.5) * w, plate_h_cm - (r + 0.5) * h, 0.45 * w, 0.45 * h
        )
    return coords


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def plot_spectra(result, cfg, segments: list[int] | None = None, max_segments: int = 12):
    """Nyquist + Bode magnitude + phase, with per-point uncertainty."""
    chosen = segments or sorted(result.segments)[:max_segments]
    chosen = [s for s in chosen if s in result.segments][:max_segments]
    if not chosen:
        return None
    asr = cfg.geometry.segment_area_cm2 * 1e3          # Ohm -> mOhm*cm^2

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colours = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(chosen)))

    for colour, seg in zip(colours, chosen):
        r = result.segments[seg]
        f, Z, sigma = r.spectrum.f, r.spectrum.Z, r.spectrum.sigma_rel
        err = np.abs(Z) * sigma * asr
        label = f"seg {seg} (K{r.card})"

        axes[0].errorbar(
            Z.real * asr, -Z.imag * asr, xerr=err, yerr=err,
            fmt="o-", ms=3.5, lw=1.2, color=colour, label=label,
            elinewidth=0.7, capsize=1.5, alpha=0.9,
        )
        axes[1].errorbar(
            f, np.abs(Z) * asr, yerr=err, fmt="o-", ms=3, lw=1.2,
            color=colour, elinewidth=0.7, alpha=0.9,
        )
        axes[2].errorbar(
            f, np.degrees(np.angle(Z)), yerr=np.degrees(sigma),
            fmt="o-", ms=3, lw=1.2, color=colour, elinewidth=0.7, alpha=0.9,
        )
        if r.ecm is not None and len(r.ecm.Z_fit):
            axes[0].plot(
                r.ecm.Z_fit.real * asr, -r.ecm.Z_fit.imag * asr,
                "--", lw=1.0, color="0.25", alpha=0.8,
            )

    axes[0].set_xlabel(r"$Z'$  [m$\Omega\cdot$cm$^2$]")
    axes[0].set_ylabel(r"$-Z''$  [m$\Omega\cdot$cm$^2$]")
    axes[0].set_title("Nyquist (dashed = ECM fit)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].axhline(0, color="0.7", lw=0.8, ls=":")

    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("f [Hz]"); axes[1].set_ylabel(r"$|Z|$ [m$\Omega\cdot$cm$^2$]")
    axes[1].set_title("Bode magnitude")

    axes[2].set_xscale("log")
    axes[2].set_xlabel("f [Hz]"); axes[2].set_ylabel("phase [deg]")
    axes[2].set_title("Bode phase")
    axes[2].axhline(0, color="0.7", lw=0.8, ls=":")

    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle(
        f"{result.measurement_id} / {result.condition}  -  "
        f"{len(result.segments)} segments, {result.segments[chosen[0]].spectrum.method} "
        f"estimate ({result.segments[chosen[0]].spectrum.estimator})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plate map
# ---------------------------------------------------------------------------

def plot_plate(
    values: dict[int, float], cfg, title: str, unit: str = "",
    uncertain: set[int] | None = None, cmap: str = "viridis",
):
    """Spatial map of one scalar per segment.

    Segments flagged as poorly determined are hatched rather than dropped, so a
    reader can see that a value exists but should not be trusted.
    """
    coords = cfg.geometry.segment_coords or default_segment_grid(
        sorted(values), cfg.geometry.plate_w_cm, cfg.geometry.plate_h_cm
    )
    usable = {s: v for s, v in values.items() if s in coords and np.isfinite(v)}
    if not usable:
        return None

    array = np.array(list(usable.values()), float)
    norm = Normalize(vmin=float(array.min()), vmax=float(array.max()))
    colours = plt.get_cmap(cmap)
    uncertain = uncertain or set()

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.add_patch(Rectangle(
        (0, 0), cfg.geometry.plate_w_cm, cfg.geometry.plate_h_cm,
        facecolor="#EEE8DC", edgecolor="#7a6a4a", lw=1.6, zorder=0,
    ))
    for seg, value in usable.items():
        x, y, hw, hh = coords[seg]
        ax.add_patch(Rectangle(
            (x - hw, y - hh), 2 * hw, 2 * hh,
            facecolor=colours(norm(value)), edgecolor="white", lw=1.2, zorder=2,
            hatch="///" if seg in uncertain else None,
        ))
        ax.text(
            x, y, f"{seg}\n{value:.3g}", ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold", zorder=3,
        )

    sm = plt.cm.ScalarMappable(cmap=colours, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(unit or title)
    ax.set_xlim(-0.2, cfg.geometry.plate_w_cm + 0.2)
    ax.set_ylim(-0.2, cfg.geometry.plate_h_cm + 0.2)
    ax.set_aspect("equal"); ax.axis("off")
    subtitle = "  (hatched = poorly determined)" if uncertain else ""
    ax.set_title(title + subtitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def plot_sync(result):
    """Measured skew per card and the phase error it would have caused."""
    frame = result.sync.to_frame()
    if frame.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cards = frame["card"].to_numpy()
    tau = frame["tau_ns"].to_numpy(float)
    err = np.nan_to_num(frame["tau_sigma_ns"].to_numpy(float), nan=0.0, posinf=0.0)
    axes[0].bar(cards, tau, yerr=err, color="#3b6ea5", capsize=4)
    axes[0].set_xlabel("card"); axes[0].set_ylabel(r"measured skew $\tau$ [ns]")
    axes[0].set_title("Inter-card skew (from the shared cell voltage)")
    axes[0].grid(alpha=0.3, axis="y")

    f = np.logspace(0, np.log10(5000), 200)
    for card, t_ns in zip(cards, tau):
        if t_ns:
            axes[1].plot(f, 360.0 * f * t_ns * 1e-9, label=f"card {card}")
    axes[1].axhline(0.1, color="crimson", ls="--", lw=1,
                    label="0.1 deg accuracy target")
    axes[1].axhline(-0.1, color="crimson", ls="--", lw=1)
    axes[1].set_xscale("log"); axes[1].set_yscale("symlog", linthresh=0.1)
    axes[1].set_xlabel("f [Hz]"); axes[1].set_ylabel("phase error if uncorrected [deg]")
    axes[1].set_title("Why it matters")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle(f"{result.measurement_id} / {result.condition} - synchronisation",
                 fontsize=11)
    fig.tight_layout()
    return fig


def plot_kk(result, max_segments: int = 8):
    """Kramers-Kronig residuals; the shape names the fault."""
    chosen = [s for s in sorted(result.segments) if result.segments[s].kk][:max_segments]
    if not chosen:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    colours = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(chosen)))
    for colour, seg in zip(colours, chosen):
        r = result.segments[seg]
        axes[0].plot(r.spectrum.f, r.kk.residual_real * 100, "o-", ms=3,
                     color=colour, lw=1, label=f"seg {seg}")
        axes[1].plot(r.spectrum.f, r.kk.residual_imag * 100, "o-", ms=3,
                     color=colour, lw=1)
    for ax, name in zip(axes, [r"$\Delta_{\rm re}$", r"$\Delta_{\rm im}$"]):
        ax.set_xscale("log"); ax.axhline(0, color="0.6", lw=0.8)
        ax.axhspan(-1, 1, color="green", alpha=0.08)
        ax.set_xlabel("f [Hz]"); ax.set_ylabel(f"{name} [%]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    shapes = {}
    for s in result.segments.values():
        if s.kk:
            shapes[s.kk.shape_class] = shapes.get(s.kk.shape_class, 0) + 1
    fig.suptitle(f"Kramers-Kronig residuals - shapes across all segments: {shapes}",
                 fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report_figures(result, cfg, out_dir: str | Path) -> list[Path]:
    """Write the standard figure set for one condition."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    asr = cfg.geometry.segment_area_cm2 * 1e3

    def save(fig, name: str) -> None:
        if fig is None:
            return
        path = out / name
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(path)

    save(plot_spectra(result, cfg), "spectra.png")
    save(plot_sync(result), "synchronisation.png")
    save(plot_kk(result), "kramers_kronig.png")

    rs = {s: r.rs_hf_ohm * asr for s, r in result.segments.items()}
    save(plot_plate(rs, cfg, f"{result.condition}: high-frequency resistance",
                    r"$R_s$ [m$\Omega\cdot$cm$^2$]"), "plate_rs.png")

    if any(r.ecm for r in result.segments.values()):
        ecm_rs, ecm_rp, poor = {}, {}, set()
        for seg, r in result.segments.items():
            if r.ecm is None:
                continue
            ecm_rs[seg] = r.ecm.params["Rs"] * asr
            ecm_rp[seg] = r.ecm.r_polarisation * asr
            if r.ecm.poorly_determined:
                poor.add(seg)
        save(plot_plate(ecm_rs, cfg, f"{result.condition}: ECM $R_s$",
                        r"$R_s$ [m$\Omega\cdot$cm$^2$]", uncertain=poor),
             "plate_ecm_rs.png")
        save(plot_plate(ecm_rp, cfg, f"{result.condition}: polarisation resistance",
                        r"$R_p$ [m$\Omega\cdot$cm$^2$]", uncertain=poor, cmap="magma"),
             "plate_ecm_rp.png")

    coherence = {
        s: float(np.median(r.spectrum.coherence)) for s, r in result.segments.items()
    }
    save(plot_plate(coherence, cfg, f"{result.condition}: median coherence",
                    r"$\gamma^2$", cmap="cividis"), "plate_coherence.png")
    return written
