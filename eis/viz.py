"""Figures: spectra with error bars, plate maps, and synchronisation diagnostics.

Matplotlib only, so the figures render anywhere the pipeline runs (including a
headless cluster).  Two rules run through all of it:

* **The uncertainty is shown, not hidden.**  Error bars on every spectrum, and
  a hatch over segments whose value is too poorly determined to be read as a
  number.
* **Every segment is drawn.**  A plate map with holes in it invites the reader
  to forget the holes.  Segments that failed are drawn desaturated with their
  status written in them, and the colour scale is computed from the segments
  that are trustworthy so that one broken channel cannot flatten the map.

For the same picture with the spectra attached to it, see
:mod:`eis.dashboard`.
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
    """Nyquist + Bode magnitude + phase, with per-point uncertainty.

    With eighty segments an overview figure has to choose.  It shows the
    *active* ones, spread evenly across the plate rather than taking the first
    twelve, so the figure reflects the spatial variation instead of one corner.
    """
    max_segments = getattr(
        getattr(cfg, "report", None), "max_spectra_per_figure", max_segments
    )
    if segments is None:
        active = [
            s for s in sorted(result.segments)
            if result.segments[s].active and len(result.segments[s].spectrum.f)
        ]
        if len(active) > max_segments:
            picks = np.linspace(0, len(active) - 1, max_segments).round().astype(int)
            segments = [active[i] for i in dict.fromkeys(picks)]
        else:
            segments = active
    chosen = [
        s for s in segments
        if s in result.segments and len(result.segments[s].spectrum.f)
    ][:max_segments]
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
    all_segments: list[int] | None = None,
    status: dict[int, str] | None = None,
    quality: dict[int, float] | None = None,
):
    """Spatial map of one scalar per segment, with every segment on it.

    ``values`` supplies the numbers.  ``all_segments`` names the segments that
    should appear even when they have no number - they are drawn as empty
    outlines carrying their status, so a gap in the data reads as a gap in the
    data rather than as a plate with fewer segments.

    ``quality`` desaturates the doubtful ones and, more importantly, keeps them
    out of the colour scale: a single segment reading ten times the plate
    median would otherwise compress every real variation into one shade.
    """
    catalogue = sorted(set(all_segments or []) | set(values))
    coords = cfg.geometry.segment_coords or default_segment_grid(
        catalogue, cfg.geometry.plate_w_cm, cfg.geometry.plate_h_cm
    )
    usable = {s: v for s, v in values.items() if s in coords and np.isfinite(v)}
    if not usable:
        return None

    good_cut = getattr(getattr(cfg, "quality", None), "good_quality", 0.0)
    trusted = [
        v for s, v in usable.items()
        if quality is None or quality.get(s, 1.0) >= good_cut
    ]
    basis = np.array(trusted if len(trusted) >= 3 else list(usable.values()), float)
    norm = Normalize(vmin=float(basis.min()), vmax=float(basis.max()))
    colours = plt.get_cmap(cmap)
    uncertain = uncertain or set()
    status = status or {}

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.add_patch(Rectangle(
        (0, 0), cfg.geometry.plate_w_cm, cfg.geometry.plate_h_cm,
        facecolor="#EEE8DC", edgecolor="#7a6a4a", lw=1.6, zorder=0,
    ))
    n_missing = 0
    for seg in catalogue:
        if seg not in coords:
            continue
        x, y, hw, hh = coords[seg]
        value = usable.get(seg)
        if value is None:
            n_missing += 1
            ax.add_patch(Rectangle(
                (x - hw, y - hh), 2 * hw, 2 * hh, facecolor="none",
                edgecolor="#b4531f", lw=1.2, ls="--", zorder=2,
            ))
            ax.text(
                x, y, f"{seg}\n{status.get(seg, 'no value')}", ha="center",
                va="center", fontsize=6.5, color="#b4531f", zorder=3,
            )
            continue
        doubtful = (
            seg in uncertain
            or (quality is not None and quality.get(seg, 1.0) < good_cut)
        )
        ax.add_patch(Rectangle(
            (x - hw, y - hh), 2 * hw, 2 * hh,
            facecolor=colours(norm(value)), edgecolor="white", lw=1.2, zorder=2,
            alpha=0.45 if doubtful else 1.0,
            hatch="///" if doubtful else None,
        ))
        ax.text(
            x, y, f"{seg}\n{value:.3g}", ha="center", va="center",
            fontsize=7.5, color="white" if not doubtful else "#222",
            fontweight="bold", zorder=3,
        )

    sm = plt.cm.ScalarMappable(cmap=colours, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(unit or title)
    ax.set_xlim(-0.2, cfg.geometry.plate_w_cm + 0.2)
    ax.set_ylim(-0.2, cfg.geometry.plate_h_cm + 0.2)
    ax.set_aspect("equal"); ax.axis("off")
    marks = []
    if any(
        seg in uncertain or (quality is not None and quality.get(seg, 1.0) < good_cut)
        for seg in usable
    ):
        marks.append("hatched = poorly determined")
    if n_missing:
        marks.append(f"{n_missing} dashed = no usable value")
    subtitle = f"  ({'; '.join(marks)})" if marks else ""
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

def plot_plate_over_frequency(
    result, cfg, key: str = "z_mod@f", n_panels: int = 6, cmap: str = "viridis"
):
    """Small multiples of one frequency-resolved map across the band.

    This is the figure that separates the loss mechanisms by eye: an ohmic term
    is spatially smooth and barely changes from panel to panel, while a kinetic
    or transport term appears in a band and follows the flow field.  It is only
    possible because the coherence gate marks rather than deletes, so every
    segment still has a value at every frequency.
    """
    from eis.pipeline.gold import map_label, scalar_map

    frequencies = np.asarray(result.frequencies, float)
    if frequencies.size < 2:
        return None
    picks = np.unique(np.geomspace(
        max(frequencies.min(), 1e-6), frequencies.max(),
        min(n_panels, frequencies.size)
    ))
    coords = cfg.geometry.segment_coords or default_segment_grid(
        sorted(result.segments), cfg.geometry.plate_w_cm, cfg.geometry.plate_h_cm
    )
    area = cfg.geometry.segment_area_cm2
    panels = [
        (f, scalar_map(result, key, area, frequency_hz=float(f))) for f in picks
    ]
    panels = [(f, m) for f, m in panels if m]
    if not panels:
        return None

    everything = np.array([v for _, m in panels for v in m.values()], float)
    norm = Normalize(vmin=float(np.nanpercentile(everything, 2)),
                     vmax=float(np.nanpercentile(everything, 98)))
    colours = plt.get_cmap(cmap)
    title, unit = map_label(key)

    rows = int(np.ceil(len(panels) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.6 * rows), squeeze=False)
    for ax, (frequency, values) in zip(axes.ravel(), panels):
        ax.add_patch(Rectangle(
            (0, 0), cfg.geometry.plate_w_cm, cfg.geometry.plate_h_cm,
            facecolor="#EEE8DC", edgecolor="#7a6a4a", lw=1.0, zorder=0,
        ))
        for seg, value in values.items():
            if seg not in coords:
                continue
            x, y, hw, hh = coords[seg]
            ax.add_patch(Rectangle(
                (x - hw, y - hh), 2 * hw, 2 * hh, facecolor=colours(norm(value)),
                edgecolor="white", lw=0.5, zorder=2,
            ))
        ax.set_xlim(-0.2, cfg.geometry.plate_w_cm + 0.2)
        ax.set_ylim(-0.2, cfg.geometry.plate_h_cm + 0.2)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{frequency:.4g} Hz", fontsize=10)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=colours, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label(unit or title)
    fig.suptitle(f"{result.condition}: {title} across the band", fontsize=12,
                 fontweight="bold")
    return fig


def write_report_figures(result, cfg, out_dir: str | Path) -> list[Path]:
    """Write the standard figure set for one condition."""
    from eis.pipeline.gold import map_label, scalar_map

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    area = cfg.geometry.segment_area_cm2
    catalogue = sorted(result.segments)
    status = {s: r.status for s, r in result.segments.items()}
    quality = {s: r.quality for s, r in result.segments.items()}

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

    # One code path, N maps: everything the configuration asks for.
    poor = {
        s for s, r in result.segments.items()
        if r.ecm is not None and r.ecm.poorly_determined
    }
    for key in cfg.report.heatmap_parameters:
        if key.endswith("@f"):
            continue
        values = scalar_map(result, key, area)
        if not values:
            continue
        title, unit = map_label(key)
        save(
            plot_plate(
                values, cfg, f"{result.condition}: {title}", unit,
                uncertain=poor if key in ("rp", "ecm_rs", "chi2_reduced") else None,
                cmap="magma" if key == "rp" else
                     "cividis" if key in ("coherence", "quality") else "viridis",
                all_segments=catalogue, status=status, quality=quality,
            ),
            f"plate_{key}.png",
        )

    for key in cfg.report.heatmap_parameters:
        if key.endswith("@f"):
            save(plot_plate_over_frequency(result, cfg, key),
                 f"plate_over_frequency_{key.replace('@', '_at_')}.png")
    return written
