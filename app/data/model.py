"""The canonical shape every loader produces.

Two pipelines write results in this repository's world: the notebook-era
bronze/silver/gold layout (``gold/plate_summary.csv``,
``silver/spectra_clean.csv``) and the packaged :mod:`eis` pipeline
(``impedance.parquet``, ``segments.parquet``).  A third shape turns up
whenever somebody hands over a plain CSV export.

Rather than teach every figure and every callback about all three, each loader
normalises into the frames below.  Adding a fourth producer later is one
adapter function, not a change to the views.

Units are fixed here once: resistances and impedances in mOhm*cm2, frequency in
Hz, lengths in mm, areas in cm2, current density in A/cm2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: Canonical per-point spectrum columns.  ``z_*_model`` are optional and hold
#: the pipeline's own smoothed/KK model when it wrote one.
SPECTRA_COLUMNS = [
    "segment", "freq_hz",
    "z_re_mohm_cm2", "z_im_mohm_cm2",
    "sigma_rel", "z_re_model_mohm_cm2", "z_im_model_mohm_cm2",
]

#: Canonical per-segment scalar columns.  Everything else a producer supplies
#: is carried through untouched, so a new parameter shows up in the heat-map
#: dropdown without any code change.
SEGMENT_COLUMNS = [
    "segment", "seg_class", "tier", "card", "area_cm2", "cx_mm", "cy_mm",
    "flags", "fault",
]

#: Label, unit and colour scale for the scalars the viewer knows about.
#: ``scale`` multiplies the stored value for display.
PARAM_META: dict[str, dict] = {
    "R_ohmic":     dict(label="RΩ (HF intercept)", unit="mΩ·cm²", colorscale="RdYlBu_r"),
    "R_ct":        dict(label="R_ct (charge transfer)", unit="mΩ·cm²", colorscale="RdYlBu_r"),
    "R_mt":        dict(label="R_mt (mass transport)", unit="mΩ·cm²", colorscale="RdYlBu_r"),
    "R_pol":       dict(label="R_pol (total polarisation)", unit="mΩ·cm²", colorscale="RdYlBu_r"),
    "R_hf_extra":  dict(label="Hidden HF resistance (bound)", unit="mΩ·cm²", colorscale="Reds"),
    "Z_mag_100Hz": dict(label="|Z| @ 100 Hz", unit="mΩ·cm²", colorscale="RdYlBu_r"),
    "phase_100Hz": dict(label="Phase @ 100 Hz", unit="°", colorscale="RdYlBu_r"),
    "j_dc":        dict(label="DC current density", unit="A/cm²", colorscale="Viridis"),
    "tau_peak":    dict(label="Dominant relaxation time", unit="s", colorscale="Plasma"),
    "T_degC":      dict(label="Temperature", unit="°C", colorscale="Inferno"),
    "sigma_rel":   dict(label="Relative uncertainty", unit="%", colorscale="Greys"),
    "hf_closure":  dict(label="HF closure", unit="-", colorscale="RdYlBu"),
    "kk_res_pct":  dict(label="Kramers-Kronig residual", unit="%", colorscale="Greys"),
    "snr_med_db":  dict(label="Median SNR", unit="dB", colorscale="Viridis"),
}


def param_label(name: str) -> str:
    meta = PARAM_META.get(name)
    if not meta:
        return name
    unit = meta.get("unit")
    return f"{meta['label']} [{unit}]" if unit else meta["label"]


@dataclass
class RunData:
    """Everything the viewer needs about one measurement at one condition."""

    measurement_id: str
    condition: str
    plate_key: str
    #: per-point spectra, canonical columns above
    spectra: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SPECTRA_COLUMNS))
    #: per-segment scalars
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: area-weighted cell spectrum, if the producer computed one
    cell: pd.DataFrame | None = None
    #: run manifests, config, DC closure, skew table
    meta: dict = field(default_factory=dict)
    #: provenance: which loader, which files
    source: str = ""
    loader: str = ""
    warnings: list[str] = field(default_factory=list)

    # -- convenience ---------------------------------------------------------

    @property
    def segment_names(self) -> list[str]:
        if self.segments.empty:
            names = sorted(self.spectra["segment"].unique()) if not self.spectra.empty else []
        else:
            names = list(self.segments["segment"])
        return sorted(names, key=lambda n: (0, int(n)) if str(n).isdigit() else (1, str(n)))

    def mappable_params(self) -> list[str]:
        """Numeric per-segment columns worth putting on a map.

        Known parameters first and in a stable order, then anything else the
        producer supplied, so a new pipeline output is visible immediately.
        """
        if self.segments.empty:
            return []
        numeric = [
            c for c in self.segments.columns
            if c not in SEGMENT_COLUMNS
            and not c.endswith("_sd")
            and pd.api.types.is_numeric_dtype(self.segments[c])
            and self.segments[c].notna().any()
        ]
        known = [p for p in PARAM_META if p in numeric]
        return known + sorted(set(numeric) - set(known))

    def spectrum(self, segment: str) -> pd.DataFrame:
        if self.spectra.empty:
            return self.spectra
        sub = self.spectra[self.spectra["segment"].astype(str) == str(segment)]
        return sub.sort_values("freq_hz")

    def value(self, param: str) -> dict[str, float]:
        if self.segments.empty or param not in self.segments.columns:
            return {}
        frame = self.segments[["segment", param]].dropna()
        return {str(k): float(v) for k, v in zip(frame["segment"], frame[param])}

    def sd(self, param: str) -> dict[str, float]:
        return self.value(f"{param}_sd")

    def classes(self) -> dict[str, str]:
        if self.segments.empty or "seg_class" not in self.segments.columns:
            return {}
        return {str(k): str(v) for k, v in
                zip(self.segments["segment"], self.segments["seg_class"])}

    def label(self) -> str:
        return f"{self.measurement_id} / {self.condition}"
