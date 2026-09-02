"""Turn whatever is on disk into a :class:`~app.data.model.RunData`.

Three producers are understood today:

``gold_silver``
    The bronze/silver/gold layout: ``gold/plate_summary.csv``,
    ``silver/spectra_clean.csv``, ``silver/cell_aggregate.csv`` plus the JSON
    manifests.  Values in ``plate_summary.csv`` are already display-scaled
    (mOhm*cm2, %, degrees), which is the canonical unit set here.

``eis_tables``
    The packaged :mod:`eis` pipeline: ``impedance`` and ``segments`` as Parquet
    (or CSV when pyarrow is absent), one file covering several conditions.

``generic_csv``
    A hand-made export.  Columns are matched by pattern, so ``freq``, ``f_Hz``
    and ``frequency_hz`` all resolve, and anything unmatched is reported rather
    than guessed at.

A loader never raises on a missing optional part; it records a warning on the
run and carries on, because a viewer that shows a Nyquist plot with a note
"no gold layer found" is more use than one that shows a stack trace.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.model import RunData

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        # index_col=False matters. Without it, a row with more fields than the
        # header makes pandas silently promote the first column to the index,
        # which shifts every value one place left and produces a table that is
        # entirely wrong and entirely plausible. With it, the same row either
        # raises or warns - and the warning is promoted to an error here,
        # because its own text is "this leads to a loss of data".
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            return pd.read_csv(path, index_col=False)
    except (pd.errors.ParserError, pd.errors.ParserWarning):
        return _read_csv_ragged(path)


def _read_csv_ragged(path: Path) -> pd.DataFrame:
    """Recover a CSV whose last column contains unquoted commas.

    ``silver/skew_model.csv`` ends in a free-text ``note`` such as
    "... (degenerate with series L, absorbed there)", written without quoting.
    Splitting each line at most ``n_columns - 1`` times lets that final column
    absorb the rest, which is exactly right when the ragged field is the last
    one - and it is, because free text is what gets appended to a row.
    """
    lines = path.read_text().splitlines()
    if not lines:
        return pd.DataFrame()
    header = lines[0].split(",")
    rows = [line.split(",", len(header) - 1) for line in lines[1:] if line.strip()]
    rows = [row + [""] * (len(header) - len(row)) for row in rows]
    frame = pd.DataFrame(rows, columns=header)
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            frame[column] = converted
    return frame


def _first(base: Path, *names: str) -> Path | None:
    for name in names:
        p = base / name
        if p.is_file():
            return p
    return None


def _read_json(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _numeric(frame: pd.DataFrame, skip: set[str]) -> pd.DataFrame:
    """Empty strings in the CSVs mean "not finite"; make them NaN, not text."""
    for col in frame.columns:
        if col in skip:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


# ---------------------------------------------------------------------------
# layout detection
# ---------------------------------------------------------------------------

def detect_layout(directory: str | Path) -> str | None:
    """Which producer wrote this directory, if any."""
    d = Path(directory)
    if not d.is_dir():
        return None
    if (d / "gold" / "plate_summary.csv").is_file() or \
       (d / "silver" / "spectra_clean.csv").is_file():
        return "gold_silver"
    if _first(d, "impedance.parquet", "impedance.csv"):
        return "eis_tables"
    if any(d.glob("*.csv")):
        return "generic_csv"
    return None


# ---------------------------------------------------------------------------
# bronze / silver / gold
# ---------------------------------------------------------------------------

def load_gold_silver(
    directory: str | Path, measurement_id: str = "", condition: str = "",
    plate_key: str = "",
) -> RunData:
    d = Path(directory)
    run = RunData(measurement_id=measurement_id or d.parent.name,
                  condition=condition or d.name,
                  plate_key=plate_key, source=str(d), loader="gold_silver")

    # -- spectra ------------------------------------------------------------
    spectra_path = d / "silver" / "spectra_clean.csv"
    if spectra_path.is_file():
        sp = _read_table(spectra_path)
        sp = sp.rename(columns={
            "zmodel_re_mohm_cm2": "z_re_model_mohm_cm2",
            "zmodel_im_mohm_cm2": "z_im_model_mohm_cm2",
        })
        sp["segment"] = sp["segment"].astype(str)
        # "card" is written by the per-card evaluator (evaluate_per_card.py),
        # which concatenates one silver table per acquisition card; it is a
        # label such as "Karte_3", so it must not be coerced to a number.
        run.spectra = _numeric(sp, {"segment", "card"})
    else:
        run.warnings.append("no silver/spectra_clean.csv - Nyquist and ECM unavailable")

    # -- per-segment scalars ------------------------------------------------
    gold_path = d / "gold" / "plate_summary.csv"
    if gold_path.is_file():
        seg = _read_table(gold_path)
        seg = seg.rename(columns={"class": "seg_class"})
        seg["segment"] = seg["segment"].astype(str)
        for col in ("seg_class", "tier", "fault", "flags"):
            if col in seg.columns:
                seg[col] = seg[col].fillna("").astype(str)
        run.segments = _numeric(
            seg, {"segment", "seg_class", "tier", "fault", "flags", "card"})
    else:
        run.warnings.append("no gold/plate_summary.csv - heat maps unavailable")

    # Silver's own summary carries acquisition quality (card, SNR, THD, KK)
    # that gold does not repeat.  Merge the columns gold is missing.
    sil_path = d / "silver" / "segments_summary.csv"
    if sil_path.is_file():
        sil = _read_table(sil_path)
        sil["segment"] = sil["segment"].astype(str)
        sil = _numeric(sil, {"segment", "card", "tier", "flags"})
        if run.segments.empty:
            run.segments = sil.rename(columns={"R_ohmic_mohm_cm2": "R_ohmic",
                                               "R_pol_mohm_cm2": "R_pol"})
        else:
            extra = [c for c in sil.columns
                     if c not in run.segments.columns or c == "segment"]
            run.segments = run.segments.merge(sil[extra], on="segment", how="left")

    # -- cell aggregate -----------------------------------------------------
    cell_path = d / "silver" / "cell_aggregate.csv"
    if cell_path.is_file():
        run.cell = _numeric(_read_table(cell_path), set())

    # -- manifests ----------------------------------------------------------
    run.meta = {
        "gold": _read_json(d / "gold" / "gold_manifest.json"),
        "silver": _read_json(d / "silver" / "silver_manifest.json"),
        "run": _read_json(d / "run_manifest.json"),
        "config": _read_json(d / "config_used.json"),
    }
    skew_path = d / "silver" / "skew_model.csv"
    if skew_path.is_file():
        run.meta["skew_table"] = _read_table(skew_path).to_dict("records")
    return run


# ---------------------------------------------------------------------------
# packaged eis pipeline
# ---------------------------------------------------------------------------

def load_eis_tables(
    directory: str | Path, measurement_id: str = "", condition: str = "",
    plate_key: str = "",
) -> RunData:
    d = Path(directory)
    run = RunData(measurement_id=measurement_id or d.name, condition=condition,
                  plate_key=plate_key, source=str(d), loader="eis_tables")

    imp_path = _first(d, "impedance.parquet", "impedance.csv")
    if imp_path is not None:
        imp = _read_table(imp_path)
        if condition and "condition" in imp.columns:
            imp = imp[imp["condition"].astype(str) == str(condition)]
        run.spectra = pd.DataFrame({
            "segment": imp["segment"].astype(str),
            "freq_hz": imp["frequency_hz"],
            "z_re_mohm_cm2": imp["z_real_mohm_cm2"],
            "z_im_mohm_cm2": imp["z_imag_mohm_cm2"],
            "sigma_rel": imp.get("sigma_rel", np.nan),
        })
        if not measurement_id and "measurement_id" in imp.columns and len(imp):
            run.measurement_id = str(imp["measurement_id"].iloc[0])
    else:
        run.warnings.append("no impedance table found")

    seg_path = _first(d, "segments.parquet", "segments.csv")
    if seg_path is not None:
        seg = _read_table(seg_path)
        if condition and "condition" in seg.columns:
            seg = seg[seg["condition"].astype(str) == str(condition)]
        seg = seg.rename(columns={
            "status": "seg_class",
            "rs_hf_mohm_cm2": "R_ohmic",
            "temperature_c": "T_degC",
            "median_sigma_rel": "sigma_rel",
            "kk_max_residual_pct": "kk_res_pct",
        })
        seg["segment"] = seg["segment"].astype(str)
        if "seg_class" in seg.columns:
            # 'ok' / 'rejected' -> the viewer's measured / bad vocabulary
            seg["seg_class"] = seg["seg_class"].map(
                lambda s: "measured" if str(s).lower() in ("ok", "good", "measured")
                else "bad").astype(str)
        run.segments = seg
    else:
        run.warnings.append("no segments table found")

    cfg_path = d / "config.yaml"
    if cfg_path.is_file():
        run.meta["config_yaml"] = cfg_path.read_text()
    return run


# ---------------------------------------------------------------------------
# generic CSV
# ---------------------------------------------------------------------------

#: canonical name -> patterns tried in order, matched case-insensitively
_SPECTRA_ALIASES = {
    "segment": ["segment", "seg", "channel", "sensor"],
    "freq_hz": ["freq_hz", "frequency_hz", "frequency", "freq", "f_hz", "f"],
    "z_re_mohm_cm2": ["z_re_mohm_cm2", "z_real_mohm_cm2", "zre", "z_re", "z_real",
                      "real", "re_z", "z'"],
    "z_im_mohm_cm2": ["z_im_mohm_cm2", "z_imag_mohm_cm2", "zim", "z_im", "z_imag",
                      "imag", "im_z", "z''"],
    "sigma_rel": ["sigma_rel", "sigma", "uncertainty", "rel_err"],
}


def map_columns(columns, aliases=None) -> dict[str, str]:
    """Best-effort mapping of arbitrary column names onto canonical ones."""
    aliases = aliases or _SPECTRA_ALIASES
    lowered = {str(c).strip().lower().replace(" ", "_"): c for c in columns}
    out: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for cand in candidates:
            if cand in lowered:
                out[canonical] = lowered[cand]
                break
        else:
            for key, original in lowered.items():
                if any(key.startswith(c) for c in candidates):
                    out[canonical] = original
                    break
    return out


def load_generic_csv(
    path: str | Path, measurement_id: str = "", condition: str = "",
    plate_key: str = "", overrides: dict[str, str] | None = None,
) -> RunData:
    """Read one CSV of spectra whose column names are not known in advance."""
    p = Path(path)
    frame = pd.read_csv(p)
    mapping = map_columns(frame.columns)
    mapping.update({k: v for k, v in (overrides or {}).items() if v})

    run = RunData(measurement_id=measurement_id or p.stem, condition=condition,
                  plate_key=plate_key, source=str(p), loader="generic_csv")
    missing = [k for k in ("freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2")
               if k not in mapping]
    if missing:
        run.warnings.append(
            f"{p.name}: could not identify column(s) {', '.join(missing)}; "
            f"saw {list(frame.columns)}")
        return run

    data = {canonical: frame[source] for canonical, source in mapping.items()}
    spectra = pd.DataFrame(data)
    if "segment" not in spectra.columns:
        spectra["segment"] = "cell"
        run.warnings.append(
            f"{p.name}: no segment column; treating the file as one spectrum")
    spectra["segment"] = spectra["segment"].astype(str)
    run.spectra = _numeric(spectra, {"segment"})
    run.meta["column_mapping"] = mapping
    return run


# ---------------------------------------------------------------------------
# front door
# ---------------------------------------------------------------------------

LOADERS = {
    "gold_silver": load_gold_silver,
    "eis_tables": load_eis_tables,
}


def load(directory: str | Path, measurement_id: str = "", condition: str = "",
         plate_key: str = "", layout: str | None = None) -> RunData:
    """Load a results directory, detecting the layout unless told."""
    d = Path(directory)
    layout = layout or detect_layout(d)
    if layout in LOADERS:
        return LOADERS[layout](d, measurement_id, condition, plate_key)
    if layout == "generic_csv":
        csvs = sorted(d.glob("*.csv"))
        run = load_generic_csv(csvs[0], measurement_id, condition, plate_key)
        if len(csvs) > 1:
            run.warnings.append(
                f"{len(csvs)} CSVs present; read {csvs[0].name}. "
                "Point the app at a single file to choose another.")
        return run
    run = RunData(measurement_id=measurement_id, condition=condition,
                  plate_key=plate_key, source=str(d), loader="none")
    run.warnings.append(f"{d}: no recognised results layout")
    return run
