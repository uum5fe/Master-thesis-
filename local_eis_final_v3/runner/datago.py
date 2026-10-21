"""
datago.py
=========
Reading an order's Gamry sweeps and bench log from datago (Spark SQL).

Datago holds, per order (metadata.orderId = 'RO<leepa>'), every file as it
was uploaded (metafiles_delta.raw_file_name):

    RO2612086-01_V26_109_HFR_111_CurrVal_60.mf4   a Gamry sweep (GALVEIS)
    20261020_..._RO2612086-01_..._Boxen_6.mf4      the bench log of the test
    ..._Current_60A_Test_02_Karte_3.DAT            FAMOS cards (NOT used here)

Every Gamry sweep is written back as a .dta, with its STARTTIME, so the
pipeline reads it unchanged -- and ALL sweeps of a current are written (an
up/down ramp measures each current twice), so the pipeline can pick the one
whose time matches the FAMOS recording (gamry_compare.select_sweep).

Nothing here works without `spark`; every function takes it explicitly.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings as S

CACHE = Path(tempfile.gettempdir()) / "datago_gamry"
_FILES: dict = {}

_DTA_COLS = [("Pt", "#", "pt"), ("Time", "s", None), ("Freq", "Hz", "freq"),
             ("Zreal", "ohm", "zreal"), ("Zimag", "ohm", "zimag"),
             ("Zsig", "V", "zsig"), ("Zmod", "ohm", "zmod"),
             ("Zphz", "deg", "zphz"), ("Idc", "A", "idc"),
             ("Vdc", "V", "vdc"), ("IERange", "#", "ie_range")]


def current_label(token) -> str:
    """'450' -> '450A', '7,5' -> '7.5A'."""
    return f"{float(str(token).replace(',', '.')):g}A"


def classify(name, mtype):
    """(kind, condition) of one datago file from its uploaded name."""
    low = str(name or "").lower()
    m = re.search(r"currval[_-]?(\d+(?:[.,]\d+)?)", low)
    if m and (mtype == "GALVEIS" or low.endswith((".mf4", ".dta"))):
        return "gamry", current_label(m.group(1))
    if low.endswith(".dat"):
        return "famos", None
    if low.endswith(".mf4"):
        return "bench", None
    return "other", None


def orders(spark) -> list[str]:
    """Every order with a Gamry sweep in datago, as '2612086'."""
    rows = spark.sql(f"""
        SELECT DISTINCT m.orderId FROM {S.DATAGO_META_TBL} m
        JOIN {S.DATAGO_GP_TBL} gp ON m.measurement_id = gp.measurement_id
        WHERE gp.measurement_type = 'GALVEIS' ORDER BY m.orderId""").collect()
    return [r["orderId"].replace("RO", "") for r in rows if r["orderId"]]


def files(spark, leepa, refresh=False) -> pd.DataFrame:
    """Every datago file of the order, one row per file, classified."""
    key = str(leepa)
    if key in _FILES and not refresh:
        return _FILES[key]
    df = spark.sql(f"""
        SELECT mf.file_id, FIRST(mf.raw_file_name) AS raw_file_name,
               FIRST(gp.measurement_type, true) AS measurement_type,
               FIRST(COALESCE(gp.measurementBegin, mf.measurementBegin), true) AS t_begin,
               FIRST(gp.measurementEnd, true) AS t_end,
               FIRST(gp.STARTTIME, true) AS starttime,
               FIRST(gp.IACREQ, true) AS iacreq,
               FIRST(gp.FREQINIT, true) AS freqinit,
               FIRST(gp.FREQFINAL, true) AS freqfinal,
               FIRST(gp.PTSPERDEC, true) AS ptsperdec
        FROM {S.DATAGO_FILES_TBL} mf
        JOIN (SELECT DISTINCT measurement_id FROM {S.DATAGO_META_TBL}
              WHERE orderId = 'RO{key}') m
          ON m.measurement_id = mf.measurement_id
        LEFT JOIN {S.DATAGO_GP_TBL} gp ON gp.file_id = mf.file_id
        GROUP BY mf.file_id""").toPandas()
    cls = [classify(n, t) for n, t in zip(df["raw_file_name"],
                                          df["measurement_type"])]
    df["kind"] = [c[0] for c in cls]
    df["condition"] = [c[1] for c in cls]
    df = df.sort_values(["t_begin", "raw_file_name"], na_position="last"
                        ).reset_index(drop=True)
    _FILES[key] = df
    return df


def values(spark, file_ids, channels=None) -> dict:
    """{file_id: {channel: (t_s, value)}} in the order the samples were
    written (the Gamry's time_value has 1 s resolution, so sorting by it
    would mix up points of the same second)."""
    ids = [str(f) for f in file_ids if f]
    if not ids:
        return {}
    flt = "file_id IN (" + ", ".join(f"'{f}'" for f in ids) + ")"
    if channels:
        flt += " AND channel IN (" + ", ".join(f"'{c}'" for c in channels) + ")"
    pdf = spark.sql(f"""
        SELECT file_id, channel, CAST(time_value AS DOUBLE) AS t,
               CAST(value AS DOUBLE) AS v,
               _metadata.file_name AS _fn, _metadata.row_index AS _ri
        FROM {S.DATAGO_VAL_TBL} WHERE {flt}""").toPandas()
    pdf = pdf.sort_values(["file_id", "_fn", "_ri"], kind="stable")
    out: dict = {}
    for (fid, ch), g in pdf.groupby(["file_id", "channel"], sort=False):
        out.setdefault(fid, {})[ch] = (g["t"].to_numpy(float),
                                       g["v"].to_numpy(float))
    return out


def bench_row(spark, leepa):
    """The bench log of the order (the longest .mf4 that is not an
    _Anfang/_Ende snapshot), or None."""
    df = files(spark, leepa)
    b = df[(df["kind"] == "bench") & ~df["raw_file_name"].str.lower()
           .str.contains("anfang|ende", na=False)]
    if b.empty:
        b = df[df["kind"] == "bench"]
    if b.empty:
        return None
    dur = (pd.to_datetime(b["t_end"]) - pd.to_datetime(b["t_begin"])
           ).dt.total_seconds()
    return b.loc[dur.fillna(-1).idxmax()]


def _starttime(row):
    """(STARTTIME text, Time offset of the first point in s)."""
    st = str(getattr(row, "starttime", "") or "").strip()
    off = 0.0
    try:
        t_st = (pd.Timestamp(datetime.strptime(st, "%d.%m.%Y %H:%M:%S"))
                .tz_localize(S.DATAGO_LOCAL_TZ).tz_convert("UTC")
                .tz_localize(None))
        o = (pd.Timestamp(row.t_begin) - t_st).total_seconds()
        off = float(round(o)) if 0 <= o < 600 else 0.0
    except Exception:                                       # noqa: BLE001
        if row.t_begin is not None and pd.notna(row.t_begin):
            st = (pd.Timestamp(row.t_begin).tz_localize("UTC")
                  .tz_convert(S.DATAGO_LOCAL_TZ).strftime("%d.%m.%Y %H:%M:%S"))
    return st, off


def _table(row, vals) -> pd.DataFrame:
    if "freq" not in vals:
        return pd.DataFrame()
    used = {c[2] for c in _DTA_COLS if c[2]}
    n = min(len(v[1]) for k, v in vals.items() if k in used)
    _, off = _starttime(row)
    tab = {"Time": vals["freq"][0][:n] + off}
    for col, _u, ch in _DTA_COLS:
        if ch is not None:
            tab[col] = vals[ch][1][:n] if ch in vals else np.full(n, np.nan)
    df = pd.DataFrame(tab)[[c[0] for c in _DTA_COLS]]
    if df["Pt"].isna().any():
        df["Pt"] = np.arange(1, len(df) + 1)
    return df.sort_values("Pt", kind="stable").reset_index(drop=True)


def _write_dta(path, row, tab) -> None:
    st, _ = _starttime(row)
    d, t = (st.split(" ") + [""])[:2]

    def q(v):
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    lines = ["EXPLAIN", "TAG\tGALVEIS",
             "TITLE\tLABEL\tGalvanostatic Impedance Spectroscopy\tTest Identifier",
             "NOTES\tNOTES\t1\tNotes",
             f"\tfrom datago, file_id {row.file_id} ({row.raw_file_name})",
             f"DATE\tLABEL\t{d}\tDate", f"TIME\tLABEL\t{t}\tTime",
             f"IACREQ\tQUANT\t{q(row.iacreq)}\tAC Current (A rms)",
             f"FREQINIT\tQUANT\t{q(row.freqinit)}\tInitial Freq. (Hz)",
             f"FREQFINAL\tQUANT\t{q(row.freqfinal)}\tFinal Freq. (Hz)",
             f"PTSPERDEC\tQUANT\t{q(row.ptsperdec)}\tPoints/decade",
             f"STARTTIME\tLABEL\t{st}\tStart Time",
             "ZCURVE\tTABLE",
             "\t" + "\t".join(c[0] for c in _DTA_COLS),
             "\t" + "\t".join(c[1] for c in _DTA_COLS)]
    for r in tab.itertuples(index=False):
        cells = []
        for (col, _u, _c), v in zip(_DTA_COLS, r):
            if not np.isfinite(v):
                cells.append("")
            elif col in ("Pt", "IERange"):
                cells.append(str(int(round(v))))
            else:
                cells.append(f"{v:.7g}")
        lines.append("\t" + "\t".join(cells))
    Path(path).write_text("\n".join(lines) + "\n", encoding="latin-1")


def gamry_dir(spark, leepa, volume_root=None, say=print):
    """A folder with EVERY Gamry sweep of the order from datago, as .dta, or
    None. A sweep that is incomplete in datago is taken from the Volume
    folder `volume_root` (same build, current and start time) when there."""
    df = files(spark, leepa)
    rows = list(df[df["kind"] == "gamry"].itertuples())
    if not rows:
        return None
    out = CACHE / f"RO{leepa}"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.dta"):
        old.unlink()
    vals_all = values(spark, [r.file_id for r in rows])
    vol = (sorted(list(Path(volume_root).glob("*.dta"))
                  + list(Path(volume_root).glob("*.DTA")))
           if volume_root else [])
    n_dg = n_vol = 0
    for row in rows:
        stem = Path(str(row.raw_file_name)).stem
        try:
            tab = _table(row, vals_all.get(str(row.file_id), {}))
        except Exception as exc:                            # noqa: BLE001
            say(f"    datago sweep {stem} not read ({exc})")
            tab = pd.DataFrame()
        if len(tab) >= S.DATAGO_MIN_SWEEP_POINTS:
            _write_dta(out / f"{stem}.dta", row, tab)
            n_dg += 1
            continue
        # the Volume file of the same name (build + HFR number), if any
        alt = [f for f in vol if f.stem in stem or stem.endswith(f.stem)]
        if alt:
            shutil.copy2(alt[0], out / alt[0].name)
            n_vol += 1
            say(f"    {stem}: incomplete in datago -> Volume {alt[0].name}")
        else:
            say(f"    {stem}: {len(tab)} point(s) in datago, no Volume copy "
                f"-- left out")
    if n_dg + n_vol == 0:
        return None
    say(f"  Gamry sweeps of {leepa}: {n_dg} from datago"
        + (f", {n_vol} from the Volume" if n_vol else ""))
    return out


def is_datago_dir(p) -> bool:
    try:
        return Path(p).resolve().is_relative_to(CACHE.resolve())
    except Exception:                                       # noqa: BLE001
        return False


def bench_channels(spark, leepa):
    """(start time UTC, {channel: (t_s, v)}, file name) of the bench log, or
    (None, {}, '') when datago has none."""
    row = bench_row(spark, leepa)
    if row is None:
        return None, {}, ""
    vals = values(spark, [row.file_id]).get(str(row.file_id), {})
    t0 = pd.Timestamp(row.t_begin) if pd.notna(row.t_begin) else None
    return t0, vals, str(row.raw_file_name)


# ---------------------------------------------------------------------------
# FAMOS card recordings (settings.FAMOS_FROM_DATAGO)
# ---------------------------------------------------------------------------
#
# Datago keeps a card as rows (channel, time_value, value), not as the .DAT.
# Each card of the run is rebuilt as a FAMOS-v1 .DAT (float32, every name in
# one |CP, |NT start time), so the pipeline reads it like any Volume file --
# with every check it runs on those. Before a rebuilt card is used it must
# pass two tests:
#
#   1. one signal per channel (eis_local.channel_integrity). Datago's import
#      of FAMOS-v2 cards (DASYLab 14, one |CN per channel, column given by the
#      |CP byte offset) ignores that offset and returns column 0 -- UC2 --
#      under every name (RO2612085 450 A Karte_3). v1 cards are imported
#      correctly.
#   2. when the Volume holds the same card: the same samples as the .DAT
#      (probes spread over the record), so a channel stored under another
#      name is caught too.
#
# A card that fails is taken from the Volume, or left out when there is none.
#
# What datago does NOT keep is the column order of the card, and the column
# is the multiplexer slot (the pipeline corrects the slot-to-slot sampling
# delay with it). It is taken from the header of a .DAT of the same card on
# the Volume -- this test, else any test or order with the same channels --
# and only when there is none does it fall back to UC, segments by number,
# temperatures (flagged: phase above ~1 kHz then carries a slot error).

_FAMOS_NAME = re.compile(r"_current_(\d+(?:[.,]\d+)?)\s*a_test_(\d+)_karte_(\d+)",
                         re.I)


def famos_key(name):
    """(condition, test, card) of a card upload, e.g.
    'RO2612085-01_Current_450A_Test_01_Karte_3.DAT' -> ('450A', '01', 3)."""
    m = _FAMOS_NAME.search(Path(str(name or "")).name)
    if not m:
        return None
    return current_label(m.group(1)), m.group(2).zfill(2), int(m.group(3))


def famos_cards(spark, leepa) -> dict:
    """{(condition, test): {card: row}} of the order's FAMOS uploads; when a
    card was uploaded twice the earliest upload is used."""
    df = files(spark, leepa)
    out: dict = {}
    for r in df[df["kind"] == "famos"].itertuples():
        k = famos_key(r.raw_file_name)
        if k:
            out.setdefault(k[:2], {}).setdefault(k[2], r)
    return out


def card_channels(spark, file_id):
    """({channel: samples}, fs, info) of one card, each channel in time order.

    A sample that datago holds more than once under the same time (the card
    ingested twice: RO2611976 holds every sample of its cards twice) is kept
    once; info counts them. Repeats with DIFFERENT values cannot be resolved
    and are counted as conflicts (famos_card rejects such a card)."""
    q = spark.sql(f"""
        SELECT channel, CAST(time_value AS DOUBLE) AS t,
               CAST(value AS DOUBLE) AS v
        FROM {S.DATAGO_VAL_TBL} WHERE file_id = '{file_id}'""")
    try:                         # one Arrow transfer: ~20 s for 90 M rows
        import pyarrow.compute as pc
        tb = q.toArrow()
        enc = pc.dictionary_encode(tb["channel"]).combine_chunks()
        codes = enc.indices.to_numpy(zero_copy_only=False)
        names = enc.dictionary.to_pylist()
        t = np.asarray(tb["t"].to_numpy(), float)
        v = np.asarray(tb["v"].to_numpy(), float)
        del tb, enc
    except (ImportError, AttributeError):
        pdf = q.toPandas()
        cat = pdf["channel"].astype(str).astype("category")
        codes = cat.cat.codes.to_numpy()
        names = [str(c) for c in cat.cat.categories]
        t = pdf["t"].to_numpy(float)
        v = pdf["v"].to_numpy(float)
        del pdf, cat
    info = {"rows": int(len(t)), "repeated": 0, "conflicts": 0,
            "t_span_s": 0.0}
    if len(t) == 0:
        return {}, 0.0, info
    # one channel at a time: a whole-card sort needs ~5x the card in memory
    out, dts = {}, []
    for c, nm in enumerate(names):
        idx = np.flatnonzero(codes == c)
        if idx.size == 0:
            continue
        tc = t[idx]
        o = np.argsort(tc, kind="stable")
        ts, vs = tc[o], v[idx[o]]
        del idx, tc, o
        rep = np.flatnonzero(ts[1:] == ts[:-1]) + 1
        if rep.size:
            info["repeated"] += int(rep.size)
            info["conflicts"] += int(np.count_nonzero(vs[rep] != vs[rep - 1]))
            keep = np.ones(ts.size, bool)
            keep[rep] = False
            ts, vs = ts[keep], vs[keep]
        out[str(nm)] = np.ascontiguousarray(vs)
        info["t_span_s"] = max(info["t_span_s"], float(ts[-1] - ts[0]))
        d = np.diff(ts[:200001])
        d = d[d > 0]
        if d.size:
            dts.append(float(np.median(d)))
        del ts, vs
    del codes, t, v
    dt = float(np.median(dts)) if dts else 0.0
    fs = 1.0 / dt if dt > 0 else 0.0
    if fs and abs(fs - round(fs)) < 1e-6 * fs:
        fs = float(round(fs))
    return out, fs, info


def default_slot_order(names) -> list[str]:
    """UC channels, segments by number, temperatures, the rest -- only when
    no .DAT header of the card states the real order."""
    uc = sorted(n for n in names if n.upper().startswith("UC"))
    seg = sorted((n for n in names if re.fullmatch(r"\d+", n)), key=int)
    tmp = sorted(n for n in names if n.lower().startswith("temp"))
    rest = sorted(set(names) - set(uc) - set(seg) - set(tmp))
    return uc + seg + tmp + rest


def write_famos_v1(path, names, cols, fs, start=None) -> Path:
    """A FAMOS-v1 .DAT that eis_local.FamosV1 reads: float32 samples,
    interleaved in the order of `names`, every name in one |CP."""
    n = min(len(c) for c in cols)
    data = np.empty((n, len(names)), dtype="<f4")
    for j, c in enumerate(cols):
        data[:, j] = c[:n]
    cp = ",".join(f"7,32,{nm}" for nm in names)
    nt = (f"|NT,1,13,{start:%d.%m.%Y},0,{start:%H:%M:%S}.00;"
          if start is not None else "")
    hdr = (f"|CF,2,1,1;|CK,1,3,1,1;{nt}|CD,2,{1.0 / fs!r},1;"
           f"|CR,1,{len(names)},1,0,1;|CP,{cp};|CS,1,{data.nbytes},")
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(hdr.encode("latin-1"))
        data.tofile(fh)
    return path


def _local_start(row):
    """t_begin (UTC) as the naive local time a FAMOS |NT holds."""
    try:
        t = pd.Timestamp(row.t_begin)
        if pd.isna(t):
            return None
        t = t.tz_localize("UTC") if t.tzinfo is None else t
        return t.tz_convert(S.DATAGO_LOCAL_TZ).tz_localize(None).to_pydatetime()
    except Exception:                                       # noqa: BLE001
        return None


def compare_with_dat(fam_dg, dat, n_probe=4, probe_len=4000) -> str:
    """'' when the rebuilt card holds the samples of the .DAT `dat` under the
    same names, else what differs."""
    from eis_local import FamosFile
    if not isinstance(dat, (str, Path)):
        ref, dat = dat, dat.path
    else:
        try:
            ref = FamosFile(dat)
        except Exception as exc:                            # noqa: BLE001
            return f"{Path(dat).name} not readable ({exc})"
    if sorted(ref.names) != sorted(fam_dg.names):
        return f"other channels than {Path(dat).name}"
    n = min(ref.n_samples, fam_dg.n_samples)
    if abs(ref.n_samples - fam_dg.n_samples) > 0.01 * ref.n_samples:
        return (f"{fam_dg.n_samples} samples per channel "
                f"({fam_dg.n_samples / fam_dg.fs:.1f} s at {fam_dg.fs:g} Hz), "
                f"{Path(dat).name} has {ref.n_samples} "
                f"({ref.n_samples / ref.fs:.1f} s at {ref.fs:g} Hz)")
    L = int(min(probe_len, max(1, n // (n_probe + 1))))
    starts = [int(k * (n - L) / max(1, n_probe - 1)) for k in range(n_probe)]
    bad = []
    for nm in fam_dg.names:
        a, b = fam_dg.channel(nm), ref.channel(nm)
        pa = np.concatenate([a[s:s + L] for s in starts])
        pb = np.concatenate([b[s:s + L] for s in starts])
        if not np.allclose(pa, pb, rtol=1e-5, atol=1e-6):
            bad.append(nm)
    return (f"channel(s) {', '.join(bad)} differ from {Path(dat).name}"
            if bad else "")


def famos_card(spark, row, out_path, slot_dat=None, same_dat=None,
               say=print) -> tuple[Path | None, str]:
    """Rebuild one card from datago at `out_path` and check it.

    slot_dat: a .DAT (or a list of candidates) whose header gives this
    card's channel (slot) order -- the first with the same channels is used;
    same_dat: the Volume copy of THIS recording, compared sample by sample.
    Returns (path, note); path None when datago's card is not usable."""
    from eis_local import FamosFile, channel_integrity, famos_start_time
    t0 = datetime.now()
    chans, fs, info = card_channels(spark, row.file_id)
    if not chans or fs <= 0:
        return None, "no samples in datago"
    if info["conflicts"]:
        return None, (f"{info['conflicts']:,} samples are held twice with "
                      f"different values (two recordings under one file_id?)")
    dup = (f", {info['repeated']:,} repeated rows dropped"
           if info["repeated"] else "")
    names, basis = None, "approximate (UC, segments by number, temperatures)"
    start = _local_start(row)
    slots = (list(slot_dat) if isinstance(slot_dat, (list, tuple))
             else [slot_dat])
    same_hdr = None
    for src in [same_dat, *slots]:
        if src is None:
            continue
        try:
            hdr = FamosFile(src)
        except Exception:                                   # noqa: BLE001
            continue
        if src is same_dat:
            same_hdr = hdr
        if sorted(hdr.names) == sorted(chans):
            names = sorted(hdr.names, key=hdr.position)
            basis = f"from {Path(src).name}"
            if abs(hdr.fs - fs) < 1e-3 * fs:
                fs = float(hdr.fs)
            if src is same_dat:
                start = famos_start_time(src) or start
            break
    if names is None:
        names = default_slot_order(chans)
    lens = {len(chans[n]) for n in names}
    if max(lens) - min(lens) > 0.01 * max(lens):
        return None, f"channels of unequal length {sorted(lens)}"
    path = write_famos_v1(out_path, names, [chans[n] for n in names], fs, start)
    del chans
    fam = FamosFile(path)
    ref = next((n for n in fam.uc_names if n.upper() == "UC2"),
               (fam.uc_names or [None])[0])
    bad = channel_integrity(fam, ref)
    if not bad and same_dat is not None:
        diff = compare_with_dat(fam, same_hdr if same_hdr is not None
                                else same_dat)
        bad = [diff] if diff else []
    if bad:
        Path(path).unlink(missing_ok=True)
        return None, "; ".join(bad) + dup
    secs = (datetime.now() - t0).total_seconds()
    return path, (f"{fam.n_ch} ch, {fam.n_samples:,} samples at "
                  f"{fam.fs:g} Hz, slots {basis}{dup}, {secs:.0f} s")


# ---------------------------------------------------------------------------
# the ORIGINAL uploaded .DAT (any FAMOS version)
# ---------------------------------------------------------------------------
#
# The datago ROWS of a FAMOS-v2 card (DASYLab 14) hold UC2 under every channel
# name: the importer ignored the |CP byte offset, so the other 15 columns were
# never written to the values table and cannot be rebuilt from it. If datago
# keeps the uploaded file itself, that file is complete, and it is read here
# with the pipeline's own reader (which handles v2).

_URI = ("/Volumes/", "/dbfs/", "dbfs:/", "abfss://", "abfs://", "wasbs://",
        "s3://", "s3a://", "gs://", "file:/")


def file_record(spark, file_id) -> dict:
    """Every column of the datago file record (metafiles_delta) of one file."""
    pdf = spark.sql(f"SELECT * FROM {S.DATAGO_FILES_TBL} "
                    f"WHERE file_id = '{file_id}' LIMIT 1").toPandas()
    return {} if pdf.empty else {k: pdf.iloc[0][k] for k in pdf.columns}


def raw_file_candidates(rec: dict, order: str = "") -> list[tuple[str, str]]:
    """[(where it came from, path or URI)] the original upload may be at."""
    name = Path(str(rec.get("raw_file_name") or "")).name
    out = []
    if S.DATAGO_RAW_FILE_TEMPLATE:
        out.append(("DATAGO_RAW_FILE_TEMPLATE", S.DATAGO_RAW_FILE_TEMPLATE.format(
            raw_file_name=name, file_id=rec.get("file_id", ""),
            order=order)))
    for col, val in rec.items():
        if not isinstance(val, str) or col == "raw_file_name":
            continue
        v = val.strip()
        if (v.startswith(_URI) or (v.startswith("/") and v.count("/") >= 2)
                or ("/" in v and v.lower().endswith(".dat"))):
            out.append((f"column {col}", v))
            if name and not v.lower().endswith(".dat"):   # a folder
                out.append((f"column {col} + name", v.rstrip("/") + "/" + name))
    seen, uniq = set(), []
    for w, v in out:
        if v not in seen:
            seen.add(v)
            uniq.append((w, v))
    return uniq


def _fetch(spark, uri: str, dest: Path) -> bool:
    """Copy `uri` to `dest`: a local / Volume / DBFS-fuse path directly, any
    other URI through Spark's binaryFile reader."""
    local = "/dbfs/" + uri[len("dbfs:/"):].lstrip("/") if uri.startswith("dbfs:/") else uri
    if local.startswith("/") and Path(local).is_file():
        shutil.copyfile(local, dest)
        return True
    if local.startswith("/"):
        return False
    rows = (spark.read.format("binaryFile").load(uri)
            .select("content").limit(1).collect())
    if not rows:
        return False
    dest.write_bytes(bytes(rows[0]["content"]))
    return True


def raw_card(spark, row, out_path, order="", say=print):
    """(path, note) of the ORIGINAL upload of this card, copied to
    `out_path` and checked; path None (and why) when datago keeps none."""
    from eis_local import FamosFile, channel_integrity
    try:
        rec = file_record(spark, row.file_id)
    except Exception as exc:                                # noqa: BLE001
        return None, f"file record not read ({exc})"
    tried = []
    for where, uri in raw_file_candidates(rec, order):
        out_path = Path(out_path)
        try:
            ok = _fetch(spark, uri, out_path)
        except Exception as exc:                            # noqa: BLE001
            tried.append(f"{where}: {uri} ({type(exc).__name__})")
            continue
        if not ok:
            tried.append(f"{where}: {uri} (not found)")
            continue
        try:
            fam = FamosFile(out_path)
            bad = channel_integrity(
                fam, next((n for n in fam.uc_names if n.upper() == "UC2"),
                          (fam.uc_names or [None])[0]))
        except Exception as exc:                            # noqa: BLE001
            bad = [f"not a readable FAMOS file ({exc})"]
        if bad:
            out_path.unlink(missing_ok=True)
            tried.append(f"{where}: {uri} ({'; '.join(bad)})")
            continue
        return out_path, (f"original upload {uri} ({type(fam).__name__[-2:]}, "
                          f"{fam.n_ch} ch, {fam.n_samples:,} samples)")
    if not tried:
        cols = ", ".join(sorted(rec)) or "none"
        return None, (f"no path to the original file in its datago record "
                      f"(columns: {cols}); set DATAGO_RAW_FILE_TEMPLATE if "
                      f"datago keeps the uploads somewhere")
    return None, "original upload not usable: " + " | ".join(tried)
