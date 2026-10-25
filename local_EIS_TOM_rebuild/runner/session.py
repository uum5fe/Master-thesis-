"""
session.py
==========
Everything the runner notebook does, behind a handful of calls:

    R = runner.Session(globals())        # setup cell
    R.widgets()                          # order, conditions, test, plate, ...
    R.run()                              # cache or rerun, one folder each
    R.show_frequency_response()
    R.show_nyquist()
    R.show_median_nyquist()
    R.show_heatmaps()
    R.show_mf4()
    R.show_ecm()

Each show_* draws ONE viewer with drop-downs and saves it automatically as
<EIS_Results>/<order>/plots/<name>.html (and the static PNGs beside it).

Saved results, one ordered folder per order / condition / test:

    EIS_Results/2612086/60A_Test02/{bronze,silver,gold,ecm,plots}/
    EIS_Results/2612086/plots/{nyquist,median_nyquist,heatmaps,mf4,ecm,...}.html

Run mode 'cache' uses a saved result and runs only what is missing; 'rerun'
recomputes and OVERWRITES the saved result. A saved result made with other
settings than the current ones is still used under 'cache', but announced.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import datago
from . import settings as S
from . import views

#: Settings that decide what a run keeps: their digest is recorded with every
#: saved result, and a mismatch is announced when it is loaded.
IDENTITY_KEYS = (
    "plate", "plate_orientation", "test_id", "f_min_hz", "f_max_hz", "full_band", "eval_f_max_hz",
    "min_snr_db", "preset_name", "gamry_version", "exclude_segments",
    "substitute_segments", "fill_missing_from_neighbours", "channel_lag",
    "gamry_sync", "gamry_match", "card_gain", "uc_series_mohm_cm2", "csv_mirror_x",
    "param_method", "ecm_compare", "r_ohmic_method", "ref_channel",
    "skew_model", "hf_use_ensemble", "hf_ladder_prune", "f_hi_frac_fs",
    "window_min_snr_db",
)

#: identity keys added after results were already saved, with the value
#: every older result implicitly had -- a saved result without the key is
#: not "made with other settings" when the current value is this one
_IDENTITY_SINCE = {"plate_orientation": "standard", "gamry_match": "time"}

#: widgets of earlier runner versions, taken off the bar
_OLD_WIDGETS = ("csv_path", "csv_dialect", "csv_tones", "gain_file",
                "gamry_dir", "bench_log", "condition", "param_profile",
                "f_min_hz", "f_max_hz", "f_band", "min_snr_db", "stop_after",
                "evaluation_mode", "cache_write")


def condition_key(cond) -> tuple:
    """'45A' < '60A' < '150A' < '450A'."""
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(cond))
    return (float(m.group(1)) if m else float("inf"), str(cond))


class Session:
    """The notebook's state: environment, selection, results."""

    def __init__(self, env: dict | None = None, **paths):
        env = env or {}
        self.dbutils = env.get("dbutils")
        self.spark = env.get("spark")
        self._display_html = env.get("displayHTML")
        self._display = env.get("display")
        # paths can be overridden (tests, another site)
        self.famos_root = Path(paths.get("famos_root", S.FAMOS_ROOT))
        self.gamry_root = Path(paths.get("gamry_root", S.GAMRY_ROOT))
        self.results_root = Path(paths.get("results_root", S.RESULTS_ROOT))
        self.config_root = Path(paths.get("config_root", S.EIS_CONFIG_ROOT))
        self.package_dir = paths.get("package_dir")
        self.scratch = Path(paths.get("scratch", Path(tempfile.gettempdir())
                                      / f"eis_{os.getuid()}"))
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, Path] = {}
        self.sel: dict = {}

    # ------------------------------------------------------------------
    # output helpers
    # ------------------------------------------------------------------
    def say(self, *a):
        print(*a)

    def html(self, fragment: str):
        if self._display_html is not None:
            self._display_html(fragment)
        else:                                   # plain python: keep a copy
            p = self.scratch / "last_view.html"
            import viewers
            viewers.save_page(fragment, p)
            self.say(f"  (no displayHTML -- written to {p})")

    def _w(self, name, default=""):
        try:
            return self.dbutils.widgets.get(name)
        except Exception:                                   # noqa: BLE001
            return self.sel.get(f"widget:{name}", default)

    def _make(self, kind, name, default, choices=None, label=""):
        if self.dbutils is None:
            return
        try:
            w = self.dbutils.widgets
            if kind == "dropdown":
                w.dropdown(name, default, list(choices), label)
            elif kind == "multiselect":
                w.multiselect(name, default, list(choices), label)
            else:
                w.text(name, default, label)
        except Exception:                                   # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    def _famos_files(self, leepa) -> list[Path]:
        out = []
        for pat in (f"Leepa_{leepa}_Current_*_Karte_*.DAT",
                    f"Leepa_RO{leepa}_Current_*_Karte_*.DAT",
                    f"RO{leepa}-*_Current_*_Karte_*.DAT",
                    f"{leepa}_Current_*_Karte_*.DAT"):
            try:
                out += sorted(self.famos_root.glob(pat))
            except OSError:
                pass
        return out

    def orders(self) -> list[str]:
        found = set()
        if self.spark is not None:
            try:
                found |= set(datago.orders(self.spark))
            except Exception as exc:                        # noqa: BLE001
                self.say(f"  datago order list not read ({exc})")
        try:
            for f in self.famos_root.glob("*_Current_*_Karte_*.DAT"):
                m = re.search(r"(?:Leepa_RO|Leepa_|RO)(\d{7})", f.name)
                if m:
                    found.add(m.group(1))
        except OSError:
            pass
        return sorted(found)

    def _famos_datago_on(self) -> bool:
        return bool(S.DATAGO_FIRST and S.FAMOS_FROM_DATAGO
                    and self.spark is not None)

    def _datago_famos_cards(self, leepa) -> dict:
        """{(condition, test): {card: row}} of the FAMOS uploads in datago,
        {} when FAMOS_FROM_DATAGO is off or datago cannot be read."""
        if not self._famos_datago_on():
            return {}
        try:
            return datago.famos_cards(self.spark, leepa)
        except Exception as exc:                            # noqa: BLE001
            self.say(f"  datago FAMOS uploads not read ({exc}) -> Volume")
            return {}

    def conditions_and_tests(self, leepa):
        """({condition: [test ids]}) from the FAMOS file names on the Volume
        and, with FAMOS_FROM_DATAGO, the FAMOS uploads in datago."""
        out: dict[str, set] = {}
        for f in self._famos_files(leepa):
            m = re.search(r"_Current_([^_]+)_Test_(\d+)_", f.name)
            if m:
                out.setdefault(m.group(1), set()).add(m.group(2).zfill(2))
        for cond, test in self._datago_famos_cards(leepa):
            out.setdefault(cond, set()).add(test)
        return {c: sorted(t) for c, t in sorted(out.items(),
                                                 key=lambda kv: condition_key(kv[0]))}

    def plate_version(self, leepa) -> str:
        import gamry_compare as gc_
        if leepa in S.PLATE_VERSION_OVERRIDE:
            return S.PLATE_VERSION_OVERRIDE[leepa]
        digits = "".join(ch for ch in str(leepa) if ch.isdigit())
        seen: dict[str, int] = {}
        names = []
        if self.spark is not None:
            try:
                names += list(datago.files(self.spark, leepa)["raw_file_name"]
                              .dropna())
            except Exception:                               # noqa: BLE001
                pass
        for root in (self.gamry_root, self.gamry_root.parent, self.famos_root):
            try:
                names += [f.name for f in Path(root).iterdir()
                          if digits in f.name]
            except OSError:
                pass
        try:                 # e.g. Gamry/RO2612090/..._V26_112_lokale_EIS.mf4
            names += [f.name for f in self.gamry_root.rglob(f"*{digits}*")]
        except OSError:
            pass
        for n in names:
            v = gc_.version_of(str(n))
            if v:
                seen[v] = seen.get(v, 0) + 1
        return max(seen, key=seen.get) if seen else ""

    def plate_hint(self, leepa, plate) -> str:
        """'' or a warning when the order's file names (the bench .mf4:
        ..._V26_113_lokale_EIS_6_Boxen_Naboo_1.mf4) name the OTHER plate."""
        import site_paths
        digits = "".join(ch for ch in str(leepa) if ch.isdigit())
        names = []
        for root in (self.gamry_root, self.gamry_root.parent):
            try:
                names += [f.name for f in Path(root).iterdir()
                          if digits in f.name]
            except OSError:
                pass
        if self.spark is not None:
            try:
                names += list(datago.files(self.spark, leepa)["raw_file_name"]
                              .dropna().astype(str))
            except Exception:                               # noqa: BLE001
                pass
        said = {p for p, toks in site_paths.PLATE_TOKENS.items()
                for n in names for t in toks[:1] if t in n.lower()}
        if said and plate not in said:
            return (f"the files of RO{leepa} name plate "
                    f"{'/'.join(sorted(said))} (e.g. Naboo = gen2, Kashyyyk "
                    f"= gen1), but {plate} is selected -- check the Plate "
                    f"drop-down")
        return ""

    def _volume_cards(self, leepa, cond, test, deep: bool = False) -> dict:
        """{card: .DAT} of this order / condition / test on the Volume.

        deep: also look up to three folders below FAMOS_ROOT, for any file
        naming this order whose name reads as this condition / test / card
        (Leepa_<order>_..., RO<order>-01_..., <order>_...)."""
        out = {}
        for f in self._famos_files(leepa):
            k = datago.famos_key(f.name)
            if k and k[0] == cond and k[1] == test:
                out.setdefault(k[2], f)
        if deep:
            digits = "".join(ch for ch in str(leepa) if ch.isdigit())
            for depth in range(1, 4):
                try:
                    hits = sorted(self.famos_root.glob(
                        "*/" * depth + f"*{digits}*_Karte_*"))
                except OSError:
                    hits = []
                for f in hits:
                    if f.suffix.lower() != ".dat" or not f.is_file():
                        continue
                    k = datago.famos_key(f.name)
                    if k and k[0] == cond and k[1] == test:
                        out.setdefault(k[2], f)
        return out

    def _report_missing_cards(self, leepa, cond, test, have):
        """Say which card of the order is absent from this condition / test,
        and where datago does have it -- a card that is simply not uploaded
        must not look like a card the pipeline dropped."""
        try:
            allc = datago.famos_cards(self.spark, leepa)
        except Exception:                                   # noqa: BLE001
            allc = {}
        expected = set().union(*[set(v) for v in allc.values()]) if allc else set()
        for card in sorted(expected - set(have)):
            elsewhere = sorted(f"{c} Test_{t}" for (c, t), v in allc.items()
                               if card in v)
            self.say(f"    Karte_{card}: NOT in datago for {cond} Test_{test} "
                     f"and not on the Volume -- nothing to evaluate"
                     + (f" (datago has Karte_{card} for "
                        f"{', '.join(elsewhere[:6])})" if elsewhere else ""))
        try:
            odd = datago.famos_unparsed(self.spark, leepa)
        except Exception:                                   # noqa: BLE001
            odd = []
        if odd:
            self.say(f"    datago .DAT uploads of RO{leepa} whose name does not "
                     f"read as <current>_Test_<nn>_Karte_<n> (not used): "
                     f"{', '.join(odd[:8])}")

    @staticmethod
    def _why_rows_broken(leepa, card, dats) -> str:
        """The FAMOS format of this card's other recordings on the Volume:
        datago reads v1 files correctly and v2 files wrongly."""
        from eis_local import FamosFile
        import contextlib
        import io
        for d in dats or []:
            if d is None:
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    kind = type(FamosFile(d)).__name__[-2:].lower()
            except Exception:                               # noqa: BLE001
                continue
            if kind == "v2":
                return (f"why: Karte_{card} is recorded as a FAMOS-v2 file "
                        f"({Path(d).name} is v2; cards written as v1 are read "
                        f"correctly). datago's importer does not apply v2's "
                        f"per-channel byte offset and stores the first "
                        f"column, UC2, under all 16 names -- the segment "
                        f"channels are not in datago at all")
        return ""

    def _find_on_volume(self, name, depth: int = 3):
        """A file of exactly this name under the FAMOS folder, up to `depth`
        sub-folders down (an order's cards kept in their own folder), or
        None."""
        name = Path(str(name or "")).name
        if not name:
            return None
        for k in range(depth + 1):
            try:
                hit = next(iter(sorted(self.famos_root.glob("*/" * k + name))),
                           None)
            except OSError:
                hit = None
            if hit is not None and hit.is_file():
                return hit
        return None

    def _slot_source(self, leepa, card) -> list:
        """A Volume .DAT of the same card (this order first, then any) whose
        header gives the card's channel order."""
        mine = [f for f in self._famos_files(leepa)
                if (datago.famos_key(f.name) or (0, 0, 0))[2] == card]
        try:
            others = sorted(self.famos_root.glob(f"*_Karte_{card}.DAT"),
                            reverse=True)[:30]
        except OSError:
            others = []
        return mine + [f for f in others if f not in mine]

    def famos_dir(self, cond):
        """(folder, {card: source}) with the FAMOS cards of this run.

        FAMOS_FROM_DATAGO off (or nothing in datago): the Volume folder.
        On: a scratch folder with every card of the condition and test,
        each from the first source that passes its checks:
          1. datago's ORIGINAL upload of the card (datago.raw_card) -- exact,
             any FAMOS version, the only datago source of a v2 card;
          2. the card rebuilt from datago's rows (datago.famos_card) -- v1
             cards only, v2 rows hold UC2 under every name;
          3. the Volume .DAT."""
        s = self.sel
        leepa, test = s["leepa"], s["test_id"]
        dg = self._datago_famos_cards(leepa).get((cond, test), {})
        if not dg:
            if self._famos_datago_on():
                self.say(f"    {cond} Test_{test}: no FAMOS cards in datago "
                         f"-> Volume")
            return self.famos_root, {}
        fallback = bool(getattr(S, "FAMOS_VOLUME_FALLBACK", True))
        vol = self._volume_cards(leepa, cond, test, deep=True)
        for card, row in dg.items():          # the same file in a sub-folder
            if card not in vol:
                hit = self._find_on_volume(getattr(row, "raw_file_name", ""))
                if hit is not None:
                    vol[card] = hit
        out = self.scratch / "datago_famos" / f"RO{leepa}" / f"{cond}_Test{test}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        self._report_missing_cards(leepa, cond, test, set(dg) | set(vol))
        # Every card of the run under ONE naming convention -- the datago
        # upload's own (RO2612091-01_Current_...), so the files read as what
        # they are. One convention, because bronze takes the first naming
        # pattern that matches: two conventions in one folder would split the
        # cards and silently drop some.
        prefix = next((m.group(1) for r in dg.values()
                       for m in [re.match(r"(.+?)_Current_",
                                          Path(str(r.raw_file_name)).name)]
                       if m), f"Leepa_{leepa}")
        src, raw_said = {}, False
        for card in sorted(set(dg) | set(vol)):
            name = f"{prefix}_Current_{cond}_Test_{test}_Karte_{card}.DAT"
            path = None
            if card in dg:
                try:
                    path, raw_note = datago.raw_card(
                        self.spark, dg[card], out / name, order=f"RO{leepa}",
                        say=self.say)
                except Exception as exc:                    # noqa: BLE001
                    path, raw_note = None, f"{type(exc).__name__}: {exc}"
                if path is not None:
                    src[card] = "datago (original file)"
                    self.say(f"    Karte_{card}: datago {raw_note}")
                    continue
                if not raw_said:          # once per run: why not the original
                    self.say(f"    datago original .DAT files: {raw_note}")
                    raw_said = True
                same = vol.get(card)
                slot = [] if same else self._slot_source(leepa, card)
                try:
                    path, note = datago.famos_card(
                        self.spark, dg[card], out / name, slot_dat=slot,
                        same_dat=same, say=self.say)
                except Exception as exc:                    # noqa: BLE001
                    path, note = None, f"{type(exc).__name__}: {exc}"
                if path is not None:
                    src[card] = "datago (rows)"
                    self.say(f"    Karte_{card}: datago rows ({note})")
                    if "approximate" in note:
                        self.say(f"      no .DAT of Karte_{card} on the Volume "
                                 f"states its channel order -- slot delay "
                                 f"correction approximate (phase above ~1 kHz)")
                    continue
                self.say(f"    Karte_{card}: datago REJECTED -- {note}")
                why = self._why_rows_broken(leepa, card, slot or [same])
                if why:
                    self.say(f"      {why}")
            if card in vol and (fallback or card not in dg):
                try:
                    (out / name).symlink_to(vol[card])
                except OSError:
                    shutil.copy2(vol[card], out / name)
                src[card] = "Volume"
                self.say(f"    Karte_{card}: Volume {vol[card]}"
                         + (" -- used for this card only, in place of the "
                            "rejected datago copy" if card in dg else ""))
            elif card in vol:
                self.say(f"    Karte_{card}: left out -- the Volume has "
                         f"{vol[card].name}, but FAMOS_VOLUME_FALLBACK is "
                         f"False")
            else:
                self.say(f"    Karte_{card}: left out -- datago's copy was "
                         f"rejected and no {cond} Test_{test} Karte_{card} "
                         f"file of RO{leepa} is on the Volume (searched "
                         f"{self.famos_root} and three folders below). Put "
                         f"{dg[card].raw_file_name} there and rerun; it is "
                         f"then used for this card only")
        return out, src

    @staticmethod
    def _has_sweeps(r) -> bool:
        """.dta sweeps, or Gamry sweeps exported as .mf4 (..._CurrVal_450.mf4)."""
        import gamry_compare as gc_
        try:
            return r.is_dir() and (
                any(r.glob("*.dta")) or any(r.glob("*.DTA"))
                or any(gc_.is_sweep_mf4(f.name) for f in r.glob("*.mf4")))
        except OSError:
            return False

    def _volume_gamry_root(self, leepa):
        for t in S.GAMRY_SEARCH_ROOT_TEMPLATES:
            r = Path(t.format(ev=self.gamry_root.parent, leepa=leepa))
            if self._has_sweeps(r):
                return r
        if self._has_sweeps(self.gamry_root):
            return self.gamry_root
        return None

    def gamry_dir(self, leepa):
        """Folder with every whole-cell sweep of the order: datago first
        (all sweeps, as .dta with their start times), then the Volume. The
        pipeline picks the sweep of each run by file or by time
        (settings.GAMRY_MATCH, gamry_compare.select_sweep)."""
        vol = self._volume_gamry_root(leepa)
        why = ""
        if not (S.DATAGO_FIRST and S.GAMRY_FROM_DATAGO):
            why = ("switched off (settings DATAGO_FIRST / GAMRY_FROM_DATAGO "
                   "is False)")
        elif self.spark is None:
            why = "no Spark session, so datago cannot be read"
        else:
            try:
                d = datago.gamry_dir(self.spark, leepa, vol, say=self.say)
                if d is not None:
                    return d
                why = f"datago holds no Gamry sweep of RO{leepa}"
            except Exception as exc:                        # noqa: BLE001
                why = f"datago Gamry sweeps not read ({exc})"
        self.say(f"  Gamry sweeps NOT from datago: {why} -> "
                 + (f"Volume folder {vol}" if vol else "no Gamry reference"))
        return vol

    # ------------------------------------------------------------------
    # widgets
    # ------------------------------------------------------------------
    def widgets(self):
        """Create / read the widgets, resolve every file of the selection."""
        import r2d2_geometry as geom
        import site_paths
        import csv_source
        if self.dbutils is not None:
            for n in _OLD_WIDGETS:
                try:
                    self.dbutils.widgets.remove(n)
                except Exception:                           # noqa: BLE001
                    pass
        orders = self.orders() or ["2612086"]
        default = orders[-1]
        self._make("dropdown", "leepa_id", default, orders, "Order ID (Leepa)")
        leepa = self._w("leepa_id", default)
        if leepa not in orders:
            leepa = default
        self._make("dropdown", "source_format", "famos", ["famos", "csv"],
                   "Measurement file format")
        source = self._w("source_format", "famos").strip().lower()
        self._make("dropdown", "plate", "gen1", ["gen1", "gen2"],
                   "Plate (gen1 green/Kashyyyk, gen2 blue/Naboo)")
        plate = geom.plate_key(self._w("plate", "gen1"))
        self._make("dropdown", "plate_orientation", "standard",
                   list(geom.ORIENTATIONS),
                   "Plate position (standard = as drawn; rot180 = turned "
                   "180°, inlet and outlet segments swapped)")
        orientation = geom.orientation_key(
            self._w("plate_orientation", "standard"))

        csv_folders, campaign = {}, ""
        if source == "csv":
            camps = {}
            try:
                camps = csv_source.campaigns(S.CSV_ROOT)
            except Exception as exc:                        # noqa: BLE001
                self.say(f"  CSV discovery failed ({exc})")
            names = sorted(camps) or ["(none found)"]
            self._make("dropdown", "csv_campaign", names[0], names,
                       "CSV folder (under csv_files)")
            campaign = self._w("csv_campaign", names[0])
            csv_folders = csv_source.condition_map(camps.get(campaign, []))
            cond_tests = {c: ["01"] for c in sorted(csv_folders, key=condition_key)}
        else:
            cond_tests = self.conditions_and_tests(leepa)
        conditions = list(cond_tests) or ["45A", "60A", "150A", "450A"]
        tests = sorted({t for ts in cond_tests.values() for t in ts}) or ["01"]

        self._make("multiselect", "conditions", "ALL", ["ALL"] + conditions,
                   "Conditions")
        self._make("dropdown", "test_id", tests[0], tests,
                   "FAMOS test (Test_xx: 01 up-ramp, 02 down-ramp, ...)")
        self._make("text", "exclude_segments", "", "Exclude segments (e.g. 33)")
        self._make("text", "substitute_segments", "",
                   "Rebuild from neighbours (e.g. 33)")
        self._make("dropdown", "fill_gaps", "no", ["no", "yes"],
                   "Fill unmeasured segments from neighbours")
        self._make("dropdown", "run_mode", "cache", list(S.RUN_MODES),
                   "Run mode (rerun overwrites the saved result)")

        raw = [x.strip() for x in str(self._w("conditions", "ALL")).split(",")
               if x.strip()]
        selected = (conditions if not raw or any(x.upper() == "ALL" for x in raw)
                    else sorted(dict.fromkeys(raw), key=condition_key))
        test_id = str(self._w("test_id", tests[0]) or "01").strip().zfill(2)

        def seglist(name):
            return frozenset(x.strip() for x in str(self._w(name, ""))
                             .replace(";", ",").split(",") if x.strip())

        geom.use_plate(plate, orientation)
        files = site_paths.resolve(plate, root=self.config_root,
                                   package_dir=self.package_dir)
        if source == "csv":
            leepa = ("csv" if campaign in ("", ".") else
                     "csv_" + re.sub(r"[^A-Za-z0-9]+", "_", campaign).strip("_"))
        sel = dict(leepa=str(leepa), source=source, plate=plate,
                   plate_orientation=orientation,
                   plate_tag=(("" if plate == "gen1" else f"_{plate}")
                              + ("" if orientation == "standard"
                                 else f"_{orientation}")),
                   conditions=selected, all_conditions=conditions,
                   tests_by_condition=cond_tests, test_id=test_id,
                   exclude=seglist("exclude_segments"),
                   substitute=seglist("substitute_segments"),
                   fill_gaps=self._w("fill_gaps", "no") == "yes",
                   run_mode=self._w("run_mode", "cache"),
                   plate_files=files, csv_folders=csv_folders,
                   csv_campaign=campaign)
        if source == "famos":
            sel["gain_file"] = self._chain_gain(plate, files)
            sel["gamry_version"] = self.plate_version(leepa)
            gdir = self.gamry_dir(leepa)
            sel["gamry_dir"] = str(gdir) if gdir else ""
            sel["bench_log"] = ""
            try:
                import gamry_compare as gc_
                bl = gc_.find_bench_log(self.gamry_root, order_id=leepa,
                                        version=sel["gamry_version"] or None)
                sel["bench_log"] = str(bl) if bl else ""
            except Exception:                               # noqa: BLE001
                pass
        else:
            sel.update(gain_file="", gamry_version="", gamry_dir="",
                       bench_log="")
        self.sel = sel
        self._print_selection()
        return sel

    def _chain_gain(self, plate, files) -> str:
        """The plate's chain_gain.csv; built once from the Abgleich bode
        sweeps when it does not exist yet."""
        if files.get("gain"):
            return str(files["gain"])
        ab = files.get("abgleich")
        if not ab:
            self.say("  chain response: no chain_gain.csv and no Abgleich "
                     "delivery for this plate -- the top decade keeps the "
                     "amplifier roll-off (the in-situ channel lag still runs)")
            return ""
        try:
            import gamry_dta
            ab = Path(ab)
            sweeps = gamry_dta.read_bode_folder(ab / "bode")
            curr = ab / "coefficients" / "curr.csv"
            chk = gamry_dta.cross_check_abgleich(sweeps, curr)
            out = Path(files["target_dir"]) / "chain_gain.csv"
            gamry_dta.write_gain_csv(sweeps, out, curr_csv=curr,
                                     shared=not chk["ok"])
            self.say(f"  chain response built once from {len(sweeps)} Abgleich "
                     f"sweeps: {out}")
            return str(out)
        except Exception as exc:                            # noqa: BLE001
            self.say(f"  chain response not built ({exc})")
            return ""

    def _print_selection(self):
        s = self.sel
        import r2d2_geometry as geom
        lines = [
            f"  Order       {s['leepa']}   ({s['source']}, {geom.plate(s['plate']).title})",
            f"  Plate in    {geom.ORIENTATION_LABELS[s['plate_orientation']]}"
            f"   (gas and coolant ports fixed to the cell)",
            f"  Conditions  {', '.join(s['conditions'])}   Test_{s['test_id']}",
            f"  Run mode    {s['run_mode']}"
            + ("   (recompute and OVERWRITE the saved results)"
               if s['run_mode'] == 'rerun' else "   (saved results are used)"),
            f"  Gamry       {s.get('gamry_dir') or '(none)'}"
            + (f"   build {s['gamry_version']}" if s.get('gamry_version') else ""),
            f"  Chain gain  {s.get('gain_file') or '(none)'}",
            f"  Results     {self.order_dir()}",
        ]
        if s["exclude"]:
            lines.append(f"  Excluded    {', '.join(sorted(s['exclude'], key=int))}")
        if s["substitute"]:
            lines.append(f"  Rebuilt     {', '.join(sorted(s['substitute'], key=int))}")
        if s["source"] == "famos":
            hint = self.plate_hint(s["leepa"], s["plate"])
            if hint:
                lines.append("  !! " + hint)
        if s["plate_files"].get("missing") and s["source"] == "famos":
            import site_paths
            lines.append("  !! " + site_paths.missing_message(s["plate_files"]))
        self.say("\n".join(lines))

    # ------------------------------------------------------------------
    # result folders
    # ------------------------------------------------------------------
    def order_dir(self) -> Path:
        return self.results_root / f"{self.sel['leepa']}{self.sel['plate_tag']}"

    def cond_name(self, cond) -> str:
        """'60A_Test02' (+ '_<mode>' for a non-default evaluation mode)."""
        name = (str(cond) if self.sel["source"] == "csv"
                else f"{cond}_Test{self.sel['test_id']}")
        return name + ("" if S.EVALUATION_MODE == "default"
                       else f"_{S.EVALUATION_MODE}")

    def saved_dir(self, cond) -> Path:
        return self.order_dir() / self.cond_name(cond)

    def plots_dir(self) -> Path:
        d = self.order_dir() / "plots"
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / ".w").touch()
            (d / ".w").unlink()
            return d
        except OSError:
            d = self.scratch / self.order_dir().name / "plots"
            d.mkdir(parents=True, exist_ok=True)
            return d

    @staticmethod
    def _usable(d) -> bool:
        d = Path(d)
        return any((d / s / "spectra_clean.csv").is_file()
                   for s in ("silver", "csv"))

    # ------------------------------------------------------------------
    # the run
    # ------------------------------------------------------------------
    def _config(self, cond, out_dir):
        from config import DEFAULT
        s = self.sel
        files = s["plate_files"]
        cfg = DEFAULT.replace(
            plate=s["plate"], plate_orientation=s["plate_orientation"],
            source_format=s["source"],
            csv_path=s["csv_folders"].get(cond) if s["source"] == "csv" else None,
            csv_mirror_x=S.CSV_MIRROR_X,
            gain_file=Path(s["gain_file"]) if s["gain_file"] else None,
            hf_response_file=self._hf_response_file(),
            channel_lag=S.CHANNEL_LAG, gamry_sync=S.GAMRY_SYNC,
            gamry_match=getattr(S, "GAMRY_MATCH", "file"),
            card_gain=S.CARD_GAIN, uc_series_mohm_cm2=S.UC_SERIES_MOHM_CM2,
            freq_response=S.FREQ_RESPONSE,
            abgleich_bode_dir=(Path(files["abgleich"]) / "bode"
                               if files.get("abgleich") else None),
            heatmap_style=S.HEATMAP_STYLE,
            gamry_dir=Path(s["gamry_dir"]) if s["gamry_dir"] else None,
            gamry_version=s["gamry_version"],
            bench_log=Path(s["bench_log"]) if s["bench_log"] else None,
            dat_dir=self.famos_root, out_dir=out_dir,
            curr_cal=files.get("curr"), temp_cal=files.get("temp"),
            leepa=s["leepa"], condition=cond, test_id=s["test_id"],
            f_min_hz=S.F_MIN_HZ, f_max_hz=S.F_MAX_HZ,
            eval_f_max_hz=0.0 if S.FULL_BAND else 1200.0,
            full_band=S.FULL_BAND, write_png=True, write_html=True,
            infer_missing_segments=False, min_snr_db=S.MIN_SNR_DB,
            min_ref_channels=2,
            skew_model=getattr(S, "SKEW_MODEL", DEFAULT.skew_model),
            hf_use_ensemble=getattr(S, "HF_USE_ENSEMBLE", DEFAULT.hf_use_ensemble),
            hf_ladder_prune=getattr(S, "HF_LADDER_PRUNE", DEFAULT.hf_ladder_prune),
            f_hi_frac_fs=getattr(S, "F_HI_FRAC_FS", DEFAULT.f_hi_frac_fs),
            window_min_snr_db=getattr(S, "WINDOW_MIN_SNR_DB",
                                      DEFAULT.window_min_snr_db),
            exclude_segments=s["exclude"],
            substitute_segments=s["substitute"],
            fill_missing_from_neighbours=s["fill_gaps"])
        if S.EVALUATION_MODE != "default":
            cfg = cfg.preset(S.EVALUATION_MODE)
        return cfg

    def _hf_response_file(self):
        if not S.HF_RESPONSE:
            return None
        if S.HF_RESPONSE != "auto":
            p = Path(S.HF_RESPONSE)
            return p if p.is_file() else None
        p = self.order_dir() / "hf_response" / "hf_response.csv"
        return p if p.is_file() else None

    def identity(self, cfg) -> dict:
        def norm(v):
            if isinstance(v, (set, frozenset)):
                return sorted(str(x) for x in v)
            if isinstance(v, (tuple, list)):
                return [str(x) for x in v]
            return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
        payload = {k: norm(getattr(cfg, k, None)) for k in IDENTITY_KEYS}
        for k, f in (("gain_file", cfg.gain_file),
                     ("hf_response_file", cfg.hf_response_file)):
            try:
                payload[k + "_sha"] = (hashlib.sha256(Path(f).read_bytes())
                                       .hexdigest()[:12] if f else "")
            except OSError:
                payload[k + "_sha"] = ""
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True)
                                .encode()).hexdigest()[:10]
        return {"digest": digest, "settings": payload}

    def run(self):
        """Use or compute every selected condition (run mode), one saved
        folder each. Returns {condition: run folder}."""
        import bronze
        import main as pipeline_main
        import r2d2_geometry as geom
        s = self.sel
        geom.use_plate(s["plate"], s["plate_orientation"])
        if s["source"] == "famos" and s["plate_files"].get("missing"):
            import site_paths
            raise FileNotFoundError(site_paths.missing_message(s["plate_files"]))
        self.results = {}
        self._pub = None                     # read the runs table afresh
        self.say(f"  {'condition':<12}{'action':<10}folder")
        for cond in s["conditions"]:
            if (s["source"] == "famos"
                    and s["test_id"] not in s["tests_by_condition"].get(cond, [])):
                self.say(f"  {cond:<12}{'skip':<10}no Test_{s['test_id']} "
                         f"files for {cond}")
                continue
            saved = self.saved_dir(cond)
            scratch = self.scratch / self.order_dir().name / self.cond_name(cond)
            cfg = self._config(cond, scratch)
            ident = self.identity(cfg)
            if (s["run_mode"] == "cache" and not self._usable(saved)
                    and S.CACHE_FROM_TABLES):
                got = self.restore_run(cond)
                if got is not None:
                    saved = self._save(got, saved)
            if s["run_mode"] == "cache" and self._usable(saved):
                self.results[cond] = saved
                note = self._identity_note(saved, ident)
                self.say(f"  {cond:<12}{'saved':<10}{saved}{note}")
                continue
            self.say(f"  {cond:<12}{'run':<10}{saved}")
            if scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)
            scratch.mkdir(parents=True, exist_ok=True)
            n_cards, famos_src, famos_tmp = 0, {}, None
            if s["source"] == "famos" and self._famos_datago_on():
                dat_dir, famos_src = self.famos_dir(cond)
                if famos_src:
                    famos_tmp = dat_dir
                    cfg = cfg.replace(dat_dir=dat_dir)
            if s["source"] == "famos":
                try:
                    n_cards = len(bronze.discover_files(cfg))
                except SystemExit as exc:
                    self.say(f"    no files: {exc}")
                    continue
                if n_cards == 1:
                    cfg = cfg.replace(min_ref_channels=1)
            t0 = time.time()
            try:
                manifest = pipeline_main.run_pipeline(cfg, stop_after="gold")
            except BaseException as exc:                    # noqa: BLE001
                if isinstance(exc, KeyboardInterrupt):
                    raise
                self.say(f"    {cond}: FAILED -- {exc}")
                continue
            finally:
                if famos_tmp is not None:     # ~350 MB per rebuilt card
                    shutil.rmtree(famos_tmp, ignore_errors=True)
            (scratch / "run_identity.json").write_text(json.dumps(
                {**ident, "runner": S.RUNNER_VERSION, "condition": cond,
                 "test_id": s["test_id"], "order": s["leepa"],
                 "cards": n_cards,
                 "famos_source": ({f"Karte_{k}": v for k, v in famos_src.items()}
                                  or "Volume"),
                 "elapsed_s": round(time.time() - t0, 1),
                 "written": datetime.now().isoformat(timespec="seconds")},
                indent=2))
            self.results[cond] = self._save(scratch, saved)
            gs = manifest.get("stages", {}).get("gold", {})
            self.say(f"    {cond}: {gs.get('n_measured', '?')} segments, "
                     f"{time.time() - t0:.0f} s -> {self.results[cond]}")
            gc.collect()
            if self.spark is not None:
                try:
                    self.spark.catalog.clearCache()
                except Exception:                           # noqa: BLE001
                    pass
        return self.results

    # ------------------------------------------------------------------
    # selection without widgets, ECM, tables, summary
    # ------------------------------------------------------------------
    def select(self, **values):
        """The selection of widgets(), from these values instead of the
        notebook's widgets (jobs, tests), e.g. select(leepa_id="2612090",
        conditions="450A", test_id="01", run_mode="rerun", plate="gen2")."""
        keep = self.dbutils
        self.dbutils = None
        self.sel = {f"widget:{k}": str(v) for k, v in values.items()}
        try:
            return self.widgets()
        finally:
            self.dbutils = keep

    def fit_ecm(self, refit: bool = False) -> dict:
        """Standard and DRT-informed ECM of every selected run, saved in
        RUN_DIR/ecm/ (what show_ecm draws and the ecm table holds)."""
        return {cond: views.ecm_fits(d, refit=refit, say=self.say)
                for cond, d in self.run_dirs().items()}

    def publish(self) -> dict:
        """Copy every selected, saved run into the Delta tables
        (runner/tables.py). A run's earlier rows are replaced."""
        from . import tables
        if self.spark is None:
            self.say("  tables: no Spark session -- not published")
            return {}
        out = {}
        for cond, d in self.run_dirs().items():
            try:
                out[cond] = tables.publish(self.spark, d, say=self.say)
            except Exception as exc:                        # noqa: BLE001
                self.say(f"    {self.label(cond)}: tables not written "
                         f"({type(exc).__name__}: {exc})")
        return out

    def _published(self, refresh: bool = False) -> dict:
        """{run_key: publication time} of the runs table ({} without)."""
        if self.spark is None:
            return {}
        if refresh or getattr(self, "_pub", None) is None:
            from . import tables
            try:
                df = tables.runs(self.spark)
                self._pub = (dict(zip(df["run_key"].astype(str),
                                      df["published"].astype(str)))
                             if len(df) else {})
            except Exception as exc:                        # noqa: BLE001
                self.say(f"  runs table not read ({type(exc).__name__}: {exc})")
                self._pub = {}
        return self._pub

    def run_key_of(self, cond) -> str:
        return f"{self.order_dir().name}/{self.cond_name(cond)}"

    def restore_run(self, cond):
        """The published run of `cond`, written back from the run_files
        table into scratch/from_tables/<run_key>; None when not published."""
        from . import tables
        key = self.run_key_of(cond)
        pub = self._published().get(key)
        if pub is None:
            return None
        try:
            d = tables.restore(self.spark, key,
                               self.scratch / "from_tables" / key,
                               published=pub, say=self.say)
        except Exception as exc:                            # noqa: BLE001
            self.say(f"    {key}: not restored from the tables "
                     f"({type(exc).__name__}: {exc})")
            return None
        return d if d is not None and self._usable(d) else None

    def summary(self) -> pd.DataFrame:
        """One line per selected run: segments, Gamry sweep, checks, source."""
        from . import tables
        rows = [tables.run_record(d) for d in self.run_dirs().values()]
        cols = ["run_key", "n_measured", "n_inferred", "gamry_sweep",
                "plausibility_fail", "plausibility_warn", "famos_source",
                "runner", "written"]
        df = pd.DataFrame(rows)
        return df[[c for c in cols if c in df]] if len(df) else df

    def _identity_note(self, saved, ident) -> str:
        p = Path(saved) / "run_identity.json"
        try:
            old = json.loads(p.read_text())
        except Exception:                                   # noqa: BLE001
            return "   (no settings record -- choose rerun to renew it)"
        if old.get("digest") != ident["digest"]:
            was = {**_IDENTITY_SINCE, **old.get("settings", {})}
            diff = [k for k in ident["settings"]
                    if was.get(k) != ident["settings"][k]]
            if diff:
                return (f"   <-- made with OTHER settings ({', '.join(diff)}); "
                        f"choose rerun to recompute")
        if old.get("runner") != S.RUNNER_VERSION:
            return (f"   (saved by runner {old.get('runner')}; rerun to "
                    f"evaluate with {S.RUNNER_VERSION})")
        return f"   ({old.get('written', '')})"

    def _save(self, src: Path, dst: Path) -> Path:
        """Copy a finished run to its saved folder, replacing what was there.
        Falls back to the scratch folder when the results root is not
        writable (the result is then lost with the cluster)."""
        try:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            return dst
        except OSError as exc:
            self.say(f"    not saved to {dst} ({exc}); kept in {src}")
            return src

    def run_dirs(self) -> dict:
        """{condition: folder} of the selected conditions: this session's
        runs, else the saved results."""
        out = {}
        for cond in self.sel.get("conditions", []):
            d = self.results.get(cond) or self.saved_dir(cond)
            if self._usable(d):
                out[cond] = Path(d)
        if not out:
            self.say("  no results for the selected conditions -- run R.run() first")
        return out

    def label(self, cond) -> str:
        return (f"{cond}" if self.sel["source"] == "csv"
                else f"{cond} Test{self.sel['test_id']}")

    # ------------------------------------------------------------------
    # showing and saving
    # ------------------------------------------------------------------
    def _show(self, html, name, title=""):
        import viewers
        self.html(html)
        p = self.plots_dir() / f"{name}.html"
        try:
            viewers.save_page(html, p, title=title or name)
            self.say(f"  saved: {p}")
        except OSError as exc:
            self.say(f"  not saved ({exc})")
        return p

    def _by_condition(self, figs, name, title, note=""):
        import viewers
        html = viewers.by_condition(figs, title=title, note=note,
                                    panel_height=S.PANEL_HEIGHT,
                                    panel_width=S.PANEL_WIDTH)
        return self._show(html, name, title)

    def show_nyquist(self):
        """Every segment's Nyquist / |Z| / phase with the Gamry sweep of that
        run (the one matched by time to its test)."""
        figs = {}
        for cond, d in self.run_dirs().items():
            figs[self.label(cond)] = views.nyquist_figure(
                d, title=f"Local EIS — {self.sel['leepa']}, {self.label(cond)}")
        return self._by_condition(
            figs, "nyquist", f"Nyquist / Bode — {self.sel['leepa']}",
            note=("colour: position along the air path, cathode inlet (blue) "
                  "-> outlet (red); black: the Gamry sweep of this test"))

    def show_median_nyquist(self):
        """Plate median, card medians, aggregate and Gamry per condition."""
        figs = {}
        for cond, d in self.run_dirs().items():
            figs[self.label(cond)] = views.median_figure(
                d, title=f"Median spectra — {self.sel['leepa']}, {self.label(cond)}")
        return self._by_condition(
            figs, "median_nyquist", f"Median Nyquist — {self.sel['leepa']}")

    def show_heatmaps(self):
        import viewers
        import r2d2_geometry as geom
        summ = {}
        for cond, d in self.run_dirs().items():
            p = Path(d) / "gold" / "plate_summary.csv"
            if p.is_file():
                summ[self.label(cond)] = p
        title = f"{self.sel['leepa']} — {geom.plate(self.sel['plate']).title}"
        html = viewers.plate_maps(dict(sorted(summ.items(),
                                              key=lambda kv: condition_key(kv[0]))),
                                  title=title)
        p = self._show(html, "heatmaps", f"Plate maps — {title}")
        # the static PNGs gold wrote, collected beside the viewer
        for cond, d in self.run_dirs().items():
            for png in sorted((Path(d) / "gold").glob("plate_*.png")):
                self._copy(png, p.parent / self.cond_name(cond) / png.name)
        return p

    def _copy(self, src, dst):
        try:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except OSError:
            pass

    def show_frequency_response(self):
        import viewers
        files = self.sel["plate_files"]
        bode = (Path(files["abgleich"]) / "bode") if files.get("abgleich") else None
        items, ex = {}, None
        for cond, d in self.run_dirs().items():
            r = views.frequency_response(
                d, gain_file=self.sel.get("gain_file") or None, bode_dir=bode,
                scratch=self.scratch / "frequency_response" / self.cond_name(cond))
            for c in r["checks"]:
                self.say(f"  {self.label(cond)}: {c}")
            ex = ex or r["exsitu"]
            if r["insitu"]:
                items[(f"In-situ — {self.label(cond)}",)] = viewers.img_html(r["insitu"])
        if ex:
            items = {("Ex-situ amplifiers (Abgleich)",): viewers.img_html(ex),
                     **items}
        html = viewers.selector_html(items, ["Frequency response"],
                                     title=f"Frequency response — {self.sel['leepa']}")
        return self._show(html, "frequency_response", "Frequency response")

    # ------------------------------------------------------------------
    # the whole bench log
    # ------------------------------------------------------------------
    def show_mf4(self):
        """The WHOLE bench .mf4 of the order: every channel, over the whole
        test, one parameter drop-down; every Gamry sweep shaded."""
        import viewers
        leepa = self.sel["leepa"]
        chans, title, sweeps = [], "", []
        if S.DATAGO_FIRST and S.BENCH_FROM_DATAGO and self.spark is not None:
            try:
                t0, vals, name = datago.bench_channels(self.spark, leepa)
                if vals:
                    chans = views.channels_from_values(
                        vals, include_others=S.MF4_INCLUDE_ALL_CHANNELS)
                    title = f"{leepa} — {name} (datago, whole test)"
                    sweeps = self._sweeps_datago(t0)
            except Exception as exc:                        # noqa: BLE001
                self.say(f"  datago bench log not read ({exc}) -> Volume")
        if not chans:
            mf4 = self._bench_mf4()
            if mf4 is None:
                self.say(f"  no bench .mf4 for {leepa} in datago or "
                         f"{self.gamry_root}")
                return None
            import bench_plots as bp
            chans = bp.read_mf4(mf4, include_others=S.MF4_INCLUDE_ALL_CHANNELS,
                                max_points=S.MF4_MAX_POINTS)
            title = f"{leepa} — {mf4.name} (whole test)"
            sweeps = self._sweeps_volume(bp.mf4_start_time(mf4))
        groups = {}
        for c in chans:
            groups[c.group] = groups.get(c.group, 0) + 1
        self.say(f"  {len(chans)} channels: "
                 + ", ".join(f"{g} {n}" for g, n in groups.items())
                 + f"; {len(sweeps)} Gamry sweeps marked")
        fig = views.bench_figure_whole(chans, title, sweeps,
                                       reference_lines=self._famos_temps())
        html = viewers.selector_html({("Bench log",): fig}, ["MF4"], title="")
        return self._show(html, "mf4_bench_log", title)

    def _bench_mf4(self):
        leepa = self.sel["leepa"]
        try:
            cands = [f for f in list(self.gamry_root.glob("*.mf4"))
                     + list(self.gamry_root.glob("*.MF4"))
                     if f"RO{leepa}" in f.name or leepa in f.name]
        except OSError:
            cands = []
        main = [f for f in cands if not re.search(r"anfang|ende", f.name, re.I)]
        pool = main or cands
        return max(pool, key=lambda f: f.stat().st_size) if pool else None

    def _sweeps_volume(self, t0) -> list:
        """[(label, start_s, end_s)] of the order's .dta sweeps on the bench
        log's clock (both are local wall-clock; the bench header may be
        tz-aware)."""
        import gamry_compare as gc_
        import gamry_sync
        if t0 is None or not self.sel.get("gamry_dir"):
            return []
        try:
            t0n = (pd.Timestamp(t0).tz_convert(S.DATAGO_LOCAL_TZ)
                   .tz_localize(None) if pd.Timestamp(t0).tzinfo
                   else pd.Timestamp(t0))
        except Exception:                                   # noqa: BLE001
            return []
        out = []
        for sw in gc_.find_cell_sweeps(Path(self.sel["gamry_dir"]),
                                       order_id=self.sel["leepa"],
                                       version=self.sel["gamry_version"] or None):
            if sw.started is None:
                continue
            a = (pd.Timestamp(sw.started) - t0n).total_seconds()
            try:
                dur = float(np.nanmax(gamry_sync.read_timeline(sw.path).t_s))
            except Exception:                               # noqa: BLE001
                dur = 120.0
            m = re.search(r"HFR_(\d+)", sw.path.name)
            out.append((f"{sw.condition}" + (f" #{m.group(1)}" if m else ""),
                        a, a + dur))
        return sorted(out, key=lambda x: x[1])

    def _sweeps_datago(self, t0) -> list:
        if t0 is None:
            return []
        df = datago.files(self.spark, self.sel["leepa"])
        out = []
        for r in df[df["kind"] == "gamry"].itertuples():
            if pd.isna(r.t_begin):
                continue
            a = (pd.Timestamp(r.t_begin) - t0).total_seconds()
            b = ((pd.Timestamp(r.t_end) - t0).total_seconds()
                 if pd.notna(r.t_end) else a + 150)
            m = re.search(r"HFR_(\d+)", str(r.raw_file_name))
            out.append((f"{r.condition}" + (f" #{m.group(1)}" if m else ""), a, b))
        return sorted(out, key=lambda x: x[1])

    def _famos_temps(self) -> list:
        """FAMOS plate sensors of the selected runs as °C reference lines."""
        import r2d2_geometry as geom
        refs = []
        for cond, d in self.run_dirs().items():
            p = Path(d) / "bronze" / "bronze_manifest.json"
            try:
                sens = json.loads(p.read_text()).get("sensor_T_degC", {})
            except Exception:                               # noqa: BLE001
                sens = {}
            for k in sorted(sens):
                x = geom.TEMP_SENSOR_X_MM.get(k)
                refs.append((f"FAMOS {k}" + (f" ({x:.0f} mm)" if x is not None else "")
                             + f" — {self.label(cond)}", "°C", float(sens[k])))
        return refs

    # ------------------------------------------------------------------
    # ECM
    # ------------------------------------------------------------------
    def show_ecm(self, refit: bool = False):
        """Standard ECM and DRT-informed ECM for the aggregate and every
        segment: Condition / Spectrum / Fit drop-downs. Fits are saved in
        RUN_DIR/ecm/ (ecm_fits.json, ecm_parameters.csv) and the aggregate's
        figure as PNG; the viewer as plots/ecm_fits.html."""
        figs = {}
        for cond, d in self.run_dirs().items():
            fits = views.ecm_fits(d, refit=refit, say=self.say)
            lab = self.label(cond)
            for key, entry in fits.items():
                name = "Aggregate (whole cell)" if key == "Aggregate" else f"Segment {key}"
                figs[(lab, name)] = views.ecm_figure(
                    entry, f"ECM — {self.sel['leepa']}, {lab}, {name}")
            if "Aggregate" in fits:
                png = views.ecm_png(fits["Aggregate"],
                                    self.plots_dir() / self.cond_name(cond)
                                    / "ecm_aggregate.png",
                                    f"{self.sel['leepa']} {lab} — aggregate: "
                                    f"standard vs DRT-informed ECM")
                self.say(f"  saved: {png}")
            for f_ in ("ecm_parameters.csv", "ecm_fits.json"):
                if (Path(d) / "ecm" / f_).is_file():
                    self._copy(Path(d) / "ecm" / f_,
                               self.plots_dir() / self.cond_name(cond) / f_)
        html = views.ecm_viewer(figs, title=f"ECM fits — {self.sel['leepa']}")
        return self._show(html, "ecm_fits", "ECM fits")

    # ------------------------------------------------------------------
    # optional: the frozen high-frequency response of the order
    # ------------------------------------------------------------------
    def build_hf_response(self, mag: str | None = None, prior=None):
        """Build <order>/hf_response/hf_response.csv from all selected runs
        (needs two or more conditions). Applied automatically to later runs
        (settings.HF_RESPONSE = 'auto'); rerun them to use it. `mag`:
        'segment' | 'card', default settings.HF_RESPONSE_MAG. The file it
        replaces goes to hf_response/history/. Refused, with nothing
        written, for runs whose band ends below 5 kHz or that were
        evaluated with another version of the file (hf_response.build)."""
        import hf_response
        runs = list(self.run_dirs().values())
        if len(runs) < 2:
            self.say("  needs two or more conditions with results")
            return None
        mag = mag or getattr(S, "HF_RESPONSE_MAG", "segment")
        try:
            return hf_response.build(runs, self.order_dir() / "hf_response",
                                     mag=mag, prior=prior)
        except ValueError as exc:
            self.say(f"  high-frequency response NOT built: {exc}")
            return None
