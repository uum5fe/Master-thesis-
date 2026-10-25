"""
site_paths.py
=============
Where the plate-specific input files live, for ANY user.

The pipeline needs four kinds of file that belong to the PLATE, not to a
measurement:

    curr.csv          72 rows "c0;c1"   per-segment Abgleich      (required)
    temp.csv           4 rows "c0;c1"   per-sensor Abgleich       (required)
    chain_gain.csv    measuring-chain response, written once by the
                      runner's "Chain response" cell                 (optional)
    Abgleich/         the calibration delivery: coefficients/, bode/,
                      Step<k>_<T>Grad.csv (for the chain response and
                      the via-resistance check)                      (optional)

They used to be read from one person's workspace folder, so the notebook
failed for everybody else. They now live in ONE shared folder on the Volume,
one sub-folder per plate generation:

    /Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/EIS_config/
        gen1/   curr.csv  temp.csv  chain_gain.csv  Abgleich/      green / Kashyyyk
        gen2/   curr.csv  temp.csv  chain_gain.csv  Abgleich/      blue  / Naboo

Every file is looked up in this order, and the first hit wins:

    1. the shared Volume folder   <config_root>/<plate>/
    2. the package folder         <pipeline dir>/EIS_config/<plate>/
    3. legacy locations           only for gen1 and only the CURRENT user's
                                  own workspace folder (never someone else's)

The plates are never mixed: a gen2 run never falls back to a gen1 file. The
two calibrations differ by ~1.45x in absolute level, so borrowing the other
plate's curr.csv would scale every impedance wrongly without any error.

Override the shared root with the environment variable EIS_CONFIG_ROOT or by
passing `root=`.
"""

from __future__ import annotations

import os
from pathlib import Path

EV_ROOT = Path("/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev")
DEFAULT_CONFIG_ROOT = EV_ROOT / "EIS_config"
PLATES = ("gen1", "gen2")

#: File names inside a plate folder, and what each one is for.
FILES = {
    "curr": ("curr.csv", True, "per-segment current calibration (72 rows c0;c1)"),
    "temp": ("temp.csv", True, "temperature-sensor calibration (4 rows c0;c1)"),
    "gain": ("chain_gain.csv", False, "measuring-chain response (Chain response cell)"),
}
ABGLEICH_SUBDIR = "Abgleich"

#: Words that identify a calibration delivery of each plate in a folder path.
PLATE_TOKENS = {"gen1": ("kashyyyk", "green", "gen1"),
                "gen2": ("naboo", "blue", "gen2")}


def config_root(root=None) -> Path:
    """The shared folder: `root`, else $EIS_CONFIG_ROOT, else the Volume."""
    if root:
        return Path(root)
    env = os.environ.get("EIS_CONFIG_ROOT", "").strip()
    return Path(env) if env else DEFAULT_CONFIG_ROOT


def _plate(plate: str) -> str:
    p = str(plate or "gen1").strip().lower()
    if p not in PLATES:
        raise ValueError(f"unknown plate {plate!r}; expected one of {PLATES}")
    return p


def plate_dirs(plate: str, root=None, package_dir=None) -> list[Path]:
    """The folders searched for this plate, in order."""
    p = _plate(plate)
    out = [config_root(root) / p]
    if package_dir:
        out.append(Path(package_dir) / "EIS_config" / p)
    return out


def legacy_dirs(plate: str, user_home=None, package_dir=None) -> list[Path]:
    """Pre-Volume locations, gen1 only: the current user's own workspace
    folder and the package folder itself (older zips shipped curr.csv /
    temp.csv / chain_gain_gen1.csv at the top level)."""
    if _plate(plate) != "gen1":
        return []
    out = []
    if user_home:
        out.append(Path(user_home))
    if package_dir:
        out.append(Path(package_dir))
    return out


def _legacy_names(kind: str, plate: str) -> list[str]:
    name = FILES[kind][0]
    return [name, f"chain_gain_{plate}.csv"] if kind == "gain" else [name]


def resolve(plate: str, root=None, package_dir=None, user_home=None) -> dict:
    """Find every plate file. Returns

        {"plate", "root", "target_dir",
         "curr", "temp", "gain", "abgleich"   -> Path or None,
         "found_in": {kind: folder},
         "missing":  [kinds that are required and were not found],
         "searched": [folders, in order]}

    `target_dir` is where the runner should WRITE a new chain_gain.csv: the
    shared folder, so the next user finds it too.
    """
    p = _plate(plate)
    primary = plate_dirs(p, root, package_dir)
    legacy = legacy_dirs(p, user_home, package_dir)
    out = {"plate": p, "root": config_root(root), "target_dir": primary[0],
           "found_in": {}, "missing": [], "searched": [str(d) for d in primary + legacy]}
    for kind, (name, required, _why) in FILES.items():
        hit = None
        for d in primary:
            if (d / name).is_file():
                hit = d / name
                break
        if hit is None:
            for d in legacy:
                for n in _legacy_names(kind, p):
                    if (d / n).is_file():
                        hit = d / n
                        break
                if hit is not None:
                    break
        out[kind] = hit
        if hit is not None:
            out["found_in"][kind] = str(hit.parent)
        elif required:
            out["missing"].append(kind)
    out["abgleich"] = next((d / ABGLEICH_SUBDIR for d in primary
                            if (d / ABGLEICH_SUBDIR / "bode").is_dir()), None)
    return out


def matches_plate(path, plate: str) -> bool:
    """True when nothing in `path` names the OTHER plate.

    Used to filter an automatic search for Abgleich deliveries, so a gen2 run
    cannot pick up the green Kashyyyk delivery just because it is the only
    one on the Volume.
    """
    p = _plate(plate)
    other = "gen2" if p == "gen1" else "gen1"
    s = str(path).lower()
    if any(t in s for t in PLATE_TOKENS[other]):
        return False
    return True


def describe(res: dict) -> list[str]:
    """Human-readable lines for the notebook printout."""
    lines = [f"plate files ({res['plate']}), shared folder {res['target_dir']}"]
    for kind, (name, required, why) in FILES.items():
        f = res.get(kind)
        tag = str(f) if f else ("MISSING" if required else "not present")
        lines.append(f"  {name:<15} {tag}")
    a = res.get("abgleich")
    lines.append(f"  {ABGLEICH_SUBDIR + '/':<15} {a if a else 'not present'}")
    return lines


def missing_message(res: dict) -> str:
    """The error text when a required file is missing."""
    need = [FILES[k][0] for k in res["missing"]]
    return (
        f"plate {res['plate']}: {', '.join(need)} not found.\n"
        f"  Put the {res['plate']} calibration files here (shared, readable by "
        f"everyone who runs the notebook):\n"
        f"      {res['target_dir']}/curr.csv\n"
        f"      {res['target_dir']}/temp.csv\n"
        f"  Searched: {'; '.join(res['searched'])}\n"
        f"  The plates are never mixed: a {res['plate']} run does not borrow "
        f"the other plate's calibration, because the two differ in absolute "
        f"level and every impedance would be scaled wrongly.")
