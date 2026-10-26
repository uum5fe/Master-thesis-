"""
auto.py
=======
The incremental job: evaluate what has not been evaluated yet.

    plan(R)   every (order, condition, test) with FAMOS data -- datago and
              Volume, as the process notebook sees them -- whose run is not
              in the runs table, or (AUTO_RERUN_CHANGED) is there with
              another settings digest
    run(R)    evaluates the first AUTO_MAX_RUNS of the plan exactly as
              "01 Local EIS - Process" does with Run mode = rerun, fits
              the ECM and publishes the tables; the rest follow at the next
              execution

The selection a person makes in the widgets comes from settings here:
plate from AUTO_PLATE, else the order's file names (bench .mf4 "..._Naboo_1"
= gen2, "Kashyyyk" = gen1), else gen1; orientation from AUTO_ORIENTATION,
else standard; every condition and every test found.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import settings as S
from . import tables
from .session import calibration_changed


def plate_of(R, leepa) -> str:
    """gen1 / gen2 of an order: AUTO_PLATE, else its file names, else gen1."""
    if str(leepa) in S.AUTO_PLATE:
        return S.AUTO_PLATE[str(leepa)]
    import site_paths
    from . import datago
    digits = "".join(ch for ch in str(leepa) if ch.isdigit())
    names = []
    for root in (R.gamry_root, R.gamry_root.parent, R.famos_root):
        try:
            names += [f.name.lower() for f in Path(root).iterdir()
                      if digits in f.name]
        except OSError:
            pass
    if R.spark is not None:
        try:
            names += [str(n).lower() for n in
                      datago.files(R.spark, leepa)["raw_file_name"].dropna()]
        except Exception:                                   # noqa: BLE001
            pass
    said = {p for p, toks in site_paths.PLATE_TOKENS.items()
            for n in names if toks[0] in n}
    return said.pop() if len(said) == 1 else "gen1"


def orientation_of(leepa) -> str:
    return S.AUTO_ORIENTATION.get(str(leepa), "standard")


def published(spark) -> dict:
    """{run_key: digest} of the runs table ({} without Spark or table)."""
    if spark is None:
        return {}
    df = tables.runs(spark)
    if not len(df):
        return {}
    return dict(zip(df["run_key"].astype(str), df["digest"].astype(str)))


def published_calibration(spark) -> dict:
    """{run_key: calibration record} of the runs table ({} without)."""
    if spark is None:
        return {}
    df = tables.runs(spark)
    if not len(df) or "calibration_json" not in df:
        return {}
    out = {}
    for k, v in zip(df["run_key"].astype(str), df["calibration_json"]):
        try:
            out[k] = json.loads(v) if isinstance(v, str) and v else {}
        except ValueError:
            out[k] = {}
    return out


def _select(R, leepa, test_id="", conditions="ALL", run_mode="cache"):
    vals = dict(leepa_id=leepa, plate=plate_of(R, leepa),
                plate_orientation=orientation_of(leepa), conditions=conditions,
                run_mode=run_mode, source_format="famos")
    if test_id:
        vals["test_id"] = test_id
    return R.select(**vals)


def plan(R, orders=None, say=None) -> list[dict]:
    """What the job would evaluate, in order (oldest order first)."""
    say = say or R.say
    have = published(R.spark)
    cal_have = published_calibration(R.spark)
    known = set(R.orders())
    todo = []
    for leepa in (orders or S.AUTO_ORDERS or sorted(known)):
        leepa = str(leepa)
        if leepa not in known:
            say(f"  {leepa}: no FAMOS / Gamry data found -- skipped")
            continue
        sel = _select(R, leepa)
        tests = sorted({t for ts in sel["tests_by_condition"].values()
                        for t in ts})
        for test in tests:
            sel = _select(R, leepa, test_id=test)
            for cond in sel["all_conditions"]:
                if test not in sel["tests_by_condition"].get(cond, []):
                    continue
                key = f"{R.order_dir().name}/{R.cond_name(cond)}"
                cfg = R._config(cond, R.scratch / "auto_plan")
                ident = R.identity(cfg)
                if key not in have:
                    reason = "new"
                elif have[key] != ident["digest"] and S.AUTO_RERUN_CHANGED:
                    reason = "settings changed"
                elif S.AUTO_RERUN_CHANGED and calibration_changed(
                        cal_have.get(key), ident.get("calibration")):
                    reason = "calibration changed"
                else:
                    continue
                todo.append(dict(order=leepa, plate=sel["plate"],
                                 orientation=sel["plate_orientation"],
                                 condition=cond, test_id=test, run_key=key,
                                 reason=reason))
    say(f"  plan: {len(todo)} run(s) to evaluate"
        + "".join(f"\n    {t['run_key']:<34} {t['reason']}" for t in todo))
    return todo


def run(R, max_runs=None, orders=None) -> list[dict]:
    """Evaluate the first `max_runs` (AUTO_MAX_RUNS) of the plan, fit the
    ECM and publish. Returns the plan entries handled, with 'status'."""
    todo = plan(R, orders)[: (max_runs or S.AUTO_MAX_RUNS)]
    groups: dict = {}
    for t in todo:
        groups.setdefault((t["order"], t["test_id"]), []).append(t)
    for (leepa, test), items in groups.items():
        conds = ",".join(t["condition"] for t in items)
        _select(R, leepa, test_id=test, conditions=conds, run_mode="rerun")
        done = R.run()
        R.fit_ecm()
        R.publish()
        for t in items:
            t["status"] = "done" if t["condition"] in done else "failed"
    R.say(f"  job: {sum(t.get('status') == 'done' for t in todo)} evaluated, "
          f"{sum(t.get('status') == 'failed' for t in todo)} failed")
    return todo
