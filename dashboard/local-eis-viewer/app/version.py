r"""What this copy of the app provides, and what the scripts need from it.

THE PROBLEM THIS SOLVES
-----------------------
The app is distributed as a folder that gets copied onto a working machine.
``app\`` and the scripts beside it change together, and copying one without
the other leaves a new script calling into an old module.  Python's report
of that is an ``AttributeError`` raised at the moment of the call --

    AttributeError: module 'app.services.runner' has no attribute
                    'run_pipeline'

-- which is both too late and too vague.  Too late because ``run_evaluation``
stages gigabytes of cards to local disk before it reaches the call, so the
failure arrives after the expensive part.  Too vague because the name it
reports is the symptom; the cause is that two files came from different
copies, and nothing in the message says so or says which folder to update.

So the contract is written down and checked at startup, before any work.
``API_VERSION`` rises whenever the surface below changes in a way that would
break a script; ``REQUIRED`` names the attributes each script actually calls,
which is what makes the diagnosis specific rather than "something is old".

WHEN YOU CHANGE THE API
-----------------------
Adding a function to a service: add its name to ``REQUIRED`` if a script
calls it, and bump ``API_VERSION``.  Renaming or removing one: the same, and
leave an alias behind if the old name might still be in a copy somewhere
(``runner.run_famos`` is such an alias).
"""

from __future__ import annotations

import importlib

#: Bumped whenever the surface in REQUIRED changes.
API_VERSION = 4

#: module path -> attributes the shipped scripts call on it.
REQUIRED: dict[str, tuple[str, ...]] = {
    "app.settings": ("SETTINGS", "Settings"),
    "app.data.sources": ("RunRef", "FamosSource", "CsvLoggerSource",
                         "_condition_sort_key"),
    "app.services.runner": ("run_pipeline", "build_command", "preflight",
                            "output_dir", "pipeline_dir", "warnings_for",
                            "self_test", "PipelineUnavailable"),
    "app.services.staging": ("stage", "clear", "is_network_path",
                             "staged_size_mb"),
    "app.plates.registry": ("get", "default_key"),
}


class VersionSkew(RuntimeError):
    """Raised when the app folder and the scripts beside it disagree."""


class MissingDependency(RuntimeError):
    """Raised when a module is fine but something it imports is not installed.

    Kept separate from :class:`VersionSkew` on purpose.  Both surface as an
    ImportError here, and telling the user their folder is half-copied when
    the real answer is `pip install pandas` sends them to re-copy files that
    were never the problem -- which is worse than no check at all, because
    it is confidently wrong.
    """


def _describe(missing: dict[str, list[str]], where: str) -> str:
    lines = [
        "this copy of the app is only partly updated.",
        "",
    ]
    for module, names in sorted(missing.items()):
        lines.append(f"  {module} is missing: {', '.join(sorted(names))}")
        try:
            path = importlib.import_module(module).__file__
        except Exception:                                   # noqa: BLE001
            path = "(not importable)"
        lines.append(f"      {path}")
    lines += [
        "",
        "These files are updated together, so a folder holding some new "
        "files and some old",
        "ones is not a state the app is ever tested in. Update the whole "
        "folder rather than",
        "the file the message happens to name:",
        "",
        "    git pull                     (if this folder is a checkout)",
        f"    or re-copy app\\ and the scripts beside it from {where}",
        "",
        "Copying one at a time is what produces this: the new script calls "
        "something the",
        "old module does not have, and without this check that only "
        "surfaces at the call,",
        "after the cards have been staged.",
    ]
    return "\n".join(lines)


def _is_ours(name: str) -> bool:
    """Is this import one of the project's own modules?"""
    root = (name or "").split(".")[0]
    return root in ("app", "eis", "local_eis") or root in set(
        m.split(".")[0] for m in REQUIRED)


def check(required: dict[str, tuple[str, ...]] | None = None,
          where: str = "the source you copied it from") -> None:
    """Verify every attribute the scripts need is present.

    Raises :class:`VersionSkew` naming exactly what is missing, rather than
    letting the first call fail on its own with the name of one attribute --
    or :class:`MissingDependency` when the module is not old at all and a
    third-party package simply is not installed.
    """
    missing: dict[str, list[str]] = {}
    absent_packages: dict[str, str] = {}
    for module, names in (required or REQUIRED).items():
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            culprit = getattr(exc, "name", None) or ""
            if not _is_ours(culprit):
                # Not a stale copy: this file is present and current, and
                # something it imports is not installed.
                absent_packages[culprit or module] = str(exc)
            else:
                missing[module] = [f"(the module itself: {exc})"]
            continue
        absent = [n for n in names if not hasattr(mod, n)]
        if absent:
            missing[module] = absent

    if missing:
        raise VersionSkew(_describe(missing, where))
    if absent_packages:
        names = sorted(absent_packages)
        raise MissingDependency(
            "this copy of the app is complete, but its dependencies are "
            "not installed.\n\n"
            + "\n".join(f"  missing package: {n}" for n in names)
            + "\n\n    pip install -r requirements.txt\n\n"
              "Nothing needs re-copying -- the files are current; it is the "
              "environment that\n  is missing packages.")


def report() -> dict:
    """What is present, for `--check` output and bug reports."""
    out: dict[str, object] = {"api_version": API_VERSION, "modules": {}}
    for module, names in REQUIRED.items():
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            culprit = getattr(exc, "name", None) or ""
            out["modules"][module] = {
                "ok": False, "error": str(exc),
                "cause": "stale copy" if _is_ours(culprit)
                         else "missing dependency"}
            continue
        absent = [n for n in names if not hasattr(mod, n)]
        out["modules"][module] = {
            "ok": not absent,
            "file": getattr(mod, "__file__", None),
            "missing": absent,
        }
    return out
