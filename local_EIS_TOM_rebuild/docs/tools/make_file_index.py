#!/usr/bin/env python3
"""
make_file_index.py
==================
Writes docs/TRAINING_GUIDE_FILE_INDEX.md: for every Python file of the
package its size, what it imports from the package, who imports it, the
libraries it uses, the first paragraph of its docstring and an index of its
top-level classes and functions (first docstring line each).

It parses the source with `ast` and never imports the package, so it runs
without numpy, Spark or Databricks:

    python docs/tools/make_file_index.py            # from the package root

Run it again after a change to the code so the index matches the package.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "TRAINING_GUIDE_FILE_INDEX.md"

#: folder -> one line on its role, in the order the index lists them
FOLDERS = {
    "": "Entry points: the four Databricks notebooks, the sys.path helper "
        "and the command-line launcher.",
    "runner": "What the notebooks do, as importable code: settings, datago "
              "access, the Session behind 01/03, the read-only Results "
              "behind 02, the Delta tables, the automatic job, the figures.",
    "core": "Shared foundations: the configuration, numerical utilities, "
            "plate geometry, plate statistics, plate conditions and the "
            "lookup of the plate calibration files.",
    "pipeline": "The processing: the orchestrator (main.py), the layers "
                "bronze, silver, gold, the separate CSV pipeline and the "
                "signal-processing modules they call.",
    "checks": "Diagnostics that judge whether a finished result is "
              "believable. None of them can stop a run.",
    "analysis": "The DRT and the DRT-informed equivalent-circuit fit used "
                "by the ECM cells.",
    "plotting": "Everything that draws: Nyquist/Bode, plate maps (static and "
                "interactive), the drop-down viewers, the common style, the "
                "bench log.",
    "readers": "Readers for everything that is not a FAMOS card: Gamry .DTA, "
               "MF4, the Abgleich calibration, CSV measurements, and the "
               "zstd shim for asammdf.",
    "docs": "Tools that build this documentation.",
    "tests": "The pytest suite, its synthetic-data generators and fixtures.",
}

RUNNER_MODULES = {"settings", "datago", "session", "results", "tables",
                  "auto", "views"}


def folder_of(rel: Path) -> str:
    return "" if len(rel.parts) == 1 else rel.parts[0]


def summary(doc: str, limit: int = 600) -> str:
    """The first paragraph of a module docstring, without its title."""
    lines = doc.strip().splitlines()
    if len(lines) > 1 and re.fullmatch(r"[=\-]{3,}", lines[1].strip()):
        lines = lines[2:]
    elif lines and "--" in lines[0] and len(lines) > 1 and \
            re.fullmatch(r"[=\-]{3,}", lines[1].strip() if len(lines) > 1 else ""):
        lines = lines[2:]
    para = []
    for ln in lines:
        if not ln.strip():
            if para:
                break
            continue
        if re.fullmatch(r"[=\-]{3,}", ln.strip()):
            break
        para.append(ln.strip())
    text = " ".join(para)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def scan() -> dict:
    files = sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    stems = {p.stem for p in files if p.relative_to(ROOT).parts[0] != "tests"}
    stems |= RUNNER_MODULES | {"runner"}
    std = set(sys.stdlib_module_names)
    data = {}
    for p in files:
        rel = p.relative_to(ROOT)
        src = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
        imports, libs = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    if n.module:
                        imports.add(n.module.split(".")[0])
                    imports |= {a.name for a in n.names if a.name in RUNNER_MODULES}
                    continue
                names = [n.module or ""]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in stems:
                    imports.add(top)
                elif top and top != "__future__":
                    libs.add(top)
        items = []
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in n.args.args + n.args.kwonlyargs]
                items.append(("def", n.name, args, [], [],
                              (ast.get_docstring(n) or "").strip().split("\n")[0]))
            elif isinstance(n, ast.ClassDef):
                decos = [ast.unparse(d) for d in n.decorator_list]
                methods = [m.name for m in n.body
                           if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and not m.name.startswith("__")]
                items.append(("class", n.name, [], decos, methods,
                              (ast.get_docstring(n) or "").strip().split("\n")[0]))
        data[rel] = dict(lines=src.count("\n") + 1, doc=ast.get_docstring(tree) or "",
                         imports=imports - {p.stem}, libs=libs, items=items,
                         std=sorted(libs & std), third=sorted(libs - std))
    for rel, d in data.items():
        d["used_by"] = sorted({r.stem for r, e in data.items()
                               if r != rel and folder_of(r) != "tests"
                               and rel.stem in e["imports"]})
        d["tests"] = sum(1 for r, e in data.items()
                         if folder_of(r) == "tests" and rel.stem in e["imports"])
    return data


def write(data: dict) -> str:
    n_py = len(data)
    n_tests = sum(1 for r in data if folder_of(r) == "tests")
    n_lines = sum(d["lines"] for d in data.values())
    out = [
        "# Local EIS pipeline — file reference (generated)",
        "",
        "Generated by `docs/tools/make_file_index.py` from the source with "
        "Python's `ast` module, so it lists every file, class and function "
        "that exists in this package and nothing else. The explanations of "
        "*why* each file exists are in [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) "
        "Part 4; this is the look-up table beside it.",
        "",
        f"**{n_py} Python files** ({n_tests} in `tests/`), "
        f"**{n_lines:,} lines** including the four notebooks.",
        "",
        "For each file: size, the package modules it imports (*depends on*), "
        "the modules that import it (*used by*, tests counted separately), "
        "third-party libraries, the first paragraph of its docstring and every "
        "top-level class and function with the first line of its docstring.",
        "",
    ]
    for folder, role in FOLDERS.items():
        rels = sorted(r for r in data if folder_of(r) == folder)
        if not rels:
            continue
        out += [f"## {folder + '/' if folder else '(top level)'}", "", role, ""]
        if folder == "tests":
            out += ["| Test file | Tests | What it checks (first docstring line) |",
                    "|---|---|---|"]
            for r in rels:
                d = data[r]
                n = sum(1 for i in d["items"] if i[0] == "def" and i[1].startswith("test_"))
                first = (d["doc"].strip().split("\n")[0] if d["doc"] else "—")
                out.append(f"| `{r.name}` | {n} | {md(first)} |")
            out.append("")
            continue
        for r in rels:
            d = data[r]
            n_cls = sum(1 for i in d["items"] if i[0] == "class")
            n_fn = sum(1 for i in d["items"] if i[0] == "def")
            out += [f"### `{r.as_posix()}`", "",
                    f"*{d['lines']:,} lines · {n_cls} classes · {n_fn} functions*", ""]
            s = summary(d["doc"])
            if s:
                out += [md(s), ""]
            dep = ", ".join(sorted(d["imports"])) or "only standard / third-party libraries"
            used = ", ".join(d["used_by"]) or "—"
            if d["tests"]:
                used += f" (+ {d['tests']} test files)"
            out += ["| | |", "|---|---|",
                    f"| **Depends on** | {dep} |",
                    f"| **Used by** | {used} |",
                    f"| **Third-party** | {', '.join(d['third']) or '—'} |",
                    f"| **Standard library** | {', '.join(d['std']) or '—'} |", ""]
            if d["items"]:
                out += ["| Class / function | What it does (first docstring line) |",
                        "|---|---|"]
                for kind, name, args, decos, methods, doc in d["items"]:
                    if kind == "class":
                        sig = f"class {name}" + (f" @{', @'.join(decos)}" if decos else "")
                        if methods:
                            doc = (doc + " " if doc else "") + "Methods: " + ", ".join(methods) + "."
                    else:
                        a = ", ".join(args)
                        sig = f"{name}({a if len(a) < 60 else a[:57] + '…'})"
                    out.append(f"| `{md(sig)}` | {md(doc) or '—'} |")
                out.append("")
    text = "\n".join(out) + "\n"
    OUT.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    t = write(scan())
    print(f"wrote {OUT} ({len(t.splitlines())} lines)")
