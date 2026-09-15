"""Discovering conditions and launching the pipeline.

The dashboard never imports the pipeline.  It shells out to ``main.py`` in
EIS_PIPELINE_DIR, which keeps the two dependency sets apart (the pipeline wants
scipy and matplotlib; the dashboard does not) and means a run that dies takes a
subprocess with it rather than the web server.
"""

from __future__ import annotations

import ntpath
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import settings
from .settings import Settings

#: "..._Current_150A_Test_01_Karte_3.DAT" -> "150A"; also "450 A", "1.5A".
_CURRENT_IN_NAME = re.compile(r"(\d+(?:[.,]\d+)?)\s*A(?![A-Za-z])")
#: "Leepa_RO2612030_..." / "Leepa_2612030_..." -> "2612030"
_LEEPA_IN_NAME = re.compile(r"RO?(\d{7})")
#: The same id on a FOLDER, where it usually carries no R prefix:
#: "2612030_07_09" -> "2612030".  Bounded so it cannot bite a chunk out of a
#: longer number.
_LEEPA_BARE = re.compile(r"(?<!\d)(\d{7})(?!\d)")


#: Extensions a measurement can arrive in, by source format.
_SUFFIXES = {"famos": (".dat",), "csv": (".csv", ".txt", ".tsv")}


def discover_conditions(root: Path | None, source: str = "famos") -> list[str]:
    """Current setpoints present in the measurement filenames, in order.

    A share that is offline raises rather than returning an empty listing, so
    the walk is guarded: the dashboard then says "nothing found" on the Setup
    page instead of showing a traceback.
    """
    if not root:
        return []
    want = _SUFFIXES.get(source, _SUFFIXES["famos"])
    found: set[str] = set()
    try:
        entries = list(Path(root).iterdir())
    except OSError:
        return []
    for p in entries:
        if p.suffix.lower() not in want:
            continue
        m = _CURRENT_IN_NAME.search(p.name)
        if m:
            found.add(_normalise_current(m.group(1)))
    return sorted(found, key=_condition_sort_key)


def _normalise_current(num: str) -> str:
    """"150" -> "150A"; "45,0" -> "45A"; "1.5" -> "1.5A".

    The pipeline matches --condition against the filename, so the spelling has
    to be the one the files use: an integer setpoint keeps no decimal point.
    """
    v = float(num.replace(",", "."))
    return (f"{v:g}" if v != int(v) else str(int(v))) + "A"


def _condition_sort_key(c: str) -> tuple[float, str]:
    m = _CURRENT_IN_NAME.search(c)
    return (float(m.group(1).replace(",", ".")) if m else float("inf"), c)


def discover_leepa(root: Path | None, source: str = "famos") -> str:
    """The order id in the first measurement filename that carries one."""
    if not root:
        return ""
    want = _SUFFIXES.get(source, _SUFFIXES["famos"])
    try:
        entries = sorted(Path(root).iterdir())
    except OSError:
        entries = []
    for p in entries:
        if p.suffix.lower() in want:
            m = _LEEPA_IN_NAME.search(p.name)
            if m:
                return m.group(1)
    # The id often sits on the FOLDER rather than the file, and there it is
    # normally bare: "2612030_07_09".  This branch needs no I/O, so it still
    # answers when the share cannot be listed -- which is the difference
    # between results landing under <root>/2612030/150A and under
    # <root>/plate/150A.
    m = _LEEPA_BARE.search(_basename(root))
    return m.group(1) if m else ""


def _basename(root: Path | str) -> str:
    """Last path component, for a POSIX or a Windows/UNC spelling alike."""
    text = str(root).rstrip("\\/")
    for sep in ("\\", "/"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text


@dataclass
class RunSpec:
    """Everything a single pipeline invocation needs."""

    condition: str
    stop_after: str = "gold"
    skew: str = "structural"
    phasor: str = "joint7"
    preset: str = "default"
    f_min: float | None = None
    f_max: float | None = None
    drt: bool = True
    spatial: bool = True
    png: bool = True
    html: bool = True
    extra: list[str] = field(default_factory=list)


def out_dir_for(st: Settings, spec: RunSpec, leepa: str) -> Path:
    r"""Where this run writes: <EIS_RESULTS_ROOT>/<leepa>/<condition>.

    Joined with ntpath when the root is a Windows drive or UNC share, so the
    command shown on the Run page stays in one spelling and can be copied into
    a shell as-is, instead of coming out as ``\\share\dir/2612030/150A``.
    """
    root = str(st.results_root)
    parts = (leepa or "plate", spec.condition)
    if settings.is_windows_absolute(root):
        return Path(ntpath.join(root, *parts))
    return Path(root).joinpath(*parts)


def build_command(st: Settings, spec: RunSpec, leepa: str) -> list[str]:
    """The argv for one run.  Pure -- tested without launching anything."""
    py = st.python or sys.executable
    cmd = [py, "main.py",
           "--plate", st.plate,
           "--condition", spec.condition,
           "--out", str(out_dir_for(st, spec, leepa)),
           "--stop-after", spec.stop_after,
           "--skew", spec.skew,
           "--phasor", spec.phasor,
           "--preset", spec.preset]
    # The source decides which root is handed over: --dat reads FAMOS .DAT,
    # --csv reads an already-extracted measurement and bypasses bronze.
    if st.source == "csv" and st.csv_root:
        cmd += ["--source", "csv", "--csv", str(st.csv_root)]
    elif st.famos_root:
        cmd += ["--dat", str(st.famos_root)]
    if leepa:
        cmd += ["--leepa", leepa]
    for flag, val in (("--curr-cal", st.curr_cal), ("--temp-cal", st.temp_cal),
                      ("--areas", st.areas_file),
                      ("--gamry", st.gamry_root)):
        if val:
            cmd += [flag, str(val)]
    if spec.f_min is not None:
        cmd += ["--f-min", repr(float(spec.f_min))]
    if spec.f_max is not None:
        cmd += ["--f-max", repr(float(spec.f_max))]
    if not spec.drt:
        cmd.append("--no-drt")
    if not spec.spatial:
        cmd.append("--no-spatial")
    if not spec.png:
        cmd.append("--no-png")
    if not spec.html:
        cmd.append("--no-html")
    cmd += spec.extra
    return cmd


class Launcher:
    """Runs one pipeline invocation, streaming its output into a buffer.

    Streamlit reruns the script on every interaction, so the run itself cannot
    live in the request: it is a subprocess plus a reader thread, and the page
    polls `lines` and `returncode`.  The object is put in st.session_state so
    it survives those reruns.
    """

    def __init__(self, cmd: list[str], cwd: Path, log_path: Path | None = None):
        self.cmd = cmd
        self.cwd = Path(cwd)
        self.log_path = log_path
        self.lines: list[str] = []
        self.returncode: int | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            return
        env = dict(os.environ)
        # Line-buffered child output, and UTF-8 so the pipeline's box-drawing
        # banners do not raise UnicodeEncodeError on a cp1252 console.
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            self.cmd, cwd=str(self.cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        fh = None
        try:
            if self.log_path:
                fh = self.log_path.open("w", encoding="utf-8")
            for line in self._proc.stdout:
                line = line.rstrip("\n")
                with self._lock:
                    self.lines.append(line)
                if fh:
                    fh.write(line + "\n")
                    fh.flush()
        finally:
            if fh:
                fh.close()
            self._proc.wait()
            with self._lock:
                self.returncode = self._proc.returncode

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    # -- state --------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._proc is not None and self.returncode is None

    def text(self, tail: int = 600) -> str:
        with self._lock:
            return "\n".join(self.lines[-tail:])
