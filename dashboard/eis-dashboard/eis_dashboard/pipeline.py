"""Discovering conditions and launching the pipeline.

The dashboard never imports the pipeline.  It shells out to ``main.py`` in
EIS_PIPELINE_DIR, which keeps the two dependency sets apart (the pipeline wants
scipy and matplotlib; the dashboard does not) and means a run that dies takes a
subprocess with it rather than the web server.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .settings import Settings

#: "..._Current_150A_Test_01_Karte_3.DAT" -> "150A"; also "450 A", "1.5A".
_CURRENT_IN_NAME = re.compile(r"(\d+(?:[.,]\d+)?)\s*A(?![A-Za-z])")
#: "Leepa_RO2612030_..." / "Leepa_2612030_..." -> "2612030"
_LEEPA_IN_NAME = re.compile(r"RO?(\d{7})")


def discover_conditions(dat_dir: Path | None) -> list[str]:
    """Current setpoints present in the .DAT filenames, naturally ordered."""
    if not dat_dir or not Path(dat_dir).is_dir():
        return []
    found: set[str] = set()
    for p in Path(dat_dir).iterdir():
        if p.suffix.lower() != ".dat":
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


def discover_leepa(dat_dir: Path | None) -> str:
    if not dat_dir or not Path(dat_dir).is_dir():
        return ""
    for p in sorted(Path(dat_dir).iterdir()):
        if p.suffix.lower() == ".dat":
            m = _LEEPA_IN_NAME.search(p.name)
            if m:
                return m.group(1)
    return ""


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
    """Where this run writes: <EIS_OUT_DIR>/<leepa>/<condition>."""
    return Path(st.out_dir) / (leepa or "plate") / spec.condition


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
    if st.dat_dir:
        cmd += ["--dat", str(st.dat_dir)]
    if leepa:
        cmd += ["--leepa", leepa]
    for flag, val in (("--curr-cal", st.curr_cal), ("--temp-cal", st.temp_cal),
                      ("--areas", st.areas), ("--gain", st.gain),
                      ("--gamry", st.gamry_dir),
                      ("--bench-log", st.bench_log)):
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
