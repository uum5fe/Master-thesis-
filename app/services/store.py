"""Server-side caching and background jobs.

A Dash callback runs on every dropdown change, so anything that touches the
file system or the optimiser has to be memoised or the app spends its life
re-reading the same CSVs.  Two mechanisms here:

``cached_run`` / ``cached_catalog``
    Plain LRU memoisation keyed on the selection *and* a generation counter, so
    the Refresh button genuinely re-reads the disk while ordinary interaction
    does not.

``JOBS``
    A tiny thread-backed job table for the two operations that are too slow to
    block on: fitting the whole plate (tens of seconds) and running the
    pipeline over raw recordings (minutes).  A job reports progress, which the
    UI polls with a ``dcc.Interval``; nothing is silently synchronous.

This is deliberately in-process.  Databricks Apps runs a single container, so a
process-local cache is the right size of solution; if this ever runs behind
several replicas, swap the dictionaries for a shared cache and nothing else
changes.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from app.data.model import RunData
from app.data.sources import Catalog, RunRef
from app.settings import SETTINGS

_GENERATION = 0


def bump_generation() -> int:
    """Invalidate every cache; called by the Refresh button."""
    global _GENERATION
    _GENERATION += 1
    cached_catalog.cache_clear()
    cached_run.cache_clear()
    return _GENERATION


def generation() -> int:
    return _GENERATION


@lru_cache(maxsize=4)
def cached_catalog(generation: int) -> Catalog:
    return Catalog(SETTINGS).refresh()


@lru_cache(maxsize=32)
def cached_run(generation: int, kind: str, measurement_id: str,
               condition: str, plate_key: str) -> RunData:
    catalog = cached_catalog(generation)
    ref = catalog.find(measurement_id, condition, kind)
    if ref is None:
        run = RunData(measurement_id=measurement_id, condition=condition,
                      plate_key=plate_key)
        run.warnings.append(
            f"no {kind} entry for {measurement_id} / {condition}; "
            "press Refresh if the run has just finished")
        return run
    return catalog.load(ref, plate_key)


def current_catalog() -> Catalog:
    return cached_catalog(generation())


def current_run(kind: str, measurement_id: str, condition: str,
                plate_key: str) -> RunData:
    return cached_run(generation(), kind, measurement_id, condition, plate_key)


def find_ref(kind: str, measurement_id: str, condition: str) -> RunRef | None:
    return current_catalog().find(measurement_id, condition, kind)


# ---------------------------------------------------------------------------
# background jobs
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    label: str
    state: str = "running"            # running | done | failed
    done: int = 0
    total: int = 0
    message: str = ""
    result: object = None
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float = 0.0

    @property
    def percent(self) -> int:
        return int(100 * self.done / self.total) if self.total else 0

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def status_text(self) -> str:
        if self.state == "running":
            head = f"{self.label}: {self.done}/{self.total}" if self.total else self.label
            return f"{head}  ({self.elapsed:.0f} s)"
        if self.state == "done":
            return f"{self.label}: finished in {self.elapsed:.1f} s"
        return f"{self.label}: failed — {self.error}"


class JobTable:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, label: str, target, *args, **kwargs) -> str:
        """Start ``target(progress, *args, **kwargs)`` on a worker thread."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, label=label)
        with self._lock:
            self._jobs[job_id] = job

        def progress(done: int, total: int, message: str = "") -> None:
            job.done, job.total = done, total
            if message:
                job.message = message

        def run() -> None:
            try:
                job.result = target(progress, *args, **kwargs)
                job.state = "done"
            except Exception as exc:                      # surfaced in the UI
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()

        threading.Thread(target=run, daemon=True, name=f"job-{job_id}").start()
        return job_id

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def prune(self, keep_seconds: float = 3600.0) -> None:
        cutoff = time.time() - keep_seconds
        with self._lock:
            for key in [k for k, j in self._jobs.items()
                        if j.finished and j.finished < cutoff]:
                self._jobs.pop(key, None)


JOBS = JobTable()


# ---------------------------------------------------------------------------
# ECM results, kept per (run, fit settings)
# ---------------------------------------------------------------------------

_ECM: dict[tuple, tuple[pd.DataFrame, dict]] = {}
_ECM_LOCK = threading.Lock()


def ecm_key(kind: str, measurement_id: str, condition: str, model: str,
            f_min, f_max, n_starts: int) -> tuple:
    return (generation(), kind, measurement_id, condition, model,
            f_min, f_max, n_starts)


def ecm_get(key: tuple):
    with _ECM_LOCK:
        return _ECM.get(key)


def ecm_put(key: tuple, table: pd.DataFrame, fits: dict) -> None:
    with _ECM_LOCK:
        if len(_ECM) > 24:                      # keep memory bounded
            _ECM.pop(next(iter(_ECM)))
        _ECM[key] = (table, fits)
