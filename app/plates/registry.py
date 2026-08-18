"""Plate geometry as data, not as code.

The Gen-1 R2-D2 plate is a 72-segment measuring plate whose segments are
*unequal* in area (0.678 .. 8.470 cm2, a factor of 12.5).  Gen 2 and Gen 3
will have a different segment count and a different arrangement.  If the
arrangement lives in Python, every new generation is a code change, a review
and a redeploy; if it lives in a JSON spec, a new generation is a file that a
colleague drops into ``app/plates/specs/`` (or into ``$EIS_PLATE_SPEC_DIR``)
and picks from the dropdown.

A spec may describe its segments in either of two ways.

``segments`` -- explicit
    A list of rectangles on the pad grid, one per segment.  Universal: any
    arrangement whatsoever can be written this way, including one that is not
    built from full-height strips.

``strips`` -- generative
    The Gen-1 construction: the plate is cut into full-height vertical strips,
    each strip is cut into bands of pad rows, and the segments are numbered
    strip-major.  Much shorter to write when the plate has that structure, and
    it expands to the explicit form on load, so nothing downstream cares which
    form was used.

Either way a segment is a union of whole pads, so its area is exact and does
not have to be trusted to a drawing scale.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

SPEC_DIR = Path(__file__).parent / "specs"

#: Colleagues can add generations without touching the repository by pointing
#: this at a Volumes directory of JSON specs.
ENV_SPEC_DIR = "EIS_PLATE_SPEC_DIR"


class PlateSpecError(ValueError):
    """A spec that cannot be turned into a consistent plate."""


@dataclass(frozen=True)
class PlateSegment:
    """One segment, as a rectangle on the pad grid.

    Pad indices are 1-based and inclusive on both ends, which is how the
    numbering is printed on the plate drawings.
    """

    name: str
    col0: int
    col1: int
    row0: int
    row1: int
    card: str | None = None
    channel: str | None = None
    area_override_cm2: float | None = None

    @property
    def n_pads(self) -> int:
        return (self.col1 - self.col0 + 1) * (self.row1 - self.row0 + 1)


@dataclass(frozen=True)
class PlateGeometry:
    """A plate generation: pad grid, segments, sensors, known faults."""

    key: str
    name: str
    pad_w_mm: float
    pad_h_mm: float
    n_cols: int
    n_rows: int
    segments: dict[str, PlateSegment]
    flow_axis: str = "x"
    temp_sensor_x_mm: dict[str, float] = field(default_factory=dict)
    flow_channel_y_mm: tuple[float, ...] = ()
    known_bad: dict[str, str] = field(default_factory=dict)
    source: str = ""
    notes: str = ""

    # -- derived plate dimensions -------------------------------------------

    @property
    def plate_w_mm(self) -> float:
        return self.pad_w_mm * self.n_cols

    @property
    def plate_h_mm(self) -> float:
        return self.pad_h_mm * self.n_rows

    @property
    def pad_area_cm2(self) -> float:
        return (self.pad_w_mm / 10.0) * (self.pad_h_mm / 10.0)

    @property
    def cell_area_cm2(self) -> float:
        return self.plate_w_mm * self.plate_h_mm / 100.0

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    # -- per-segment accessors ----------------------------------------------

    def bounds_mm(self, name: str) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` of one segment in plate millimetres."""
        s = self.segments[name]
        return (
            (s.col0 - 1) * self.pad_w_mm,
            (s.row0 - 1) * self.pad_h_mm,
            s.col1 * self.pad_w_mm,
            s.row1 * self.pad_h_mm,
        )

    def area_cm2(self, name: str) -> float:
        s = self.segments[name]
        if s.area_override_cm2 is not None:
            return float(s.area_override_cm2)
        return s.n_pads * self.pad_area_cm2

    def centroid_mm(self, name: str) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds_mm(name)
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def areas(self) -> dict[str, float]:
        return {k: self.area_cm2(k) for k in self.segments}

    def equal_areas(self) -> dict[str, float]:
        """Every segment at ``A_cell / n_segments``.

        A deliberate simplification, not a correction: local ASR is area-free
        and does not move, but DC closure and every area-weighted average do.
        """
        a = self.cell_area_cm2 / max(self.n_segments, 1)
        return {k: a for k in self.segments}

    def centroids(self) -> dict[str, tuple[float, float]]:
        return {k: self.centroid_mm(k) for k in self.segments}

    def order(self) -> list[str]:
        """Segment names in numeric order where possible, else lexicographic."""
        def key(n: str) -> tuple[int, object]:
            return (0, int(n)) if n.lstrip("-").isdigit() else (1, n)
        return sorted(self.segments, key=key)

    # -- integrity -----------------------------------------------------------

    def self_check(self) -> dict[str, object]:
        """Do the segments tile the pad grid exactly once?

        This is the check that catches a mistyped Gen-2 spec before it silently
        produces a heat map with a hole or a double-counted area.
        """
        grid: dict[tuple[int, int], list[str]] = {}
        for name, s in self.segments.items():
            for c in range(s.col0, s.col1 + 1):
                for r in range(s.row0, s.row1 + 1):
                    grid.setdefault((c, r), []).append(name)

        overlaps = {k: v for k, v in grid.items() if len(v) > 1}
        covered = set(grid)
        full = {(c, r) for c in range(1, self.n_cols + 1)
                for r in range(1, self.n_rows + 1)}
        area_sum = sum(self.areas().values())
        return {
            "n_segments": self.n_segments,
            "pads_covered": len(covered),
            "pads_total": len(full),
            "uncovered_pads": sorted(full - covered),
            "overlapping_pads": {f"{c},{r}": v for (c, r), v in overlaps.items()},
            "area_sum_cm2": area_sum,
            "cell_area_cm2": self.cell_area_cm2,
            "area_closes": abs(area_sum - self.cell_area_cm2) < 1e-6,
            "tiles_exactly": not overlaps and covered == full,
        }

    def describe(self) -> str:
        a = self.areas().values()
        return (
            f"{self.name}: {self.n_segments} segments, "
            f"{self.plate_w_mm:.1f} x {self.plate_h_mm:.1f} mm "
            f"({self.cell_area_cm2:.2f} cm2), "
            f"areas {min(a):.3f}..{max(a):.3f} cm2"
        )


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

def _segments_from_strips(spec: dict) -> dict[str, PlateSegment]:
    """Expand the generative strip form into explicit rectangles.

    ``strips`` is a list of ``{"cols": [c0, c1], "bands": "<name>"}`` and
    ``bands`` is a dict of named row-span lists.  Segments are numbered by
    walking the strip groups in the order given by ``numbering``, each group
    naming the strips it covers and the first segment number in it.
    """
    bands: dict[str, list[list[int]]] = spec["bands"]
    strips: list[dict] = spec["strips"]
    groups: list[dict] = spec.get(
        "numbering",
        [{"strips": list(range(len(strips))), "start": 1}],
    )

    out: dict[str, PlateSegment] = {}
    for group in groups:
        n = int(group.get("start", 1))
        for idx in group["strips"]:
            strip = strips[idx]
            c0, c1 = strip["cols"]
            for r0, r1 in bands[strip["bands"]]:
                name = str(n)
                if name in out:
                    raise PlateSpecError(f"segment {name} defined twice")
                out[name] = PlateSegment(name, int(c0), int(c1), int(r0), int(r1))
                n += 1
    return out


def _segments_explicit(spec: dict) -> dict[str, PlateSegment]:
    out: dict[str, PlateSegment] = {}
    for entry in spec["segments"]:
        name = str(entry["name"] if "name" in entry else entry["segment"])
        if name in out:
            raise PlateSpecError(f"segment {name} defined twice")
        out[name] = PlateSegment(
            name=name,
            col0=int(entry["col0"]), col1=int(entry["col1"]),
            row0=int(entry["row0"]), row1=int(entry["row1"]),
            card=entry.get("card"), channel=entry.get("channel"),
            area_override_cm2=entry.get("area_cm2"),
        )
    return out


def load_spec(path: str | Path) -> PlateGeometry:
    """Read one plate spec file."""
    path = Path(path)
    spec = json.loads(path.read_text())

    if "segments" in spec:
        segments = _segments_explicit(spec)
    elif "strips" in spec:
        segments = _segments_from_strips(spec)
    else:
        raise PlateSpecError(f"{path.name}: needs either 'segments' or 'strips'")

    # Per-segment wiring, kept separate from the layout so that a re-wiring
    # does not force the geometry to be re-entered.
    wiring: dict[str, dict] = spec.get("wiring", {})
    if wiring:
        segments = {
            k: PlateSegment(
                **{**vars(s),
                   "card": wiring.get(k, {}).get("card", s.card),
                   "channel": wiring.get(k, {}).get("channel", s.channel)}
            )
            for k, s in segments.items()
        }

    geom = PlateGeometry(
        key=spec.get("key", path.stem),
        name=spec.get("name", path.stem),
        pad_w_mm=float(spec["pad_w_mm"]),
        pad_h_mm=float(spec["pad_h_mm"]),
        n_cols=int(spec["n_cols"]),
        n_rows=int(spec["n_rows"]),
        segments=segments,
        flow_axis=spec.get("flow_axis", "x"),
        temp_sensor_x_mm=spec.get("temp_sensor_x_mm", {}),
        flow_channel_y_mm=tuple(spec.get("flow_channel_y_mm", ())),
        known_bad=spec.get("known_bad", {}),
        source=str(path),
        notes=spec.get("notes", ""),
    )

    for name, s in geom.segments.items():
        if not (1 <= s.col0 <= s.col1 <= geom.n_cols):
            raise PlateSpecError(
                f"{path.name}: segment {name} columns {s.col0}..{s.col1} "
                f"outside 1..{geom.n_cols}")
        if not (1 <= s.row0 <= s.row1 <= geom.n_rows):
            raise PlateSpecError(
                f"{path.name}: segment {name} rows {s.row0}..{s.row1} "
                f"outside 1..{geom.n_rows}")
    return geom


def spec_dirs() -> list[Path]:
    """Built-in specs first, then any directory the operator has added."""
    dirs = [SPEC_DIR]
    extra = os.environ.get(ENV_SPEC_DIR, "")
    dirs += [Path(p) for p in extra.split(os.pathsep) if p.strip()]
    return [d for d in dirs if d.is_dir()]


@lru_cache(maxsize=1)
def _scan() -> dict[str, PlateGeometry]:
    found: dict[str, PlateGeometry] = {}
    for directory in spec_dirs():
        for path in sorted(directory.glob("*.json")):
            if path.name.startswith("_"):
                continue                      # templates and examples
            try:
                geom = load_spec(path)
            except Exception as exc:          # a bad spec must not hide good ones
                print(f"[plates] skipping {path.name}: {exc}")
                continue
            found[geom.key] = geom            # later directories win
    return found


def available() -> dict[str, PlateGeometry]:
    """Every plate generation this deployment can draw, keyed by spec key."""
    return dict(_scan())


def get(key: str) -> PlateGeometry:
    plates = available()
    if key not in plates:
        raise KeyError(
            f"unknown plate generation {key!r}; have {sorted(plates)}")
    return plates[key]


def default_key() -> str:
    plates = available()
    if not plates:
        raise PlateSpecError("no plate specs found in " +
                             os.pathsep.join(str(d) for d in spec_dirs()))
    return os.environ.get("EIS_DEFAULT_PLATE") or sorted(plates)[0]


def reload() -> None:
    """Forget the cached scan so a newly dropped spec is picked up."""
    _scan.cache_clear()


def options() -> list[dict[str, str]]:
    """Dash dropdown options for the generation selector."""
    return [{"label": g.name, "value": k} for k, g in sorted(available().items())]


if __name__ == "__main__":
    for key, geom in sorted(available().items()):
        print(f"\n{key}\n  {geom.describe()}")
        check = geom.self_check()
        status = "OK" if check["tiles_exactly"] and check["area_closes"] else "PROBLEM"
        print(f"  tiling: {check['pads_covered']}/{check['pads_total']} pads, "
              f"area {check['area_sum_cm2']:.2f}/{check['cell_area_cm2']:.2f} cm2  "
              f"[{status}]")
        if check["uncovered_pads"]:
            print(f"  uncovered: {check['uncovered_pads'][:10]} ...")
        if check["overlapping_pads"]:
            print(f"  overlapping: {list(check['overlapping_pads'])[:10]} ...")
