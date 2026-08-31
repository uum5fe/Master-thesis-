#!/usr/bin/env python3
"""
plate_conditions.py  --  temperature, pressure and humidity across the plate
============================================================================

The bench measures the operating point at the PORTS: gas in and gas out.  The
plate is 252 mm long, and what happens between those two ports is most of what
makes a local EIS map interesting -- the cathode dries at the inlet and floods
at the outlet, and the impedance follows.  This module turns port measurements
into a per-segment field so the two can be looked at on the same geometry.

HOW MUCH OF EACH FIELD IS MEASURED
----------------------------------
Not equally, and the difference matters more than the pictures do.  Every
field this module returns carries a `provenance` string, and the viewer prints
it on the map, because a modelled field drawn in the same colours as a
measured one invites conclusions the data cannot support.

    TEMPERATURE   MEASURED.  Four sensors sit on the plate at x = 0, 84, 168
                  and 252 mm and are interpolated along the flow direction.
                  This is a real spatial measurement, not a model.  Without
                  them (a CSV sweep that carries no temp channels) it falls
                  back to interpolating the gas inlet and outlet, which is a
                  much weaker statement and says so.

    PRESSURE      INTERPOLATED between two measurements.  Inlet and outlet are
                  both measured (p_Si_C, p_So_C); the profile between them is
                  assumed linear.  For a straight channel at these flow rates
                  that is a good assumption, but it is an assumption: a real
                  channel loses more pressure where the gas is fastest.

    HUMIDITY      MODELLED.  Only the INLET humidity is measured -- there is no
                  outlet RH sensor on this bench.  The rest is a water balance
                  along the cathode channel: the vapour that entered, plus the
                  water produced by the current that has passed so far, minus
                  the oxygen consumed, evaluated against the saturation
                  pressure at the local temperature.  It is standard and
                  defensible, and it is still a model.  Where it predicts more
                  than 100 % the gas is saturated and liquid water is present;
                  that is reported rather than clipped away silently.

THE WATER BALANCE
-----------------
Along the flow coordinate xi in [0, 1], with F the Faraday constant and I the
cell current:

    n_dry(xi)  = n_dry_in  -  q(xi) * I / (4F)        oxygen consumed
    n_vap(xi)  = n_vap_in  +  q(xi) * I / (2F)        product water
    x_vap(xi)  = n_vap / (n_vap + n_dry)
    RH(xi)     = x_vap(xi) * p(xi) / p_sat(T(xi))

`q(xi)` is the fraction of the total current that has been produced upstream of
xi.  It is NOT assumed uniform when the local map is available: the measured
per-segment current density is used, so the humidity field is informed by the
current distribution that was actually measured rather than by an average that
the local EIS exists to disprove.  With no local map it falls back to the
area-weighted uniform assumption and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FARADAY = 96485.332          # C/mol
MOLAR_VOLUME_NL = 22.414     # Nl/mol at 0 C, 1013.25 mbar -- the "N" in Nl/min


# ---------------------------------------------------------------------------
# 1. Water vapour
# ---------------------------------------------------------------------------

def saturation_pressure_pa(t_c) -> np.ndarray:
    """Saturation vapour pressure of water, Buck (1996).

    p_sat = 611.21 * exp((18.678 - T/234.5) * (T / (257.14 + T)))  [Pa], T in C.

    Within 0.05 % of the steam tables from 0 to 100 C -- it gives 101.34 kPa at
    100 C against the definition of 101.325.  The simpler Magnus form is 2.7 %
    high there, and while a fuel cell never runs at 100 C, the saturation
    pressure is the DENOMINATOR of every humidity number in this module, so an
    error in it is an error in the whole field.
    """
    t = np.asarray(t_c, float)
    return 611.21 * np.exp((18.678 - t / 234.5) * (t / (257.14 + t)))


def dew_point_to_mole_fraction(dew_c, p_total_pa) -> np.ndarray:
    """Vapour mole fraction from a dew point and a total pressure."""
    return saturation_pressure_pa(dew_c) / np.asarray(p_total_pa, float)


def cathode_stoichiometry(current_a: float, dry_air_nl_min: float,
                          o2_fraction: float = 0.2095) -> float:
    """Air supplied / air needed, lambda.

    The one number that says whether to believe the humidity field.  Every
    mole of oxygen consumed is two moles of electrons and one mole of product
    water, so lambda fixes both how much water is added and how much gas there
    is to carry it.  At lambda = 3 the outlet is comfortably below saturation
    and the model is forgiving; at lambda near 1 almost all the oxygen is used,
    the outlet floods, and small errors in the flow reading move the predicted
    RH a long way.  Below 1 the operating point is impossible and the log is
    wrong, not the cell.
    """
    if not (np.isfinite(current_a) and np.isfinite(dry_air_nl_min)):
        return float("nan")
    need = current_a / (4.0 * FARADAY) / o2_fraction * MOLAR_VOLUME_NL * 60.0
    return float(dry_air_nl_min / need) if need > 0 else float("inf")


# ---------------------------------------------------------------------------
# 2. What the bench has to supply
# ---------------------------------------------------------------------------

@dataclass
class PortState:
    """The operating point at the ports, in SI-ish units the bench reports."""

    current_a: float = float("nan")
    n_cells: float = 1.0
    #: cathode gas
    t_in_c: float = float("nan")
    t_out_c: float = float("nan")
    p_in_bara: float = float("nan")
    p_out_bara: float = float("nan")
    rh_in_pct: float = float("nan")
    dew_in_c: float = float("nan")
    air_flow_nl_min: float = float("nan")        # total (wet) cathode air
    air_dry_flow_nl_min: float = float("nan")    # dry fraction, if reported
    #: plate-mounted sensors, name -> degrees C
    plate_t: dict[str, float] = field(default_factory=dict)

    #: The channel names this was read from, so a map can be traced back.
    source: dict[str, str] = field(default_factory=dict)


#: Bench channel -> PortState field, for the MF4 layout this rig writes.
MF4_CHANNELS: dict[str, str] = {
    "I_S": "current_a",
    "n_Cells": "n_cells",
    "T_Si_C": "t_in_c",
    "T_So_C": "t_out_c",
    "p_Si_C": "p_in_bara",
    "p_So_C": "p_out_bara",
    "RH_Si_C_gas": "rh_in_pct",
    "DPT_Si_C": "dew_in_c",
    "FN_Si_Air_C": "air_flow_nl_min",
    "FN_Si_Air_C_dry": "air_dry_flow_nl_min",
}


def port_state_from_bench(state: dict, plate_t: dict | None = None) -> PortState:
    """Turn a `gamry_compare.BenchLog.state_at()` reading into a PortState."""
    out = PortState(plate_t=dict(plate_t or {}))
    for channel, attr in MF4_CHANNELS.items():
        if channel in state and np.isfinite(state[channel]):
            setattr(out, attr, float(state[channel]))
            out.source[attr] = channel
    return out



# ---------------------------------------------------------------------------
# 2b. Where the gases actually enter and leave
# ---------------------------------------------------------------------------
#
# The plate has FOUR ports, one per corner, and two gas circuits that cross:
#
#        (0,0) top-left                              top-right (x_max, 0)
#              O2 out  <------------------------------  H2 out
#                 ^  \                              /    ^
#                 |    \                          /      |
#                 |      \   O2 path      H2 path       |
#                 |        \                    /        |
#              H2 in   ------------------------------>  O2 in
#        bottom-left (0, y_max)                    bottom-right (x_max, y_max)
#
#   hydrogen : bottom-left  ->  top-right
#   oxygen   : bottom-right ->  top-left
#
# Both paths run bottom-to-top, and they run in OPPOSITE directions across
# the plate.  That is why a single "flow axis with an inlet end", which is
# what this module used to model, cannot describe it: whichever direction
# that axis is given, it is right for one gas and backwards for the other,
# and every field built on it is mirrored for the gas it got wrong.
#
# It matters most exactly where the user is looking.  Water is produced at
# the CATHODE, so the liquid-water gradient runs along the oxygen path --
# right to left -- while the hydrogen along its own path runs left to right.
# At 450 A there is an order of magnitude more product water than at 45 A,
# so the two conditions differ most at the oxygen outlet, in the TOP-LEFT
# corner.  Reading that map against the hydrogen coordinate puts the wet end
# at the wrong corner.

#: y increases DOWNWARD in this geometry -- the centroids have their origin
#: at the top-left pad -- so "top" is minimum y and "bottom" is maximum y.
CORNERS = {
    "top-left": ("min", "min"),
    "top-right": ("max", "min"),
    "bottom-left": ("min", "max"),
    "bottom-right": ("max", "max"),
}


@dataclass(frozen=True)
class Port:
    """One corner connection."""

    gas: str            #: "H2" | "O2"
    role: str           #: "in" | "out"
    corner: str         #: a key of CORNERS

    @property
    def name(self) -> str:
        return f"{self.gas}_{self.role}"


#: The arrangement on this bench, looking at the plate as the maps draw it.
DEFAULT_PORTS: tuple[Port, ...] = (
    Port("O2", "out", "top-left"),
    Port("H2", "in", "bottom-left"),
    Port("H2", "out", "top-right"),
    Port("O2", "in", "bottom-right"),
)


def corner_xy(centroids: dict[str, tuple[float, float]],
              corner: str) -> tuple[float, float]:
    """The (x, y) of a named corner of the segment field."""
    if corner not in CORNERS:
        raise ValueError(f"unknown corner {corner!r}; expected one of "
                         f"{sorted(CORNERS)}")
    xs = [c[0] for c in centroids.values()]
    ys = [c[1] for c in centroids.values()]
    kx, ky = CORNERS[corner]
    return ((min(xs) if kx == "min" else max(xs)),
            (min(ys) if ky == "min" else max(ys)))


def gas_path(ports: tuple[Port, ...] = DEFAULT_PORTS,
             gas: str = "O2") -> tuple[str, str]:
    """(inlet corner, outlet corner) for one gas."""
    try:
        inlet = next(p.corner for p in ports if p.gas == gas and p.role == "in")
        outlet = next(p.corner for p in ports if p.gas == gas and p.role == "out")
    except StopIteration:
        raise ValueError(f"no complete path for {gas!r} in {ports!r}") from None
    return inlet, outlet


def flow_coordinate(centroids: dict[str, tuple[float, float]],
                    gas: str = "O2",
                    ports: tuple[Port, ...] = DEFAULT_PORTS
                    ) -> dict[str, float]:
    """Position of each segment along one gas's path: 0 inlet, 1 outlet.

    The path is the straight line from the inlet corner to the outlet
    corner, and each segment is projected onto it.  For the diagonal
    arrangement above this is the natural generalisation of the old
    single-axis coordinate: give it two corners that differ in x only and it
    reduces exactly to it.

    Projection rather than a channel-following path length: the segments are
    a coarse 12 x 6 grid and the serpentine inside each of them is far finer
    than the grid, so any attempt to trace the actual channel would be
    inventing detail the measurement cannot resolve. What the coordinate has
    to get right is the ORDER -- which segments the gas reaches before which
    -- and the projection does.
    """
    inlet, outlet = gas_path(ports, gas)
    x0, y0 = corner_xy(centroids, inlet)
    x1, y1 = corner_xy(centroids, outlet)
    dx, dy = x1 - x0, y1 - y0
    norm = dx * dx + dy * dy
    if norm <= 0:
        raise ValueError(f"the {gas} inlet and outlet are the same corner")
    raw = {k: ((c[0] - x0) * dx + (c[1] - y0) * dy) / norm
           for k, c in centroids.items()}
    lo, hi = min(raw.values()), max(raw.values())
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in raw.items()}


def port_layout(centroids: dict[str, tuple[float, float]],
                ports: tuple[Port, ...] = DEFAULT_PORTS) -> dict:
    """The port map in a form a figure or a manifest can use directly."""
    return {
        "ports": [{"gas": p.gas, "role": p.role, "corner": p.corner,
                   "x_mm": corner_xy(centroids, p.corner)[0],
                   "y_mm": corner_xy(centroids, p.corner)[1]}
                  for p in ports],
        "paths": {gas: {"inlet": gas_path(ports, gas)[0],
                        "outlet": gas_path(ports, gas)[1]}
                  for gas in sorted({p.gas for p in ports})},
        "note": "y increases downward; 'top' is minimum y",
    }


# ---------------------------------------------------------------------------
# 3. The fields
# ---------------------------------------------------------------------------

@dataclass
class Field:
    """One scalar field over the segments, and how much of it was measured."""

    name: str
    unit: str
    values: dict[str, float]
    provenance: str
    notes: list[str] = field(default_factory=list)
    #: Cathode lambda, on the humidity field only; NaN elsewhere.
    stoichiometry: float = float("nan")

    @property
    def measured(self) -> bool:
        return self.provenance.startswith("measured")

    def summary(self) -> dict:
        v = [x for x in self.values.values() if np.isfinite(x)]
        return {
            "field": self.name, "unit": self.unit,
            "n": len(v),
            "min": float(min(v)) if v else float("nan"),
            "max": float(max(v)) if v else float("nan"),
            "provenance": self.provenance,
            "notes": "; ".join(self.notes),
        }


def _flow_coordinate(centroids: dict[str, tuple[float, float]],
                     axis: str = "x", inlet: str = "min") -> dict[str, float]:
    """Position of each segment along the flow path, 0 at the inlet, 1 at the
    outlet.

    WHICH END IS THE INLET IS AN ASSUMPTION.  Nothing in the plate drawing or
    the bench log states it, so it is a parameter with a default rather than
    something inferred and presented as fact.  Getting it backwards mirrors
    every field, which is why it is worth saying out loud.
    """
    i = 0 if axis == "x" else 1
    pos = {k: c[i] for k, c in centroids.items()}
    lo, hi = min(pos.values()), max(pos.values())
    span = (hi - lo) or 1.0
    if inlet == "min":
        return {k: (v - lo) / span for k, v in pos.items()}
    return {k: (hi - v) / span for k, v in pos.items()}


def _current_fraction(xi: dict[str, float], areas: dict[str, float],
                      j_dc: dict[str, float] | None) -> tuple[dict[str, float], bool]:
    """Fraction of the total current produced upstream of each segment.

    Uses the MEASURED local current density when there is one.  The whole point
    of a local EIS is that the current is not uniform, so assuming uniformity
    to build the humidity field would bake in the very thing the measurement
    is there to test.
    """
    order = sorted(xi, key=lambda k: xi[k])
    if j_dc and any(np.isfinite(j_dc.get(k, np.nan)) for k in order):
        weight = {k: max(float(j_dc.get(k, 0.0)), 0.0) * areas.get(k, 0.0)
                  for k in order}
        used_measured = True
    else:
        weight = {k: areas.get(k, 0.0) for k in order}
        used_measured = False
    total = sum(weight.values()) or 1.0

    out: dict[str, float] = {}
    run = 0.0
    for k in order:
        # the midpoint of this segment's own contribution
        out[k] = (run + 0.5 * weight[k]) / total
        run += weight[k]
    return out, used_measured


def condition_fields(centroids: dict[str, tuple[float, float]],
                     areas: dict[str, float],
                     ports: PortState,
                     temp_sensor_x_mm: dict[str, float] | None = None,
                     j_dc: dict[str, float] | None = None,
                     axis: str = "x", inlet: str = "min",
                     gas: str | None = "O2",
                     port_map: tuple[Port, ...] = DEFAULT_PORTS
                     ) -> dict[str, Field]:
    """Per-segment temperature, pressure and relative humidity.

    `gas` selects which circuit the fields run along.  The default is the
    OXYGEN path, because the fields this builds are about the cathode: water
    is produced there, so the humidity gradient -- the reason the whole
    module exists -- develops from the oxygen inlet to the oxygen outlet.
    Passing gas=None falls back to the old single-axis coordinate, which is
    kept for plates whose ports really are on opposite edges rather than at
    the corners.
    """
    if gas is not None:
        xi = flow_coordinate(centroids, gas, port_map)
        inlet_corner, outlet_corner = gas_path(port_map, gas)
        # The projection axis for the sensor interpolation: whichever of x
        # or y the path travels further along.
        x0, y0 = corner_xy(centroids, inlet_corner)
        x1, y1 = corner_xy(centroids, outlet_corner)
        i = 0 if abs(x1 - x0) >= abs(y1 - y0) else 1
        flow_label = f"{gas}: {inlet_corner} to {outlet_corner}"
    else:
        xi = _flow_coordinate(centroids, axis, inlet)
        i = 0 if axis == "x" else 1
        flow_label = f"axis {axis}, inlet at {inlet}"
    fields: dict[str, Field] = {}

    # ---- temperature ------------------------------------------------------
    sensors = {k: v for k, v in (ports.plate_t or {}).items()
               if np.isfinite(v) and k in (temp_sensor_x_mm or {})}
    if len(sensors) >= 2:
        xs = sorted((temp_sensor_x_mm[k], sensors[k]) for k in sensors)
        t_vals = {k: float(np.interp(c[i], [p[0] for p in xs], [p[1] for p in xs]))
                  for k, c in centroids.items()}
        t_field = Field("temperature", "°C", t_vals,
                        f"measured — {len(sensors)} plate sensors interpolated "
                        f"along {axis}")
    elif np.isfinite(ports.t_in_c) and np.isfinite(ports.t_out_c):
        t_vals = {k: ports.t_in_c + (ports.t_out_c - ports.t_in_c) * f
                  for k, f in xi.items()}
        t_field = Field("temperature", "°C", t_vals,
                        "interpolated — gas inlet and outlet only",
                        ["no plate temperature sensors in this recording, so "
                         "this is the port-to-port gradient, not a measurement "
                         "of the plate"])
    else:
        t_field = Field("temperature", "°C",
                        {k: float("nan") for k in centroids}, "unavailable",
                        ["no plate sensors and no gas inlet/outlet temperature"])
    fields["temperature"] = t_field

    # ---- pressure ---------------------------------------------------------
    if np.isfinite(ports.p_in_bara) and np.isfinite(ports.p_out_bara):
        p_vals = {k: ports.p_in_bara
                  + (ports.p_out_bara - ports.p_in_bara) * f
                  for k, f in xi.items()}
        drop = 1e3 * (ports.p_in_bara - ports.p_out_bara)
        p_field = Field("pressure", "bar(a)", p_vals,
                        "interpolated — measured at both ports",
                        [f"{drop:.0f} mbar across the plate, assumed linear"])
    else:
        p_field = Field("pressure", "bar(a)",
                        {k: float("nan") for k in centroids}, "unavailable",
                        ["inlet or outlet pressure missing"])
    fields["pressure"] = p_field

    # ---- relative humidity ------------------------------------------------
    fields["humidity"] = _humidity_field(
        xi, areas, ports, t_field, p_field, j_dc)
    for field_ in fields.values():
        field_.notes.append(f"along the flow path {flow_label}")
    return fields


def _humidity_field(xi, areas, ports: PortState, t_field: Field,
                    p_field: Field, j_dc) -> Field:
    notes: list[str] = []

    dry_nl = ports.air_dry_flow_nl_min
    if not np.isfinite(dry_nl) or dry_nl <= 0:
        dry_nl = ports.air_flow_nl_min
        if np.isfinite(dry_nl):
            notes.append("dry-air flow not reported; total cathode flow used, "
                         "which overstates the dry fraction and therefore "
                         "understates RH")
    missing = [n for n, v in (("current", ports.current_a),
                              ("cathode air flow", dry_nl),
                              ("inlet humidity", ports.dew_in_c
                               if np.isfinite(ports.dew_in_c)
                               else ports.rh_in_pct))
               if not np.isfinite(v)]
    if missing or not np.isfinite(ports.p_in_bara):
        return Field("humidity", "% RH", {k: float("nan") for k in xi},
                     "unavailable",
                     [f"missing: {', '.join(missing) or 'inlet pressure'}"])

    p_in_pa = ports.p_in_bara * 1e5
    if np.isfinite(ports.dew_in_c):
        x_v_in = float(dew_point_to_mole_fraction(ports.dew_in_c, p_in_pa))
        notes.append(f"inlet from the measured dew point {ports.dew_in_c:.1f} °C")
    else:
        x_v_in = float(0.01 * ports.rh_in_pct
                       * saturation_pressure_pa(ports.t_in_c) / p_in_pa)
        notes.append(f"inlet from the measured RH {ports.rh_in_pct:.0f} % at "
                     f"{ports.t_in_c:.1f} °C")
    x_v_in = float(np.clip(x_v_in, 0.0, 0.95))

    # Nl/min of the wet stream -> mol/s of dry gas and of vapour
    n_total = dry_nl / MOLAR_VOLUME_NL / 60.0
    n_dry_in = n_total * (1.0 - x_v_in) if dry_nl is ports.air_flow_nl_min \
        else n_total
    n_vap_in = n_dry_in * x_v_in / max(1.0 - x_v_in, 1e-9)

    cells = ports.n_cells if np.isfinite(ports.n_cells) and ports.n_cells > 0 else 1.0
    i_cell = ports.current_a / cells

    lam = cathode_stoichiometry(i_cell, dry_nl)
    if np.isfinite(lam):
        notes.append(f"cathode stoichiometry λ = {lam:.2f}")
        if lam < 1.0:
            notes.append("λ BELOW 1: more current is being drawn than the "
                         "measured air can supply, so the flow reading and the "
                         "current do not describe the same moment — treat this "
                         "field as unusable rather than as a wet cell")
        elif lam < 1.3:
            notes.append("λ close to 1: nearly all the oxygen is consumed, so "
                         "the outlet is genuinely near flooding AND the model "
                         "is at its most sensitive to the flow reading here")
    q, used_measured = _current_fraction(xi, areas, j_dc)
    notes.append("current distribution from the measured local map"
                 if used_measured else
                 "no local current map available, so the current is assumed "
                 "uniform over the plate")

    values: dict[str, float] = {}
    saturated: list[str] = []
    for k, frac in q.items():
        n_dry = n_dry_in - frac * i_cell / (4.0 * FARADAY)
        n_vap = n_vap_in + frac * i_cell / (2.0 * FARADAY)
        n_dry = max(n_dry, 1e-12)
        x_v = n_vap / (n_vap + n_dry)
        p_pa = p_field.values.get(k, float("nan")) * 1e5
        t_c = t_field.values.get(k, float("nan"))
        rh = 100.0 * x_v * p_pa / saturation_pressure_pa(t_c)
        if np.isfinite(rh) and rh > 100.0:
            saturated.append(k)
        values[k] = float(rh)

    if saturated:
        notes.append(f"{len(saturated)} segment(s) above 100 % — the gas is "
                     "saturated there and liquid water is present; the value "
                     "is left uncapped so the degree of oversaturation stays "
                     "visible")
    out = Field("humidity", "% RH", values,
                "modelled — cathode water balance", notes)
    out.stoichiometry = lam
    return out


# ---------------------------------------------------------------------------
# 4. Self-test
# ---------------------------------------------------------------------------

def _selftest(log=None) -> int:
    import utils
    log = log or utils.get_logger(True)
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        log.info(f"    {name:<38}: {'PASS' if ok else 'FAIL'}  {detail}")

    # p_sat against the steam tables
    check("p_sat(100 C) = 1 atm",
          abs(float(saturation_pressure_pa(100.0)) / 101325.0 - 1.0) < 0.01,
          f"{float(saturation_pressure_pa(100.0)):.0f} Pa")
    check("p_sat(60 C) ~ 19.9 kPa",
          abs(float(saturation_pressure_pa(60.0)) / 19946.0 - 1.0) < 0.01,
          f"{float(saturation_pressure_pa(60.0)):.0f} Pa")

    cent = {str(i): (float(x), 60.0) for i, x in
            enumerate(np.linspace(0, 252, 10), start=1)}
    areas = {k: 30.0 for k in cent}
    ports = PortState(current_a=450.0, n_cells=1.0, t_in_c=70.0, t_out_c=62.0,
                      p_in_bara=1.5, p_out_bara=1.39, dew_in_c=55.0,
                      air_dry_flow_nl_min=8.0,
                      plate_t={"temp1": 58.0, "temp2": 62.0,
                               "temp3": 66.0, "temp4": 70.0})
    sensors = {"temp1": 0.0, "temp2": 84.0, "temp3": 168.0, "temp4": 252.0}
    f = condition_fields(cent, areas, ports, sensors)

    # ORDER BY THE FLOW COORDINATE, NOT BY THE SEGMENT LABEL.  The label
    # order only matched the flow while the model was a single left-to-right
    # axis. With the real four-corner port map the oxygen runs right to left,
    # so "segment 1 is the inlet" stopped being true -- and an assertion
    # written that way fails for a model that is correct, which is the worst
    # kind of test to leave in place.
    order = sorted(cent, key=lambda k: flow_coordinate(cent, "O2")[k])
    first, last = order[0], order[-1]

    t = f["temperature"]
    check("T uses the plate sensors", t.measured, t.provenance)
    check("T spans the sensor range",
          abs(min(t.values.values()) - 58.0) < 0.1
          and abs(max(t.values.values()) - 70.0) < 0.1)

    p = f["pressure"]
    check("p falls from inlet to outlet",
          p.values[first] > p.values[last]
          and abs(p.values[first] - 1.5) < 1e-6,
          f"{p.values[first]:.3f} -> {p.values[last]:.3f} bara")

    h = f["humidity"]
    check("RH is modelled, and says so",
          not h.measured and h.provenance.startswith("modelled"),
          h.provenance)
    rh = [h.values[k] for k in order]
    check("RH rises along the channel", all(np.diff(rh) > 0),
          f"{rh[0]:.0f} -> {rh[-1]:.0f} %")

    # More current must make it wetter, and no current must leave it at inlet RH
    wetter = condition_fields(cent, areas,
                              PortState(**{**ports.__dict__, "current_a": 900.0}),
                              sensors)["humidity"]
    check("more current -> wetter", wetter.values["10"] > h.values["10"],
          f"{h.values['10']:.0f} -> {wetter.values['10']:.0f} %")

    # With no current no water is added, so the VAPOUR FRACTION is constant.
    # RH is not: it is x_v * p / p_sat(T), and p_sat doubles over the 12 K the
    # plate spans, so a flat RH here would mean the temperature field was being
    # ignored.  What must be constant is what the balance actually conserves.
    dry = condition_fields(cent, areas,
                           PortState(**{**ports.__dict__, "current_a": 0.0}),
                           sensors)["humidity"]
    p_f, t_f = f["pressure"], f["temperature"]
    x_v = [dry.values[k] / 100.0 * saturation_pressure_pa(t_f.values[k])
           / (p_f.values[k] * 1e5) for k in cent]
    check("no current -> constant vapour fraction",
          (max(x_v) - min(x_v)) / max(x_v) < 1e-9,
          f"x_v = {x_v[0]:.5f}, RH still varies {min(dry.values.values()):.0f}"
          f"-{max(dry.values.values()):.0f} % because T does")

    # The measured current map must actually change the answer
    skewed = {k: (5.0 if float(k) <= 5 else 0.2) for k in cent}
    tilted = condition_fields(cent, areas, ports, sensors, j_dc=skewed)["humidity"]
    check("a measured current map changes RH",
          abs(tilted.values["5"] - h.values["5"]) > 1.0,
          f"{h.values['5']:.0f} -> {tilted.values['5']:.0f} % at mid-plate")

    # Stoichiometry: at 450 A a cell needs ~7.5 Nl/min of dry air per lambda
    check("lambda = 1 at the stoichiometric flow",
          abs(cathode_stoichiometry(450.0, 7.466) - 1.0) < 0.01,
          f"{cathode_stoichiometry(450.0, 7.466):.3f}")
    starved = condition_fields(cent, areas,
                               PortState(**{**ports.__dict__,
                                            "air_dry_flow_nl_min": 3.0}),
                               sensors)["humidity"]
    check("starved air is called out",
          starved.stoichiometry < 1.0
          and any("BELOW 1" in n for n in starved.notes),
          f"λ = {starved.stoichiometry:.2f}")

    # Reversing the inlet must mirror the field, not silently agree
    rev = condition_fields(cent, areas, ports, sensors, inlet="max")["humidity"]
    check("reversing the inlet mirrors RH",
          abs(rev.values["1"] - h.values["10"]) > 1.0
          or abs(rev.values["10"] - h.values["1"]) > 1.0)

    return int(fails)
