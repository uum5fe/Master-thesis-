"""Synthetic segmented-cell measurement with known ground truth.

Generates FAMOS files that the real reader can parse, containing signals built
from impedances the test knows exactly.  This is what makes the accuracy claims
checkable: the pipeline is asked to recover numbers that were put in on
purpose, including deliberately imposed card skew and clock drift.

Physical model
--------------
The segments share one cell voltage and are electrically in parallel, so the
consistent way to build the data is to choose the voltage and derive each
segment's current::

    U(f)  = designed multisine
    I_k(f) = U(f) / Z_k(f)
    V_shunt,k(t) = I_k(t) * H_k / A_segment
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# FAMOS writer (matches the dialect the reader expects)
# ---------------------------------------------------------------------------

def write_famos(
    path: str | Path,
    channels: dict[str, np.ndarray],
    dt: float,
    start_time: datetime | None = None,
) -> Path:
    """Write an interleaved float32 FAMOS file with the given channels."""
    names = list(channels)
    n_ch = len(names)
    n = min(len(v) for v in channels.values())
    block = np.empty((n, n_ch), dtype="<f4")
    for i, name in enumerate(names):
        block[:, i] = np.asarray(channels[name][:n], dtype="<f4")
    payload = block.tobytes()

    parts = ["|CF,2,1,1;", "|CK,1,1,1,1;"]
    if start_time is not None:
        parts.append(
            f"|NT,1,{start_time.day},{start_time.month},{start_time.year},"
            f"{start_time.hour},{start_time.minute},"
            f"{start_time.second + start_time.microsecond / 1e6:.6f};"
        )
    parts.append(f"|CD,4,{dt:.10g},1,s,0,0,0;")
    parts.append(f"|CR,1,{n_ch},1,1,0,V;")
    cp = ",".join(f"7,32,{name}" for name in names)
    parts.append(f"|CP,3,10,64,{n_ch},{cp};")
    header = "".join(parts) + f"|CS,1,{len(payload)},"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header.encode("latin-1"))
        fh.write(payload)
    return path


# ---------------------------------------------------------------------------
# Ground-truth impedance
# ---------------------------------------------------------------------------

def true_impedance(
    f: np.ndarray, Rs: float, L: float, Rct1: float, tau1: float,
    Rct2: float, tau2: float, n1: float = 0.9, n2: float = 0.85,
) -> np.ndarray:
    """Rs + jwL + two depressed arcs - the reference the test fits back."""
    w = 2 * np.pi * np.asarray(f, float)
    Y01 = tau1 ** n1 / Rct1
    Y02 = tau2 ** n2 / Rct2
    z = Rs + 1j * w * L
    z = z + Rct1 / (1.0 + (1j * w) ** n1 * Y01 * Rct1)
    z = z + Rct2 / (1.0 + (1j * w) ** n2 * Y02 * Rct2)
    return z


@dataclass
class SyntheticTruth:
    """Everything the generator knows, for the test to compare against."""

    tones_hz: np.ndarray
    base_frequency_hz: float
    fs: float
    duration_s: float
    segment_params: dict[int, dict[str, float]] = field(default_factory=dict)
    Z_true: dict[int, np.ndarray] = field(default_factory=dict)
    card_of_segment: dict[int, int] = field(default_factory=dict)
    card_delays_s: dict[int, float] = field(default_factory=dict)
    card_drift_ppm: dict[int, float] = field(default_factory=dict)
    channel_delays_s: dict[int, dict[str, float]] = field(default_factory=dict)
    shunt_c0: np.ndarray = field(default_factory=lambda: np.array([]))
    shunt_c1: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_c: float = 60.0
    files: dict[int, str] = field(default_factory=dict)

    def Z_at(self, segment: int, f: np.ndarray) -> np.ndarray:
        p = self.segment_params[segment]
        return true_impedance(f, **p)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _sample_at(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """High-accuracy resampling used only to *impose* known delays."""
    from eis.sync.resample import resample_at

    return resample_at(x, positions)


def simulate_measurement(
    out_dir: str | Path,
    measurement_id: str = "SYNTH01",
    condition: str = "150A",
    n_cards: int = 5,
    segments_per_card: int = 16,
    fs: float = 10_000.0,
    duration_s: float = 40.0,
    base_frequency_hz: float = 1.0,
    tones_hz: list[float] | None = None,
    card_delays_s: dict[int, float] | None = None,
    card_drift_ppm: dict[int, float] | None = None,
    channel_delays_s: dict[int, dict[str, float]] | None = None,
    noise_v: float = 5e-7,
    voltage_amplitude_v: float = 4e-3,
    temperature_c: float = 62.0,
    dead_uc_cards: tuple[int, ...] = (),
    seed: int = 7,
) -> SyntheticTruth:
    """Create a full synthetic measurement on disk and return its ground truth.

    ``card_delays_s`` imposes a bulk timing offset on a whole card (as a start
    trigger difference would), ``card_drift_ppm`` a clock-rate error, and
    ``channel_delays_s`` a per-channel offset *within* a card - the one that
    does not cancel in the impedance ratio.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    n = int(round(fs * duration_s))
    t = np.arange(n) / fs

    tones = np.asarray(
        tones_hz if tones_hz is not None
        else [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
              987, 1597, 2584, 3571],
        dtype=float,
    )

    # ---- cell voltage: an odd-ish multisine with random phases -----------
    phases = rng.uniform(0, 2 * np.pi, len(tones))
    u_ac = np.zeros(n)
    for f0, ph in zip(tones, phases):
        u_ac += np.sin(2 * np.pi * f0 * t + ph)
    u_ac *= voltage_amplitude_v / np.sqrt(np.mean(u_ac**2))
    u_dc = 0.65
    U_spec = np.fft.rfft(u_ac)
    freqs = np.fft.rfftfreq(n, 1 / fs)

    # ---- per-segment truth ------------------------------------------------
    n_segments = n_cards * segments_per_card
    truth = SyntheticTruth(
        tones_hz=tones, base_frequency_hz=base_frequency_hz, fs=fs,
        duration_s=duration_s, temperature_c=temperature_c,
    )
    area = 4.235
    shunt_c0 = np.zeros(n_segments)
    shunt_c1 = np.zeros(n_segments)

    currents: dict[int, np.ndarray] = {}
    for seg in range(1, n_segments + 1):
        # Smooth spatial variation plus a little scatter.
        u_pos = (seg - 1) / max(n_segments - 1, 1)
        params = {
            "Rs": 0.0140 * (1.0 + 0.30 * u_pos) * (1 + 0.01 * rng.standard_normal()),
            "L": 8e-9,
            "Rct1": 0.0060 * (1.0 + 0.20 * np.sin(3 * np.pi * u_pos)),
            "tau1": 1.2e-4,
            "Rct2": 0.0180 * (1.0 + 0.45 * u_pos**2),
            "tau2": 2.5e-3,
            "n1": 0.92, "n2": 0.86,
        }
        truth.segment_params[seg] = params
        truth.Z_true[seg] = true_impedance(tones, **params)

        Z_all = true_impedance(np.maximum(freqs, 1e-6), **params)
        I_spec = np.where(freqs > 0, U_spec / Z_all, 0.0)
        current = np.fft.irfft(I_spec, n)                       # A, AC only
        currents[seg] = current + 150.0 / n_segments            # + DC operating point

        shunt_c0[seg - 1] = 1.05e-4 * (1 + 0.02 * rng.standard_normal())
        shunt_c1[seg - 1] = 3.9e-4 * shunt_c0[seg - 1] * 1e3     # copper TCR
    truth.shunt_c0, truth.shunt_c1 = shunt_c0, shunt_c1

    # ---- assemble each card ----------------------------------------------
    card_delays_s = card_delays_s or {}
    card_drift_ppm = card_drift_ppm or {}
    channel_delays_s = channel_delays_s or {}
    truth.card_delays_s = dict(card_delays_s)
    truth.card_drift_ppm = dict(card_drift_ppm)
    truth.channel_delays_s = {k: dict(v) for k, v in channel_delays_s.items()}
    t0 = datetime(2026, 3, 14, 9, 15, 0)

    for card in range(1, n_cards + 1):
        tau_card = float(card_delays_s.get(card, 0.0))
        drift = float(card_drift_ppm.get(card, 0.0)) * 1e-6
        channels: dict[str, np.ndarray] = {}

        def delayed(signal: np.ndarray, extra: float = 0.0) -> np.ndarray:
            """Impose tau(t) = tau_card + drift*t + extra on a channel."""
            if abs(tau_card + extra) < 1e-15 and abs(drift) < 1e-15:
                return signal.copy()
            positions = np.arange(n) * (1.0 - drift) - (tau_card + extra) * fs
            return _sample_at(signal, positions)

        first = (card - 1) * segments_per_card + 1
        for seg in range(first, first + segments_per_card):
            truth.card_of_segment[seg] = card
            name = str(seg)
            H = shunt_c0[seg - 1] + 1e-3 * shunt_c1[seg - 1] * temperature_c
            v_shunt = currents[seg] * H / area                    # V
            extra = float(channel_delays_s.get(card, {}).get(name, 0.0))
            channels[name] = delayed(v_shunt, extra) + noise_v * rng.standard_normal(n)

        uc_name = f"UC{card}"
        if card in dead_uc_cards:
            channels[uc_name] = 1e-5 * rng.standard_normal(n)     # dead channel
        else:
            extra = float(channel_delays_s.get(card, {}).get(uc_name, 0.0))
            channels[uc_name] = (
                delayed(u_ac + u_dc, extra) + noise_v * rng.standard_normal(n)
            )

        # Temperature sensor: T = (V - c0)/c1 with c0 = 0, c1 = 0.01 V/K
        channels[f"Temp_{card}"] = (
            temperature_c * 0.01 + 1e-5 * rng.standard_normal(n)
        )

        path = out_dir / (
            f"Leepa_{measurement_id}_Current_{condition}_Test_01_Karte_{card}.DAT"
        )
        write_famos(path, channels, 1.0 / fs, t0 + timedelta(milliseconds=card - 1))
        truth.files[card] = str(path)

    # calibration files in the format the pipeline expects
    np.savetxt(
        out_dir / "curr.csv",
        np.column_stack([shunt_c0, shunt_c1]), delimiter=";", fmt="%.10g",
    )
    np.savetxt(
        out_dir / "temp.csv",
        np.column_stack([np.zeros(n_cards), np.full(n_cards, 0.01)]),
        delimiter=";", fmt="%.10g",
    )
    return truth
