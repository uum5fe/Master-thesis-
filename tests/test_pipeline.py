"""End-to-end: FAMOS files with imposed timing faults -> correct impedances."""

from __future__ import annotations

import numpy as np
import pytest

from eis.config import load_config
from eis.io.famos import FamosFormatError, discover_files, open_famos, parse_famos_header
from eis.pipeline import inventory_card, measure_sync, run_measurement
from tests.synthetic import simulate_measurement, write_famos


@pytest.fixture(scope="module")
def measurement(tmp_path_factory):
    """One synthetic measurement carrying every timing fault at once."""
    raw = tmp_path_factory.mktemp("raw")
    truth = simulate_measurement(
        raw, n_cards=4, segments_per_card=4, duration_s=40.0,
        card_delays_s={2: 1.7e-4, 3: -5.2e-4, 4: 3.1e-3},
        card_drift_ppm={4: 6.0},
        channel_delays_s={2: {"7": 50e-6}},
        dead_uc_cards=(3,),
    )
    return raw, truth


def make_config(raw, truth, tmp_path):
    cfg = load_config(
        measurement_id="SYNTH01", raw_dir=str(raw),
        output_dir=str(tmp_path), conditions=["150A"],
    )
    cfg.calibration.shunt_csv = str(raw / "curr.csv")
    cfg.calibration.temperature_csv = str(raw / "temp.csv")
    cfg.spectral.base_frequency_hz = 1.0
    cfg.spectral.excitation_tones_hz = list(truth.tones_hz)
    cfg.acquisition.channel_delay_table = {2: {"7": 50e-6}}
    cfg.acquisition.channel_delay_table_version = "test-known-truth"
    cfg.model.n_starts = 3
    return cfg


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def test_reader_recovers_header_and_metadata(measurement):
    raw, truth = measurement
    header = parse_famos_header(truth.files[1])
    assert header.fs == pytest.approx(10_000.0)
    assert header.n_channels == len(header.channel_names)
    assert header.start_time is not None                   # |NT was parsed
    assert "UC1" in header.channel_names


def test_reader_refuses_an_inconsistent_channel_count(tmp_path):
    """A wrong channel count de-interleaves every signal; it must not be guessed."""
    path = write_famos(
        tmp_path / "x.DAT", {"1": np.zeros(100), "2": np.zeros(100)}, 1e-4
    )
    data = path.read_bytes().replace(b"|CR,1,2,", b"|CR,1,3,", 1)
    path.write_bytes(data)
    with pytest.raises(FamosFormatError):
        parse_famos_header(path)


def test_discovery_groups_cards_by_condition(measurement):
    raw, truth = measurement
    cfg = load_config()
    found = discover_files(raw, cfg.acquisition.discovery_regex)
    assert set(found) == {"150A"}
    assert sorted(found["150A"]) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Synchronisation
# ---------------------------------------------------------------------------

def test_measured_skew_matches_the_imposed_skew(measurement, tmp_path):
    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    files = discover_files(raw, cfg.acquisition.discovery_regex)["150A"]
    inventories = {c: inventory_card(p, c) for c, p in files.items()}
    report, _ = measure_sync(inventories, cfg)

    reference = report.reference_card
    for card, sync in report.cards.items():
        if card == reference or not sync.healthy_uc:
            continue
        expected = (
            truth.card_delays_s.get(card, 0.0)
            - truth.card_delays_s.get(reference, 0.0)
        )
        assert abs(sync.tau_s - expected) < 5e-6, (
            f"card {card}: measured {sync.tau_s * 1e9:.0f} ns, "
            f"imposed {expected * 1e9:.0f} ns"
        )

    drifting = report.cards[4]
    assert abs(drifting.delay_slope_ppm - truth.card_drift_ppm[4]) < 0.5
    assert drifting.drift_corrected


def test_dead_cell_voltage_falls_back_to_a_segment_proxy(measurement, tmp_path):
    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    files = discover_files(raw, cfg.acquisition.discovery_regex)["150A"]
    report, _ = measure_sync({c: inventory_card(p, c) for c, p in files.items()}, cfg)

    proxied = report.cards[3]
    assert not proxied.healthy_uc
    assert proxied.method == "segment_proxy"
    assert proxied.ok
    # Coarse but honest: the claimed uncertainty must cover the actual error.
    assert abs(proxied.tau_s - truth.card_delays_s[3]) < 3 * proxied.tau_sigma_s


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_impedances_match_the_ground_truth(measurement, tmp_path):
    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    results = run_measurement(cfg, write=False, verbose=False)
    result = results["150A"]

    assert not result.rejected, result.rejected
    assert len(result.segments) == 16

    for segment, r in result.segments.items():
        Z_true = truth.Z_at(segment, r.spectrum.f)
        error = np.abs(r.spectrum.Z - Z_true) / np.abs(Z_true)
        # Cards timed from a proxy are held to a looser, stated bound.
        limit = 0.02 if r.card == 3 else 0.005
        assert np.median(error) < limit, (
            f"segment {segment} (card {r.card}): median error {np.median(error):.3%}"
        )


def test_intra_card_channel_skew_must_be_corrected(measurement, tmp_path):
    """The offset that does *not* cancel in the ratio.

    Card 2 (segments 5-8) has a 50 us offset on channel "7".  Without the table
    that is 68 degrees of phase error at 3.8 kHz.
    """
    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    cfg.acquisition.channel_delay_table = {}          # pretend we never measured it
    cfg.validation.run_lin_kk = False
    cfg.model.run_ecm = False
    result = run_measurement(cfg, write=False, verbose=False)["150A"]

    def hf_phase_error_deg(segment: int) -> float:
        """Phase error at the top tone, where a delay does its damage."""
        r = result.segments[segment]
        top = int(np.argmax(r.spectrum.f))
        measured = np.angle(r.spectrum.Z[top])
        expected = np.angle(truth.Z_at(segment, r.spectrum.f)[top])
        return float(abs(np.degrees(np.angle(np.exp(1j * (measured - expected))))))

    # 50 us at 3571 Hz is 64 degrees.
    assert hf_phase_error_deg(7) > 30.0, "the uncorrected 50 us skew should show up"
    assert hf_phase_error_deg(5) < 2.0, "an unaffected channel on the same card is fine"


def test_same_card_pairing_is_immune_to_bulk_card_skew(measurement, tmp_path):
    """Bulk offsets cancel in the ratio when segment and voltage share a card."""
    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    cfg.sync.uc_strategy = "same_card"
    cfg.validation.run_lin_kk = False
    cfg.model.run_ecm = False
    result = run_measurement(cfg, write=False, verbose=False)["150A"]

    r = result.segments[5]                            # card 2, offset by 170 us
    Z_true = truth.Z_at(5, r.spectrum.f)
    assert np.median(np.abs(r.spectrum.Z - Z_true) / np.abs(Z_true)) < 0.005


def test_validation_and_modelling_recover_the_truth(measurement, tmp_path):
    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    result = run_measurement(cfg, write=True, verbose=False)["150A"]

    passed = sum(1 for r in result.segments.values() if r.kk and r.kk.passed)
    assert passed >= 0.9 * len(result.segments)

    for segment, r in result.segments.items():
        if r.ecm is None or r.card == 3:
            continue
        true_rs = truth.segment_params[segment]["Rs"]
        assert abs(r.ecm.params["Rs"] - true_rs) / true_rs < 0.02

    # the whole-plate parallel-admittance identity
    assert result.plate is not None
    assert result.plate.n_segments == len(result.segments)

    files = list((tmp_path).glob("*"))
    assert any(f.name.startswith("impedance") for f in files)
    assert any(f.name.startswith("sync") for f in files)


def test_output_tables_carry_provenance(measurement, tmp_path):
    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    cfg.validation.run_lin_kk = False
    cfg.model.run_ecm = False
    result = run_measurement(cfg, write=False, verbose=False)["150A"]

    frame = result.impedance_frame(cfg.geometry.segment_area_cm2)
    for column in ("param_hash", "pipeline_version", "git_sha", "sigma_real_ohm"):
        assert column in frame.columns
    assert frame["param_hash"].nunique() == 1

    other = make_config(raw, truth, tmp_path)
    other.spectral.estimator = "h1"
    assert other.param_hash != cfg.param_hash, "parameters must change the hash"
