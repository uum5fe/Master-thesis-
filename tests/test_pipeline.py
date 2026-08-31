"""End-to-end: FAMOS files with imposed faults -> correct impedances."""

from __future__ import annotations

import numpy as np
import pytest

from eis.io.famos import FamosFormatError, discover_files, parse_famos_header
from eis.pipeline import inventory_card, measure_sync, run_measurement
from eis.pipeline.config import load_config
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


# ---------------------------------------------------------------------------
# High frequency: the correction that timing alone cannot make
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def inductive_shunt(tmp_path_factory):
    """A clean measurement whose only fault is a real via-shunt inductance."""
    raw = tmp_path_factory.mktemp("shunt")
    truth = simulate_measurement(
        raw, n_cards=2, segments_per_card=4, duration_s=20.0,
        shunt_inductance_nh=0.6,
    )
    return raw, truth


def _run_with_shunt_inductance(raw, truth, tmp_path, nh):
    cfg = make_config(raw, truth, tmp_path)
    cfg.hf.shunt_inductance_nh = nh
    cfg.validation.run_stationarity = False
    cfg.model.run_ecm = False
    return run_measurement(cfg, write=False, verbose=False)["150A"]


def _error_at_the_top_tone(result, truth) -> float:
    errors = []
    for segment, r in result.segments.items():
        top = int(np.argmax(r.spectrum.f))
        Z_true = truth.Z_at(segment, r.spectrum.f)[top]
        errors.append(abs(r.spectrum.Z[top] - Z_true) / abs(Z_true))
    return float(np.median(errors))


def test_shunt_inductance_wrecks_the_high_frequency_end_if_uncorrected(
    inductive_shunt, tmp_path
):
    """A 0.6 nH via shunt is 25 degrees at 4 kHz and no delay correction sees it.

    The shunt is tens of microohms, so its own L/R time constant is tens of
    microseconds - three orders of magnitude above the nanosecond timing
    budget the synchronisation stage meets.  Treating H(T) as a real number
    therefore costs more accuracy at the top of the band than every timing
    error in the pipeline put together.
    """
    raw, truth = inductive_shunt
    uncorrected = _run_with_shunt_inductance(raw, truth, tmp_path, nh=0.0)
    corrected = _run_with_shunt_inductance(raw, truth, tmp_path, nh=0.6)

    assert _error_at_the_top_tone(uncorrected, truth) > 0.30
    assert _error_at_the_top_tone(corrected, truth) < 0.01

    # And the sync stage is blameless: it met its budget in both runs.
    for card, sync in corrected.sync.cards.items():
        if sync.residual_tau_s is not None:
            assert abs(sync.residual_tau_s) < 100e-9


def test_the_applied_instrument_response_is_recorded_in_the_output(
    inductive_shunt, tmp_path
):
    raw, truth = inductive_shunt
    result = _run_with_shunt_inductance(raw, truth, tmp_path, nh=0.6)
    frame = result.segment_frame(4.235)

    assert (frame["instrument_terms"].str.contains("shunt_tau")).all()
    assert (frame["instrument_phase_deg_at_fmax"].abs() > 10.0).all()
    assert "hf_crossover_hz" in frame.columns


# ---------------------------------------------------------------------------
# Multi-resolution analysis
# ---------------------------------------------------------------------------

def test_multiresolution_buys_averages_at_the_top_of_the_band(
    measurement, tmp_path
):
    """One window length cannot serve four decades.

    A window long enough for eight periods of 1 Hz throws away almost all the
    averaging a 40 s record offers at 3 kHz.  The multi-resolution plan spends
    a short window there instead, and the uncertainty falls accordingly.
    """
    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    cfg.spectral.base_frequency_hz = None      # force the Welch family
    cfg.spectral.excitation_tones_hz = None
    cfg.validation.run_lin_kk = False
    cfg.validation.run_stationarity = False
    cfg.model.run_ecm = False

    cfg.spectral.method = "welch"
    fixed = run_measurement(cfg, write=False, verbose=False)["150A"]
    cfg.spectral.method = "multiresolution"
    multi = run_measurement(cfg, write=False, verbose=False)["150A"]

    segment = sorted(fixed.segments)[0]
    a, b = fixed.segments[segment].spectrum_all, multi.segments[segment].spectrum_all

    def top_decade(spectrum):
        m = spectrum.f >= spectrum.f.max() / 10.0
        return m

    assert b.n_eff_f is not None
    n_fixed = float(np.median(a.n_eff_f[top_decade(a)]))
    n_multi = float(np.median(b.n_eff_f[top_decade(b)]))
    assert n_multi > 4 * n_fixed, (
        f"expected many more averages at the top of the band, got "
        f"{n_multi:.0f} against {n_fixed:.0f}"
    )
    assert float(np.median(b.sigma_rel[top_decade(b)])) < float(
        np.median(a.sigma_rel[top_decade(a)])
    )
    # Window length has to fall with frequency, which is the mechanism.
    assert b.nperseg_f[0] > b.nperseg_f[-1]


# ---------------------------------------------------------------------------
# Every segment stays on the plate
# ---------------------------------------------------------------------------

def test_a_segment_without_calibration_is_classified_not_deleted(
    measurement, tmp_path
):
    """Quality decisions belong in the data, not in a list of missing numbers."""
    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    cfg.validation.run_lin_kk = False
    cfg.validation.run_stationarity = False
    cfg.model.run_ecm = False
    # Truncate the shunt file so the last four segments have no calibration.
    cfg.calibration.shunt_csv = str(tmp_path / "short_curr.csv")
    rows = np.column_stack([truth.shunt_c0[:12], truth.shunt_c1[:12]])
    np.savetxt(cfg.calibration.shunt_csv, rows, delimiter=";", fmt="%.10g")

    result = run_measurement(cfg, write=False, verbose=False)["150A"]

    assert len(result.segments) == 16, "no segment may vanish"
    uncalibrated = [s for s in range(13, 17)]
    for segment in uncalibrated:
        record = result.segments[segment]
        assert record.status == "no_calibration"
        assert not record.physical_units
        assert record.active, "it still has a spectrum, just not in ohms"
        assert len(record.spectrum.f) > 0
    for segment in range(1, 13):
        assert result.segments[segment].physical_units

    frame = result.segment_frame(4.235)
    assert set(frame["segment"]) == set(range(1, 17))
    assert frame["status"].eq("no_calibration").sum() == 4


def test_low_coherence_points_are_marked_and_kept_on_a_shared_grid(
    measurement, tmp_path
):
    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    cfg.spectral.min_coherence = 0.999999    # gate almost everything out
    cfg.validation.run_lin_kk = False
    cfg.validation.run_stationarity = False
    cfg.model.run_ecm = False
    result = run_measurement(cfg, write=False, verbose=False)["150A"]

    grids = [r.spectrum_all.f for r in result.segments.values()]
    for grid in grids[1:]:
        assert np.array_equal(grid, grids[0]), (
            "every segment must keep every frequency it measured, or a "
            "frequency-resolved plate map has holes in it"
        )
    total = sum(len(r.spectrum_all.f) for r in result.segments.values())
    used = sum(r.spectrum_all.n_used for r in result.segments.values())
    assert used < total, "the gate has to have rejected something"

    frame = result.impedance_frame(4.235)
    assert len(frame) == total, "rejected points are still rows"
    assert not frame["used"].all()


# ---------------------------------------------------------------------------
# The interactive plate view
# ---------------------------------------------------------------------------

def test_dashboard_is_self_contained_and_carries_every_segment(
    measurement, tmp_path
):
    from eis.dashboard import write_dashboard

    raw, truth = measurement
    cfg = make_config(raw, truth, tmp_path)
    cfg.validation.run_stationarity = False
    cfg.model.n_starts = 2
    results = run_measurement(cfg, write=False, verbose=False)

    path = write_dashboard(results, cfg, tmp_path / "plate.html")
    html = path.read_text(encoding="utf-8")

    assert "<script src=" not in html and "<link rel=\"stylesheet\"" not in html, (
        "the page has to open with no network and no libraries"
    )
    for segment in results["150A"].segments:
        assert f'"{segment}":' in html
    for key in ("z_mod@f", "phase@f"):
        assert key in html, "the frequency slider needs its map keys"
    assert "prefers-color-scheme" in html
