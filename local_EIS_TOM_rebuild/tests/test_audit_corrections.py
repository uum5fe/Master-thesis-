"""The corrections that came out of the raw-signal audit of 2612030 450 A:
a coherent tone at a step frequency before the sweep, a channel fault seen
in the DC current, a window at the ADC full scale, and each card's UC2
magnitude against the cleanest card above the gain band."""

from __future__ import annotations

import numpy as np
import pytest

import card_gain
import silver
import utils
from config import DEFAULT

FS = 25000.0


def test_a_mains_tone_at_a_step_frequency_is_seen_before_the_sweep():
    t = np.arange(int(40 * FS)) / FS
    rng = np.random.default_rng(3)
    x = 1e-3 * rng.standard_normal(t.size) + 1e-4 * np.sin(2 * np.pi * 150 * t)
    coh, noise = utils.background_at(x, FS, 150.24, 12500)
    assert coh == pytest.approx(1e-4, rel=0.15)        # the tone
    coh2, _ = utils.background_at(x, FS, 119.17, 12500)
    assert coh2 < 3e-5                                  # nothing there
    assert noise < 3e-5                                 # broadband, separate


def test_too_short_a_quiet_stretch_gives_no_verdict():
    x = np.zeros(int(0.5 * FS))
    assert np.isnan(utils.background_at(x, FS, 0.19, int(20 * FS))[0])


class _Sp:
    def __init__(self, j):
        self._j = j

    def j_dc(self):
        return self._j


def test_a_segment_far_below_its_neighbours_is_a_channel_fault():
    import neighbours
    adj = neighbours.adjacency()
    spectra = {s: _Sp(1.40) for s in adj}
    spectra["33"] = _Sp(0.255)                         # 18 % of neighbours
    out = silver.dc_share_faults(spectra, DEFAULT)
    assert set(out) == {"33"} and "18 %" in out["33"]
    assert silver.dc_share_faults(spectra, DEFAULT.replace(
        segment_fault_ratio=0.0)) == {}


def test_uc_magnitude_is_equalised_to_the_reference_card_above_the_band():
    f = np.array([1, 3, 10, 30, 100, 300, 1000, 2400, 3800, 9500.0])
    pick = np.where(f > 300, 1 + 0.3 * f / 9500, 1.0)  # card 4's pickup
    A = {"Karte_1": np.ones(f.size, complex),
         "Karte_4": 1.012 * pick.astype(complex)}      # + a static gain
    snr = {c: np.full(f.size, 45.0) for c in A}
    ref, h = card_gain.hf_response(f, A, snr, DEFAULT, ref_card="Karte_1")
    assert ref == "Karte_1"
    assert np.all(h["Karte_1"] == 1.0)
    assert np.allclose(h["Karte_4"][f <= 300], 1.0)    # static: card gain
    hi = f > 300
    assert np.allclose(h["Karte_4"][hi][1:-1], pick[hi][1:-1], rtol=0.03)


def test_clipped_windows_are_dropped(tmp_path):
    """A window touching the ADC full scale is not fitted."""
    import make_synth_famos as M
    import bronze
    import eis_local
    from pathlib import Path
    d = M.main(str(tmp_path / "raw")) or tmp_path / "raw"
    files = sorted(Path(tmp_path / "raw").glob("*.DAT"))
    assert files
    fam = eis_local.FamosFile(files[0])
    seg = fam.segment_names[0]
    lim = float(np.max(np.abs(fam.channel(seg)))) * 0.999
    cfg = DEFAULT.replace(dat_dir=tmp_path / "raw", out_dir=tmp_path / "o",
                          curr_cal=Path("/tmp/curr.csv"),
                          temp_cal=Path("/tmp/temp.csv"), ref_channel="UC1",
                          verbose=False, adc_full_scale_v=lim)
    run = bronze.run(cfg)
    sp = run.spectra.get(seg)
    assert sp is not None and sp.n_clipped >= 1
