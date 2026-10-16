"""Fixed heat-map colour scales (config.HEATMAP_LIMITS)."""
import config


def test_every_resistance_map_has_a_fixed_scale():
    for p in ("R_ohmic", "R_ct", "R_mt", "R_pol"):
        lo, hi = config.heatmap_limits(p)
        assert lo < hi


def test_ecm_names_share_the_pipeline_scale():
    assert config.heatmap_limits("Rs (ECM)") == config.heatmap_limits("R_ohmic")
    assert config.heatmap_limits("R_pol (ECM)") == config.heatmap_limits("R_pol")


def test_switching_fixed_scaling_off_restores_automatic(monkeypatch):
    monkeypatch.setattr(config, "HEATMAP_FIXED_SCALE", False)
    assert config.heatmap_limits("R_ct") is None
    assert config.heatmap_limits("j_dc") is None
