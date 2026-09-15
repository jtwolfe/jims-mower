"""YAML config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jims_mower.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    EnvConfig,
    load_config,
    validate_config,
)


def test_default_yaml_exists() -> None:
    assert DEFAULT_CONFIG_PATH.is_file() or isinstance(load_config(), EnvConfig)


def test_default_safe_state_knobs() -> None:
    cfg = load_config()
    assert 0.0 < cfg.planner.safe_state.limp_scale <= 1.0
    assert cfg.planner.safe_state.limp_after_stops >= 1


def test_rejects_bad_limp_scale() -> None:
    with pytest.raises(ConfigError):
        load_config({"planner": {"safe_state": {"limp_scale": 0.0}}})


def test_load_default_has_six_cameras() -> None:
    cfg = load_config()
    cams = cfg.resolved_cameras()
    assert len(cams) == 6
    assert cfg.robot.length_m == pytest.approx(0.5)
    assert cfg.robot.width_m == pytest.approx(0.5)
    assert cfg.robot.height_m == pytest.approx(0.5)


def test_camera_count_presets() -> None:
    for n in (4, 5, 6):
        cfg = load_config({"sensors": {"camera_count": n}})
        assert len(cfg.resolved_cameras()) == n


def test_explicit_camera_list() -> None:
    cams = [
        {
            "name": f"c{i}",
            "x": 0.1 * i,
            "y": 0.0,
            "z": 0.3,
            "yaw_deg": 0.0,
            "pitch_deg": -10.0,
        }
        for i in range(4)
    ]
    cfg = load_config({"sensors": {"cameras": cams, "camera_count": 4}})
    assert [c.name for c in cfg.resolved_cameras()] == ["c0", "c1", "c2", "c3"]


def test_rejects_three_cameras() -> None:
    with pytest.raises(ConfigError):
        load_config({"sensors": {"camera_count": 3}})


def test_rejects_seven_cameras() -> None:
    with pytest.raises(ConfigError):
        load_config({"sensors": {"camera_count": 7}})


def test_rejects_bad_dt() -> None:
    with pytest.raises(ConfigError):
        load_config({"dt": 0.0})


def test_rejects_unknown_key() -> None:
    with pytest.raises(ConfigError):
        load_config({"not_a_field": 1})


def test_rejects_missing_file() -> None:
    with pytest.raises(ConfigError):
        load_config("/tmp/does-not-exist-jims-mower.yaml")


def test_load_from_path(tmp_path: Path) -> None:
    payload = {"dt": 0.05, "sensors": {"camera_count": 4}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.dt == pytest.approx(0.05)
    assert len(cfg.resolved_cameras()) == 4


def test_duplicate_camera_names() -> None:
    cams = [
        {"name": "front", "x": 0.2, "y": 0.0, "z": 0.3, "yaw_deg": 0.0, "pitch_deg": 0.0}
        for _ in range(4)
    ]
    with pytest.raises(ConfigError):
        load_config({"sensors": {"cameras": cams}})


def test_validate_body_positive() -> None:
    cfg = EnvConfig()
    cfg.robot.length_m = 0.0
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_tiny_resolution_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config({"sensors": {"width": 4, "height": 4}})


def test_default_has_terrain_and_imu() -> None:
    cfg = load_config()
    assert cfg.world.terrain.enabled is True
    assert cfg.world.terrain.n_drains >= 1
    assert cfg.sensors.imu.enabled is True
    assert cfg.sensors.gps.enabled is True
    assert cfg.perception.terrain_mode == "heuristic"
    assert cfg.world.terrain.base_gradient.effective_slope_rad() > 0.0
    assert cfg.world.terrain.base_gradient.undulation_m > 0.0


def test_steep_yard_has_stronger_gradient() -> None:
    root = Path(__file__).resolve().parents[1]
    default = load_config()
    steep = load_config(root / "configs" / "steep_yard.yaml")
    assert steep.world.terrain.base_gradient.effective_slope_rad() > (
        default.world.terrain.base_gradient.effective_slope_rad() + 0.02
    )


def test_rejects_negative_base_gradient() -> None:
    with pytest.raises(ConfigError):
        load_config({"world": {"terrain": {"base_gradient": {"slope_rad": -0.1}}}})
    with pytest.raises(ConfigError):
        load_config({"world": {"terrain": {"base_gradient": {"undulation_m": -0.01}}}})


def test_slope_pct_overrides_slope_rad() -> None:
    cfg = load_config({"world": {"terrain": {"base_gradient": {"slope_pct": 10.0}}}})
    import math

    assert cfg.world.terrain.base_gradient.effective_slope_rad() == pytest.approx(
        math.atan(0.10), abs=1e-9
    )


def test_rejects_bad_terrain_mode() -> None:
    with pytest.raises(ConfigError):
        load_config({"perception": {"terrain_mode": "slam"}})


def test_accepts_learned_terrain_mode() -> None:
    cfg = load_config({"perception": {"terrain_mode": "learned", "weights_path": "w.npz"}})
    assert cfg.perception.terrain_mode == "learned"
    assert cfg.perception.weights_path == "w.npz"


def test_rejects_bad_gps_dropout() -> None:
    with pytest.raises(ConfigError):
        load_config({"sensors": {"gps": {"dropout_prob": 1.5}}})


def test_suburban_inherits_mild_gradient() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "scenarios" / "suburban.yaml")
    assert cfg.world.terrain.base_gradient.effective_slope_rad() > 0.03


def test_rejects_bad_drain_width() -> None:
    with pytest.raises(ConfigError):
        load_config({"world": {"terrain": {"drain_width_m": 0.0}}})


def test_default_has_planner_knobs() -> None:
    cfg = load_config()
    assert cfg.planner.max_climb_slope_rad > 0
    assert cfg.planner.drain_clearance_m >= 0
    assert 0.0 < cfg.planner.slow_speed_factor <= 1.0
    assert cfg.planner.pose_filter == "ekf"
    assert cfg.planner.ekf.r_gps_xy > 0
    assert cfg.planner.uncertainty.inflate >= 0


def test_rejects_bad_slow_speed_factor() -> None:
    with pytest.raises(ConfigError):
        load_config({"planner": {"slow_speed_factor": 0.0}})
    with pytest.raises(ConfigError):
        load_config({"planner": {"slow_speed_factor": 1.5}})


def test_rejects_bad_max_climb() -> None:
    with pytest.raises(ConfigError):
        load_config({"planner": {"max_climb_slope_rad": 0.0}})


def test_rejects_bad_world_layout() -> None:
    with pytest.raises(ConfigError):
        load_config({"world": {"layout": "moon_base"}})


def test_rejects_bad_weather_pack() -> None:
    with pytest.raises(ConfigError):
        load_config({"weather": {"pack": "hailstorm"}})


def test_rejects_bad_grass_frac() -> None:
    with pytest.raises(ConfigError):
        load_config({"world": {"grass": {"regenerate_frac": 1.5}}})


def test_wave1c_defaults() -> None:
    cfg = load_config()
    assert cfg.world.layout == "random"
    assert cfg.world.n_hoses == 0
    assert cfg.weather.pack == "clear"
    assert cfg.domain_randomization.enabled is False
    assert cfg.world.grass.enabled is False


def test_rejects_bad_pose_filter() -> None:
    with pytest.raises(ConfigError):
        load_config({"planner": {"pose_filter": "ukf-paper"}})


def test_tof_count_presets() -> None:
    for n in (0, 2, 4):
        cfg = load_config({"sensors": {"tof": {"count": n}}})
        assert cfg.sensors.tof.count == n


def test_rejects_bad_tof_count() -> None:
    with pytest.raises(ConfigError):
        load_config({"sensors": {"tof": {"count": 3}}})


def test_faults_and_radio_defaults_off() -> None:
    cfg = load_config()
    assert cfg.faults.enabled is False
    assert cfg.radio.enabled is False
    assert cfg.radio.on_loss == "stop_beacon"
    assert cfg.radio.wifi.range_m > 0


def test_rejects_bad_radio_on_loss() -> None:
    with pytest.raises(ConfigError):
        load_config({"radio": {"on_loss": "shout"}})


def test_runtime_budget_defaults_off() -> None:
    cfg = load_config()
    assert cfg.runtime.enabled is False
    assert cfg.sensors.tof.count == 4
    assert 0.0 <= cfg.runtime.battery.limp_soc <= 1.0


def test_rejects_bad_battery_soc() -> None:
    with pytest.raises(ConfigError):
        load_config({"runtime": {"battery": {"soc": 1.5}}})


def test_rejects_bad_uncertainty_floor() -> None:
    with pytest.raises(ConfigError):
        load_config({"planner": {"uncertainty": {"confidence_floor": 1.5}}})


def test_planner_yaml_override() -> None:
    cfg = load_config({"planner": {"drain_clearance_m": 0.55, "cruise_speed": 0.4}})
    assert cfg.planner.drain_clearance_m == pytest.approx(0.55)
    assert cfg.planner.cruise_speed == pytest.approx(0.4)
