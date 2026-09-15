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
