"""Gymnasium env contract, cameras, and safety-in-the-loop."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import jims_mower  # noqa: F401  registers the env id
from jims_mower.env import MowerEnv, encode_detections
from jims_mower.perception import BlindDetector
from jims_mower.types import Detection

from tests.conftest import tiny_config_dict


def _env(**kwargs) -> MowerEnv:
    cfg = tiny_config_dict()
    return MowerEnv(config=cfg, **kwargs)


def test_reset_returns_obs_and_info() -> None:
    env = _env()
    obs, info = env.reset(seed=0)
    assert set(obs) >= {
        "cameras",
        "coverage",
        "occupancy",
        "detections",
        "pose",
        "trimmer_enabled",
        "hand_signal",
    }
    assert "coverage_fraction" in info
    assert "detections" in info
    env.close()


def test_step_five_tuple() -> None:
    env = _env()
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(np.array([0.2, 0.2, 0.0]))
    assert obs["pose"].shape == (3,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "newly_cut" in info
    env.close()


def test_action_must_be_length_three() -> None:
    env = _env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.array([0.1, 0.1]))
    env.close()


def test_default_four_cameras_in_tiny_cfg() -> None:
    env = _env()
    obs, _ = env.reset(seed=2)
    assert set(obs["cameras"]) == {"front", "rear", "left", "right"}
    for frame in obs["cameras"].values():
        assert frame.shape == (24, 32, 3)
        assert frame.dtype == np.uint8
    env.close()


def test_six_camera_default_config() -> None:
    env = MowerEnv(
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 6},
            "world": {
                "width_m": 8.0,
                "height_m": 8.0,
                "resolution_m": 0.25,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 1,
                "n_furniture": 0,
                "n_toys": 0,
            },
        }
    )
    obs, _ = env.reset(seed=3)
    assert len(obs["cameras"]) == 6
    env.close()


def test_registered_gym_id() -> None:
    import gymnasium as gym

    env = gym.make("jims_mower/Mower-v0", config=tiny_config_dict())
    obs, _ = env.reset(seed=4)
    assert "cameras" in obs
    env.close()


def test_check_env() -> None:
    env = _env(render_mode="rgb_array")
    check_env(env, skip_render_check=False)
    env.close()


def test_seed_reproducible_reset() -> None:
    env = _env()
    obs_a, _ = env.reset(seed=11)
    env.close()
    env = _env()
    obs_b, _ = env.reset(seed=11)
    assert np.allclose(obs_a["pose"], obs_b["pose"])
    assert np.array_equal(obs_a["coverage"], obs_b["coverage"])
    env.close()


def test_trimmer_cuts_grass_when_safe() -> None:
    env = MowerEnv(
        config={
            "max_steps": 20,
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "world": {
                "width_m": 8.0,
                "height_m": 8.0,
                "resolution_m": 0.2,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
            },
        }
    )
    _, info0 = env.reset(seed=5)
    assert info0["coverage_fraction"] == pytest.approx(0.0)
    cut = 0
    for _ in range(8):
        _, _, _, _, info = env.step(np.array([0.6, 0.6, 1.0], dtype=np.float32))
        cut += int(info["newly_cut"])
        assert info["trimmer_enabled"] is True
    assert cut > 0
    env.close()


def test_trimmer_blocked_near_person() -> None:
    env = _env()
    env.reset(seed=6)
    # Drop a person on the trimmer hub.
    if env._yard.obstacles:
        env._yard.obstacles[0].kind = "person"
        env._yard.obstacles[0].x = env._pose.x + 0.32
        env._yard.obstacles[0].y = env._pose.y
        env._yard.obstacles[0].z = 0.9
    else:
        from jims_mower.types import Obstacle

        env._yard.obstacles.append(
            Obstacle("person", env._pose.x + 0.32, env._pose.y, 0.25, z=0.9)
        )
    _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert info["trimmer_requested"] is True
    assert info["trimmer_enabled"] is False
    env.close()


def test_zero_turn_in_env() -> None:
    env = _env()
    env.reset(seed=7)
    x0, y0, th0 = env._pose.x, env._pose.y, env._pose.theta
    env.step(np.array([-0.8, 0.8, 0.0], dtype=np.float32))
    assert env._pose.x == pytest.approx(x0, abs=1e-6)
    assert env._pose.y == pytest.approx(y0, abs=1e-6)
    assert env._pose.theta != pytest.approx(th0)
    env.close()


def test_blind_detector_swappable() -> None:
    env = _env(detector=BlindDetector())
    obs, info = env.reset(seed=8)
    assert int((obs["detections"][:, 0] > 0).sum()) == 0
    assert info["detections"] == []
    env.close()


def test_hand_signal_obs_when_enabled() -> None:
    env = _env(hand_signals=True)
    obs, info = env.reset(seed=9)
    assert info["hand_signals_enabled"] is True
    assert 0 <= int(obs["hand_signal"]) <= 4
    env.close()


def test_hand_signal_zero_when_disabled() -> None:
    env = _env(hand_signals=False)
    obs, info = env.reset(seed=9)
    assert info["hand_signals_enabled"] is False
    assert int(obs["hand_signal"]) == 0
    env.close()


def test_render_rgb_array() -> None:
    env = _env(render_mode="rgb_array")
    env.reset(seed=10)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 3
    env.close()


def test_truncates_at_max_steps() -> None:
    env = MowerEnv(
        config={
            "max_steps": 3,
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "world": {
                "width_m": 8.0,
                "height_m": 8.0,
                "resolution_m": 0.25,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
            },
        }
    )
    env.reset(seed=12)
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
        if terminated:
            break
    assert truncated is True
    env.close()


def test_encode_detections_padding() -> None:
    dets = [
        Detection("person", "front", (2, 3, 4, 5), 0.8, hand_signal="go"),
    ]
    buf = encode_detections(dets, {"front": 0})
    assert buf.shape[1] == 8
    assert buf[0, 0] == pytest.approx(1.0)
    assert buf[0, 7] == pytest.approx(2.0)  # go
    assert buf[1, 0] == pytest.approx(0.0)


def test_out_of_bounds_terminates() -> None:
    env = MowerEnv(
        config={
            "sensors": {"width": 16, "height": 12, "camera_count": 4},
            "world": {
                "width_m": 6.0,
                "height_m": 6.0,
                "resolution_m": 0.25,
                "n_people": 0,
                "n_dogs": 0,
                "n_cats": 0,
                "n_birds": 0,
                "n_trees": 0,
                "n_furniture": 0,
                "n_toys": 0,
            },
            "robot": {"max_wheel_speed_mps": 2.0},
        }
    )
    env.reset(seed=13)
    terminated = False
    for _ in range(40):
        _, _, terminated, _, info = env.step(
            np.array([1.0, 1.0, 0.0], dtype=np.float32)
        )
        if terminated:
            assert info["out_of_bounds"] is True
            break
    assert terminated
    env.close()


def test_observation_in_spaces() -> None:
    env = _env()
    obs, _ = env.reset(seed=14)
    assert env.observation_space.contains(obs)
    env.step(env.action_space.sample())
    env.close()
