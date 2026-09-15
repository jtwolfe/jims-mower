"""Radio simulator: Wi-Fi → BT → LoRa failover and heartbeat loss."""

from __future__ import annotations

import numpy as np
import pytest

from jims_mower.config import ConfigError, load_config
from jims_mower.env import MowerEnv
from jims_mower.radio import RadioLink, RadioSim, default_links


def test_preference_wifi_then_bt_then_lora() -> None:
    links = {
        "wifi": RadioLink("wifi", 1e6, 40.0, drop_prob=0.0),
        "bt": RadioLink("bt", 1e5, 12.0, drop_prob=0.0),
        "lora": RadioLink("lora", 1e3, 2000.0, drop_prob=0.0),
    }
    sim = RadioSim(enabled=True, links=links, seed=0)
    near = sim.send(distance_m=5.0)
    assert near.ok and near.channel == "wifi"
    mid = sim.send(distance_m=20.0)
    assert mid.ok and mid.channel == "bt"
    far = sim.send(distance_m=100.0)
    assert far.ok and far.channel == "lora"
    assert far.attempts == ["wifi", "bt", "lora"]


def test_drop_probability_failsover() -> None:
    links = {
        "wifi": RadioLink("wifi", 1e6, 40.0, drop_prob=1.0),
        "bt": RadioLink("bt", 1e5, 12.0, drop_prob=0.0),
        "lora": RadioLink("lora", 1e3, 2000.0, drop_prob=0.0),
    }
    sim = RadioSim(enabled=True, links=links, seed=1, distance_m=5.0)
    delivery = sim.send()
    assert delivery.ok
    assert delivery.channel == "bt"
    assert delivery.attempts[0] == "wifi"


def test_all_channels_fail() -> None:
    links = {
        "wifi": RadioLink("wifi", 1e6, 1.0, drop_prob=1.0),
        "bt": RadioLink("bt", 1e5, 1.0, drop_prob=1.0),
        "lora": RadioLink("lora", 1e3, 1.0, drop_prob=1.0),
    }
    sim = RadioSim(enabled=True, links=links, seed=2, distance_m=50.0)
    delivery = sim.send()
    assert delivery.ok is False
    assert delivery.channel is None


def test_heartbeat_loss_stop_beacon() -> None:
    links = {
        name: RadioLink(name, 1e6, 0.1, drop_prob=1.0) for name in ("wifi", "bt", "lora")
    }
    sim = RadioSim(
        enabled=True,
        links=links,
        heartbeat_timeout_s=0.3,
        on_loss="stop_beacon",
        distance_m=10.0,
        seed=3,
    )
    sim.tick(0.1)
    assert sim.lost is False
    sim.tick(0.1)
    sim.tick(0.1)
    assert sim.lost is True
    assert sim.on_loss == "stop_beacon"


def test_heartbeat_loss_limp_home_in_env() -> None:
    cfg = {
        "dt": 0.1,
        "max_steps": 20,
        "sensors": {"width": 32, "height": 24, "camera_count": 4},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.20,
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": False},
        },
        "perception": {"terrain_mode": "oracle"},
        "radio": {
            "enabled": True,
            "distance_m": 500.0,
            "heartbeat_timeout_s": 0.15,
            "on_loss": "limp_home",
            "wifi": {"range_m": 1.0, "drop_prob": 1.0, "bandwidth_bps": 1e6},
            "bt": {"range_m": 1.0, "drop_prob": 1.0, "bandwidth_bps": 1e5},
            "lora": {"range_m": 1.0, "drop_prob": 1.0, "bandwidth_bps": 1e3},
        },
    }
    env = MowerEnv(config=cfg)
    env.reset(seed=0)
    _obs, _r, _t, _tr, info = env.step(np.array([0.8, 0.8, 1.0], dtype=np.float32))
    _obs, _r, _t, _tr, info = env.step(np.array([0.8, 0.8, 1.0], dtype=np.float32))
    assert info["radio_lost"] is True
    assert info["radio_on_loss"] == "limp_home"
    assert info["fault"]["code"] == "RADIO_LOSS"
    assert info["fault"]["retrieve"] is False
    env.close()


def test_heartbeat_loss_stop_beacon_sos_in_env() -> None:
    cfg = {
        "dt": 0.1,
        "max_steps": 20,
        "sensors": {"width": 32, "height": 24, "camera_count": 4},
        "world": {
            "width_m": 8.0,
            "height_m": 8.0,
            "resolution_m": 0.20,
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": 0,
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": False},
        },
        "perception": {"terrain_mode": "oracle"},
        "radio": {
            "enabled": True,
            "distance_m": 500.0,
            "heartbeat_timeout_s": 0.15,
            "on_loss": "stop_beacon",
            "wifi": {"range_m": 1.0, "drop_prob": 1.0, "bandwidth_bps": 1e6},
            "bt": {"range_m": 1.0, "drop_prob": 1.0, "bandwidth_bps": 1e5},
            "lora": {"range_m": 1.0, "drop_prob": 1.0, "bandwidth_bps": 1e3},
        },
    }
    env = MowerEnv(config=cfg)
    env.reset(seed=0)
    env.step(np.array([0.8, 0.8, 1.0], dtype=np.float32))
    _obs, _r, _t, _tr, info = env.step(np.array([0.8, 0.8, 1.0], dtype=np.float32))
    assert info["radio_lost"] is True
    assert info["fault"]["retrieve"] is True
    assert info["fault"]["code"] == "RADIO_LOSS"
    assert info["trimmer_enabled"] is False
    env.close()


def test_default_links_are_stubs() -> None:
    links = default_links()
    assert tuple(links) == ("wifi", "bt", "lora")
    assert links["wifi"].range_m < links["lora"].range_m
    assert links["wifi"].bandwidth_bps > links["lora"].bandwidth_bps


def test_rejects_bad_on_loss() -> None:
    with pytest.raises(ConfigError):
        load_config({"radio": {"on_loss": "jam_the_spectrum"}})
