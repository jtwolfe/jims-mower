"""Teach trail → polygon and YardProfile load into the env."""

from __future__ import annotations

from pathlib import Path

from jims_mower.env import MowerEnv
from jims_mower.profile import (
    YardProfile,
    load_yard_profile,
    trail_to_polygon,
    write_yard_profile,
)
from jims_mower.teach import TeachPolicy, main as teach_main, run_teach
from jims_mower.scenarios import load_source


def test_trail_to_polygon_square() -> None:
    trail = []
    for x in [i * 0.1 for i in range(40)]:
        trail.append((1.0 + x, 1.0))
    for y in [i * 0.1 for i in range(40)]:
        trail.append((5.0, 1.0 + y))
    for x in [i * 0.1 for i in range(40)]:
        trail.append((5.0 - x, 5.0))
    for y in [i * 0.1 for i in range(40)]:
        trail.append((1.0, 5.0 - y))
    poly = trail_to_polygon(trail, epsilon_m=0.25)
    assert len(poly) >= 4
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert min(xs) < 1.4
    assert max(xs) > 4.6
    assert min(ys) < 1.4
    assert max(ys) > 4.6


def test_trail_falls_back_when_short() -> None:
    fallback = [(0.5, 0.5), (4.0, 0.5), (4.0, 4.0), (0.5, 4.0)]
    poly = trail_to_polygon([(1.0, 1.0)], fallback=fallback)
    assert poly == fallback


def test_teach_policy_records_trail() -> None:
    cfg, scenario = load_source(None)
    cfg.sensors.width = 16
    cfg.sensors.height = 12
    cfg.sensors.camera_count = 4
    cfg.sensors.cameras = []
    env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
    obs, info = env.reset(seed=2)
    policy = TeachPolicy(env.cfg, spec=env.geofence_spec())
    policy.reset(obs, info)
    for _ in range(6):
        action = policy.act(obs, info)
        obs, _r, term, trunc, info = env.step(action)
        if term or trunc:
            break
    env.close()
    assert len(policy.trail) >= 2
    profile = policy.to_profile()
    assert len(profile.keep_in) >= 3
    assert profile.not_a_benchmark is True


def test_run_teach_writes_profile(tmp_path: Path) -> None:
    out = tmp_path / "teach"
    summary = run_teach(out, steps=6, seed=3, cameras=4)
    assert (out / "profile.json").is_file()
    assert (out / "viewer.json").is_file()
    assert (out / "yard.glb").is_file()
    prof = load_yard_profile(out / "profile.json")
    assert len(prof.keep_in) >= 3
    assert summary["keep_in_vertices"] >= 3
    assert summary["policy"] == "teach"


def test_teach_cli(tmp_path: Path) -> None:
    dest = tmp_path / "cli-teach"
    teach_main(["--out", str(dest), "--steps", "4", "--cameras", "4", "--seed", "1"])
    assert (dest / "profile.json").is_file()


def test_profile_loads_into_env(tmp_path: Path) -> None:
    keep_in = [(1.0, 1.0), (6.0, 1.0), (6.0, 6.0), (1.0, 6.0)]
    profile = YardProfile(
        name="unit",
        width_m=8.0,
        height_m=8.0,
        resolution_m=0.20,
        keep_in=keep_in,
        keep_out=[[(3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 4.0)]],
        home={"x": 2.0, "y": 2.0, "theta": 0.4},
    )
    dest = tmp_path / "yard.json"
    write_yard_profile(dest, profile)
    env = MowerEnv(config={"world": {"width_m": 8.0, "height_m": 8.0, "resolution_m": 0.20,
                                     "n_people": 0, "n_dogs": 0, "n_cats": 0, "n_birds": 0,
                                     "n_trees": 0, "n_furniture": 0, "n_toys": 0,
                                     "terrain": {"enabled": False}},
                           "sensors": {"width": 16, "height": 12, "camera_count": 4}})
    obs, info = env.reset(seed=1, options={"yard_profile": str(dest)})
    spec = info["geofence_spec"]
    assert len(spec["keep_in"]) >= 4
    assert spec["keep_out"]
    assert abs(info["pose"]["x"] - 2.0) < 0.05
    assert abs(info["pose"]["y"] - 2.0) < 0.05
    env.close()
    assert "pose" in obs
