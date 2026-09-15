"""RL scaffold: 1-episode CPU smoke + hazard mask. torch/sb3 optional."""

from __future__ import annotations

import pytest

from jims_mower.rl import sb3_available, smoke
from jims_mower.rl_cli import main as rl_main


def test_reinforce_one_episode_cpu() -> None:
    result = smoke(algo="reinforce", seed=0, steps=4)
    assert result["algo"] == "reinforce"
    assert result["episodes"] == 1
    assert result["steps"] >= 1
    assert result["gpu"] is False
    assert result["not_a_benchmark"] is True


def test_random_search_one_episode() -> None:
    result = smoke(algo="random-search", seed=1, steps=3)
    assert result["algo"] == "random_search"
    assert result["episodes"] == 1
    assert result["not_a_benchmark"] is True
    assert result["gpu"] is False


def test_rl_cli_smoke(tmp_path) -> None:
    out = tmp_path / "rl.json"
    rl_main(["--algo", "reinforce", "--episodes", "1", "--steps", "3", "--out", str(out)])
    assert out.is_file()


@pytest.mark.skipif(not sb3_available(), reason="optional [rl] extra (sb3/torch) not installed")
def test_sb3_optional_smoke() -> None:
    result = smoke(algo="sb3", seed=0, steps=8)
    assert result["algo"] == "sb3-ppo"
    assert result["gpu"] is False
    assert result["not_a_benchmark"] is True
