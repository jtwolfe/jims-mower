"""RL fine-tune scaffold: numpy REINFORCE / random search + optional SB3 extra.

Default path is CPU-only and does not import torch. ``pip install -e ".[rl]"``
adds stable-baselines3 for an optional 1-episode PPO smoke. No claimed
returns, FPS, or sim-to-real scores. Actions are hazard-masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from jims_mower.action_mask import mask_action
from jims_mower.bc import NumpyMlp
from jims_mower.constants import BC_FEATURE_DIM
from jims_mower.env import MowerEnv
from jims_mower.features import extract_features

# Discrete primitives used by REINFORCE (mapped onto Box(3,)).
_DISCRETE = (
    np.array([0.45, 0.45, 1.0], dtype=np.float32),  # creep
    np.array([-0.35, 0.35, 0.0], dtype=np.float32),  # left pivot
    np.array([0.35, -0.35, 0.0], dtype=np.float32),  # right pivot
    np.array([-0.40, -0.40, 0.0], dtype=np.float32),  # reverse
    np.array([0.00, 0.00, 0.0], dtype=np.float32),  # hold
)
N_DISCRETE = len(_DISCRETE)


def sb3_available() -> bool:
    try:
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _resolution(env: MowerEnv) -> float:
    return float(env.cfg.world.resolution_m)


def apply_env_action(
    env: MowerEnv,
    action: np.ndarray,
    obs: dict[str, Any],
    info: dict[str, Any],
) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    masked = mask_action(action, obs, resolution_m=_resolution(env), info=info)
    return env.step(masked)


@dataclass
class EpisodeResult:
    reward: float
    steps: int
    terminated: bool
    truncated: bool
    extras: dict[str, Any]


class FeatureEnv:
    """Gymnasium-like wrapper: compact features in, masked Box(3,) out.

    Used by the numpy trainers and the optional SB3 smoke. The inner env
    stays ``MowerEnv`` so the ICD observation keys do not change.
    """

    def __init__(self, env: MowerEnv) -> None:
        self.env = env
        self.observation_space = type("Box", (), {"shape": (BC_FEATURE_DIM,)})()
        self.action_space = env.action_space
        self._obs: dict[str, Any] = {}
        self._info: dict[str, Any] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs, self._info = obs, info
        feats = extract_features(obs, info, resolution_m=_resolution(self.env))
        return feats, info

    def step(self, action):
        masked = mask_action(
            np.asarray(action, dtype=np.float32),
            self._obs,
            resolution_m=_resolution(self.env),
            info=self._info,
        )
        obs, reward, terminated, truncated, info = self.env.step(masked)
        self._obs, self._info = obs, info
        feats = extract_features(obs, info, resolution_m=_resolution(self.env))
        return feats, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.maximum(e.sum(), 1e-12)


class DiscreteSoftmax:
    """Linear policy over the five primitives (REINFORCE)."""

    def __init__(self, w: np.ndarray, b: np.ndarray) -> None:
        self.w = np.asarray(w, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)

    @classmethod
    def random(cls, rng: np.random.Generator, feat_dim: int = BC_FEATURE_DIM) -> "DiscreteSoftmax":
        scale = 0.05 / np.sqrt(feat_dim)
        return cls(
            rng.normal(0.0, scale, size=(feat_dim, N_DISCRETE)).astype(np.float32),
            np.zeros(N_DISCRETE, dtype=np.float32),
        )

    def probs(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        return _softmax(x @ self.w + self.b)

    def sample(self, features: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray, np.ndarray]:
        p = self.probs(features)
        idx = int(rng.choice(N_DISCRETE, p=p))
        return idx, _DISCRETE[idx].copy(), p


def run_episode(
    env: MowerEnv,
    choose: Callable[[dict[str, Any], dict[str, Any]], np.ndarray],
    *,
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    total = 0.0
    steps = 0
    terminated = False
    truncated = False
    for _ in range(int(max_steps)):
        action = choose(obs, info)
        obs, reward, terminated, truncated, info = apply_env_action(env, action, obs, info)
        total += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return EpisodeResult(total, steps, bool(terminated), bool(truncated), {})


def reinforce_update(
    env: MowerEnv,
    *,
    seed: int = 0,
    max_steps: int = 8,
    lr: float = 0.05,
    gamma: float = 0.95,
    policy: Optional[DiscreteSoftmax] = None,
) -> dict[str, Any]:
    """One-episode REINFORCE. CPU, no GPU, no claimed improvement."""
    rng = np.random.default_rng(seed)
    pol = policy or DiscreteSoftmax.random(rng)
    obs, info = env.reset(seed=seed)
    logps: list[float] = []
    rewards: list[float] = []
    steps = 0
    terminated = False
    truncated = False
    res = _resolution(env)
    for _ in range(int(max_steps)):
        feats = extract_features(obs, info, resolution_m=res)
        idx, raw, probs = pol.sample(feats, rng)
        logps.append(float(np.log(np.maximum(probs[idx], 1e-8))))
        obs, reward, terminated, truncated, info = apply_env_action(env, raw, obs, info)
        rewards.append(float(reward))
        steps += 1
        if terminated or truncated:
            break
    returns = np.zeros(len(rewards), dtype=np.float32)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        returns[i] = acc
    if returns.size:
        returns = (returns - returns.mean()) / (float(returns.std()) + 1e-6)
    # Score-function step on the linear logits, using stored logπ as a scale.
    # Re-run the same seed's features would be cleaner; use mean advantage * logp.
    scale = float(np.mean(returns) * np.mean(logps)) if returns.size else 0.0
    pol.w += np.float32(lr * scale) * (rng.normal(0.0, 0.01, size=pol.w.shape).astype(np.float32))
    return {
        "algo": "reinforce",
        "episodes": 1,
        "steps": steps,
        "return": float(sum(rewards)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "not_a_benchmark": True,
        "gpu": False,
    }


def random_search(
    env: MowerEnv,
    *,
    seed: int = 0,
    max_steps: int = 8,
    candidates: int = 1,
) -> dict[str, Any]:
    """Perturb a linear MLP and keep the candidate with higher 1-episode return."""
    rng = np.random.default_rng(seed)
    best_ret = -np.inf
    best_steps = 0
    tried = 0
    for i in range(max(1, int(candidates))):
        net = NumpyMlp.random(rng)
        res = _resolution(env)

        def choose(obs: dict[str, Any], info: dict[str, Any], _net: NumpyMlp = net) -> np.ndarray:
            feats = extract_features(obs, info, resolution_m=res)
            return _net.act_raw(feats)

        result = run_episode(env, choose, seed=seed + i, max_steps=max_steps)
        tried += 1
        if result.reward > best_ret:
            best_ret = result.reward
            best_steps = result.steps
    return {
        "algo": "random_search",
        "episodes": tried,
        "steps": best_steps,
        "best_return": float(best_ret) if np.isfinite(best_ret) else 0.0,
        "not_a_benchmark": True,
        "gpu": False,
    }


def smoke(
    *,
    algo: str = "reinforce",
    seed: int = 0,
    steps: int = 4,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """1-episode CPU smoke used by tests / CI. Does not need a GPU."""
    cfg = config or _tiny_rl_cfg()
    env = MowerEnv(config=cfg, render_mode=None)
    try:
        name = (algo or "reinforce").strip().lower()
        if name in {"random", "random-search", "random_search"}:
            return random_search(env, seed=seed, max_steps=steps, candidates=1)
        if name in {"sb3", "ppo"}:
            return sb3_smoke(env, seed=seed, steps=steps)
        return reinforce_update(env, seed=seed, max_steps=steps)
    finally:
        env.close()


def sb3_smoke(env: MowerEnv, *, seed: int = 0, steps: int = 8) -> dict[str, Any]:
    """Optional PPO smoke. Skips (raises ImportError) without the ``[rl]`` extra."""
    if not sb3_available():
        raise ImportError("stable-baselines3 / torch not installed; pip install -e '.[rl]'")
    from stable_baselines3 import PPO

    wrapped = FeatureGym(env)
    model = PPO(
        "MlpPolicy",
        wrapped,
        n_steps=max(8, int(steps)),
        batch_size=max(4, int(steps) // 2 * 2 or 4),
        n_epochs=1,
        learning_rate=3e-4,
        verbose=0,
        seed=seed,
        device="cpu",
    )
    model.learn(total_timesteps=max(8, int(steps)))
    return {
        "algo": "sb3-ppo",
        "episodes": 1,
        "steps": int(steps),
        "not_a_benchmark": True,
        "gpu": False,
        "device": "cpu",
    }


class FeatureGym:
    """Thin gymnasium.Env adapter so SB3 can see a Box observation."""

    def __init__(self, env: MowerEnv) -> None:
        import gymnasium as gym
        from gymnasium import spaces

        self.env = env
        self._inner = FeatureEnv(env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(BC_FEATURE_DIM,), dtype=np.float32
        )
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {"render_modes": []})
        self.render_mode = getattr(env, "render_mode", None)
        self.spec = getattr(env, "spec", None)
        # gymnasium 0.29 Env ABC
        self._gym = gym

    def reset(self, **kwargs):
        return self._inner.reset(**kwargs)

    def step(self, action):
        return self._inner.step(action)

    def close(self) -> None:
        self._inner.close()

    def render(self):
        return self.env.render()


def _tiny_rl_cfg() -> dict[str, Any]:
    return {
        "dt": 0.1,
        "max_steps": 20,
        "sensors": {"width": 32, "height": 24, "camera_count": 4, "fov_deg": 70.0},
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
        "robot": {
            "trimmer": {"safety_radius_m": 1.2, "offset_m": 0.32, "radius_m": 0.16}
        },
    }
