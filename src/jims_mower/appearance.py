"""Weather packs and seeded domain-randomisation for the geometric renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from jims_mower.config import DomainRandomizationConfig, EnvConfig, WeatherConfig
from jims_mower.constants import SKY_RGB
from jims_mower.types import Obstacle


# Pack colour temperatures (Kelvin). 0 in YAML means "use the pack default".
_PACK_KELVIN = {
    "clear": 6500.0,
    "dawn": 3600.0,
    "dusk": 3000.0,
    "night": 8500.0,
    "rain": 6200.0,
}

_PACK_AMBIENT = {
    "clear": 1.00,
    "dawn": 0.72,
    "dusk": 0.62,
    "night": 0.18,
    "rain": 0.78,
}

_PACK_SKY = {
    "clear": SKY_RGB,
    "dawn": (232, 164, 118),
    "dusk": (210, 120, 92),
    "night": (12, 16, 32),
    "rain": (112, 128, 148),
}


def kelvin_rgb_scale(kelvin: float) -> tuple[float, float, float]:
    """Cheap Tanner-Helland-style CCT → RGB scale centred on ~6500 K."""
    k = float(np.clip(kelvin, 1000.0, 12000.0)) / 100.0
    if k <= 66.0:
        r = 255.0
        g = 99.4708025861 * np.log(k) - 161.1195681661
    else:
        r = 329.698727446 * ((k - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((k - 60.0) ** -0.0755148492)
    if k >= 66.0:
        b = 255.0
    elif k <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(k - 10.0) - 305.0447927307
    rgb = np.clip(np.array([r, g, b], dtype=np.float64), 0.0, 255.0)
    # Normalise so 6500 K is ~identity.
    ref = kelvin_rgb_scale_raw(6500.0)
    scale = rgb / np.maximum(ref, 1e-6)
    return float(scale[0]), float(scale[1]), float(scale[2])


def kelvin_rgb_scale_raw(kelvin: float) -> np.ndarray:
    k = float(np.clip(kelvin, 1000.0, 12000.0)) / 100.0
    if k <= 66.0:
        r = 255.0
        g = 99.4708025861 * np.log(k) - 161.1195681661
    else:
        r = 329.698727446 * ((k - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((k - 60.0) ** -0.0755148492)
    if k >= 66.0:
        b = 255.0
    elif k <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(k - 10.0) - 305.0447927307
    return np.clip(np.array([r, g, b], dtype=np.float64), 0.0, 255.0)


@dataclass
class Appearance:
    """Per-episode look. Neutral matches the pre-1C renderer."""

    light_dir: np.ndarray
    ambient: float = 1.0
    sky_rgb: tuple[int, int, int] = SKY_RGB
    colour_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    colour_temperature_k: float = 6500.0
    shadow_blobs: list[tuple[float, float, float, float, float]] = field(
        default_factory=list
    )
    motion_blur: bool = False
    camera_dirt: bool = False
    vignette: bool = False
    wet_specular: bool = False
    porch_lights: list[tuple[float, float, float, tuple[int, int, int]]] = field(
        default_factory=list
    )
    dirt_specks: list[tuple[float, float, float, float]] = field(default_factory=list)
    night: bool = False
    pack: str = "clear"
    long_grass: bool = False
    leaf_clutter: bool = False

    @classmethod
    def neutral(cls) -> "Appearance":
        light = np.array([-0.35, 0.25, 0.90], dtype=np.float64)
        light = light / np.linalg.norm(light)
        return cls(light_dir=light)


def appearance_rng(
    episode_rng: np.random.Generator,
    dr: DomainRandomizationConfig,
) -> np.random.Generator:
    extra = int(dr.seed)
    seed = int(episode_rng.integers(0, 2**31 - 1))
    return np.random.default_rng([seed, extra])


def sample_appearance(
    cfg: EnvConfig,
    rng: np.random.Generator,
    yard_size: tuple[float, float],
    obstacles: Optional[list[Obstacle]] = None,
) -> Appearance:
    weather: WeatherConfig = cfg.weather
    dr: DomainRandomizationConfig = cfg.domain_randomization
    pack = weather.pack if weather.pack in _PACK_KELVIN else "clear"
    kelvin = weather.colour_temperature_k or _PACK_KELVIN[pack]
    ambient = _PACK_AMBIENT[pack]
    sky = _PACK_SKY[pack]
    light = np.array([-0.35, 0.25, 0.90], dtype=np.float64)
    if pack == "dawn":
        light = np.array([0.75, 0.15, 0.55], dtype=np.float64)
    elif pack == "dusk":
        light = np.array([-0.70, 0.20, 0.50], dtype=np.float64)
    elif pack == "night":
        light = np.array([0.15, -0.10, 0.85], dtype=np.float64)
    elif pack == "rain":
        light = np.array([-0.20, 0.10, 0.97], dtype=np.float64)
    wet = pack == "rain" or dr.wet_specular
    app = Appearance(
        light_dir=light / np.linalg.norm(light),
        ambient=ambient,
        sky_rgb=sky,
        colour_scale=kelvin_rgb_scale(kelvin),
        colour_temperature_k=kelvin,
        wet_specular=wet,
        night=pack == "night",
        pack=pack,
        long_grass=cfg.world.season == "long_grass",
        leaf_clutter=cfg.world.season == "leaf_clutter",
    )
    if weather.porch_lights or pack == "night":
        app.porch_lights = _porch_lights(yard_size, obstacles)

    if not dr.enabled:
        return app

    if dr.lighting:
        jitter = rng.uniform(-0.22, 0.22, size=3)
        jitter[2] = abs(jitter[2]) * 0.15
        light = app.light_dir + jitter
        light[2] = max(0.35, light[2])
        app.light_dir = light / np.linalg.norm(light)
        app.ambient = float(np.clip(app.ambient * rng.uniform(0.85, 1.12), 0.12, 1.2))
    if dr.colour_jitter:
        jitter = rng.uniform(0.86, 1.14, size=3)
        app.colour_scale = (
            app.colour_scale[0] * float(jitter[0]),
            app.colour_scale[1] * float(jitter[1]),
            app.colour_scale[2] * float(jitter[2]),
        )
    if dr.shadow_blobs:
        n = int(rng.integers(1, 4))
        blobs = []
        for _ in range(n):
            cx = float(rng.uniform(0.5, yard_size[0] - 0.5))
            cy = float(rng.uniform(0.5, yard_size[1] - 0.5))
            rx = float(rng.uniform(0.6, 1.8))
            ry = float(rng.uniform(0.4, 1.4))
            dark = float(rng.uniform(0.35, 0.62))
            blobs.append((cx, cy, rx, ry, dark))
        app.shadow_blobs = blobs
    app.motion_blur = bool(dr.motion_blur)
    app.camera_dirt = bool(dr.camera_dirt)
    app.vignette = bool(dr.vignette)
    if dr.wet_specular:
        app.wet_specular = True
    if dr.camera_dirt:
        n = int(rng.integers(4, 10))
        specks = []
        for _ in range(n):
            specks.append(
                (
                    float(rng.uniform(0.02, 0.98)),
                    float(rng.uniform(0.02, 0.98)),
                    float(rng.uniform(0.012, 0.045)),
                    float(rng.uniform(0.25, 0.65)),
                )
            )
        app.dirt_specks = specks
    return app


def _porch_lights(
    yard_size: tuple[float, float],
    obstacles: Optional[list[Obstacle]],
) -> list[tuple[float, float, float, tuple[int, int, int]]]:
    lights: list[tuple[float, float, float, tuple[int, int, int]]] = []
    warm = (255, 196, 110)
    furniture = [o for o in (obstacles or []) if o.kind == "furniture"]
    if furniture:
        for obst in furniture[:3]:
            lights.append((obst.x, obst.y, 2.8, warm))
    else:
        w, h = yard_size
        lights.append((0.8, 0.8, 2.6, warm))
        lights.append((w - 0.8, 0.8, 2.2, warm))
        lights.append((0.8, h - 0.8, 1.8, warm))
    return lights
