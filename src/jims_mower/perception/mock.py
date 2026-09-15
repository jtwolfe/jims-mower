"""Ground-truth projector used in simulation. Not a trained detector."""

from __future__ import annotations

from typing import Optional

import numpy as np

from jims_mower.cameras import camera_world_pose, focal_length_px, project_point
from jims_mower.constants import ANIMAL_KINDS, LIVING_KINDS
from jims_mower.types import CameraSpec, Detection, Obstacle, PerceptionContext, Pose


def category_for(label: str) -> str:
    if label == "person":
        return "person"
    if label in ANIMAL_KINDS:
        return "animal"
    return "static"


def detection_from_obstacle(
    obst: Obstacle,
    cam: CameraSpec,
    pose: Pose,
    width: int,
    height: int,
    *,
    include_signal: bool,
    min_pixels: float = 2.0,
) -> Optional[Detection]:
    world_cam = camera_world_pose(pose, cam)
    proj = project_point(obst.x, obst.y, obst.z, world_cam, width, height)
    if proj is None:
        return None
    u, v, depth = proj
    fx = focal_length_px(width, world_cam.fov_deg)
    size = fx * (2.0 * obst.radius) / max(depth, 1e-3)
    if size < min_pixels:
        return None
    x0 = int(round(u - 0.5 * size))
    y0 = int(round(v - 0.5 * size))
    w = max(1, int(round(size)))
    h = max(1, int(round(size)))
    if x0 + w < 0 or y0 + h < 0 or x0 >= width or y0 >= height:
        return None
    x0 = int(np.clip(x0, 0, width - 1))
    y0 = int(np.clip(y0, 0, height - 1))
    w = min(w, width - x0)
    h = min(h, height - y0)
    signal = obst.hand_signal if include_signal and obst.kind == "person" else None
    return Detection(
        label=obst.kind,
        camera=cam.name,
        bbox=(x0, y0, w, h),
        confidence=float(np.clip(0.97 * np.exp(-depth / 25.0), 0.15, 0.99)),
        world_xy=(obst.x, obst.y),
        hand_signal=signal,
        category=category_for(obst.kind),
        depth_m=depth,
    )


class MockDetector:
    """Project world obstacles into each camera. For training in this gym."""

    def detect(
        self,
        images: dict[str, np.ndarray],
        context: PerceptionContext,
    ) -> list[Detection]:
        width, height = context.image_size
        if images:
            any_img = next(iter(images.values()))
            height, width = any_img.shape[:2]
        out: list[Detection] = []
        for cam in context.cameras:
            if images and cam.name not in images:
                continue
            for obst in context.obstacles:
                det = detection_from_obstacle(
                    obst,
                    cam,
                    context.pose,
                    width,
                    height,
                    include_signal=context.hand_signals_enabled,
                )
                if det is not None:
                    out.append(det)
        return out


def living_detections(detections: list[Detection]) -> list[Detection]:
    return [d for d in detections if d.label in LIVING_KINDS]
