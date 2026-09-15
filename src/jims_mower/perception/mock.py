"""Ground-truth projector used in simulation. Not a trained detector."""

from __future__ import annotations

from typing import Optional

import numpy as np

from jims_mower.cameras import camera_world_pose, focal_length_px, project_point
from jims_mower.constants import ANIMAL_KINDS, LIVING_KINDS
from jims_mower.perception.classify import HandSignalClassifier, refine_category
from jims_mower.types import CameraSpec, Detection, Obstacle, PerceptionContext, Pose


def category_for(label: str) -> str:
    if label == "person":
        return "person"
    if label in ANIMAL_KINDS:
        return "animal"
    if label == "toy":
        return "toy"
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
    proj = project_point(obst.x, obst.y, obst.visual_z, world_cam, width, height)
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
    """Project world obstacles into each camera. For training in this gym.

    After the geometric projection, a cheap appearance model (palette +
    crop stats) refines person / animal / toy categories. Not a claimed
    detector. Optional ``HandSignalClassifier`` sits behind the curriculum.
    """

    def __init__(self, *, appearance: bool = True) -> None:
        self.appearance = bool(appearance)
        self._signals = HandSignalClassifier()

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
        use_classifier = bool(
            context.hand_signals_enabled and context.hand_signal_classifier
        )
        for cam in context.cameras:
            if images and cam.name not in images:
                continue
            frame = images.get(cam.name) if images else None
            for obst in context.obstacles:
                det = detection_from_obstacle(
                    obst,
                    cam,
                    context.pose,
                    width,
                    height,
                    include_signal=context.hand_signals_enabled and not use_classifier,
                )
                if det is None:
                    continue
                if self.appearance and frame is not None:
                    _label, category, score = refine_category(frame, det.bbox, det.label)
                    signal = det.hand_signal
                    if use_classifier and det.label == "person":
                        signal = self._signals.classify(frame, det.bbox)
                    det = Detection(
                        label=det.label,
                        camera=det.camera,
                        bbox=det.bbox,
                        confidence=float(np.clip(0.65 * det.confidence + 0.35 * score, 0.05, 0.99)),
                        world_xy=det.world_xy,
                        hand_signal=signal,
                        category=category,
                        depth_m=det.depth_m,
                    )
                elif use_classifier and frame is not None and det.label == "person":
                    det = Detection(
                        label=det.label,
                        camera=det.camera,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        world_xy=det.world_xy,
                        hand_signal=self._signals.classify(frame, det.bbox),
                        category=det.category,
                        depth_m=det.depth_m,
                    )
                out.append(det)
        return out


def living_detections(detections: list[Detection]) -> list[Detection]:
    return [d for d in detections if d.label in LIVING_KINDS]
