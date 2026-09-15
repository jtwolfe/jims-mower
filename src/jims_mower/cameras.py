"""Camera-rig math. Body frame is ROS-style: x forward, y left, z up."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jims_mower.kinematics import wrap_angle
from jims_mower.types import CameraSpec, Pose


@dataclass(frozen=True)
class CameraWorldPose:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    fov_deg: float
    name: str


def camera_world_pose(robot: Pose, cam: CameraSpec) -> CameraWorldPose:
    c = math.cos(robot.theta)
    s = math.sin(robot.theta)
    return CameraWorldPose(
        x=robot.x + cam.x * c - cam.y * s,
        y=robot.y + cam.x * s + cam.y * c,
        z=cam.z,
        yaw=wrap_angle(robot.theta + math.radians(cam.yaw_deg)),
        pitch=math.radians(cam.pitch_deg),
        fov_deg=cam.fov_deg,
        name=cam.name,
    )


def focal_length_px(width: int, fov_deg: float) -> float:
    if fov_deg <= 0 or fov_deg >= 180:
        raise ValueError("fov_deg must be in (0, 180)")
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


def world_to_body(wx: float, wy: float, wz: float, robot: Pose) -> tuple[float, float, float]:
    dx = wx - robot.x
    dy = wy - robot.y
    c = math.cos(robot.theta)
    s = math.sin(robot.theta)
    bx = dx * c + dy * s
    by = -dx * s + dy * c
    return bx, by, wz


def body_to_world(bx: float, by: float, bz: float, robot: Pose) -> tuple[float, float, float]:
    c = math.cos(robot.theta)
    s = math.sin(robot.theta)
    return robot.x + bx * c - by * s, robot.y + bx * s + by * c, bz


def world_to_optical(
    wx: float,
    wy: float,
    wz: float,
    cam: CameraWorldPose,
) -> tuple[float, float, float]:
    """Point in OpenCV optical frame (x right, y down, z forward)."""
    dx = wx - cam.x
    dy = wy - cam.y
    dz = wz - cam.z
    c = math.cos(cam.yaw)
    s = math.sin(cam.yaw)
    # Yawed camera frame: x forward, y left, z up.
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    lz = dz
    cp = math.cos(cam.pitch)
    sp = math.sin(cam.pitch)
    fwd = lx * cp + lz * sp
    up = -lx * sp + lz * cp
    right = -ly
    return right, -up, fwd


def project_point(
    wx: float,
    wy: float,
    wz: float,
    cam: CameraWorldPose,
    width: int,
    height: int,
    near_m: float = 0.05,
) -> tuple[float, float, float] | None:
    """Return (u, v, depth) or None if behind the near plane."""
    ox, oy, oz = world_to_optical(wx, wy, wz, cam)
    if oz <= near_m:
        return None
    fx = focal_length_px(width, cam.fov_deg)
    fy = fx
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    u = cx + fx * (ox / oz)
    v = cy + fy * (oy / oz)
    return float(u), float(v), float(oz)


def pixel_rays_world(
    cam: CameraWorldPose,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """World-space ray origins (broadcast) and direction components.

    Returns ``(ox, oy, oz, dirs)`` where ``dirs`` has shape (H, W, 3).
    """
    fx = focal_length_px(width, cam.fov_deg)
    fy = fx
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    us = np.arange(width, dtype=np.float32)
    vs = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    # Optical rays.
    X = (uu - cx) / fx
    Y = (vv - cy) / fy
    Z = np.ones_like(X)
    fwd = Z
    left = -X
    up = -Y
    cp = math.cos(cam.pitch)
    sp = math.sin(cam.pitch)
    lx = fwd * cp - up * sp
    lz = fwd * sp + up * cp
    ly = left
    c = math.cos(cam.yaw)
    s = math.sin(cam.yaw)
    dx = lx * c - ly * s
    dy = lx * s + ly * c
    dz = lz
    dirs = np.stack([dx, dy, dz], axis=-1)
    return (
        np.float32(cam.x),
        np.float32(cam.y),
        np.float32(cam.z),
        dirs,
    )


def ground_hits(
    cam: CameraWorldPose,
    width: int,
    height: int,
    ground_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect camera rays with a horizontal plane. Invalid hits are NaN."""
    ox, oy, oz, dirs = pixel_rays_world(cam, width, height)
    dz = dirs[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ground_z - oz) / dz
    valid = (dz < -1e-6) & (t > 0.0)
    hx = np.where(valid, ox + t * dirs[:, :, 0], np.nan)
    hy = np.where(valid, oy + t * dirs[:, :, 1], np.nan)
    return hx, hy, valid
