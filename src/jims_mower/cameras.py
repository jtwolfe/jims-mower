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
    roll: float = 0.0


def _body_to_world_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Body (x forward, y left, z up) to world. R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _matrix_to_ypr(rm: np.ndarray) -> tuple[float, float, float]:
    """Extract (yaw, pitch, roll) from a body-to-world matrix."""
    pitch = math.asin(float(np.clip(-rm[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) < 1e-8:
        yaw = math.atan2(-rm[0, 1], rm[1, 1])
        roll = 0.0
    else:
        yaw = math.atan2(rm[1, 0], rm[0, 0])
        roll = math.atan2(rm[2, 1], rm[2, 2])
    return wrap_angle(yaw), float(pitch), float(roll)


def camera_world_pose(robot: Pose, cam: CameraSpec) -> CameraWorldPose:
    r_robot = _body_to_world_matrix(robot.theta, robot.pitch, robot.roll)
    offset = r_robot @ np.array([cam.x, cam.y, cam.z], dtype=np.float64)
    r_cam_body = _body_to_world_matrix(math.radians(cam.yaw_deg), math.radians(cam.pitch_deg), 0.0)
    yaw, pitch, roll = _matrix_to_ypr(r_robot @ r_cam_body)
    return CameraWorldPose(
        x=robot.x + float(offset[0]),
        y=robot.y + float(offset[1]),
        z=robot.z + float(offset[2]),
        yaw=yaw,
        pitch=pitch,
        fov_deg=cam.fov_deg,
        name=cam.name,
        roll=roll,
    )


def focal_length_px(width: int, fov_deg: float) -> float:
    if fov_deg <= 0 or fov_deg >= 180:
        raise ValueError("fov_deg must be in (0, 180)")
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


def world_to_body(wx: float, wy: float, wz: float, robot: Pose) -> tuple[float, float, float]:
    r = _body_to_world_matrix(robot.theta, robot.pitch, robot.roll)
    d = np.array([wx - robot.x, wy - robot.y, wz - robot.z], dtype=np.float64)
    b = r.T @ d
    return float(b[0]), float(b[1]), float(b[2])


def body_to_world(bx: float, by: float, bz: float, robot: Pose) -> tuple[float, float, float]:
    r = _body_to_world_matrix(robot.theta, robot.pitch, robot.roll)
    w = r @ np.array([bx, by, bz], dtype=np.float64)
    return robot.x + float(w[0]), robot.y + float(w[1]), robot.z + float(w[2])


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
    cr = math.cos(cam.roll)
    sr = math.sin(cam.roll)
    left = -right
    left2 = cr * left + sr * up
    up2 = -sr * left + cr * up
    return -left2, -up2, fwd


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
    cr = math.cos(cam.roll)
    sr = math.sin(cam.roll)
    left_r = cr * left - sr * up
    up_r = sr * left + cr * up
    cp = math.cos(cam.pitch)
    sp = math.sin(cam.pitch)
    lx = fwd * cp - up_r * sp
    lz = fwd * sp + up_r * cp
    ly = left_r
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


def heightfield_hits(
    cam: CameraWorldPose,
    width: int,
    height: int,
    sample_z,
    iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Intersect rays with a height field via a few Newton-style updates.

    ``sample_z`` maps arrays of world ``(x, y)`` to elevations. Returns
    ``(hx, hy, hz, valid)``. When the field is flat this matches
    :func:`ground_hits` at ``z=0``.
    """
    ox, oy, oz, dirs = pixel_rays_world(cam, width, height)
    dz = dirs[:, :, 2]
    safe_dz = np.where(np.abs(dz) < 1e-8, -1e-8, dz)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (0.0 - oz) / safe_dz
    t = np.clip(np.where(np.isfinite(t), t, 2.0), 0.02, 80.0)
    hz = np.zeros_like(t)
    for _ in range(max(1, int(iterations))):
        hx = ox + t * dirs[:, :, 0]
        hy = oy + t * dirs[:, :, 1]
        hz = np.asarray(sample_z(hx, hy), dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (hz - oz) / safe_dz
        t = np.clip(np.where(np.isfinite(t), t, t), 0.02, 80.0)
    hx = ox + t * dirs[:, :, 0]
    hy = oy + t * dirs[:, :, 1]
    hz = np.asarray(sample_z(hx, hy), dtype=np.float32)
    hit_z = oz + t * dz
    resid = np.abs(hit_z - hz)
    # Allow slightly upward rays so a bank face above the camera still renders.
    valid = (
        (t > 0.02)
        & np.isfinite(hx)
        & np.isfinite(hy)
        & (resid < 0.25)
        & (dz < 0.20)
    )
    hx = np.where(valid, hx, np.nan)
    hy = np.where(valid, hy, np.nan)
    hz = np.where(valid, hz, np.nan)
    return hx, hy, hz, valid
