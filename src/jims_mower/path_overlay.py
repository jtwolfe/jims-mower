"""Lightweight owner-map path overlay.

Built for the phone 2D map and live SSE. Stdlib only — do not import
``live``, ``mission_flow``, or the viewer from here (circular-import
trap for ``jims-mower-mission inspect``).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

TRAIL_MAX_POINTS = 80
PLAN_MAX_POINTS = 80
EXPLORE_MAX_POINTS = 80
FRONTIER_MAX_POINTS = 40

MAPPING_PHASES = frozenset({"teach", "calibrate_boundary", "explore", "review"})
MOWING_PHASES = frozenset({"mow", "return_home"})
TEACH_PHASES = frozenset({"teach", "calibrate_boundary"})

MODE_ROWS: dict[str, dict[str, str]] = {
    "idle": {"label": "Ready", "tone": "idle", "kind": "idle"},
    "teach": {"label": "Teaching boundary", "tone": "teach", "kind": "mapping"},
    "calibrate_boundary": {"label": "Teaching boundary", "tone": "teach", "kind": "mapping"},
    "explore": {"label": "Mapping yard", "tone": "map", "kind": "mapping"},
    "review": {"label": "Map ready — review", "tone": "review", "kind": "mapping"},
    "mow": {"label": "Mowing", "tone": "mow", "kind": "mowing"},
    "return_home": {"label": "Returning home", "tone": "home", "kind": "mowing"},
    "charging": {"label": "Charging", "tone": "home", "kind": "idle"},
    "complete": {"label": "Done", "tone": "done", "kind": "done"},
    "safe": {"label": "Hold — safe", "tone": "idle", "kind": "idle"},
    "fault": {"label": "Fault", "tone": "idle", "kind": "idle"},
}

HOLD_LABELS = {
    "paused": "Paused",
    "hold": "Hold",
    "estop": "E-STOP",
}

# Phone / viewer stroke colors — keep the live legend in sync.
OVERLAY_COLORS = {
    "fog": "#1a1e1a",
    "mapped": "#3dbe5a",
    "trail": "#aa88ff",
    "plan": "#2ad4e6",
    "cut": "#d2be50",
    "explore": "#f0a030",
    "frontier": "#42c4dc",
    "fence": "#ffcc33",
    "keepout": "#c82828",
    "target": "#ffee88",
    "pose": "#f3f6f1",
    "grass": "#2e8c3a",
    "mowable": "#7de66e",
    "path": "#80807a",
    "sand": "#d2b478",
    "building": "#b08a56",
    "water": "#1ca4d6",
    "drain": "#c46024",
    "beds": "#58763c",
    "blocked": "#c44c7a",
}


def _as_xy(raw: Any) -> Optional[list[float]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        if raw.get("x") is None or raw.get("y") is None:
            return None
        return [float(raw["x"]), float(raw["y"])]
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return [float(raw[0]), float(raw[1])]
        except (TypeError, ValueError):
            return None
    return None


def _as_pose(raw: Any) -> dict[str, float]:
    if isinstance(raw, dict):
        return {
            "x": float(raw.get("x", 0.0) or 0.0),
            "y": float(raw.get("y", 0.0) or 0.0),
            "theta": float(raw.get("theta", 0.0) or 0.0),
        }
    xy = _as_xy(raw)
    if xy is None:
        return {"x": 0.0, "y": 0.0, "theta": 0.0}
    return {"x": xy[0], "y": xy[1], "theta": 0.0}


def downsample_xy(points: Iterable[Any], limit: int) -> list[list[float]]:
    """Even-stride polyline. Always keeps first and last. Caps at ``limit``."""
    cleaned: list[list[float]] = []
    for raw in points or []:
        xy = _as_xy(raw)
        if xy is None:
            continue
        if cleaned:
            dx = xy[0] - cleaned[-1][0]
            dy = xy[1] - cleaned[-1][1]
            if dx * dx + dy * dy < 1e-8:
                cleaned[-1] = xy
                continue
        cleaned.append(xy)
    cap = max(2, int(limit))
    if len(cleaned) <= cap:
        return cleaned
    step = max(1, int(math.ceil((len(cleaned) - 1) / float(cap - 1))))
    picked = cleaned[::step]
    if picked[-1] != cleaned[-1]:
        picked.append(cleaned[-1])
    if len(picked) > cap:
        picked = [picked[0]] + picked[-(cap - 1) :]
    return picked


def trail_from_poses(poses: Iterable[Any], *, limit: int = TRAIL_MAX_POINTS) -> list[list[float]]:
    """Downsample the whole driven trail so the phone shows the job, not a tail."""
    return downsample_xy(poses or [], limit)


def normalize_phase(phase: str, job_state: str = "running") -> str:
    key = str(phase or "").strip().lower()
    if key in {"calibrate", "calibrating"}:
        return "calibrate_boundary"
    if job_state == "teach" or key == "teach":
        return "teach"
    return key or "idle"


def mode_banner_for(
    phase: str,
    job_state: str = "running",
    *,
    tipped: bool = False,
    immobilised: bool = False,
) -> dict[str, Any]:
    """Large owner chip: Mapping vs Mowing in plain words.

    Pause / hold / ESTOP stay visible as a hold badge but do not hide the
    underlying phase. A past-tip / immobilised chassis is SOS, never Ready.
    """
    key = normalize_phase(phase, job_state)
    state = str(job_state or "").strip().lower()
    if tipped or immobilised:
        row = MODE_ROWS["fault"]
        return {
            "label": "SOS — immobilised",
            "tone": row["tone"],
            "kind": row["kind"],
            "hold": "SOS",
            "phase": "fault" if key not in {"fault", "safe"} else key,
            "job_state": state or "hold",
        }
    if state == "idle" and key not in {"complete", "teach"}:
        row = MODE_ROWS["idle"]
        return {
            "label": row["label"],
            "tone": row["tone"],
            "kind": row["kind"],
            "hold": None,
            "phase": key,
            "job_state": "idle",
        }
    row = MODE_ROWS.get(key) or MODE_ROWS["idle"]
    hold = HOLD_LABELS.get(state)
    return {
        "label": row["label"],
        "tone": row["tone"],
        "kind": row["kind"],
        "hold": hold,
        "phase": key,
        "job_state": str(job_state or "idle"),
    }


def progress_kind_for(phase: str, job_state: str = "running") -> str:
    key = normalize_phase(phase, job_state)
    if key in MOWING_PHASES:
        return "mowing"
    if key in MAPPING_PHASES:
        return "mapping"
    if key == "complete":
        return "done"
    return "idle"


def mission_from_phase(
    phase: str,
    job_state: str = "running",
    *,
    tipped: bool = False,
    immobilised: bool = False,
) -> str:
    """Owner ``state.mission`` from phase — never ``running`` → mowing.

    Pause / hold / idle stay ``idle`` so schedule can re-arm. ESTOP stays
    ``estop``. A live explore / calibrate / review job is not ``mowing``.
    A past-tip chassis is not an idle Ready job.
    """
    if tipped or immobilised:
        return "fault"
    state = str(job_state or "").strip().lower()
    if state == "estop":
        return "estop"
    key = normalize_phase(phase, job_state)
    if state in {"idle", "paused", "hold"} and key != "teach":
        return "idle"
    return {
        "teach": "teach",
        "calibrate_boundary": "teach",
        "explore": "explore",
        "review": "review",
        "mow": "mowing",
        "return_home": "returning",
        "charging": "charging",
        "complete": "idle",
        "safe": "idle",
        "idle": "idle",
        "fault": "idle",
    }.get(key, "idle")


def target_from_waypoints(points: Optional[Iterable[Any]], index: int) -> Optional[list[float]]:
    cleaned = [xy for xy in (_as_xy(raw) for raw in (points or [])) if xy is not None]
    if not cleaned:
        return None
    i = min(max(0, int(index)), len(cleaned) - 1)
    return cleaned[i]


def build_path_overlay(
    *,
    phase: str,
    job_state: str = "running",
    pose: Optional[Any] = None,
    poses: Optional[Iterable[Any]] = None,
    trail: Optional[Iterable[Any]] = None,
    plan: Optional[Iterable[Any]] = None,
    explore: Optional[Iterable[Any]] = None,
    frontiers: Optional[Iterable[Any]] = None,
    n_waypoints: int = 0,
    waypoint_index: int = 0,
    tipped: bool = False,
    immobilised: bool = False,
) -> dict[str, Any]:
    """SSE-cheap overlay: trail / plan / frontiers / pose / phase."""
    key = normalize_phase(phase, job_state)
    mode = mode_banner_for(key, job_state, tipped=tipped, immobilised=immobilised)
    pose_row = _as_pose(pose)
    if trail:
        trail_xy = downsample_xy(trail, TRAIL_MAX_POINTS)
    else:
        trail_xy = trail_from_poses(poses or [], limit=TRAIL_MAX_POINTS)
    if pose_row and (pose_row["x"] or pose_row["y"]) and (
        not trail_xy or trail_xy[-1] != [pose_row["x"], pose_row["y"]]
    ):
        # Keep the live tip on the robot even when the trail was strided.
        if trail_xy:
            trail_xy = list(trail_xy)
            trail_xy[-1] = [pose_row["x"], pose_row["y"]]
        else:
            trail_xy = [[pose_row["x"], pose_row["y"]]]

    show_explore = key in TEACH_PHASES or key == "explore"
    show_plan = key in MOWING_PHASES or key in {"review", "complete", "charging"}
    plan_xy = downsample_xy(plan or [], PLAN_MAX_POINTS) if show_plan else []
    explore_xy = downsample_xy(explore or [], EXPLORE_MAX_POINTS) if show_explore else []
    frontier_xy = downsample_xy(frontiers or [], FRONTIER_MAX_POINTS) if show_explore else []

    remaining = max(0, int(n_waypoints) - int(waypoint_index))
    if int(n_waypoints) <= 0:
        remaining = 0
    active = list(plan or []) if show_plan else list(explore or [])
    target = target_from_waypoints(active, waypoint_index)

    return {
        "trail": trail_xy,
        "plan": plan_xy,
        "explore": explore_xy,
        "frontiers": frontier_xy,
        "pose": pose_row,
        "target": target,
        "phase": key,
        "job_state": str(job_state or "idle"),
        "mode": mode,
        "progress_kind": progress_kind_for(key, job_state),
        "mission": mission_from_phase(
            key, job_state, tipped=tipped, immobilised=immobilised
        ),
        "path_remaining": remaining,
        "n_waypoints": int(n_waypoints),
        "waypoint_index": int(waypoint_index),
        "colors": dict(OVERLAY_COLORS),
    }
