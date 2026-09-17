"""Fused terrain state + grade-aware stamp contract."""

from __future__ import annotations

from jims_mower.live import OWNER_COPY, owner_copy_for, snapshot_tilt_kind
from jims_mower.planning.grade_tip import KIND_GRADE, KIND_OK, KIND_TIP
from jims_mower.planning.terrain_decision import (
    TERRAIN_BLOCKED_NOGO,
    TERRAIN_CONTOUR,
    TERRAIN_IMMOBILISED,
    TERRAIN_OK,
    TERRAIN_RETRACE,
    TERRAIN_TIP_REVERSE,
    climbable_grade,
    fuse_terrain_state,
    may_stamp_blockage,
    near_level_attitude,
    retrace_waypoints,
    stamp_on_cooldown,
)
from jims_mower.types import Pose


def test_fuse_priority_immobilise_then_retrace_then_tip() -> None:
    assert (
        fuse_terrain_state(chassis_tipped=True, retracing=True, tilt_kind=KIND_TIP)
        == TERRAIN_IMMOBILISED
    )
    assert fuse_terrain_state(retracing=True, tilt_kind=KIND_TIP, advice="stop") == TERRAIN_RETRACE
    assert (
        fuse_terrain_state(tilt_kind=KIND_TIP, advice="stop") == TERRAIN_TIP_REVERSE
    )
    assert fuse_terrain_state(tilt_kind=KIND_GRADE, advice="slow") == TERRAIN_CONTOUR
    assert (
        fuse_terrain_state(stamped_recent=True, tilt_kind=KIND_OK) == TERRAIN_BLOCKED_NOGO
    )
    assert fuse_terrain_state() == TERRAIN_OK


def test_owner_copy_one_line_no_seeking_plus_tip() -> None:
    assert owner_copy_for("running", "explore", tilt_kind="tip") == OWNER_COPY["tip_risk"]
    assert (
        owner_copy_for("running", "explore", tilt_kind="tip", terrain_state="retrace")
        == OWNER_COPY["retrace"]
    )
    assert (
        owner_copy_for("running", "explore", tilt_kind="grade", terrain_state="contour")
        == OWNER_COPY["steep_grade"]
    )
    seeking = {"code": "seeking_frontier", "label": "Seeking frontier · map 40%"}
    # Fused tip state must not leak the seeking label.
    assert (
        owner_copy_for(
            "running",
            "explore",
            tilt_kind="tip",
            terrain_state="tip_reverse",
            explore_reason=seeking,
        )
        == OWNER_COPY["tip_risk"]
    )
    assert "Seeking" not in owner_copy_for(
        "running",
        "explore",
        tilt_kind="tip",
        terrain_state="tip_reverse",
        explore_reason=seeking,
    )


def test_snapshot_tilt_kind_not_ok_with_tip_risk() -> None:
    pose = Pose(1.0, 1.0, 0.0, pitch=0.02, roll=0.30)
    assert snapshot_tilt_kind(pose, last_kind="tip") == KIND_TIP
    # Contract helper: near-level + tip-risk copy needs a look-ahead note.
    assert near_level_attitude(0.02, 0.02) is True
    assert near_level_attitude(0.02, 0.30) is False


def test_may_stamp_skips_climbable_grade() -> None:
    assert (
        may_stamp_blockage(reason="no_progress", tilt_kind=KIND_GRADE, advice="slow")
        is False
    )
    assert (
        may_stamp_blockage(
            reason="tip_collision",
            tilt_kind=KIND_TIP,
            advice="stop",
            chassis_tipped=False,
            tipover=False,
        )
        is False
    )
    assert may_stamp_blockage(reason="collision", tilt_kind=KIND_GRADE) is True
    assert may_stamp_blockage(reason="no_progress", tipover=True) is True
    assert may_stamp_blockage(reason="no_progress", drain_drop=True) is True
    assert may_stamp_blockage(reason="no_progress", hard_structure=True) is True
    assert (
        may_stamp_blockage(reason="no_progress", tilt_kind=KIND_OK, advice="ok") is True
    )


def test_climbable_grade_helper() -> None:
    assert climbable_grade(tilt_kind=KIND_GRADE) is True
    assert climbable_grade(look_ahead_kind=KIND_GRADE, tilt_kind=KIND_OK) is True
    assert climbable_grade(tilt_kind=KIND_TIP, chassis_tipped=True) is False


def test_stamp_cooldown_rate_limit() -> None:
    assert stamp_on_cooldown(
        cool_left=4,
        last_xy=(1.0, 1.0),
        stamp_xy=(1.1, 1.0),
        step=10,
        last_step=8,
        cooldown_steps=20,
        min_sep_m=0.90,
    )
    assert not stamp_on_cooldown(
        cool_left=0,
        last_xy=(1.0, 1.0),
        stamp_xy=(3.0, 1.0),
        step=40,
        last_step=8,
        cooldown_steps=20,
        min_sep_m=0.90,
    )


def test_retrace_follows_trail_order_downhill() -> None:
    # Chronological climb +y. Pose at the top. Retrace must walk down.
    trail = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 2.0, 0.0), (1.0, 3.0, 0.0)]
    wps = retrace_waypoints(trail, (1.0, 3.05), length_m=2.2, skip_near_m=0.08)
    assert wps
    ys = [p[1] for p in wps]
    assert ys == sorted(ys, reverse=True)
    assert ys[-1] < ys[0]
    assert ys[-1] <= 1.0 + 1e-9
