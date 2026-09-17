"""Unknown-space semantics: observed / free / hazard with explicit confidence.

Unknown cells are not safe and not mowable. The mission layer stamps camera
ground hits (seen mask), a body/ToF disk, and IMU-local cues. Elevation is
a local sample (wheel-z + slow grade in a neighborhood), frozen after the
first stamp — not a yard-wide plane hinged to chassis tilt. Authored /
god-view structure leaked into ``obs["structure"]`` is ignored outside
the observed mask.
MAP READY calls ``lock_observed`` so later stereo / observer elev stamps
cannot flop the frozen mow cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

from jims_mower.cameras import attitude_plane_hits, camera_world_pose
from jims_mower.perception.grade import LOCAL_GRADE_RADIUS_M, local_disk_mask
from jims_mower.constants import (
    BUILDING_RGB,
    HAZARD_DRAIN,
    HAZARD_DRAIN_EDGE,
    HAZARD_STEEP,
    POND_RGB,
    STRUCTURE_BUILDING,
    STRUCTURE_BUNKER,
    STRUCTURE_GARDEN,
    STRUCTURE_GREEN,
    STRUCTURE_NONE,
    STRUCTURE_PATH,
    STRUCTURE_POND,
)
from jims_mower.geofence import GeofenceSpec, rasterize_geofence
from jims_mower.perception.cv_terrain import classify_structure_rgb
from jims_mower.types import CameraSpec, Pose

# Viewer / debug palette (unknown stays dark so the map visibly grows).
UNKNOWN_RGB = (28, 30, 34)
FREE_RGB = (46, 140, 58)
EXPLORED_RGB = (72, 176, 88)
HAZARD_RGB = (196, 96, 36)
STRUCTURE_RGB = (92, 92, 110)
BLOCKAGE_RGB = (196, 76, 122)
BLOCKAGE_VIEW_HEX = "#c44c7a"
# Owner-view water / sheds: saturated so they read through fog holes.
# Physics / CV still use POND_RGB / BUILDING_RGB on the true field.
POND_VIEW_RGB = (28, 164, 214)
BUILDING_VIEW_RGB = (176, 138, 86)
PATH_VIEW_RGB = (128, 128, 122)
BUNKER_VIEW_RGB = (210, 180, 120)
GARDEN_VIEW_RGB = (88, 118, 56)
DRAIN_VIEW_RGB = (196, 96, 36)
MOWABLE_VIEW_RGB = (125, 230, 110)
KEEPOUT_VIEW_RGB = (200, 40, 40)
FRONTIER_RGB = (42, 196, 220)

AREA_LEGEND: tuple[dict[str, str], ...] = (
    {"id": "grass", "label": "Grass", "color": "#2e8c3a"},
    {"id": "mowable", "label": "Mow this", "color": "#7de66e"},
    {"id": "path", "label": "Path", "color": "#80807a"},
    {"id": "sand", "label": "Sand", "color": "#d2b478"},
    {"id": "building", "label": "Building", "color": "#b08a56"},
    {"id": "water", "label": "Water", "color": "#1ca4d6"},
    {"id": "drain", "label": "Drain", "color": "#c46024"},
    {"id": "beds", "label": "Beds", "color": "#58763c"},
    {"id": "keepout", "label": "Keep-out", "color": "#c82828"},
    {"id": "blocked", "label": "Blocked / no-go learned", "color": "#c44c7a"},
    {"id": "fog", "label": "Fog", "color": "#1c1e22"},
)
FOG_RGB = (16, 18, 22)
FOG_ALPHA_UNKNOWN = 236
SHED_LIFT_M = 0.20
POND_DROP_M = 0.07


@dataclass
class ObservedMap:
    """Persistent onboard map. Unknown is implicit: ``~observed``."""

    observed: np.ndarray
    explored: np.ndarray
    free: np.ndarray
    hazard: np.ndarray
    structure: np.ndarray
    elevation: np.ndarray
    confidence: np.ndarray
    occupancy: np.ndarray
    elevation_set: np.ndarray
    locked: np.ndarray
    blockage: np.ndarray
    width_m: float
    height_m: float
    resolution_m: float

    @classmethod
    def empty(
        cls,
        width_m: float,
        height_m: float,
        resolution_m: float,
        shape: Optional[tuple[int, int]] = None,
    ) -> "ObservedMap":
        res = max(float(resolution_m), 1e-6)
        if shape is None:
            cols = max(1, int(round(float(width_m) / res)))
            rows = max(1, int(round(float(height_m) / res)))
        else:
            rows, cols = int(shape[0]), int(shape[1])
        return cls(
            observed=np.zeros((rows, cols), dtype=bool),
            explored=np.zeros((rows, cols), dtype=bool),
            free=np.zeros((rows, cols), dtype=bool),
            hazard=np.zeros((rows, cols), dtype=np.float32),
            structure=np.zeros((rows, cols), dtype=np.uint8),
            elevation=np.zeros((rows, cols), dtype=np.float32),
            confidence=np.zeros((rows, cols), dtype=np.float32),
            occupancy=np.zeros((rows, cols), dtype=np.float32),
            elevation_set=np.zeros((rows, cols), dtype=bool),
            locked=np.zeros((rows, cols), dtype=bool),
            blockage=np.zeros((rows, cols), dtype=bool),
            width_m=float(width_m),
            height_m=float(height_m),
            resolution_m=res,
        )

    @property
    def rows(self) -> int:
        return int(self.observed.shape[0])

    @property
    def cols(self) -> int:
        return int(self.observed.shape[1])

    def world_to_cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        if x < 0.0 or y < 0.0 or x >= self.width_m or y >= self.height_m:
            return None
        col = int(x / self.resolution_m)
        row = int(y / self.resolution_m)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            (col + 0.5) * self.resolution_m,
            (row + 0.5) * self.resolution_m,
        )

    def stamp_disk(self, x: float, y: float, radius_m: float, *, explored: bool = False) -> int:
        """Mark a world disk as observed. Returns newly observed cells."""
        cell = self._disk_indices(x, y, radius_m)
        if cell[0].size == 0:
            return 0
        rows, cols = cell
        fresh = ~self.observed[rows, cols]
        self.observed[rows, cols] = True
        self.confidence[rows, cols] = np.maximum(self.confidence[rows, cols], 0.45)
        if explored:
            self.explored[rows, cols] = True
            self.confidence[rows, cols] = np.maximum(self.confidence[rows, cols], 0.70)
        return int(fresh.sum())

    def stamp_blockage(self, x: float, y: float, radius_m: float) -> int:
        """Mark a local disk as a learned no-go (lethal for planning).

        Invisible / dynamic blockages (physics that never painted a shed,
        a fence the cameras missed, wheel-slip against a lip) become
        known-blocked: observed, not free, high occupancy. Returns newly
        blocked cells. Does not lock — a later owner remap can restamp.
        """
        cell = self._disk_indices(x, y, radius_m)
        if cell[0].size == 0:
            return 0
        rows, cols = cell
        fresh = ~np.asarray(self.blockage[rows, cols], dtype=bool)
        self.blockage[rows, cols] = True
        self.observed[rows, cols] = True
        self.occupancy[rows, cols] = np.maximum(self.occupancy[rows, cols], 0.95)
        self.confidence[rows, cols] = np.maximum(self.confidence[rows, cols], 0.80)
        self.refresh_free()
        return int(fresh.sum())

    def blockage_count(self) -> int:
        return int(np.asarray(self.blockage, dtype=bool).sum())

    def clear_blockages(self) -> int:
        """Drop learned no-go disks. Fence / observed map stay. Returns cells cleared."""
        n = int(np.asarray(self.blockage, dtype=bool).sum())
        self.blockage = np.zeros_like(self.blockage, dtype=bool)
        self.refresh_free()
        return n

    def clear_progress(self, *, clear_blockages: bool = True) -> None:
        """Wipe fog / explore paint for a fresh job. Size and fence stay."""
        self.observed.fill(False)
        self.explored.fill(False)
        self.free.fill(False)
        self.hazard.fill(0)
        self.structure.fill(0)
        self.elevation.fill(0)
        self.confidence.fill(0)
        self.occupancy.fill(0)
        self.elevation_set.fill(False)
        self.locked.fill(False)
        if clear_blockages:
            self.blockage.fill(False)

    def stamp_cameras(
        self,
        images: dict[str, np.ndarray],
        cameras: Iterable[CameraSpec],
        pose: Pose,
        *,
        max_range_m: float = 6.0,
        pixel_stride: int = 2,
    ) -> int:
        """Mark ground-plane camera hits as observed and stamp semantics."""
        added = 0
        world_size = (self.width_m, self.height_m)
        cams = {c.name: c for c in cameras}
        stride = max(1, int(pixel_stride))
        for name, frame in images.items():
            cam = cams.get(name)
            if cam is None or frame is None or frame.ndim != 3:
                continue
            h, w = frame.shape[:2]
            world_cam = camera_world_pose(pose, cam)
            hx, hy, valid = attitude_plane_hits(world_cam, w, h, pose)
            if stride > 1:
                hx = hx[::stride, ::stride]
                hy = hy[::stride, ::stride]
                valid = valid[::stride, ::stride]
                pix = frame[::stride, ::stride]
            else:
                pix = frame
            rng = np.hypot(hx - world_cam.x, hy - world_cam.y)
            inside = (
                valid
                & np.isfinite(hx)
                & np.isfinite(hy)
                & (hx >= 0.0)
                & (hy >= 0.0)
                & (hx < world_size[0])
                & (hy < world_size[1])
                & (rng < float(max_range_m))
            )
            if not np.any(inside):
                continue
            rr = np.floor(hy[inside] / self.resolution_m).astype(np.int32)
            cc = np.floor(hx[inside] / self.resolution_m).astype(np.int32)
            ok = (rr >= 0) & (cc >= 0) & (rr < self.rows) & (cc < self.cols)
            if not np.any(ok):
                continue
            rr, cc = rr[ok], cc[ok]
            fresh = ~self.observed[rr, cc]
            self.observed[rr, cc] = True
            self.confidence[rr, cc] = np.maximum(self.confidence[rr, cc], 0.55)
            added += int(fresh.sum())
            # Semantics come from the gated observer on these newly seen
            # cells. RGB colour heuristics flood path/bunker across grass.
        return added

    def ingest_observer(
        self,
        obs: dict[str, Any],
        *,
        only_observed: bool = True,
        authored_structure: Optional[np.ndarray] = None,
        pose: Optional[Pose] = None,
        grade_radius_m: float = LOCAL_GRADE_RADIUS_M,
    ) -> None:
        """Copy heuristic hazard/structure onto seen cells. Height is local.

        Cameras grow the *seen* mask. Elevation is fused only in a
        neighborhood around ``pose``. Already-stamped heights freeze.
        New cells inherit locked neighbours + wheel ``z`` — not a
        swinging IMU plane on the growing edge.
        """
        mask = self.observed if only_observed else np.ones_like(self.observed, dtype=bool)
        if not np.any(mask):
            return
        hazard = obs.get("hazard")
        if hazard is not None:
            hz = np.asarray(hazard, dtype=np.float32)
            if hz.shape == self.hazard.shape:
                self.hazard[mask] = np.maximum(self.hazard[mask], hz[mask])
        self._fuse_elevation(obs, mask, pose=pose, grade_radius_m=grade_radius_m)
        conf = obs.get("confidence")
        if conf is not None:
            cv = np.asarray(conf, dtype=np.float32)
            if cv.shape == self.confidence.shape:
                self.confidence[mask] = np.maximum(self.confidence[mask], cv[mask])
        struct = authored_structure if authored_structure is not None else obs.get("structure")
        if struct is not None:
            st = np.asarray(struct)
            if st.shape == self.structure.shape:
                # Authored polygons on cells we have seen. Live RGB stamps
                # flood path/bunker across grass and are not used here.
                self.structure[mask] = np.maximum(
                    self.structure[mask], st[mask].astype(np.uint8)
                )
        occ = obs.get("occupancy")
        if occ is not None:
            ov = np.asarray(occ, dtype=np.float32)
            if ov.shape == self.occupancy.shape:
                self.occupancy[mask] = np.maximum(self.occupancy[mask], ov[mask])
        self.refresh_free()

    def lock_observed(self) -> int:
        """Freeze currently observed elev/hazard cells (MAP READY).

        Later stereo / observer stamps must not flop these cells.
        """
        self.locked |= np.asarray(self.observed, dtype=bool)
        return int(self.locked.sum())

    def stamp_metric_elevation(
        self,
        elev: np.ndarray,
        hit_mask: Optional[np.ndarray] = None,
        *,
        respect_lock: bool = True,
    ) -> int:
        """Write metric local height onto observed, unlocked cells.

        ``hit_mask`` marks cells the stereo (or ToF) band actually saw.
        Locked cells stay put — the frozen mow map must not jitter.
        """
        ev = np.asarray(elev, dtype=np.float32)
        if ev.shape != self.elevation.shape:
            raise ValueError("elevation shape must match ObservedMap")
        writable = np.asarray(self.observed, dtype=bool)
        if hit_mask is not None:
            writable = writable & np.asarray(hit_mask, dtype=bool)
        if respect_lock:
            writable = writable & ~np.asarray(self.locked, dtype=bool)
        if not np.any(writable):
            return 0
        before = self.elevation[writable].copy()
        self.elevation[writable] = ev[writable]
        self.elevation_set[writable] = True
        self.confidence[writable] = np.maximum(self.confidence[writable], 0.60)
        return int(np.count_nonzero(np.abs(self.elevation[writable] - before) > 1e-6))

    def fuse_metric(
        self,
        result: Any,
        *,
        respect_lock: bool = True,
    ) -> int:
        """Write a gym stereo+ToF+IMU fuse onto observed cells.

        Stereo / ToF hits use ``stamp_metric_elevation`` (locked cells stay).
        IMU / prior fills only unset, unlocked cells.
        """
        ev = np.asarray(getattr(result, "elevation"), dtype=np.float32)
        if ev.shape != self.elevation.shape:
            raise ValueError("fuse elevation shape must match ObservedMap")
        stereo = np.asarray(
            getattr(result, "stereo_hits", np.zeros_like(self.observed)), dtype=bool
        )
        tof = np.asarray(getattr(result, "tof_hits", np.zeros_like(self.observed)), dtype=bool)
        imu = np.asarray(getattr(result, "imu_hits", np.zeros_like(self.observed)), dtype=bool)
        prior = np.asarray(
            getattr(result, "prior_hits", np.zeros_like(self.observed)), dtype=bool
        )
        metric = (stereo | tof) & np.asarray(self.observed, dtype=bool)
        n = self.stamp_metric_elevation(ev, metric, respect_lock=respect_lock)
        fill = (
            (imu | prior)
            & np.asarray(self.observed, dtype=bool)
            & ~np.asarray(self.elevation_set, dtype=bool)
        )
        if respect_lock:
            fill = fill & ~np.asarray(self.locked, dtype=bool)
        if np.any(fill):
            self.elevation[fill] = ev[fill]
            self.elevation_set[fill] = True
            self.confidence[fill] = np.maximum(self.confidence[fill], 0.50)
            n += int(fill.sum())
        return n

    def refresh_free(self) -> None:
        """Known-free = observed, not a drain/steep-blocked hint, not hard structure."""
        hard = np.isin(
            self.structure,
            (STRUCTURE_BUILDING, STRUCTURE_BUNKER, STRUCTURE_GARDEN, STRUCTURE_GREEN),
        )
        # Channels are forbidden. Isolated lip stamps from the colour
        # heuristic are not a ditch — the costmap can still slow them.
        channel = self.hazard >= HAZARD_DRAIN
        learned = np.asarray(self.blockage, dtype=bool)
        self.free = self.observed & ~hard & ~channel & ~learned

    def keep_in_mask(self, spec: Optional[GeofenceSpec]) -> np.ndarray:
        if spec is None or not spec.has_polygons():
            return np.ones((self.rows, self.cols), dtype=bool)
        outside = rasterize_geofence(
            (self.rows, self.cols),
            resolution_m=self.resolution_m,
            spec=spec,
        )
        return ~outside

    def completion(self, keep_in: Optional[np.ndarray] = None) -> float:
        region = np.ones((self.rows, self.cols), dtype=bool) if keep_in is None else np.asarray(keep_in, dtype=bool)
        n = int(region.sum())
        if n <= 0:
            return 1.0
        return float(np.count_nonzero(self.observed & region) / n)

    def mean_confidence(self, keep_in: Optional[np.ndarray] = None) -> float:
        region = self.observed
        if keep_in is not None:
            region = region & np.asarray(keep_in, dtype=bool)
        if not np.any(region):
            return 0.0
        return float(self.confidence[region].mean())

    def mowable_mask(self, keep_in: Optional[np.ndarray] = None) -> np.ndarray:
        """Reachable mow targets: observed free grass, no hardscape."""
        region = np.ones((self.rows, self.cols), dtype=bool) if keep_in is None else np.asarray(keep_in, dtype=bool)
        hardscape = self.structure != STRUCTURE_NONE
        return self.free & region & ~hardscape

    def unknown_blocked(self, keep_in: Optional[np.ndarray] = None) -> np.ndarray:
        """Cells the global planner must not treat as safe transit."""
        blocked = ~self.observed
        hard = np.isin(
            self.structure,
            (STRUCTURE_BUILDING, STRUCTURE_BUNKER, STRUCTURE_GARDEN, STRUCTURE_GREEN),
        )
        learned = np.asarray(self.blockage, dtype=bool)
        blocked = blocked | hard | (self.hazard >= HAZARD_DRAIN) | learned
        if keep_in is not None:
            blocked = blocked | ~np.asarray(keep_in, dtype=bool)
        return blocked

    def _paint_area_types(self, rgb: np.ndarray, *, observed_only: bool = True) -> np.ndarray:
        """Colour grass / path / sand / building / water / drain / beds."""
        mask = np.asarray(self.observed, dtype=bool) if observed_only else np.ones(
            (self.rows, self.cols), dtype=bool
        )
        rgb[mask] = FREE_RGB
        explored = mask & np.asarray(self.explored, dtype=bool)
        rgb[explored] = EXPLORED_RGB
        rgb[mask & (self.hazard >= HAZARD_STEEP)] = (210, 168, 48)
        rgb[mask & (self.hazard >= HAZARD_DRAIN_EDGE)] = DRAIN_VIEW_RGB
        rgb[mask & (self.structure == STRUCTURE_PATH)] = PATH_VIEW_RGB
        rgb[mask & (self.structure == STRUCTURE_BUNKER)] = BUNKER_VIEW_RGB
        rgb[mask & (self.structure == STRUCTURE_GARDEN)] = GARDEN_VIEW_RGB
        rgb[mask & (self.structure == STRUCTURE_GREEN)] = GARDEN_VIEW_RGB
        rgb[mask & (self.structure == STRUCTURE_BUILDING)] = BUILDING_VIEW_RGB
        rgb[mask & (self.structure == STRUCTURE_POND)] = POND_VIEW_RGB
        rgb[mask & np.asarray(self.blockage, dtype=bool)] = BLOCKAGE_RGB
        return rgb

    def as_rgb(self, *, frontiers: Optional[list[tuple[int, int]]] = None) -> np.ndarray:
        """Unknown stays dark so exploration growth is visible frame-to-frame."""
        rgb = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        rgb[:, :] = UNKNOWN_RGB
        self._paint_area_types(rgb, observed_only=True)
        if frontiers:
            for row, col in frontiers:
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    rgb[row, col] = FRONTIER_RGB
        return rgb[::-1]

    def surface_colors(self) -> np.ndarray:
        """Unflipped owner-view RGB for the observed terrain mesh."""
        rgb = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        rgb[:, :] = UNKNOWN_RGB
        self._paint_area_types(rgb, observed_only=True)
        return rgb

    def areas_rgb(
        self,
        keep_in: Optional[np.ndarray] = None,
        *,
        mowable: Optional[np.ndarray] = None,
        keep_out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Owner area-type sheet: classified cells + planned mowable mask.

        Flipped like ``as_rgb`` so a PNG lines up with the phone overlay.
        """
        rgb = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        rgb[:, :] = UNKNOWN_RGB
        self._paint_area_types(rgb, observed_only=True)
        if mowable is None:
            paint = self.mowable_mask(keep_in)
        else:
            paint = np.asarray(mowable, dtype=bool)
            if paint.shape != (self.rows, self.cols):
                paint = self.mowable_mask(keep_in)
        # Highlight the automatic mow region without hiding grass class.
        if np.any(paint):
            base = rgb[paint].astype(np.float32)
            tint = np.asarray(MOWABLE_VIEW_RGB, dtype=np.float32)
            rgb[paint] = np.clip(0.45 * base + 0.55 * tint, 0, 255).astype(np.uint8)
        if keep_out is not None:
            hole = np.asarray(keep_out, dtype=bool)
            if hole.shape == (self.rows, self.cols):
                rgb[hole & np.asarray(self.observed, dtype=bool)] = KEEPOUT_VIEW_RGB
        return rgb[::-1]

    @staticmethod
    def area_legend() -> list[dict[str, str]]:
        return [dict(row) for row in AREA_LEGEND]

    def observed_elevation(
        self,
        *,
        shed_lift_m: float = SHED_LIFT_M,
        pond_drop_m: float = POND_DROP_M,
    ) -> np.ndarray:
        """Partial height field: observed cells that have a height sample.

        Physics still uses the true field. Camera-seen cells without a
        local z stay NaN (mesh hole + fog overlay). Sheds lift a little,
        ponds sit slightly low.
        """
        z = np.full((self.rows, self.cols), np.nan, dtype=np.float32)
        mask = np.asarray(self.observed, dtype=bool)
        elev = np.asarray(self.elevation, dtype=np.float32)
        if mask.shape != z.shape or elev.shape != z.shape:
            return z
        have = np.asarray(self.elevation_set, dtype=bool)
        if have.shape == mask.shape and np.any(have):
            mask = mask & have
        if not np.any(mask):
            return z
        z[mask] = elev[mask]
        sheds = self.observed & (self.structure == STRUCTURE_BUILDING)
        ponds = self.observed & (self.structure == STRUCTURE_POND)
        if np.any(sheds):
            z[sheds] = z[sheds] + float(shed_lift_m)
        if np.any(ponds):
            z[ponds] = z[ponds] - float(pond_drop_m)
        return z

    def copy(self) -> "ObservedMap":
        return ObservedMap(
            observed=self.observed.copy(),
            explored=self.explored.copy(),
            free=self.free.copy(),
            hazard=self.hazard.copy(),
            structure=self.structure.copy(),
            elevation=self.elevation.copy(),
            confidence=self.confidence.copy(),
            occupancy=self.occupancy.copy(),
            elevation_set=np.asarray(self.elevation_set, dtype=bool).copy(),
            locked=np.asarray(self.locked, dtype=bool).copy(),
            blockage=np.asarray(self.blockage, dtype=bool).copy(),
            width_m=self.width_m,
            height_m=self.height_m,
            resolution_m=self.resolution_m,
        )

    def save_npz(self, path: Union[str, Path]) -> Path:
        """Write fog / elev / lock rasters. Cold-load companion to mission.npz."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            observed=np.asarray(self.observed, dtype=bool),
            explored=np.asarray(self.explored, dtype=bool),
            free=np.asarray(self.free, dtype=bool),
            hazard=np.asarray(self.hazard, dtype=np.float32),
            structure=np.asarray(self.structure, dtype=np.uint8),
            elevation=np.asarray(self.elevation, dtype=np.float32),
            confidence=np.asarray(self.confidence, dtype=np.float32),
            occupancy=np.asarray(self.occupancy, dtype=np.float32),
            elevation_set=np.asarray(self.elevation_set, dtype=bool),
            locked=np.asarray(self.locked, dtype=bool),
            blockage=np.asarray(self.blockage, dtype=bool),
            width_m=np.float32(self.width_m),
            height_m=np.float32(self.height_m),
            resolution_m=np.float32(self.resolution_m),
        )
        return dest

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "ObservedMap":
        src = Path(path)
        data = np.load(src, allow_pickle=False)
        omap = cls(
            observed=np.asarray(data["observed"], dtype=bool),
            explored=np.asarray(data["explored"], dtype=bool),
            free=np.asarray(data["free"], dtype=bool),
            hazard=np.asarray(data["hazard"], dtype=np.float32),
            structure=np.asarray(data["structure"], dtype=np.uint8),
            elevation=np.asarray(data["elevation"], dtype=np.float32),
            confidence=np.asarray(data["confidence"], dtype=np.float32),
            occupancy=np.asarray(data["occupancy"], dtype=np.float32),
            elevation_set=np.asarray(data["elevation_set"], dtype=bool)
            if "elevation_set" in data.files
            else np.zeros_like(data["observed"], dtype=bool),
            locked=np.asarray(data["locked"], dtype=bool)
            if "locked" in data.files
            else np.zeros_like(data["observed"], dtype=bool),
            blockage=np.asarray(data["blockage"], dtype=bool)
            if "blockage" in data.files
            else np.zeros_like(data["observed"], dtype=bool),
            width_m=float(data["width_m"]),
            height_m=float(data["height_m"]),
            resolution_m=float(data["resolution_m"]),
        )
        return omap

    def arrays_for_npz(self) -> dict[str, np.ndarray]:
        """Prefix-free arrays for embedding in a mission bundle."""
        return {
            "omap_observed": np.asarray(self.observed, dtype=bool),
            "omap_explored": np.asarray(self.explored, dtype=bool),
            "omap_free": np.asarray(self.free, dtype=bool),
            "omap_hazard": np.asarray(self.hazard, dtype=np.float32),
            "omap_structure": np.asarray(self.structure, dtype=np.uint8),
            "omap_elevation": np.asarray(self.elevation, dtype=np.float32),
            "omap_confidence": np.asarray(self.confidence, dtype=np.float32),
            "omap_occupancy": np.asarray(self.occupancy, dtype=np.float32),
            "omap_elevation_set": np.asarray(self.elevation_set, dtype=bool),
            "omap_locked": np.asarray(self.locked, dtype=bool),
            "omap_blockage": np.asarray(self.blockage, dtype=bool),
        }

    @classmethod
    def from_npz_arrays(cls, data: Any) -> Optional["ObservedMap"]:
        files = set(getattr(data, "files", []) or [])
        if "omap_observed" not in files:
            return None
        return cls(
            observed=np.asarray(data["omap_observed"], dtype=bool),
            explored=np.asarray(data["omap_explored"], dtype=bool),
            free=np.asarray(data["omap_free"], dtype=bool),
            hazard=np.asarray(data["omap_hazard"], dtype=np.float32),
            structure=np.asarray(data["omap_structure"], dtype=np.uint8),
            elevation=np.asarray(data["omap_elevation"], dtype=np.float32),
            confidence=np.asarray(data["omap_confidence"], dtype=np.float32),
            occupancy=np.asarray(data["omap_occupancy"], dtype=np.float32),
            elevation_set=np.asarray(data["omap_elevation_set"], dtype=bool)
            if "omap_elevation_set" in files
            else np.zeros_like(data["omap_observed"], dtype=bool),
            locked=np.asarray(data["omap_locked"], dtype=bool)
            if "omap_locked" in files
            else np.zeros_like(data["omap_observed"], dtype=bool),
            blockage=np.asarray(data["omap_blockage"], dtype=bool)
            if "omap_blockage" in files
            else np.zeros_like(data["omap_observed"], dtype=bool),
            width_m=float(data["width_m"]),
            height_m=float(data["height_m"]),
            resolution_m=float(data["resolution_m"]),
        )

    def _fuse_elevation(
        self,
        obs: dict[str, Any],
        mask: np.ndarray,
        *,
        pose: Optional[Pose],
        grade_radius_m: float,
    ) -> None:
        """Anchor height on first local stamp; do not hinge the sheet to IMU."""
        ev = obs.get("elevation")
        prior = obs.get("elevation_prior")
        ev_a = None if ev is None else np.asarray(ev, dtype=np.float32)
        prior_a = None if prior is None else np.asarray(prior, dtype=np.float32)
        if ev_a is not None and ev_a.shape != self.elevation.shape:
            ev_a = None
        if prior_a is not None and prior_a.shape != self.elevation.shape:
            prior_a = None
        if ev_a is None and prior_a is None and pose is None:
            return
        meas = np.zeros_like(self.elevation)
        have_meas = np.zeros_like(self.observed, dtype=bool)
        if prior_a is not None:
            meas = prior_a
            have_meas[:, :] = True
        if ev_a is not None:
            meas = ev_a
            have_meas[:, :] = True
        if pose is not None and not np.any(have_meas):
            meas[:, :] = np.float32(pose.z)
            have_meas[:, :] = True
        if pose is None:
            # Tests / callers without a chassis pose: first-stamp freeze.
            write = mask & have_meas & ~self.elevation_set & ~np.asarray(self.locked, dtype=bool)
            if np.any(write):
                self.elevation[write] = meas[write]
                self.elevation_set[write] = True
            return
        radius = max(float(grade_radius_m), self.resolution_m)
        local = local_disk_mask(
            self.elevation.shape,
            (pose.x, pose.y),
            radius,
            self.resolution_m,
        )
        neighborhood = mask & local
        fresh = neighborhood & ~self.elevation_set & ~np.asarray(self.locked, dtype=bool)
        if np.any(fresh):
            seeded = self._seed_fresh_heights(fresh, pose.z)
            self.elevation[fresh] = seeded[fresh]
            self.elevation_set[fresh] = True
        # Already-mapped cells stay put. A ridge tip must not leap them
        # or reshape the growing edge from a new IMU plane.

    def _seed_fresh_heights(self, fresh: np.ndarray, pose_z: float) -> np.ndarray:
        """Height for new cells: locked neighbours, else wheel ``z``.

        The swinging observer / IMU plane is not used. A climb creeps in
        via ``pose.z``; the bright patch must not flop at the frontier.
        """
        seed = np.full(self.elevation.shape, np.float32(pose_z))
        locked = np.asarray(self.elevation_set, dtype=bool)
        if not np.any(locked):
            return seed
        z = np.asarray(self.elevation, dtype=np.float64)
        acc = np.zeros(self.elevation.shape, dtype=np.float64)
        cnt = np.zeros(self.elevation.shape, dtype=np.float64)
        pairs = (
            ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
            ((slice(1, None), slice(None)), (slice(0, -1), slice(None))),
            ((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
            ((slice(None), slice(1, None)), (slice(None), slice(0, -1))),
        )
        for sl_src, sl_dst in pairs:
            src = locked[sl_src]
            acc[sl_dst] += np.where(src, z[sl_src], 0.0)
            cnt[sl_dst] += src.astype(np.float64)
        have = cnt > 0
        take = fresh & have
        if np.any(take):
            nb = (acc[take] / cnt[take]).astype(np.float32)
            seed[take] = np.float32(0.85) * nb + np.float32(0.15) * np.float32(pose_z)
        return seed

    def _disk_indices(self, x: float, y: float, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
        r = max(float(radius_m), self.resolution_m * 0.5)
        c0 = int((x - r) / self.resolution_m)
        c1 = int((x + r) / self.resolution_m)
        r0 = int((y - r) / self.resolution_m)
        r1 = int((y + r) / self.resolution_m)
        rows: list[int] = []
        cols: list[int] = []
        r2 = r * r
        for row in range(max(0, r0), min(self.rows, r1 + 1)):
            cy = (row + 0.5) * self.resolution_m
            for col in range(max(0, c0), min(self.cols, c1 + 1)):
                cx = (col + 0.5) * self.resolution_m
                if (cx - x) ** 2 + (cy - y) ** 2 <= r2:
                    rows.append(row)
                    cols.append(col)
        if not rows:
            return np.array([], dtype=int), np.array([], dtype=int)
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)

    def _stamp_structure_pixels(
        self,
        pix: np.ndarray,
        inside: np.ndarray,
        ok: np.ndarray,
        rr: np.ndarray,
        cc: np.ndarray,
    ) -> None:
        """Path / bunker colour only. Hazard comes from the gated observer."""
        struct_lab = classify_structure_rgb(pix)
        st = struct_lab[inside][ok].astype(np.uint8)
        self.structure[rr, cc] = np.maximum(self.structure[rr, cc], st)
        self.refresh_free()


def frontiers(
    observed: np.ndarray,
    free: np.ndarray,
    *,
    keep_in: Optional[np.ndarray] = None,
    ignore_unknown: Optional[np.ndarray] = None,
) -> list[tuple[int, int]]:
    """Known-free cells that touch unknown (and stay inside keep-in)."""
    unknown = ~np.asarray(observed, dtype=bool)
    if ignore_unknown is not None:
        unknown = unknown & ~np.asarray(ignore_unknown, dtype=bool)
    safe = np.asarray(free, dtype=bool)
    if keep_in is not None:
        region = np.asarray(keep_in, dtype=bool)
        unknown = unknown & region
        safe = safe & region
    if not np.any(safe) or not np.any(unknown):
        return []
    # 4-neighbour unknown adjacency.
    touch = np.zeros_like(safe, dtype=bool)
    touch[1:, :] |= unknown[:-1, :]
    touch[:-1, :] |= unknown[1:, :]
    touch[:, 1:] |= unknown[:, :-1]
    touch[:, :-1] |= unknown[:, 1:]
    cells = np.argwhere(safe & touch)
    return [(int(r), int(c)) for r, c in cells]


def nearest_frontier(
    cells: list[tuple[int, int]],
    start: tuple[int, int],
) -> Optional[tuple[int, int]]:
    if not cells:
        return None
    sr, sc = start
    return min(cells, key=lambda rc: (rc[0] - sr) ** 2 + (rc[1] - sc) ** 2)


def downsample_frontiers(
    cells: list[tuple[int, int]],
    *,
    min_sep: int = 3,
    limit: int = 48,
) -> list[tuple[int, int]]:
    """Thin a frontier cloud so the viewer / planner is not flooded."""
    if not cells:
        return []
    picked: list[tuple[int, int]] = []
    sep2 = max(1, int(min_sep)) ** 2
    for cell in cells:
        if all((cell[0] - p[0]) ** 2 + (cell[1] - p[1]) ** 2 >= sep2 for p in picked):
            picked.append(cell)
        if len(picked) >= int(limit):
            break
    return picked


def fog_rgba(observed: np.ndarray) -> np.ndarray:
    """Owner fog veil: unknown is nearly opaque, observed cells are holes.

    Physics may still use the true height field. This raster is the owner
    map: unknown is not shown as a finished god-view mesh.
    """
    mask = np.asarray(observed, dtype=bool)
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[:, :] = (*FOG_RGB, FOG_ALPHA_UNKNOWN)
    rgba[mask] = (0, 0, 0, 0)
    return rgba[::-1]
