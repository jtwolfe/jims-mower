"""Unknown-space semantics: observed / free / hazard with explicit confidence.

Unknown cells are not safe and not mowable. The mission layer stamps camera
ground hits, a body/ToF disk, and IMU-local cues. Observer rasters are copied
only onto cells this robot has actually seen — authored/god-view structure
leaked into ``obs["structure"]`` is ignored outside the observed mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from jims_mower.cameras import attitude_plane_hits, camera_world_pose
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
POND_VIEW_RGB = POND_RGB
BUILDING_VIEW_RGB = BUILDING_RGB
FRONTIER_RGB = (42, 196, 220)
FOG_RGB = (16, 18, 22)
FOG_ALPHA_UNKNOWN = 236


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
    ) -> None:
        """Copy heuristic elevation/hazard onto seen cells. Not a god-view merge."""
        mask = self.observed if only_observed else np.ones_like(self.observed, dtype=bool)
        if not np.any(mask):
            return
        hazard = obs.get("hazard")
        if hazard is not None:
            hz = np.asarray(hazard, dtype=np.float32)
            if hz.shape == self.hazard.shape:
                self.hazard[mask] = np.maximum(self.hazard[mask], hz[mask])
        elev = obs.get("elevation")
        if elev is not None:
            ev = np.asarray(elev, dtype=np.float32)
            if ev.shape == self.elevation.shape:
                self.elevation[mask] = ev[mask]
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

    def refresh_free(self) -> None:
        """Known-free = observed, not a drain/steep-blocked hint, not hard structure."""
        hard = np.isin(
            self.structure,
            (STRUCTURE_BUILDING, STRUCTURE_BUNKER, STRUCTURE_GARDEN, STRUCTURE_GREEN),
        )
        # Channels are forbidden. Isolated lip stamps from the colour
        # heuristic are not a ditch — the costmap can still slow them.
        channel = self.hazard >= HAZARD_DRAIN
        self.free = self.observed & ~hard & ~channel

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
        blocked = blocked | hard | (self.hazard >= HAZARD_DRAIN)
        if keep_in is not None:
            blocked = blocked | ~np.asarray(keep_in, dtype=bool)
        return blocked

    def as_rgb(self, *, frontiers: Optional[list[tuple[int, int]]] = None) -> np.ndarray:
        """Unknown stays dark so exploration growth is visible frame-to-frame."""
        rgb = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        rgb[:, :] = UNKNOWN_RGB
        rgb[self.observed] = FREE_RGB
        rgb[self.explored] = EXPLORED_RGB
        rgb[self.observed & (self.hazard >= HAZARD_STEEP)] = (210, 168, 48)
        rgb[self.observed & (self.hazard >= HAZARD_DRAIN_EDGE)] = HAZARD_RGB
        rgb[self.observed & (self.structure != STRUCTURE_NONE)] = STRUCTURE_RGB
        rgb[self.observed & (self.structure == STRUCTURE_BUILDING)] = BUILDING_VIEW_RGB
        rgb[self.observed & (self.structure == STRUCTURE_POND)] = POND_VIEW_RGB
        if frontiers:
            for row, col in frontiers:
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    rgb[row, col] = FRONTIER_RGB
        return rgb[::-1]

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
            width_m=self.width_m,
            height_m=self.height_m,
            resolution_m=self.resolution_m,
        )

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
) -> list[tuple[int, int]]:
    """Known-free cells that touch unknown (and stay inside keep-in)."""
    unknown = ~np.asarray(observed, dtype=bool)
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
