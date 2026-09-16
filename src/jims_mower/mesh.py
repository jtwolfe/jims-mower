"""Low-poly terrain mesh from an elevation / hazard field.

Writes glTF/GLB (binary, no extra deps) plus OBJ + a JSON sidecar the
World Viewer can load without a GLTF parser. Decimation is a regular
stride over the height raster — not a claimed mesh-quality metric.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from jims_mower.constants import (
    BUILDING_RGB,
    BUNKER_RGB,
    CUT_GRASS_RGB,
    DIRT_RGB,
    GARDEN_RGB,
    GREEN_RGB,
    HAZARD_DRAIN,
    HAZARD_DRAIN_EDGE,
    HAZARD_NONE,
    HAZARD_STEEP,
    MESH_SCHEMA,
    PATH_RGB,
    POND_RGB,
    STRUCTURE_BUILDING,
    STRUCTURE_BUNKER,
    STRUCTURE_GARDEN,
    STRUCTURE_GREEN,
    STRUCTURE_PATH,
    STRUCTURE_POND,
    UNCUT_GRASS_RGB,
)

HAZARD_RGB = {
    HAZARD_NONE: (52, 128, 62),
    HAZARD_STEEP: (210, 168, 48),
    HAZARD_DRAIN_EDGE: (196, 96, 36),
    HAZARD_DRAIN: (92, 48, 28),
}

STRUCTURE_MESH_RGB = {
    STRUCTURE_PATH: PATH_RGB,
    STRUCTURE_BUILDING: BUILDING_RGB,
    STRUCTURE_BUNKER: BUNKER_RGB,
    STRUCTURE_GARDEN: GARDEN_RGB,
    STRUCTURE_GREEN: GREEN_RGB,
    STRUCTURE_POND: POND_RGB,
}


@dataclass
class TerrainMesh:
    """Triangle mesh in metres. Y-up (three.js): X=world x, Y=elev, Z=world y."""

    positions: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    width_m: float
    height_m: float
    resolution_m: float
    stride: int = 1
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.size // 3)

    def is_empty(self) -> bool:
        return self.vertex_count < 3 or self.triangle_count < 1


def _row_col_indices(n: int, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    idx = list(range(0, n, stride))
    if not idx:
        idx = [0]
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return np.asarray(idx, dtype=int)


def _hazard_colors(hazard: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    if hazard is None or hazard.shape != shape:
        rgb = np.broadcast_to(
            np.asarray(HAZARD_RGB[HAZARD_NONE], dtype=np.float32) / 255.0,
            (rows, cols, 3),
        ).copy()
        return rgb
    hz = np.asarray(hazard)
    rgb = np.zeros((rows, cols, 3), dtype=np.float32)
    rgb[...] = np.asarray(HAZARD_RGB[HAZARD_NONE], dtype=np.float32) / 255.0
    for code, color in HAZARD_RGB.items():
        mask = np.isclose(hz, float(code)) if hz.dtype != np.uint8 else hz == code
        if code == HAZARD_DRAIN:
            mask = hz >= float(HAZARD_DRAIN) if hz.dtype != np.uint8 else hz >= HAZARD_DRAIN
        rgb[mask] = np.asarray(color, dtype=np.float32) / 255.0
    return rgb


def mesh_from_elevation(
    elevation: np.ndarray,
    *,
    width_m: float,
    height_m: float,
    resolution_m: float,
    hazard: Optional[np.ndarray] = None,
    coverage: Optional[np.ndarray] = None,
    structure: Optional[np.ndarray] = None,
    stride: int = 2,
    z_scale: float = 1.0,
) -> TerrainMesh:
    """Decimate a height raster into a low-poly triangle mesh."""
    elev = np.asarray(elevation, dtype=np.float32)
    if elev.ndim != 2 or elev.size == 0:
        raise ValueError("elevation must be a non-empty 2-D array")
    rows, cols = elev.shape
    res = max(float(resolution_m), 1e-6)
    row_i = _row_col_indices(rows, stride)
    col_i = _row_col_indices(cols, stride)
    sub = elev[np.ix_(row_i, col_i)]
    rr, cc = sub.shape
    xs = (col_i.astype(np.float32) + 0.5) * res
    ys = (row_i.astype(np.float32) + 0.5) * res
    grid_x, grid_y = np.meshgrid(xs, ys)
    pos = np.stack(
        [
            grid_x.reshape(-1),
            (sub * float(z_scale)).reshape(-1),
            grid_y.reshape(-1),
        ],
        axis=1,
    ).astype(np.float32)

    colors_grid = _hazard_colors(hazard, (rows, cols))
    if coverage is not None and coverage.shape == (rows, cols):
        cov = np.asarray(coverage, dtype=np.float32)
        cut = cov >= 0.5
        uncut = (cov >= 0.0) & (cov < 0.5)
        dirt = cov < 0.0
        colors_grid = colors_grid.copy()
        colors_grid[cut] = np.asarray(CUT_GRASS_RGB, dtype=np.float32) / 255.0
        colors_grid[uncut] = np.asarray(UNCUT_GRASS_RGB, dtype=np.float32) / 255.0
        colors_grid[dirt] = np.asarray(DIRT_RGB, dtype=np.float32) / 255.0
    if structure is not None and np.asarray(structure).shape == (rows, cols):
        st = np.asarray(structure)
        colors_grid = colors_grid.copy()
        for code, color in STRUCTURE_MESH_RGB.items():
            colors_grid[st == code] = np.asarray(color, dtype=np.float32) / 255.0
    colors = colors_grid[np.ix_(row_i, col_i)].reshape(-1, 3).astype(np.float32)

    uvs = np.stack(
        [
            np.clip(grid_x.reshape(-1) / max(float(width_m), 1e-6), 0.0, 1.0),
            np.clip(grid_y.reshape(-1) / max(float(height_m), 1e-6), 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)

    quads: list[int] = []
    for r in range(rr - 1):
        for c in range(cc - 1):
            i00 = r * cc + c
            i10 = r * cc + (c + 1)
            i01 = (r + 1) * cc + c
            i11 = (r + 1) * cc + (c + 1)
            quads.extend((i00, i01, i11, i00, i11, i10))
    indices = np.asarray(quads, dtype=np.uint32)
    normals = _vertex_normals(pos, indices, rr, cc)
    return TerrainMesh(
        positions=pos,
        normals=normals,
        colors=colors,
        uvs=uvs,
        indices=indices,
        width_m=float(width_m),
        height_m=float(height_m),
        resolution_m=res,
        stride=max(1, int(stride)),
        extras={"z_scale": float(z_scale), "grid": [int(rr), int(cc)]},
    )


def _vertex_normals(
    positions: np.ndarray,
    indices: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float32)
    if indices.size < 3:
        normals[:, 1] = 1.0
        return normals
    tri = positions[indices.reshape(-1, 3)]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    face = np.cross(e1, e2)
    for k in range(3):
        np.add.at(normals, indices.reshape(-1, 3)[:, k], face)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(lens, 1e-9))
    # Degenerate verts (grid corners with no area) point up.
    missing = lens.reshape(-1) < 1e-9
    normals[missing] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return normals.astype(np.float32)


def mesh_to_payload(mesh: TerrainMesh) -> dict[str, Any]:
    return {
        "schema": MESH_SCHEMA,
        "width_m": mesh.width_m,
        "height_m": mesh.height_m,
        "resolution_m": mesh.resolution_m,
        "stride": mesh.stride,
        "vertex_count": mesh.vertex_count,
        "triangle_count": mesh.triangle_count,
        "positions": mesh.positions.reshape(-1).astype(float).tolist(),
        "normals": mesh.normals.reshape(-1).astype(float).tolist(),
        "colors": mesh.colors.reshape(-1).astype(float).tolist(),
        "uvs": mesh.uvs.reshape(-1).astype(float).tolist(),
        "indices": mesh.indices.astype(int).tolist(),
        "up": "y",
        "not_a_benchmark": True,
        **mesh.extras,
    }


def write_mesh_json(mesh: TerrainMesh, path: Union[str, Path]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(mesh_to_payload(mesh), indent=2), encoding="utf-8")
    return dest


def write_obj(mesh: TerrainMesh, path: Union[str, Path]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# jims-mower low-poly terrain",
        f"# vertices={mesh.vertex_count} triangles={mesh.triangle_count}",
    ]
    for x, y, z in mesh.positions:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for nx, ny, nz in mesh.normals:
        lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")
    for u, v in mesh.uvs:
        lines.append(f"vt {u:.6f} {v:.6f}")
    idx = mesh.indices.reshape(-1, 3) + 1
    for a, b, c in idx:
        lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def _pad4(buf: bytes, pad: bytes) -> bytes:
    extra = (4 - (len(buf) % 4)) % 4
    return buf if extra == 0 else buf + pad * extra


def write_glb(mesh: TerrainMesh, path: Union[str, Path]) -> Path:
    """Minimal glTF 2.0 binary (one mesh, POSITION/NORMAL/COLOR_0/TEXCOORD_0)."""
    if mesh.is_empty():
        raise ValueError("refusing to write an empty mesh")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    pos = np.ascontiguousarray(mesh.positions, dtype=np.float32)
    nrm = np.ascontiguousarray(mesh.normals, dtype=np.float32)
    col = np.ascontiguousarray(mesh.colors, dtype=np.float32)
    uvs = np.ascontiguousarray(mesh.uvs, dtype=np.float32)
    # uint16 if it fits; otherwise uint32.
    if mesh.vertex_count <= 65535:
        idx = np.ascontiguousarray(mesh.indices, dtype=np.uint16)
        idx_type = 5123  # UNSIGNED_SHORT
    else:
        idx = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
        idx_type = 5125  # UNSIGNED_INT

    blobs = [pos.tobytes(), nrm.tobytes(), col.tobytes(), uvs.tobytes(), idx.tobytes()]
    parts: list[bytes] = []
    views: list[dict[str, Any]] = []
    cursor = 0
    for blob in blobs:
        aligned = _pad4(blob, b"\x00")
        views.append({"buffer": 0, "byteOffset": cursor, "byteLength": len(blob)})
        parts.append(aligned)
        cursor += len(aligned)
    binary = b"".join(parts)

    pos_min = pos.min(axis=0).tolist()
    pos_max = pos.max(axis=0).tolist()
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": mesh.vertex_count,
            "type": "VEC3",
            "min": pos_min,
            "max": pos_max,
        },
        {
            "bufferView": 1,
            "componentType": 5126,
            "count": mesh.vertex_count,
            "type": "VEC3",
        },
        {
            "bufferView": 2,
            "componentType": 5126,
            "count": mesh.vertex_count,
            "type": "VEC3",
        },
        {
            "bufferView": 3,
            "componentType": 5126,
            "count": mesh.vertex_count,
            "type": "VEC2",
        },
        {
            "bufferView": 4,
            "componentType": idx_type,
            "count": int(mesh.indices.size),
            "type": "SCALAR",
        },
    ]
    gltf = {
        "asset": {"version": "2.0", "generator": "jims-mower-mesh"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "yard"}],
        "meshes": [
            {
                "name": "terrain",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "COLOR_0": 2,
                            "TEXCOORD_0": 3,
                        },
                        "indices": 4,
                        "mode": 4,
                    }
                ],
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    json_bytes = _pad4(
        json.dumps(gltf, separators=(",", ":")).encode("utf-8"),
        b" ",
    )
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    header = struct.pack("<4sII", b"glTF", 2, total)
    json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    bin_chunk = struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    dest.write_bytes(header + json_chunk + bin_chunk)
    return dest


def export_mesh(
    mesh: TerrainMesh,
    out: Union[str, Path],
    *,
    also_obj: bool = True,
    also_json: bool = True,
) -> dict[str, str]:
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() in {".glb", ".gltf"}:
        glb_path = dest.with_suffix(".glb")
    else:
        glb_path = dest if dest.suffix else dest.with_suffix(".glb")
    written = {"glb": str(write_glb(mesh, glb_path))}
    if also_obj:
        written["obj"] = str(write_obj(mesh, glb_path.with_suffix(".obj")))
    if also_json:
        written["json"] = str(write_mesh_json(mesh, glb_path.with_suffix(".json")))
    return written


def coverage_to_rgb(coverage: np.ndarray) -> np.ndarray:
    cov = np.asarray(coverage, dtype=np.float32)
    rgb = np.zeros(cov.shape + (3,), dtype=np.uint8)
    rgb[cov < 0.0] = DIRT_RGB
    rgb[(cov >= 0.0) & (cov < 0.5)] = UNCUT_GRASS_RGB
    rgb[cov >= 0.5] = CUT_GRASS_RGB
    return rgb[::-1]


def occupancy_to_rgb(occupancy: np.ndarray) -> np.ndarray:
    occ = np.clip(np.asarray(occupancy, dtype=np.float32), 0.0, 1.0)
    rgb = np.zeros(occ.shape + (3,), dtype=np.uint8)
    rgb[:, :, 0] = (40 + 200 * occ).astype(np.uint8)
    rgb[:, :, 1] = (90 - 50 * occ).astype(np.uint8)
    rgb[:, :, 2] = (50 + 20 * occ).astype(np.uint8)
    return rgb[::-1]


def write_map_png(path: Union[str, Path], image: np.ndarray) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(dest)
    return dest


def write_map_npy(path: Union[str, Path], array: np.ndarray) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, np.asarray(array))
    return dest


def mesh_from_env(env: Any, *, stride: int = 2, z_scale: float = 1.0) -> TerrainMesh:
    terrain = env._terrain
    hazard = terrain.hazard_map(env.cfg.robot.steep_slope_rad)
    coverage = env._coverage.as_float()
    structure = None
    layer = getattr(env, "_structure", None)
    if layer is not None and getattr(layer, "grid", None) is not None:
        structure = np.asarray(layer.grid)
    return mesh_from_elevation(
        terrain.elevation,
        width_m=env.cfg.world.width_m,
        height_m=env.cfg.world.height_m,
        resolution_m=env.cfg.world.resolution_m,
        hazard=hazard,
        coverage=coverage,
        structure=structure,
        stride=stride,
        z_scale=z_scale,
    )


def export_mesh_from_env(
    env: Any,
    out: Union[str, Path],
    *,
    stride: int = 2,
    z_scale: float = 1.0,
) -> dict[str, str]:
    mesh = mesh_from_env(env, stride=stride, z_scale=z_scale)
    return export_mesh(mesh, out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export a low-poly yard mesh (GLB + OBJ + JSON)")
    p.add_argument("--out", type=Path, default=Path("yard.glb"))
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--stride", type=int, default=2, help="decimation stride over the height raster")
    p.add_argument("--z-scale", type=float, default=1.0)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument(
        "--episode",
        type=Path,
        default=None,
        help="optional demo/record dir with maps/elevation.npy or reset.npz",
    )
    return p


def _mesh_from_episode(episode: Path, *, stride: int, z_scale: float) -> TerrainMesh:
    npy = episode / "maps" / "elevation.npy"
    if npy.is_file():
        elev = np.load(npy)
        meta_path = episode / "viewer.json"
        width_m = height_m = 12.0
        resolution_m = 0.20
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            width_m = float(meta.get("width_m") or width_m)
            height_m = float(meta.get("height_m") or height_m)
            resolution_m = float(meta.get("resolution_m") or resolution_m)
        hazard = None
        haz_path = episode / "maps" / "hazard.npy"
        if haz_path.is_file():
            hazard = np.load(haz_path)
        coverage = None
        cov_path = episode / "maps" / "coverage.npy"
        if cov_path.is_file():
            coverage = np.load(cov_path)
        return mesh_from_elevation(
            elev,
            width_m=width_m,
            height_m=height_m,
            resolution_m=resolution_m,
            hazard=hazard,
            coverage=coverage,
            stride=stride,
            z_scale=z_scale,
        )
    from jims_mower.episode import EpisodeReader

    reader = EpisodeReader(episode)
    if reader.reset_obs is None or "elevation" not in reader.reset_obs:
        raise ValueError(f"episode {episode} has no elevation map")
    cfg = reader.config()
    return mesh_from_elevation(
        reader.reset_obs["elevation"],
        width_m=cfg.world.width_m,
        height_m=cfg.world.height_m,
        resolution_m=cfg.world.resolution_m,
        hazard=reader.reset_obs.get("hazard"),
        coverage=reader.reset_obs.get("coverage"),
        stride=stride,
        z_scale=z_scale,
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.episode is not None:
        mesh = _mesh_from_episode(args.episode, stride=args.stride, z_scale=args.z_scale)
    else:
        from jims_mower.env import MowerEnv
        from jims_mower.scenarios import load_source

        cfg, scenario = load_source(args.config)
        if args.cameras:
            cfg.sensors.camera_count = args.cameras
            cfg.sensors.cameras = []
        env = MowerEnv(config=cfg, scenario=scenario, render_mode=None)
        env.reset(seed=args.seed)
        mesh = mesh_from_env(env, stride=args.stride, z_scale=args.z_scale)
        env.close()
    written = export_mesh(mesh, args.out)
    print(
        f"mesh vertices={mesh.vertex_count} triangles={mesh.triangle_count} "
        f"→ {written['glb']}"
    )


if __name__ == "__main__":
    main()
