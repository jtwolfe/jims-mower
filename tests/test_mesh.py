"""Low-poly mesh export is non-empty (GLB + JSON)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jims_mower.constants import POND_RGB, STRUCTURE_POND
from jims_mower.mesh import (
    export_mesh,
    mesh_from_elevation,
    write_glb,
    write_mesh_json,
    write_obj,
)
from jims_mower.mesh import main as mesh_main


def _bump() -> np.ndarray:
    elev = np.zeros((16, 16), dtype=np.float32)
    elev[6:11, 6:11] = 0.35
    elev[2:4, 2:14] = -0.12
    return elev


def test_mesh_from_elevation_non_empty() -> None:
    mesh = mesh_from_elevation(
        _bump(),
        width_m=3.2,
        height_m=3.2,
        resolution_m=0.20,
        hazard=np.zeros((16, 16), dtype=np.float32),
        stride=2,
    )
    assert mesh.vertex_count >= 16
    assert mesh.triangle_count >= 8
    assert not mesh.is_empty()
    assert mesh.positions.shape == (mesh.vertex_count, 3)


def test_mesh_structure_colors_pond() -> None:
    elev = np.zeros((8, 8), dtype=np.float32)
    structure = np.zeros((8, 8), dtype=np.uint8)
    structure[2:5, 2:5] = STRUCTURE_POND
    mesh = mesh_from_elevation(
        elev,
        width_m=1.6,
        height_m=1.6,
        resolution_m=0.20,
        structure=structure,
        stride=1,
    )
    want = np.asarray(POND_RGB, dtype=np.float32) / 255.0
    assert np.any(np.linalg.norm(mesh.colors - want, axis=1) < 0.02)


def test_mesh_export_files(tmp_path: Path) -> None:
    mesh = mesh_from_elevation(
        _bump(),
        width_m=3.2,
        height_m=3.2,
        resolution_m=0.20,
        stride=2,
    )
    dest = tmp_path / "yard.glb"
    written = export_mesh(mesh, dest)
    glb = Path(written["glb"])
    assert glb.is_file()
    raw = glb.read_bytes()
    assert raw[:4] == b"glTF"
    assert len(raw) > 200
    assert Path(written["obj"]).is_file()
    assert Path(written["json"]).is_file()
    payload = __import__("json").loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert payload["vertex_count"] == mesh.vertex_count
    assert payload["triangle_count"] == mesh.triangle_count
    assert payload["not_a_benchmark"] is True


def test_write_obj_and_json_standalone(tmp_path: Path) -> None:
    mesh = mesh_from_elevation(
        np.zeros((8, 8), dtype=np.float32),
        width_m=1.6,
        height_m=1.6,
        resolution_m=0.20,
        stride=1,
    )
    obj = write_obj(mesh, tmp_path / "flat.obj")
    js = write_mesh_json(mesh, tmp_path / "flat.json")
    glb = write_glb(mesh, tmp_path / "flat.glb")
    assert "v " in obj.read_text(encoding="utf-8")
    assert js.stat().st_size > 20
    assert glb.stat().st_size > 100


def test_mesh_cli(tmp_path: Path) -> None:
    dest = tmp_path / "cli.glb"
    mesh_main(["--out", str(dest), "--seed", "1", "--cameras", "4", "--stride", "3"])
    assert dest.is_file()
    assert dest.with_suffix(".json").is_file()


def test_mesh_preserves_yard_gradient() -> None:
    elev = np.zeros((20, 20), dtype=np.float32)
    res = 0.20
    for col in range(20):
        elev[:, col] = 0.12 * ((col + 0.5) * res - 2.0)
    mesh = mesh_from_elevation(
        elev,
        width_m=4.0,
        height_m=4.0,
        resolution_m=res,
        stride=1,
    )
    xs = mesh.positions[:, 0]
    ys = mesh.positions[:, 1]
    assert float(ys.max() - ys.min()) > 0.35
    low = float(ys[xs < 0.6].mean())
    high = float(ys[xs > 3.4].mean())
    assert high > low + 0.25
