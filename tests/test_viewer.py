"""World Viewer static assets + bundle from a demo/teach run."""

from __future__ import annotations

import json
from pathlib import Path

from jims_mower.demo import run_demo
from jims_mower.viewer import (
    STATIC_NAMES,
    prepare_viewer_dir,
    static_dir,
    viewer_assets_present,
)
from jims_mower.viewer import main as viewer_main


def test_viewer_static_assets_present() -> None:
    root = static_dir()
    assert root.is_dir()
    for name in STATIC_NAMES:
        path = root / name
        assert path.is_file(), path
        assert path.stat().st_size > 20
    assert viewer_assets_present()
    html = (root / "index.html").read_text(encoding="utf-8")
    assert "three" in html.lower()
    assert "World Viewer" in html
    js = (root / "app.js").read_text(encoding="utf-8")
    assert "coverage" in js
    assert "keep_in" in js
    # Viewer-only Y lift so a property-scale grade reads on a 12 m yard.
    assert "scale.y" in js
    assert "sampleElev" in js
    assert "tog-error" in html
    assert "relief" in js
    assert "phase-bar" in html
    assert "CALIBRATE" in html
    assert "mission-metrics" in html
    assert "tog-observed" in html
    assert "tog-fog" in html
    assert "tog-god" in html
    assert "owner-bar" in html
    assert "btn-job-start" in html
    assert "btn-estop" in html
    assert "btn-start-mow" in html
    assert "owner-copy" in html
    assert "EventSource" in js
    assert "postControl" in js
    assert "/api/live/control" in js
    assert 'job_state === "idle"' in js
    assert "Yard unknown" in html
    assert "unknownPad" in js
    assert "setOwnerMeshVis" in js
    assert "applyObservedMesh" in js
    assert "observedTerrain" in js
    assert "sameVerts" in js
    assert "not a second IMU hinge" in js
    assert "observed_mesh" in js
    assert "observed_fog" in js or "fog of war" in html.lower() or "fog" in js


def test_demo_writes_viewer_bundle(tmp_path: Path) -> None:
    out = tmp_path / "demo"
    summary = run_demo(out, steps=3, seed=1, cameras=4, policy="scripted")
    assert summary["steps_run"] >= 1
    assert (out / "viewer.json").is_file()
    assert (out / "yard.glb").is_file()
    assert (out / "yard.json").is_file()
    assert (out / "poses.json").is_file()
    assert (out / "maps" / "coverage.png").is_file()
    assert (out / "maps" / "elevation.npy").is_file()
    manifest = json.loads((out / "viewer.json").read_text(encoding="utf-8"))
    assert manifest["schema"].startswith("jims_mower.viewer")
    assert manifest["vertex_count"] > 0
    assert manifest["triangle_count"] > 0
    assert manifest["not_a_benchmark"] is True
    assert manifest["health"]["placeholder"] is True
    assert manifest["radio"]["placeholder"] is True
    glb = (out / "yard.glb").read_bytes()
    assert glb[:4] == b"glTF"
    assert len(glb) > 200


def test_viewer_prepare_only(tmp_path: Path) -> None:
    out = tmp_path / "ep"
    run_demo(out, steps=2, seed=2, cameras=4, policy="scripted")
    viewer_main(["--episode", str(out), "--prepare-only"])
    again = prepare_viewer_dir(out)
    assert again["vertex_count"] > 0
    assert (out / "viewer.json").is_file()


def test_demo_teach_writes_profile(tmp_path: Path) -> None:
    out = tmp_path / "teach-demo"
    summary = run_demo(out, steps=5, seed=4, cameras=4, policy="teach")
    assert summary["policy"] == "teach"
    assert (out / "profile.json").is_file()
    assert summary.get("keep_in_vertices", 0) >= 3
    profile = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    assert profile["schema"].startswith("jims_mower.yard")
    assert len(profile["keep_in"]) >= 3
