"""Extrinsics load + gym stereo calibration bench.

Software procedure and gym acceptance only. A human on the rig still
has to tape the 6–12 cm baseline and commit *measured* YAML. This
module does **not** invent field measurements, mAP, or FPS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import numpy as np
import yaml

from jims_mower.cameras import camera_world_pose, focal_length_px
from jims_mower.config import EnvConfig, load_config
from jims_mower.perception.stereo import (
    STEREO_BASELINE_MAX_M,
    STEREO_BASELINE_MIN_M,
    STEREO_Z_MAX_M,
    STEREO_Z_MIN_M,
    StereoPair,
    StereoPairError,
    baseline_cm,
    disparity_px,
    find_stereo_pair,
    range_from_disparity,
    rasterize_points,
    require_stereo_pair,
    synthetic_stereo_points,
)
from jims_mower.types import CameraSpec, Pose

if TYPE_CHECKING:
    from jims_mower.planning.observed import ObservedMap

# Field band, centimetres. Same as stereo.py metres.
STEREO_BASELINE_MIN_CM = STEREO_BASELINE_MIN_M * 100.0
STEREO_BASELINE_MAX_CM = STEREO_BASELINE_MAX_M * 100.0

# Gym tape vs ideal-disparity identity. Not a matcher residual.
TAPE_IDENTITY_TOL_M = 0.02
# Lip cell landing: one ObservedMap cell at the fixture resolution.
LIP_CELL_TOL_CELLS = 1
# Ray-march reconstruction vs authored tape to the *near face*.
# One 0.25 m cell plus near-edge clustering; not a matcher residual.
LIP_TAPE_TOL_M = 0.20

LABEL_EXAMPLE = "EXAMPLE"
LABEL_TEMPLATE = "TEMPLATE"
LABEL_MEASURED = "MEASURED"

CHECKLIST = (
    "Tape the forward stereo baseline (6–12 cm) between optical centres.",
    "Mount / shoot a checkerboard or drain-lip fixture in the 0.8–4 m band.",
    "Copy configs/orin/extrinsics_stereo.yaml → extrinsics_stereo_measured.yaml.",
    "Replace every pose with measured numbers; set calibration.measured: true.",
    "Set calibration.template: false and fill tape_baseline_cm / measured_at.",
    "Verify disparity vs tape at 0.8, 2, and 4 m (or 1.5 / 2.5 / 3.5 m).",
    "Commit the measured YAML only after the tape check. Do not invent numbers.",
)


class CalibrationError(ValueError):
    """Extrinsics file is missing, not a pair, or mis-labelled."""


@dataclass(frozen=True)
class CalibrationMeta:
    measured: bool
    template: bool
    measured_at: str
    tape_baseline_cm: Optional[float]
    notes: str
    path: Path

    @property
    def kind(self) -> str:
        if self.template:
            return "template"
        if self.measured:
            return "measured"
        return "example"

    @property
    def label(self) -> str:
        if self.template:
            return LABEL_TEMPLATE
        if self.measured:
            return LABEL_MEASURED
        return LABEL_EXAMPLE


@dataclass
class ExtrinsicsBundle:
    path: Path
    cameras: list[CameraSpec]
    pair: Optional[StereoPair]
    meta: CalibrationMeta
    cfg: EnvConfig

    @property
    def baseline_cm(self) -> Optional[float]:
        if self.pair is None:
            return None
        return baseline_cm(self.pair)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class ValidationReport:
    path: Path
    label: str
    kind: str
    baseline_cm: Optional[float]
    pair_names: Optional[tuple[str, str]]
    checks: list[Check]
    measured: bool
    template: bool
    not_field_measurement: bool = True

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "label": self.label,
            "kind": self.kind,
            "baseline_cm": self.baseline_cm,
            "pair": list(self.pair_names) if self.pair_names else None,
            "ok": self.ok,
            "measured": self.measured,
            "template": self.template,
            "not_field_measurement": True,
            "fps_claim": None,
            "map_claim": None,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
        }


@dataclass
class TapeCheck:
    tape_m: float
    recon_m: float
    disparity_px: float
    abs_err_m: float
    tolerance_m: float

    @property
    def ok(self) -> bool:
        return self.abs_err_m <= self.tolerance_m


@dataclass
class LipFixtureResult:
    tape_m: float
    lip_height_m: float
    expected_cells: list[tuple[int, int]]
    hit_cells: list[tuple[int, int]]
    cells_stamped: int
    mean_range_m: Optional[float]
    tape_err_m: Optional[float]
    range_ok: bool
    cells_ok: bool
    tolerance_m: float
    not_colmap: bool = True
    fps_claim: Optional[float] = None
    map_claim: Optional[float] = None

    @property
    def ok(self) -> bool:
        return bool(self.cells_ok and self.range_ok)


def example_extrinsics_path() -> Path:
    """Documented EXAMPLE file (not measured)."""
    packaged = Path(__file__).resolve().parents[1] / "data" / "orin" / "extrinsics_stereo.yaml"
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[3] / "configs" / "orin" / "extrinsics_stereo.yaml"


def measured_template_path() -> Path:
    """Copy-this MEASURED template. Placeholder numbers, ``template: true``."""
    packaged = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "orin"
        / "extrinsics_stereo_measured.template.yaml"
    )
    if packaged.is_file():
        return packaged
    return (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "orin"
        / "extrinsics_stereo_measured.template.yaml"
    )


def _meta_from_cfg(cfg: EnvConfig, path: Path) -> CalibrationMeta:
    cal = cfg.calibration
    tape = cal.tape_baseline_cm
    return CalibrationMeta(
        measured=bool(cal.measured),
        template=bool(cal.template),
        measured_at=str(cal.measured_at or ""),
        tape_baseline_cm=None if tape is None else float(tape),
        notes=str(cal.notes or ""),
        path=path,
    )


def load_extrinsics(source: Union[str, Path, EnvConfig]) -> ExtrinsicsBundle:
    """Load cameras + calibration provenance. Rejects nothing by itself."""
    if isinstance(source, EnvConfig):
        cfg = source
        path = Path("<config>")
    else:
        path = Path(source)
        if not path.is_file():
            raise CalibrationError(f"extrinsics file not found: {path}")
        cfg = load_config(path)
    cameras = list(cfg.resolved_cameras())
    pair = find_stereo_pair(cameras)
    return ExtrinsicsBundle(
        path=path,
        cameras=cameras,
        pair=pair,
        meta=_meta_from_cfg(cfg, path),
        cfg=cfg,
    )


def report_baseline_cm(cameras: Iterable[CameraSpec]) -> float:
    """Baseline centimetres between the accepted stereo pair. Raises on non-pairs."""
    return baseline_cm(require_stereo_pair(cameras))


def _is_example_filename(path: Path) -> bool:
    name = path.name.lower()
    return name == "extrinsics_stereo.yaml"


def _is_measured_filename(path: Path) -> bool:
    name = path.name.lower()
    return name == "extrinsics_stereo_measured.yaml"


def _is_template_filename(path: Path) -> bool:
    name = path.name.lower()
    return "template" in name and "measured" in name


def validate_stereo_yaml(source: Union[str, Path, ExtrinsicsBundle]) -> ValidationReport:
    """Validate a YAML file's stereo pair + EXAMPLE/MEASURED labelling.

    Does **not** claim the numbers were taped on hardware.
    """
    bundle = source if isinstance(source, ExtrinsicsBundle) else load_extrinsics(source)
    checks: list[Check] = []
    names = {c.name for c in bundle.cameras}
    has_named = "stereo_left" in names and "stereo_right" in names
    checks.append(
        Check(
            "named_pair",
            has_named,
            "stereo_left + stereo_right present" if has_named else "missing stereo_left/stereo_right",
        )
    )
    if bundle.pair is None:
        checks.append(
            Check(
                "stereo_pair",
                False,
                "find_stereo_pair rejected this rig (look-around or baseline out of 6–12 cm)",
            )
        )
        pair_names = None
        b_cm = None
    else:
        pair_names = (bundle.pair.left.name, bundle.pair.right.name)
        b_cm = baseline_cm(bundle.pair)
        in_band = STEREO_BASELINE_MIN_CM - 1e-6 <= b_cm <= STEREO_BASELINE_MAX_CM + 1e-6
        checks.append(
            Check(
                "stereo_pair",
                True,
                f"{pair_names[0]} / {pair_names[1]}",
            )
        )
        checks.append(
            Check(
                "baseline_band",
                in_band,
                f"{b_cm:.2f} cm (band {STEREO_BASELINE_MIN_CM:.0f}–{STEREO_BASELINE_MAX_CM:.0f} cm)",
            )
        )

    example_name = _is_example_filename(bundle.path)
    measured_name = _is_measured_filename(bundle.path)
    template_name = _is_template_filename(bundle.path)
    if example_name:
        ok_label = (not bundle.meta.measured) and (not bundle.meta.template)
        checks.append(
            Check(
                "example_not_measured",
                ok_label,
                "EXAMPLE file must keep calibration.measured: false"
                if ok_label
                else "EXAMPLE extrinsics_stereo.yaml must not be marked measured",
            )
        )
    if template_name:
        ok_t = bundle.meta.template and bundle.meta.measured
        checks.append(
            Check(
                "measured_template_flag",
                ok_t,
                "template file has measured: true and template: true"
                if ok_t
                else "measured template must set calibration.measured and calibration.template",
            )
        )
    if measured_name and not bundle.meta.template:
        checks.append(
            Check(
                "measured_filename_flag",
                bundle.meta.measured,
                "extrinsics_stereo_measured.yaml should set calibration.measured: true"
                if bundle.meta.measured
                else "measured filename is missing calibration.measured: true",
            )
        )
    if bundle.meta.measured and not bundle.meta.template and bundle.meta.tape_baseline_cm is None:
        checks.append(
            Check(
                "tape_baseline_recorded",
                False,
                "measured YAML should record calibration.tape_baseline_cm after the bench",
            )
        )
    elif bundle.meta.measured and bundle.meta.tape_baseline_cm is not None:
        tape = float(bundle.meta.tape_baseline_cm)
        tape_ok = STEREO_BASELINE_MIN_CM - 1e-6 <= tape <= STEREO_BASELINE_MAX_CM + 1e-6
        checks.append(
            Check(
                "tape_baseline_recorded",
                tape_ok,
                f"tape_baseline_cm={tape:.2f} in the 6–12 cm band"
                if tape_ok
                else f"tape_baseline_cm={tape:.2f} is outside 6–12 cm",
            )
        )

    return ValidationReport(
        path=bundle.path,
        label=bundle.meta.label,
        kind=bundle.meta.kind,
        baseline_cm=b_cm,
        pair_names=pair_names,
        checks=checks,
        measured=bundle.meta.measured,
        template=bundle.meta.template,
    )


def tape_vs_ideal_disparity(
    tape_m: float,
    pair: StereoPair,
    *,
    width: int = 80,
    fov_deg: float = 70.0,
    tolerance_m: float = TAPE_IDENTITY_TOL_M,
) -> TapeCheck:
    """Ideal ``d = f B / Z`` vs tape. Gym identity — not a matcher score."""
    if tape_m < STEREO_Z_MIN_M - 1e-6 or tape_m > STEREO_Z_MAX_M + 1e-6:
        raise CalibrationError(
            f"tape {tape_m:.2f} m is outside the 0.8–4 m near-field band"
        )
    fx = focal_length_px(int(width), float(fov_deg))
    disp = disparity_px(range_m=tape_m, baseline_m=pair.baseline_m, focal_px=fx)
    recon = range_from_disparity(disparity_px=disp, baseline_m=pair.baseline_m, focal_px=fx)
    return TapeCheck(
        tape_m=float(tape_m),
        recon_m=float(recon),
        disparity_px=float(disp),
        abs_err_m=abs(float(recon) - float(tape_m)),
        tolerance_m=float(tolerance_m),
    )


def vertical_board_principal_range(*, tape_m: float, pitch_deg: float) -> float:
    """Range along a pitched principal ray to a vertical board at horizontal tape.

    Tape is camera-midpoint → board (horizontal). The optical axis is
    pitched down, so the ray range is ``tape / cos(pitch)``.
    """
    pitch = math.radians(float(pitch_deg))
    denom = max(abs(math.cos(pitch)), 1e-6)
    return float(tape_m) / denom


@dataclass
class LipFixture:
    """Known drain-lip / checkerboard pad at a taped distance on +x."""

    tape_m: float = 2.0
    lip_height_m: float = 0.12
    lip_thickness_m: float = 0.30
    world_m: float = 8.0
    resolution_m: float = 0.25
    robot: Pose = field(default_factory=lambda: Pose(2.0, 4.0, 0.0, z=0.10))


def _lip_x_band(fix: LipFixture) -> tuple[float, float]:
    """Near face at ``robot.x + tape_m`` (tape-to-board), then thickness."""
    x0 = float(fix.robot.x) + float(fix.tape_m)
    x1 = x0 + float(fix.lip_thickness_m)
    return x0, x1


def build_lip_elevation(fix: LipFixture) -> np.ndarray:
    """Flat yard plus a raised strip at ``robot.x + tape_m`` (drain lip / board)."""
    res = max(float(fix.resolution_m), 1e-6)
    rows = int(round(fix.world_m / res))
    cols = int(round(fix.world_m / res))
    elev = np.full((rows, cols), float(fix.robot.z), dtype=np.float32)
    x0, x1 = _lip_x_band(fix)
    y0 = float(fix.robot.y) - 0.80
    y1 = float(fix.robot.y) + 0.80
    for rr in range(rows):
        y = (rr + 0.5) * res
        if y < y0 or y > y1:
            continue
        for cc in range(cols):
            x = (cc + 0.5) * res
            if x0 <= x <= x1:
                elev[rr, cc] = float(fix.robot.z) + float(fix.lip_height_m)
    return elev


def expected_lip_cells(fix: LipFixture) -> list[tuple[int, int]]:
    res = max(float(fix.resolution_m), 1e-6)
    rows = int(round(fix.world_m / res))
    cols = int(round(fix.world_m / res))
    x0, x1 = _lip_x_band(fix)
    y0 = float(fix.robot.y) - 0.40
    y1 = float(fix.robot.y) + 0.40
    cells: list[tuple[int, int]] = []
    for rr in range(rows):
        y = (rr + 0.5) * res
        if y < y0 or y > y1:
            continue
        for cc in range(cols):
            x = (cc + 0.5) * res
            if x0 <= x <= x1:
                cells.append((rr, cc))
    return cells


def run_lip_fixture(
    pair: StereoPair,
    fix: Optional[LipFixture] = None,
    *,
    width: int = 80,
    height: int = 60,
    tolerance_m: float = LIP_TAPE_TOL_M,
) -> LipFixtureResult:
    """Stamp a known lip and compare reconstructed range / cells to tape.

    Synthetic / ideal gym stereo — correspondence is the true ray hit.
    """
    fixture = fix or LipFixture()
    if fixture.tape_m < STEREO_Z_MIN_M or fixture.tape_m > STEREO_Z_MAX_M:
        raise CalibrationError(
            f"lip tape {fixture.tape_m:.2f} m is outside the 0.8–4 m band"
        )
    elev = build_lip_elevation(fixture)
    res = float(fixture.resolution_m)
    world = float(fixture.world_m)

    def height_at(x: float, y: float) -> float:
        if x < 0.0 or y < 0.0 or x >= world or y >= world:
            return float(fixture.robot.z)
        rr = int(y / res)
        cc = int(x / res)
        if 0 <= rr < elev.shape[0] and 0 <= cc < elev.shape[1]:
            return float(elev[rr, cc])
        return float(fixture.robot.z)

    xs, ys, zs = synthetic_stereo_points(
        fixture.robot,
        pair,
        width=int(width),
        height=int(height),
        height_at=height_at,
        pixel_stride=2,
    )
    raster, hits = rasterize_points(
        xs,
        ys,
        zs,
        shape=elev.shape,
        resolution_m=res,
        width_m=world,
        height_m=world,
    )
    from jims_mower.planning.observed import ObservedMap

    omap = ObservedMap.empty(world, world, res)
    omap.stamp_disk(fixture.robot.x, fixture.robot.y, max(fixture.tape_m + 1.2, 2.5))
    written = omap.stamp_metric_elevation(raster, hits, respect_lock=True)

    expected = expected_lip_cells(fixture)
    hit_cells = [(int(r), int(c)) for r, c in zip(*np.where(hits))]
    expected_set = set(expected)
    # Allow one-cell slop: a hit next to an expected cell still counts.
    cells_ok = False
    for rr, cc in expected:
        if hits[rr, cc]:
            cells_ok = True
            break
        for dr in (-LIP_CELL_TOL_CELLS, 0, LIP_CELL_TOL_CELLS):
            for dc in (-LIP_CELL_TOL_CELLS, 0, LIP_CELL_TOL_CELLS):
                r2, c2 = rr + dr, cc + dc
                if 0 <= r2 < hits.shape[0] and 0 <= c2 < hits.shape[1] and hits[r2, c2]:
                    cells_ok = True
                    break
            if cells_ok:
                break
        if cells_ok:
            break

    cam = camera_world_pose(fixture.robot, pair.left)
    origin = np.array([cam.x, cam.y, cam.z], dtype=np.float64)
    x0, x1 = _lip_x_band(fixture)
    ranges: list[float] = []
    for x, y, z in zip(xs.tolist(), ys.tolist(), zs.tolist()):
        if x0 - res <= x <= x1 + res and abs(y - fixture.robot.y) <= 0.90:
            ranges.append(float(np.linalg.norm(np.array([x, y, z]) - origin)))
    mean_range = float(np.mean(ranges)) if ranges else None
    # Horizontal tape from the stereo-pair midpoint (body) to the lip centre.
    # Disparity range is the ray length; compare the reconstructed *ground*
    # distance from the robot to the lip against the authored tape.
    if xs.size:
        lip_mask = (xs >= x0 - res) & (xs <= x1 + res)
        if np.any(lip_mask):
            mean_x = float(np.mean(xs[lip_mask]))
            horiz = mean_x - float(fixture.robot.x)
            tape_err = abs(horiz - float(fixture.tape_m))
        else:
            tape_err = None
    else:
        tape_err = None
    range_ok = tape_err is not None and tape_err <= float(tolerance_m)

    return LipFixtureResult(
        tape_m=float(fixture.tape_m),
        lip_height_m=float(fixture.lip_height_m),
        expected_cells=expected,
        hit_cells=hit_cells,
        cells_stamped=int(written),
        mean_range_m=mean_range,
        tape_err_m=tape_err,
        range_ok=range_ok,
        cells_ok=bool(cells_ok and expected_set),
        tolerance_m=float(tolerance_m),
    )


def run_gym_acceptance(
    source: Union[str, Path, ExtrinsicsBundle, None] = None,
    *,
    tapes_m: tuple[float, ...] = (1.5, 2.0, 2.5, 3.5),
) -> dict[str, Any]:
    """Full gym bench: YAML pair + disparity-vs-tape + lip cells."""
    bundle = (
        source
        if isinstance(source, ExtrinsicsBundle)
        else load_extrinsics(source or example_extrinsics_path())
    )
    if bundle.pair is None:
        raise StereoPairError("gym acceptance needs a 6–12 cm stereo pair")
    pair = bundle.pair
    fov = float(bundle.cfg.sensors.fov_deg)
    width = int(bundle.cfg.sensors.width)
    tapes = [tape_vs_ideal_disparity(t, pair, width=width, fov_deg=fov) for t in tapes_m]
    # Vertical board: principal-ray range must stay in-band and identity-close.
    pitch = float(pair.left.pitch_deg)
    board: list[dict[str, Any]] = []
    for t in tapes_m:
        ray = vertical_board_principal_range(tape_m=t, pitch_deg=pitch)
        if ray < STEREO_Z_MIN_M or ray > STEREO_Z_MAX_M:
            continue
        chk = tape_vs_ideal_disparity(ray, pair, width=width, fov_deg=fov)
        board.append(
            {
                "horizontal_tape_m": t,
                "principal_range_m": ray,
                "recon_m": chk.recon_m,
                "ok": chk.ok,
            }
        )
    lip = run_lip_fixture(pair, LipFixture(tape_m=2.0), width=width, height=int(bundle.cfg.sensors.height))
    report = validate_stereo_yaml(bundle)
    return {
        "yaml": report.as_dict(),
        "tape_identity": [
            {
                "tape_m": c.tape_m,
                "recon_m": c.recon_m,
                "disparity_px": c.disparity_px,
                "abs_err_m": c.abs_err_m,
                "ok": c.ok,
            }
            for c in tapes
        ],
        "vertical_board": board,
        "lip": {
            "tape_m": lip.tape_m,
            "cells_stamped": lip.cells_stamped,
            "cells_ok": lip.cells_ok,
            "range_ok": lip.range_ok,
            "tape_err_m": lip.tape_err_m,
            "mean_range_m": lip.mean_range_m,
            "tolerance_m": lip.tolerance_m,
            "ok": lip.ok,
            "not_colmap": True,
            "fps_claim": None,
            "map_claim": None,
        },
        "ok": bool(report.ok and all(c.ok for c in tapes) and lip.ok and all(b["ok"] for b in board)),
        "not_field_measurement": True,
        "fps_claim": None,
        "map_claim": None,
    }


def dump_yaml_header_kind(path: Path) -> str:
    """Read the EXAMPLE / MEASURED / TEMPLATE token from the file header."""
    text = Path(path).read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:12]).upper()
    if "TEMPLATE" in head:
        return LABEL_TEMPLATE
    if "MEASURED" in head and "EXAMPLE" not in head:
        return LABEL_MEASURED
    if "EXAMPLE" in head:
        return LABEL_EXAMPLE
    raw = yaml.safe_load(text) or {}
    cal = raw.get("calibration") or {}
    if cal.get("template"):
        return LABEL_TEMPLATE
    if cal.get("measured"):
        return LABEL_MEASURED
    return LABEL_EXAMPLE
