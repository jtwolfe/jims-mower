"""Real-yard import stub: survey polygon JSON → geofence + drain polylines.

This does not georeference a GIS file. A surveyor (or a phone walk) dumps
a local-metre polygon plus drain polylines; we turn that into a WAVE 1A
``Scenario`` the gym already understands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from jims_mower.constants import SURVEY_SCHEMA
from jims_mower.scenarios import Scenario, parse_scenario
from jims_mower.terrain import DrainFeature


class SurveyError(ValueError):
    """Invalid survey JSON."""


def _xy(item: Any, *, field: str) -> tuple[float, float]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return float(item[0]), float(item[1])
    if isinstance(item, dict) and "x" in item and "y" in item:
        return float(item["x"]), float(item["y"])
    raise SurveyError(f"{field} entries must be [x, y] or {{x, y}}")


def _polygon(raw: Any, *, field: str) -> list[tuple[float, float]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise SurveyError(f"{field} must be a list of vertices")
    poly = [_xy(p, field=field) for p in raw]
    if poly and len(poly) < 3:
        raise SurveyError(f"{field} needs at least 3 vertices")
    return poly


def drains_from_polyline(
    points: list[tuple[float, float]],
    *,
    width_m: float = 0.40,
    depth_m: float = 0.16,
    side_slope: float = 1.5,
    kind: str = "drain",
) -> list[DrainFeature]:
    """Consecutive vertices become drain segments."""
    if len(points) < 2:
        raise SurveyError("drain polyline needs at least 2 points")
    out: list[DrainFeature] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        out.append(
            DrainFeature(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                width_m=float(width_m),
                depth_m=float(depth_m),
                side_slope=float(side_slope),
                kind=kind,
            )
        )
    return out


def survey_to_scenario_dict(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SurveyError("survey JSON must be a mapping")
    schema = str(data.get("schema") or "").strip()
    if schema and schema != SURVEY_SCHEMA:
        raise SurveyError(f"unsupported survey schema {schema!r}; expected {SURVEY_SCHEMA}")
    if "schema" not in data:
        raise SurveyError(f"survey JSON missing required 'schema' field ({SURVEY_SCHEMA})")
    name = str(data.get("name") or "imported").strip() or "imported"
    geofence = _polygon(data.get("geofence") or data.get("keep_in"), field="geofence")
    drains_raw = data.get("drains") or []
    if not isinstance(drains_raw, list):
        raise SurveyError("drains must be a list")
    drains: list[dict[str, Any]] = []
    for i, item in enumerate(drains_raw):
        if not isinstance(item, dict):
            raise SurveyError(f"drains[{i}] must be a mapping")
        if "polyline" in item:
            pts = [_xy(p, field=f"drains[{i}].polyline") for p in item["polyline"]]
            segs = drains_from_polyline(
                pts,
                width_m=float(item.get("width_m", 0.40)),
                depth_m=float(item.get("depth_m", 0.16)),
                side_slope=float(item.get("side_slope", 1.5)),
                kind=str(item.get("kind", "drain")),
            )
            for seg in segs:
                drains.append(
                    {
                        "x0": seg.x0,
                        "y0": seg.y0,
                        "x1": seg.x1,
                        "y1": seg.y1,
                        "width_m": seg.width_m,
                        "depth_m": seg.depth_m,
                        "side_slope": seg.side_slope,
                        "kind": seg.kind,
                    }
                )
        else:
            required = ("x0", "y0", "x1", "y1")
            missing = [k for k in required if k not in item]
            if missing:
                raise SurveyError(f"drains[{i}] missing {missing} (or use polyline)")
            drains.append(dict(item))
    width = float(data.get("width_m") or 0.0)
    height = float(data.get("height_m") or 0.0)
    if geofence and (width <= 0.0 or height <= 0.0):
        xs = [p[0] for p in geofence]
        ys = [p[1] for p in geofence]
        width = max(xs) + 0.8
        height = max(ys) + 0.8
    payload: dict[str, Any] = {
        "kind": "scenario",
        "name": name,
        "description": str(data.get("description") or "Imported survey polygon"),
        "base": data.get("base") or "default",
        "geofence": [list(p) for p in geofence],
        "drains": drains,
        "world": {
            "width_m": width or 12.0,
            "height_m": height or 12.0,
            "resolution_m": float(data.get("resolution_m") or 0.20),
            "n_people": 0,
            "n_dogs": 0,
            "n_cats": 0,
            "n_birds": 0,
            "n_trees": int(data.get("n_trees") or 0),
            "n_furniture": 0,
            "n_toys": 0,
            "terrain": {"enabled": True, "n_drains": 0, "n_banks": 0},
        },
    }
    return payload


def load_survey(source: Union[str, Path, dict]) -> Scenario:
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.is_file():
            raise SurveyError(f"survey JSON not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
    return parse_scenario(survey_to_scenario_dict(data))


def build_parser():
    import argparse

    p = argparse.ArgumentParser(description="Import a survey polygon JSON as a jims-mower scenario")
    p.add_argument("survey", type=Path, help="survey JSON (schema jims_mower.survey.v1)")
    p.add_argument("--out", type=Path, default=None, help="optional scenario YAML dump")
    return p


def main(argv=None) -> None:
    import yaml

    args = build_parser().parse_args(argv)
    scenario = load_survey(args.survey)
    print(f"imported {scenario.name} drains={len(scenario.drains)} geofence={len(scenario.geofence)}")
    if args.out is not None:
        payload = survey_to_scenario_dict(json.loads(Path(args.survey).read_text(encoding="utf-8")))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
