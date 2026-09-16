"""Dataset schema versioning for WAVE 1A exporter folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from jims_mower.constants import DATASET_SCHEMA


class DatasetSchemaError(ValueError):
    """meta.json is missing or the schema field does not match."""


def validate_dataset_meta(meta: dict[str, Any], *, expected: str = DATASET_SCHEMA) -> str:
    """Require ``meta['schema']`` == ``jims_mower.dataset.v1``."""
    if not isinstance(meta, dict):
        raise DatasetSchemaError("dataset meta must be a mapping")
    raw = meta.get("schema")
    if raw is None or str(raw).strip() == "":
        raise DatasetSchemaError(
            f"dataset meta missing required 'schema' field (expected {expected})"
        )
    schema = str(raw).strip()
    if schema != expected:
        raise DatasetSchemaError(f"unsupported dataset schema {schema!r}; expected {expected}")
    return schema


def load_dataset_meta(dataset_dir: Union[str, Path]) -> dict[str, Any]:
    root = Path(dataset_dir)
    path = root / "meta.json"
    if not path.is_file():
        raise DatasetSchemaError(f"export meta.json missing: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    validate_dataset_meta(meta)
    return meta


def assign_frame_split(
    n_frames: int,
    *,
    val_frac: float = 0.20,
) -> dict[str, Any]:
    """Deterministic last-fraction val split. Documented, not a published mAP.

    Frame indices are ``0 .. n_frames-1`` (same as exporter ``frame``).
    """
    n = max(0, int(n_frames))
    frac = float(val_frac)
    if frac < 0.0 or frac >= 1.0:
        raise DatasetSchemaError("val_frac must be in [0, 1)")
    if n <= 1:
        n_val = 0
    else:
        n_val = max(1, int(round(n * frac)))
        if n_val >= n:
            n_val = n - 1
    train = list(range(0, n - n_val))
    val = list(range(n - n_val, n))
    return {
        "rule": "last_frac_val",
        "val_frac": frac,
        "train_frames": train,
        "val_frames": val,
        "n_train": len(train),
        "n_val": len(val),
        "note": (
            "Deterministic by frame index. Fake CSI is OK in CI. "
            "Do not publish mAP / IoU from this split."
        ),
    }


def train_frame_set(meta: dict[str, Any]) -> Optional[set[int]]:
    """Return train frame indices when ``meta['split']`` is present."""
    split = meta.get("split")
    if not isinstance(split, dict):
        return None
    raw = split.get("train_frames")
    if raw is None:
        return None
    return {int(i) for i in raw}
