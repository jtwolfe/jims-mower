"""Dataset schema versioning for WAVE 1A exporter folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

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
