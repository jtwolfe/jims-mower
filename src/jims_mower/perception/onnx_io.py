"""Optional ONNX export / load for the tiny terrain (and detector) stubs.

``onnx`` and ``onnxruntime`` are extras — CI without them still passes.
Sim-trained graphs are a **dev artifact** (``sim_only``). Not field-ready.
``iou_claim`` / ``map_claim`` / ``fps_claim`` stay null.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jims_mower.constants import TERRAIN_CLASS_NAMES
from jims_mower.perception.learn import (
    N_CLASSES,
    N_FEATURES,
    TerrainMLP,
    _softmax,
)

ONNX_SCHEMA = "jims_mower.terrain_onnx.v1"
ONNX_INPUT = "features"
ONNX_OUTPUT = "logits"
DEFAULT_ONNX_NAME = "terrain_seg.onnx"


class OnnxExportError(RuntimeError):
    """ONNX extra missing or the graph could not be written."""


def try_import_onnx():
    try:
        import onnx  # type: ignore

        return onnx
    except ImportError:
        return None


def try_import_onnxruntime():
    try:
        import onnxruntime  # type: ignore

        return onnxruntime
    except ImportError:
        return None


def onnx_available() -> bool:
    return try_import_onnx() is not None


def onnxruntime_available() -> bool:
    return try_import_onnxruntime() is not None


def default_onnx_search_paths() -> tuple[Path, ...]:
    return (
        Path("artifacts") / DEFAULT_ONNX_NAME,
        Path("models") / DEFAULT_ONNX_NAME,
    )


def resolve_onnx_path(path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    if path:
        dest = Path(path)
        return dest if dest.is_file() else None
    for candidate in default_onnx_search_paths():
        if candidate.is_file():
            return candidate
    return None


def terrain_sidecar_payload(
    model: TerrainMLP,
    onnx_path: Union[str, Path],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": ONNX_SCHEMA,
        "onnx": str(onnx_path),
        "domain": "sim_only",
        "field_ready": False,
        "iou_claim": None,
        "map_claim": None,
        "fps_claim": None,
        "n_features": int(model.n_features),
        "n_classes": int(model.n_classes),
        "hidden": int(model.hidden),
        "kind": model.kind,
        "classes": list(TERRAIN_CLASS_NAMES[: int(model.n_classes)]),
        "input": ONNX_INPUT,
        "output": ONNX_OUTPUT,
        "note": (
            "Dev artifact trained on gym / FakeCsi labels. "
            "Not a field head. Do not publish IoU / mAP / FPS."
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def write_sidecar(path: Union[str, Path], payload: dict[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def export_terrain_onnx(
    model: TerrainMLP,
    path: Union[str, Path],
    *,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a Gemm(+ReLU) ONNX of ``TerrainMLP``. Requires the ``onnx`` extra."""
    onnx = try_import_onnx()
    if onnx is None:
        raise OnnxExportError(
            "onnx is not installed; pip install -e '.[onnx]' to export. "
            "Numpy weights still work without this extra."
        )
    from onnx import TensorProto, helper, numpy_helper, save  # type: ignore

    n_in = int(model.n_features)
    n_out = int(model.n_classes)
    features = helper.make_tensor_value_info(ONNX_INPUT, TensorProto.FLOAT, ["batch", n_in])
    logits = helper.make_tensor_value_info(ONNX_OUTPUT, TensorProto.FLOAT, ["batch", n_out])
    if model.is_mlp:
        assert model.W2 is not None and model.b2 is not None
        nodes = [
            helper.make_node("Gemm", [ONNX_INPUT, "W1", "b1"], ["hidden_pre"], name="gemm1"),
            helper.make_node("Relu", ["hidden_pre"], ["hidden_act"], name="relu"),
            helper.make_node("Gemm", ["hidden_act", "W2", "b2"], [ONNX_OUTPUT], name="gemm2"),
        ]
        initializers = [
            numpy_helper.from_array(np.asarray(model.W1, dtype=np.float32), name="W1"),
            numpy_helper.from_array(np.asarray(model.b1, dtype=np.float32), name="b1"),
            numpy_helper.from_array(np.asarray(model.W2, dtype=np.float32), name="W2"),
            numpy_helper.from_array(np.asarray(model.b2, dtype=np.float32), name="b2"),
        ]
    else:
        nodes = [
            helper.make_node("Gemm", [ONNX_INPUT, "W1", "b1"], [ONNX_OUTPUT], name="gemm1"),
        ]
        initializers = [
            numpy_helper.from_array(np.asarray(model.W1, dtype=np.float32), name="W1"),
            numpy_helper.from_array(np.asarray(model.b1, dtype=np.float32), name="b1"),
        ]
    graph = helper.make_graph(nodes, "terrain_seg", [features], [logits], initializers)
    onnx_model = helper.make_model(
        graph,
        producer_name="jims_mower",
        doc_string="sim_only terrain MLP; not field-ready; iou_claim null",
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    meta = {
        "schema": ONNX_SCHEMA,
        "domain": "sim_only",
        "field_ready": "false",
        "iou_claim": "",
        "map_claim": "",
        "fps_claim": "",
        "n_features": str(n_in),
        "n_classes": str(n_out),
        "hidden": str(int(model.hidden)),
        "kind": model.kind,
        "classes": ",".join(TERRAIN_CLASS_NAMES[:n_out]),
    }
    for key, value in meta.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save(onnx_model, str(dest))
    write_sidecar(dest.with_suffix(".json"), terrain_sidecar_payload(model, dest, extra))
    return dest


def load_onnx_session(path: Union[str, Path]):
    """Return an InferenceSession or None if onnxruntime is missing."""
    ort = try_import_onnxruntime()
    if ort is None:
        return None
    dest = Path(path)
    if not dest.is_file():
        raise FileNotFoundError(f"ONNX not found: {dest}")
    return ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])


class OnnxMLP:
    """TerrainMLP-shaped wrapper around an onnxruntime session."""

    def __init__(self, session, *, n_features: int = N_FEATURES, n_classes: int = N_CLASSES) -> None:
        self.session = session
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        self.input_name = inputs[0].name if inputs else ONNX_INPUT
        self.output_name = outputs[0].name if outputs else ONNX_OUTPUT

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        if h.ndim == 1:
            h = h.reshape(1, -1)
        raw = self.session.run([self.output_name], {self.input_name: h})[0]
        return np.asarray(raw, dtype=np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=-1).astype(np.uint8)
