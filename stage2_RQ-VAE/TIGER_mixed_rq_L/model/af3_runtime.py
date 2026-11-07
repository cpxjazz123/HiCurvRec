"""Shared-readout lifecycle and identities for Issue #139 (no job launcher).

The calibration is an immutable F0/epoch-1500 artifact. A consumer cannot
invent its own calibration, reuse a different support, or overwrite a producer.
These helpers do not grant training admission or read validation/test data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from .af3_structure import CALIBRATION_TOL, TAU_FLOOR

READOUT_SCHEMA = "issue139-shared-readout/1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def snapshot_state(model) -> dict:
    """Clone both tensors and the non-tensor alpha-table extra state."""
    return {name: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for name, value in model.state_dict().items()}


def snapshot_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(snapshot_state(model).items()):
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))
    return digest.hexdigest()


def input_identity(graph: dict, support: dict, graph_path: Path,
                   embedding_path: Path, train_path: Path, code_paths: list[Path]) -> dict:
    code = hashlib.sha256()
    for path in sorted(code_paths, key=lambda p: p.name):
        code.update(path.name.encode("utf-8"))
        code.update(file_sha256(path).encode("ascii"))
    return {"graph_sha256": file_sha256(graph_path),
            "support_sha256": array_sha256(support["pairs"], support["is_edge"]),
            "embedding_sha256": file_sha256(embedding_path),
            "train_sha256": file_sha256(train_path),
            "item_ids_sha256": array_sha256(graph["item_ids"]),
            "code_sha256": code.hexdigest()}


def validate_readout(record: dict, identity: dict) -> dict:
    if (record.get("schema") != READOUT_SCHEMA or record.get("source_arm") != "F0"
            or record.get("source_epoch") != 1500):
        raise ValueError("the shared readout must originate from F0 at epoch 1500")
    if not identity or record.get("identity") != identity:
        raise ValueError("shared readout input/support/code identity mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("snapshot_sha256", ""))):
        raise ValueError("shared readout needs a valid F0 snapshot SHA-256")
    values = [float(record[key]) for key in ("tau", "intercept", "density", "mean_probability")]
    tau, _, density, measured = values
    if (not all(math.isfinite(value) for value in values) or tau < TAU_FLOOR
            or not 0.0 < density < 1.0 or abs(measured - density) > CALIBRATION_TOL):
        raise ValueError("invalid shared readout or failed density calibration")
    return record


def resolve_shared_readout(arm: str, epoch: int, path: Path, calibrate_fn,
                           identity: dict) -> dict | None:
    """Create only on F0/1500, otherwise validate/reuse without refitting.

    Consumers call this at epoch zero so missing provenance fails before any
    optimiser step. Existing files are validated, never truncated/replaced.
    """
    if arm not in ("F0", "G", "L", "E", "A"):
        raise ValueError("unknown Issue-139 arm: " + str(arm))
    path = Path(path)
    if path.is_file():
        return validate_readout(json.loads(path.read_text(encoding="utf-8")), identity)
    if arm != "F0":
        raise FileNotFoundError("run F0 and freeze its epoch-1500 readout first: " + str(path))
    if epoch != 1500:
        return None
    record = dict(calibrate_fn())
    record.update(schema=READOUT_SCHEMA, source_arm="F0", source_epoch=1500,
                  identity=copy.deepcopy(identity))
    validate_readout(record, identity)
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: a concurrent producer cannot overwrite a frozen file.
    # A partial/interrupted write is invalid and will fail closed on later reads.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
    return record
