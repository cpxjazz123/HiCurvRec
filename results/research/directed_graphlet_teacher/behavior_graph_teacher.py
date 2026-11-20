"""The teacher model, in one class (Issue #229).

Two artifacts already exist: the frozen bundle (targets + carrier + recognition +
losses) and the trained joint graph model (edge/direction scoring).  Consumers
should not have to know that split, so this facade exposes the #229 capabilities
behind one object and reports the evidence status of each:

    teacher = BehaviorGraphTeacher.load()
    teacher.verify()
    teacher.recognise(matrix)                       # graphlet type / orbits
    state = teacher.encode(features, adjacency)     # graph model
    teacher.score_pairs(state, positives, negatives)
    teacher.targets_for(item_ids)                   # fixed-SID supervision rows
    teacher.predict_targets(embeddings)             # carrier fallback
    teacher.structure_loss(student_vectors, item_ids=batch)
    teacher.describe()                              # capability -> status
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "directed_graphlet_role"))

import joint_teacher_model as jtm  # noqa: E402
import structure_teacher_bundle as stb  # noqa: E402

CONFORMANCE = HERE / "conformance_229.json"
CAPABILITY_CHECKS = {
    "recognition": ("recognition_api", "recognised_type_from_adjacency", "multi_role"),
    "prediction_edge": ("prediction_edge", "structure_environment", "edge_vs_heuristic"),
    "prediction_direction": ("prediction_direction",),
    "prediction_structure": ("prediction_structure_without_adjacency",),
    "item_carrier": ("item_carrier", "coverage"),
    "sid_capacity_32d": ("sid_capacity_32d",),
    "supervision_deliverable": ("supervision_deliverable",),
}
CAPABILITY_KEYS = tuple(CAPABILITY_CHECKS)
SEVERITY = {"PASS": 0, "NOT_ESTABLISHED": 1, "FAIL": 2}

BOUNDARY_KEYS = ("rqvae_input_changed", "rqvae_encoder_input",
                 "teacher_inference_needs_graph", "used_only_in_training_loss")
EXPECTED_BOUNDARY = {"rqvae_input_changed": False,
                     "teacher_inference_needs_graph": False,
                     "used_only_in_training_loss": True}


def _read_contract(path):
    return json.loads(Path(path).read_text())


def _boundary_report(meta) -> dict:
    """Read the frozen contract instead of restating its claims."""
    candidates = []
    if meta.get("contract_source"):
        candidates.append(HERE.parents[2] / str(meta["contract_source"]))
    candidates.append(stb.BUNDLE_DIR / "structure_teacher_bundle_contract.json")
    for path in candidates:
        path = Path(path)
        if not path.is_file():
            continue
        contract = _read_contract(path)
        recorded = contract.get("boundary") or {
            key: contract[key] for key in BOUNDARY_KEYS if key in contract}
        boundary = dict(recorded)
        root = HERE.parents[2]
        boundary["recorded_from"] = (str(path.relative_to(root))
                                     if path.is_relative_to(root) else str(path))
        boundary["matches_expected"] = all(
            boundary.get(key) is value for key, value in EXPECTED_BOUNDARY.items())
        return boundary
    return {"recorded_from": None, "matches_expected": False}


class BehaviorGraphTeacher:
    def __init__(self, bundle, joint=None):
        self.bundle = bundle
        self.joint = joint

    @classmethod
    def load(cls, joint: bool = True, device=None) -> "BehaviorGraphTeacher":
        bundle = stb.load_structure_teacher(stb.BUNDLE_DIR)
        model = None
        if joint:
            model = jtm.load_joint_teacher()
            if device is not None:
                model.model.to(device)
        return cls(bundle, model)

    # -- recognition ------------------------------------------------------
    def recognise(self, matrix) -> dict:
        return self.bundle.recognise(matrix)

    # -- graph model ------------------------------------------------------
    def encode(self, features, adjacency):
        if self.joint is None:
            raise RuntimeError("load(joint=True) is required for the graph model")
        return self.joint.encode(features, adjacency)

    def score_pairs(self, state, positive, negative) -> np.ndarray:
        if self.joint is None:
            raise RuntimeError("load(joint=True) is required for the graph model")
        return self.joint.score_pairs(state, positive, negative)

    def direction_metrics(self, state, direction) -> dict:
        if self.joint is None:
            raise RuntimeError("load(joint=True) is required for the graph model")
        return self.joint.direction_accuracy(state, direction)

    # -- item-level supervision ------------------------------------------
    def targets_for(self, item_ids) -> dict:
        return self.bundle.targets_for(item_ids)

    def predict_targets(self, embeddings: torch.Tensor) -> dict:
        return self.bundle.predict_targets(embeddings)

    def structure_loss(self, student_vectors, item_ids=None, role_logits=None,
                       weights=None):
        return self.bundle.structure_loss(student_vectors, item_ids=item_ids,
                                          role_logits=role_logits, weights=weights)

    # -- status -----------------------------------------------------------
    def verify(self) -> dict:
        checks = {"bundle": self.bundle.verify()["healthy"]}
        if self.joint is not None:
            checks["joint_model"] = self.joint.verify()["healthy"]
        return {"healthy": all(checks.values()), "checks": checks}

    def describe(self) -> dict:
        if CONFORMANCE.is_file():
            audit = json.loads(CONFORMANCE.read_text())
        else:
            audit = {"checks": {}, "declared_not_established": []}
        checks = audit["checks"]
        capabilities, evidence, mapped = {}, {}, set()
        for group, names in CAPABILITY_CHECKS.items():
            present = {name: checks[name]["status"] for name in names if name in checks}
            evidence[group] = present
            mapped.update(present)
            if not present:
                capabilities[group] = "NOT_RUN"
            else:
                capabilities[group] = max(
                    present.values(), key=lambda status: SEVERITY.get(status, 1))
        other = {name: entry["status"] for name, entry in checks.items()
                 if name not in mapped}
        return {"capabilities": capabilities, "evidence": evidence,
                "other_checks": other,
                "declared_not_established": audit["declared_not_established"],
                "weights": self.bundle.weights,
                "boundary": _boundary_report(self.bundle.meta)}


def main() -> None:
    teacher = BehaviorGraphTeacher.load(joint=False)
    print(json.dumps({"verify": teacher.verify(), "describe": teacher.describe()},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
