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
CAPABILITY_KEYS = ("recognition", "prediction_edge", "prediction_direction",
                   "item_carrier", "sid_capacity_32d", "supervision_deliverable")


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
            statuses = {name: audit["checks"][name]["status"]
                        for name in CAPABILITY_KEYS if name in audit["checks"]}
            declared = audit["declared_not_established"]
        else:
            statuses, declared = {}, []
        return {"capabilities": statuses, "declared_not_established": declared,
                "weights": self.bundle.weights,
                "boundary": {"rqvae_input_changed": False,
                             "teacher_inference_needs_graph": False,
                             "used_only_in_training_loss": True}}


def main() -> None:
    teacher = BehaviorGraphTeacher.load(joint=False)
    print(json.dumps({"verify": teacher.verify(), "describe": teacher.describe()},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
