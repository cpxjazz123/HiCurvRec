"""Issue #229: the trained teacher as a frozen, loadable model (not just a table).

`train_teacher.py` saves one encoder with edge / direction / orbit / subtype
heads.  This module loads it, verifies its parameter digest, and reproduces the
recorded edge AUC and direction accuracy from the checkpoint, so the model
itself is an auditable artifact:

    model = load_joint_teacher(".../teacher_joint_v1.pth")
    model.verify()
    state = model.encode(features, adjacency)
    scores = model.score_pairs(state, positives, negatives)
    model.recognise(matrix)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "directed_graphlet_role"))

import graph_teacher3 as gt3  # noqa: E402
import structure_teacher as st  # noqa: E402
from train_teacher import JointTeacher  # noqa: E402

JOINT_DIR = HERE / "teacher" / "structure" / "joint"
CHECKPOINT = JOINT_DIR / "teacher_joint_v1.pth"
MANIFEST = JOINT_DIR / "teacher_joint_v1_manifest.json"


class JointTeacherModel:
    def __init__(self, model: JointTeacher, meta: dict):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.meta = meta

    def verify(self) -> dict:
        actual = gt3.param_sha_from_state(self.model.state_dict())
        return {"healthy": actual == self.meta["param_sha"],
                "param_sha": actual, "expected": self.meta["param_sha"]}

    def encode(self, features, adjacency):
        with torch.no_grad():
            return self.model.encode(features, adjacency)

    def score_pairs(self, state, positive, negative) -> np.ndarray:
        with torch.no_grad():
            device = state.device
            left = torch.from_numpy(positive[:, 0]).to(device)
            right = torch.from_numpy(positive[:, 1]).to(device)
            target = torch.from_numpy(negative).to(device)
            positive_score = self.model.edge(self.model.pair(state, left, right)).cpu().numpy()
            negative_score = self.model.edge(self.model.pair(state, left, target)).cpu().numpy()
        return np.stack([positive_score, negative_score], axis=1)

    def direction_accuracy(self, state, direction) -> dict:
        with torch.no_grad():
            device = state.device
            rows = torch.from_numpy(direction["rows"]).to(device)
            cols = torch.from_numpy(direction["cols"]).to(device)
            logits = self.model.direction(state, rows, cols)
            prediction = logits.argmax(-1).cpu().numpy()
        truth = direction["labels"]
        confusion = np.zeros((2, 2), dtype=np.int64)
        for expected, predicted in zip(truth, prediction):
            confusion[expected, predicted] += 1
        import graph_teacher2 as gt2
        return gt2.direction_metrics(confusion)

    def recognise(self, matrix) -> dict:
        return st.recognise_subgraph(matrix)


def load_joint_teacher(path: Path | None = None) -> JointTeacherModel:
    path = Path(path) if path is not None else CHECKPOINT
    payload = torch.load(path, map_location="cpu")
    model = JointTeacher(payload["subtype_count"])
    model.load_state_dict(payload["state"])
    meta = dict(payload["meta"])
    meta.update({"subtype_count": payload["subtype_count"], "hidden": payload["hidden"],
                 "dim": payload["dim"], "roles": payload["roles"], "steps": payload["steps"],
                 "graph_sha": payload["graph_sha"], "split_sha": payload["split_sha"],
                 "source": path.name,
                 "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if "param_sha" not in meta:
        meta["param_sha"] = gt3.param_sha_from_state(model.state_dict())
        MANIFEST.write_text(json.dumps(meta, indent=1))
    return JointTeacherModel(model, meta)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher = load_joint_teacher()
    verification = teacher.verify()
    print(json.dumps(verification, indent=1))
    if not verification["healthy"]:
        raise SystemExit(1)
    data = gt3.prepare_data(device)
    model = teacher.model
    features = data["features"]
    adjacency = data["adjacency"]
    import torch as t
    with t.no_grad():
        features = features.to(device)
    model.to(device)
    state = teacher.encode(features, adjacency)
    scores = teacher.score_pairs(state, data["test_positive"], data["test_negative"])
    edge_auc = gt3.auc_of(scores)
    direction = teacher.direction_accuracy(state, data["test_direction"])
    recorded = json.loads((HERE / "teacher" / "opt" / "joint_teacher_report.json").read_text())
    result = {"edge_auc": edge_auc, "recorded_edge_auc": recorded["edge"]["test_auc"],
              "edge_auc_matches": abs(edge_auc - recorded["edge"]["test_auc"]) < 1e-6,
              "direction_balanced_accuracy": direction["balanced_accuracy"],
              "recorded_direction": recorded["direction"]["test_balanced_accuracy"],
              "direction_matches": abs(direction["balanced_accuracy"]
                                       - recorded["direction"]["test_balanced_accuracy"]) < 1e-6,
              "recognition": teacher.recognise(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]],
                                                        dtype=bool)),
              "param_sha": verification["param_sha"]}
    print(json.dumps(result, indent=1))
    (HERE / "teacher" / "opt" / "joint_teacher_reload.json").write_text(
        json.dumps(result, indent=1))
    raise SystemExit(0 if result["edge_auc_matches"] and result["direction_matches"] else 1)


if __name__ == "__main__":
    main()
