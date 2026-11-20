"""Issue #229: the trained teacher must reload and reproduce its recorded metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TEACHER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEACHER))
sys.path.insert(0, str(TEACHER.parent / "directed_graphlet_role"))

import joint_teacher_model as jtm  # noqa: E402


def test_joint_teacher_reloads_and_reproduces_metrics():
    if not jtm.CHECKPOINT.is_file():
        pytest.skip("the joint teacher has not been trained")
    teacher = jtm.load_joint_teacher()
    assert teacher.verify()["healthy"] is True

    import graph_teacher3 as gt3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = gt3.prepare_data(device)
    teacher.model.to(device)
    state = teacher.encode(data["features"], data["adjacency"])
    scores = teacher.score_pairs(state, data["test_positive"], data["test_negative"])
    edge_auc = gt3.auc_of(scores)
    direction = teacher.direction_accuracy(state, data["test_direction"])
    recorded = json.loads((TEACHER / "teacher" / "opt" / "joint_teacher_report.json").read_text())
    assert abs(edge_auc - recorded["edge"]["test_auc"]) < 1e-6
    assert abs(direction["balanced_accuracy"]
               - recorded["direction"]["test_balanced_accuracy"]) < 1e-6
    k3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    assert teacher.recognise(k3)["family"] == "K3"
