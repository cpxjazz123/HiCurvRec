"""Issue #229 acceptance: one command a third party can run on the delivered artifacts.

    python3.9 accept_teacher_229.py

Exits 0 only when the bundle verifies, recognition returns the right families,
the target rows are aligned, the supervision loss prefers the teacher geometry,
and the conformance audit has no FAIL.  Prints one line per check.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "directed_graphlet_role"))

import structure_teacher_bundle as stb  # noqa: E402

C4 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool)
K3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)


def main() -> int:
    failures = []

    def check(name: str, condition: bool, detail: str) -> None:
        print(f"{'PASS' if condition else 'FAIL'}  {name}: {detail}")
        if not condition:
            failures.append(name)

    teacher = stb.load_structure_teacher(stb.BUNDLE_DIR)
    report = teacher.verify()
    check("bundle_sha", report["healthy"], f"{len(report['checks'])} checks")
    check("recognition_k3", teacher.recognise(K3)["family"] == "K3",
          teacher.recognise(K3)["family"])
    check("recognition_c4", teacher.recognise(C4)["family"] == "C4",
          teacher.recognise(C4)["family"])

    rows = teacher.targets_for([0, 1, 2])
    check("targets_aligned", rows["relation"].shape == (3, 128)
          and rows["neighbour_orbit"].shape == (3, 376),
          f"relation {rows['relation'].shape}, orbit {rows['neighbour_orbit'].shape}")
    check("coverage_in_table", rows["coverage"].shape == (3,), str(rows["coverage"]))

    teacher_vectors = torch.from_numpy(teacher.targets["relation"]).float()
    metrics, teacher_loss = teacher.structure_loss(teacher_vectors)
    _, random_loss = teacher.structure_loss(torch.zeros_like(teacher_vectors))
    check("loss_prefers_teacher", float(teacher_loss) < float(random_loss),
          f"{metrics['total']:.4f} < {float(random_loss):.4f}")

    contract = json.loads((stb.BUNDLE_DIR / "structure_teacher_bundle_contract.json").read_text())
    check("boundary_no_graph", contract["boundary"]["teacher_inference_needs_graph"] is False,
          "teacher_inference_needs_graph=False")
    check("boundary_training_only", contract["boundary"]["used_only_in_training_loss"] is True,
          "used_only_in_training_loss=True")

    audit = json.loads((HERE / "conformance_229.json").read_text())
    check("conformance_no_fail", audit["failed"] == [], f"failed={audit['failed']}")
    print(f"declared_not_established: {audit['declared_not_established']}")
    print(f"verdict: {audit['verdict']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
