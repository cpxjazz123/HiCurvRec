"""Issue #222 deliverable: MANIFEST.json with every input, checkpoint and output digest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

INPUTS = (
    "results/research/directed_graphlet_role/instances.npz",
    "results/research/directed_graphlet_role/subtypes.json",
    "results/research/behavior_graph/graph.npz",
    "stage1_GeneEmbedding/output/sentence_t5.npy",
    "results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth",
    "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json",
    "teacher/teacher_directed_v1.pth",
    "teacher/teacher_metadata.json",
    "teacher/teacher_probe.pt",
)

ANALYSIS = (
    "REPORT_TEACHER_DISTILL_INTERFACE.md",
    "teacher_model.py", "frozen_sid.py", "student_head.py", "kdistill.py",
    "fit_teacher.py", "gate.py", "manifest.py", "close_comment.py", "report_sections.py",
    "trainable_student.py", "evidence_backprop.py", "smoke_trainable_student.py",
    "REPORT_STUDENT_RQVAE_BACKPROP.md", "backprop_evidence.json",
    "smoke_trainable_student.json", "health_stop.json", "issue223_close_comment.md",
    "teacher_controls.json", "gate.json",
    "tests/test_teacher_contract.py", "tests/test_student_interface.py",
    "tests/test_trainable_student.py",
)


def digest(path: Path) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def main() -> None:
    payload = {"issue": 222, "inputs": {}, "analysis": {}}
    for relative in INPUTS:
        path = HERE / relative if (HERE / relative).is_file() else REPO / relative
        if path.is_file():
            payload["inputs"][relative] = digest(path)
        else:
            payload["inputs"][relative] = {"missing": True}
    for name in ANALYSIS:
        path = HERE / name
        if path.is_file():
            payload["analysis"][name] = digest(path)
        else:
            payload["analysis"][name] = {"missing": True}
    (HERE / "MANIFEST.json").write_text(json.dumps(payload, indent=1))
    missing = [key for key, value in {**payload["inputs"], **payload["analysis"]}.items()
               if "missing" in value]
    print("manifest written:", len(payload["inputs"]), "inputs,", len(payload["analysis"]),
          "analysis files; missing:", missing)


if __name__ == "__main__":
    main()
