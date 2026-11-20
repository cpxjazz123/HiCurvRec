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
    "results/research/behavior_graph_directed/graph.npz",
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
    "relational_distill.py", "relation_stability.py", "relation_gate.py",
    "relation_reference.py", "relation_ceiling.py", "REPORT_RELATIONAL_DISTILL.md",
    "relation_stability.json", "relation_gate.json", "relation_reference.json",
    "relation_ceiling.json", "tests/test_relational_distill.py",
    "issue224_close_comment.md",
    "relation_audit.py", "relation_audit.json",
    "relation_probe.py", "relation_probe.json",
    "affinity_gate.py", "affinity_gate.json",
    "item_signal_audit.py", "item_signal_audit.json",
    "item_gate.py", "item_gate.json",
    "collapse_gate.py", "collapse_gate.json",
    "graph_teacher.py", "graph_teacher_eval.py", "graph_teacher_report.json",
    "graph_teacher2.py", "graph_teacher2_report.json", "REPORT_GRAPH_TEACHER.md",
    "graph_teacher2_corrected_report.json", "REPORT_GRAPH_TEACHER_CORRECTED.md",
    "tests/test_graph_teacher_pairing.py", "issue227_close_comment.md",
    "tests/test_graph_teacher_leakage.py", "issue226_close_comment.md",
    "teacher_controls.json", "gate.json",
    "tests/test_teacher_contract.py", "tests/test_student_interface.py",
    "tests/test_trainable_student.py",
    "graph_teacher3.py", "tests/test_teacher_optimization.py",
    "teacher/opt/graph_teacher3_report.json", "REPORT_GRAPH_TEACHER_OPTIMIZATION.md",
    "issue228_close_comment.md",
    "structure_teacher.py", "tests/test_structure_teacher.py",
    "teacher/structure/teacher_structure_contract.json",
    "teacher/structure/teacher_structure_manifest.json",
    "REPORT_STRUCTURE_TEACHER.md", "structure_teacher_report.json",
    "issue229_close_comment.md",
    "probe_structure_v2.py", "tests/test_structure_teacher_v2.py",
    "REPORT_STRUCTURE_TEACHER_V2.md",
    "teacher/opt/structure_probe_v2.json",
    "teacher/structure/v2/teacher_structure_contract_v2.json",
    "teacher/structure/v2/teacher_structure_manifest_v2.json",
    "teacher/structure/v2/structure_teacher_v2_report.json",
    "probe_neighbour_roles.py", "tests/test_structure_teacher_v3.py",
    "REPORT_STRUCTURE_TEACHER_V3.md", "issue229_v3_comment.md",
    "teacher/opt/neighbour_role_probe.json",
    "teacher/structure/v3/teacher_structure_contract_v3.json",
    "teacher/structure/v3/teacher_structure_manifest_v3.json",
    "teacher/structure/v3/structure_teacher_v3_report.json",
    "structure_teacher_bundle.py", "tests/test_structure_teacher_bundle.py",
    "REPORT_STRUCTURE_TEACHER_BUNDLE.md", "issue229_bundle_comment.md",
    "teacher/structure/bundle/structure_teacher_bundle_contract.json",
    "teacher/structure/bundle/structure_teacher_bundle_manifest.json",
    "verify_conformance_229.py", "conformance_229.json",
    "tests/test_conformance_229.py", "REPORT_STRUCTURE_TEACHER_CONFORMANCE.md",
    "issue229_conformance_comment.md",
    "sid_capacity_32d.py", "tests/test_sid_capacity.py",
    "REPORT_STRUCTURE_TEACHER_SID_CAPACITY.md", "issue229_capacity_comment.md",
    "teacher/opt/sid_capacity_32d.json",
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
