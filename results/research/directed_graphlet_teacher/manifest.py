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
    "train_teacher.py", "tests/test_joint_teacher.py",
    "REPORT_STRUCTURE_TEACHER_JOINT.md", "issue229_joint_comment.md",
    "teacher/opt/joint_teacher_report.json",
    "TEACHER_229.md", "accept_teacher_229.py",
    "tests/test_acceptance_229.py",
    "recognition_coverage.py", "tests/test_recognition_coverage.py",
    "REPORT_RECOGNITION_COVERAGE.md",
    "teacher/opt/recognition_coverage.json",
    "joint_teacher_model.py", "tests/test_joint_teacher_model.py",
    "teacher/structure/joint/teacher_joint_v1_manifest.json",
    "teacher/opt/joint_teacher_reload.json",
    "behavior_graph_teacher.py", "tests/test_behavior_graph_teacher.py",
    "structure_environment.py", "tests/test_structure_environment.py",
    "REPORT_STRUCTURE_ENVIRONMENT.md",
    "teacher/opt/structure_environment.json",
    "structure_heuristics.py", "tests/test_structure_heuristics.py",
    "REPORT_STRUCTURE_HEURISTICS.md", "teacher/opt/structure_heuristics.json",
    "orbit_vectors/build_orbit_vectors.py", "tests/test_orbit_vectors.py",
    "orbit_vectors/REPORT_ORBIT_VECTORS.md", "orbit_vectors/orbit_vocab.json",
    "orbit_vectors/split_audit.json", "orbit_vectors/reliability.json",
    "orbit_vectors/learnability.json", "orbit_vectors/manifest.json",
    "relation/audit_relation.py",
    "relation/REPORT_RELATION_AUDIT.md", "tests/test_relation_audit.py",
    "relation/audit_negative_validation.py",
    "representation/build_gnn_embedding.py", "representation/manifest.json",
    "representation/role_probe.py", "representation/role_probe.json",
    "representation/export_latent_q.py", "representation/rqvae_latent.json",
    "representation/checkpoint_sid_pairs.py", "representation/checkpoint_sid_pairs.json",
    "representation/REPORT_REPRESENTATION.md", "representation/RETRACTIONS.md",
    "quantisation/run_quantisation_check.py", "quantisation/sweep.json",
    "quantisation/REPORT_QUANTISATION.md",
    "fidelity/fidelity_study.py", "fidelity/fidelity.json",
    "fidelity/fidelity_states.json", "fidelity/REPORT_FIDELITY.md",
    "structure/structure_transformer.py", "structure/structure_transformer.json",
    "structure/REPORT_STRUCTURE_TRANSFORMER.md",
    "full_attention/full_attention.py", "full_attention/full_attention.json",
    "full_attention/REPORT_FULL_ATTENTION.md",
    "role_context/role_context.py", "role_context/role_context.json",
    "role_context/REPORT_ROLE_CONTEXT.md",
    "edge_reconstruction/edge_reconstruction.py",
    "edge_reconstruction/edge_reconstruction.json",
    "edge_reconstruction/REPORT_EDGE_RECONSTRUCTION.md",
    "edge_rnn/edge_rnn.py", "edge_rnn/edge_rnn.json",
    "edge_rnn/REPORT_EDGE_RNN.md",
    "capability/capability.py", "capability/capability.json",
    "capability/REPORT_CAPABILITY.md",
    "pair_relations/pair_relations.py", "pair_relations/pair_relations.json",
    "pair_relations/REPORT_PAIR_RELATIONS.md",
    "two_node_rnn/two_node_rnn.py", "two_node_rnn/two_node_rnn.json",
    "two_node_rnn/REPORT_TWO_NODE_RNN.md",
    "inject_directed/inject_directed.py", "inject_directed/result.json",
    "inject_directed/REPORT_INJECT_DIRECTED.md",
    "augment/augment_embedding.py", "augment/augmented_manifest.json",
    "augment/REPORT_AUGMENT.md",
    "augment_gnn/augment_gnn.py", "augment_gnn/gnn_augment.json",
    "augment_gnn/REPORT_AUGMENT_GNN.md",
    "augment_gnn/dual_augment.py", "augment_gnn/dual_augment.json",
    "augment_gnn/REPORT_DUAL_PROPAGATION.md",
    "augment_gnn/direction_arms.py", "augment_gnn/direction_arms.json",
    "augment_gnn/REPORT_DIRECTION_ARMS.md",
    "augment_gnn/dir_ablation.py", "augment_gnn/dir_ablation.json",
    "augment_gnn/REPORT_DIR_ABLATION.md",
    "rrwp/rrwp_direction.py", "rrwp/rrwp_direction.json",
    "rrwp/rrwp_ablation.py", "rrwp/rrwp_ablation.json", "rrwp/REPORT_RRWP.md",
    "hard_negatives/hard_negatives.py", "hard_negatives/hard_negatives.json",
    "hard_negatives/repeat.json", "hard_negatives/REPORT_HARD_NEGATIVES.md",
    "edge_prediction/edge_prediction.py", "edge_prediction/edge_prediction.json",
    "edge_prediction/REPORT_EDGE_PREDICTION.md",
    "diage/diage_repro.py", "diage/diage_repro.json",
    "diage/REPORT_DIAGE_REPRO.md", "diage/REPORT_DIRECTION_IS_TEMPORAL.md",
    "diage/item_time_direction.py", "diage/item_time_direction.json",
    "concat_diage/concat_diage.py", "concat_diage/concat_diage.json",
    "concat_diage/REPORT_CONCAT_DIAGE.md",
    "local_pairs/local_pairs.py", "local_pairs/local_pairs.json",
    "local_pairs/REPORT_LOCAL_PAIRS.md",
    "decomposed/decomposed.py", "decomposed/decomposed.json",
    "decomposed/REPORT_DECOMPOSED.md",
    "graph_loss/graph_loss.py", "graph_loss/graph_loss.json",
    "graph_loss/REPORT_GRAPH_LOSS.md",
    "graph_loss/graph_loss_v2.py", "graph_loss/graph_loss_v2.json",
    "graph_loss/REPORT_GRAPH_LOSS_V2.md",
    "graph_loss/graph_loss_v3.py", "graph_loss/graph_loss_v3.json",
    "graph_loss/sid_structure.py", "graph_loss/sid_structure.json",
    "graph_loss/sid_alignment.py", "graph_loss/sid_alignment.json",
    "graph_loss/sid_alignment_v2.py", "graph_loss/sid_alignment_v2.json",
    "graph_loss/prototype_chain.py", "graph_loss/prototype_chain.json",
    "graph_loss/REPORT_CHAIN_PROTOTYPE.md",
    "graph_loss/REPORT_SID_ALIGNMENT_V2.md",
    "graph_loss/REPORT_SID_ALIGNMENT.md",
    "graph_loss/REPORT_SID_STRUCTURE.md",
    "graph_loss/REPORT_GRAPH_LOSS_V3.md",
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

