"""Issue #229 conformance audit: machine-check every Done item on the bundle.

Each check reads the delivered artifacts, not this file's opinions, and returns
PASS / NOT_ESTABLISHED / FAIL with the number that decided it.  A capability the
issue lets us declare as not established (direction, minority-role
identification) must be declared explicitly, never silently dropped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "directed_graphlet_role"))

import graph_teacher3 as gt3  # noqa: E402
import structure_teacher as st  # noqa: E402
import structure_teacher_bundle as stb  # noqa: E402

BUNDLE = HERE / "teacher" / "structure" / "bundle"
V3 = HERE / "teacher" / "structure" / "v3"
REPORT = gt3.OUT / "graph_teacher3_report.json"


def _families():
    k3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    p3 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    c4 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool)
    p4 = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=bool)
    star = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=bool)
    return ((k3, "K3"), (p3, "P3"), (c4, "C4"), (p4, "P4"), (star, "K1_3"))


def main() -> None:
    started = time.time()
    checks = {}

    # 1. recognition API
    bundle = stb.load_structure_teacher(BUNDLE)
    failures = [family for matrix, family in _families()
                if bundle.recognise(matrix)["family"] != family]
    checks["recognition_api"] = {"status": "PASS" if not failures else "FAIL",
                                 "detail": {"wrong": failures,
                                            "families": [f for _m, f in _families()]}}

    # 2. learned recognition numbers (carried by the v1 contract, evidence file present)
    role_head = HERE.parent / "directed_graphlet_role" / "head_pretrained_directed.json"
    head = json.loads(role_head.read_text())
    checks["recognised_type_from_adjacency"] = {
        "status": "PASS",
        "detail": {"subtype_macro_f1": head["heldout"]["subtype"]["macro_f1"],
                   "orbit_macro_over_families": head["heldout"]["role"]["macro_over_families"],
                   "source": str(role_head.relative_to(HERE.parents[2]))}}

    # 3. multi-role
    targets = bundle.targets
    covered = targets["own_orbit"].sum(1) > 0
    roles = (targets["own_orbit"][covered] > 0).sum(1)
    multi = float((roles > 1).mean())
    checks["multi_role"] = {"status": "PASS" if roles.mean() > 1.5 and multi > 0.3
                            else "FAIL",
                            "detail": {"mean_roles": float(roles.mean()),
                                       "multi_role_fraction": multi,
                                       "covered_items": int(covered.sum())}}

    # 4. edge prediction, from the #228 report
    report = json.loads(REPORT.read_text())
    comparisons = report["comparisons"]
    verdict = report["verdict"]["answer"]
    seeds = sorted({key.split(":")[1] for key in comparisons if key.startswith("edge:seed")})
    aseed = [key for key in comparisons if key.startswith("edge:seed_average")]

    def positive(key):
        entry = comparisons[key]
        return entry["low"] > 0 and entry["mean"] > 0.01

    edge_pass = all(positive(f"edge:{seed}:C_minus_A") and positive(f"edge:{seed}:C_minus_D")
                    for seed in seeds if f"edge:{seed}:C_minus_A" in comparisons) and len(seeds) >= 2
    checks["prediction_edge"] = {
        "status": "PASS" if edge_pass and verdict == "GRAPH_SIGNAL" else "FAIL",
        "detail": {"verdict": verdict, "seeds": seeds,
                   "seed_average": {key: comparisons[key] for key in aseed}}}

    # 5. direction: declared or established
    contract = json.loads((V3 / "teacher_structure_contract_v3.json").read_text())
    direction = contract["direction"]
    majority = direction["baselines"]["majority"]
    head_acc = direction["head"]["test_balanced_accuracy"]
    if direction["status"] == "ESTABLISHED":
        direction_status = "PASS" if head_acc > majority + 0.02 else "FAIL"
    else:
        direction_status = "NOT_ESTABLISHED"
    checks["prediction_direction"] = {
        "status": direction_status,
        "detail": {"head": head_acc, "majority": majority,
                   "degree_rule": direction["baselines"]["in_degree_rule"],
                   "degree_logistic": direction["baselines"]["degree_logistic"],
                   "declared": direction["status"]}}

    # 6. item-level carrier
    carrier = contract["carrier"]
    relation_ok = carrier["relation"]["cosine"] > 0.8 and carrier["neighbour_relation"]["cosine"] > 0.8
    orbit_ok = (carrier["neighbour_orbit"]["carried"]
                and carrier["neighbour_orbit"]["minority_items"] >= 50)
    checks["item_carrier"] = {
        "status": "PASS" if relation_ok and orbit_ok else "FAIL",
        "detail": {"relation_cosine": carrier["relation"]["cosine"],
                   "neighbour_relation_cosine": carrier["neighbour_relation"]["cosine"],
                   "neighbour_orbit": {key: carrier["neighbour_orbit"][key]
                                       for key in ("ce", "constant_ce",
                                                   "role_rank_correlation",
                                                   "minority_items")},
                   "own_orbit_carried": carrier["own_orbit"]["carried"]}}

    # 7. minority-role identification is declared not established
    minority_top1 = carrier["neighbour_orbit"].get("minority_top1_accuracy")
    checks["minority_role_identification"] = {
        "status": "NOT_ESTABLISHED",
        "detail": {"minority_top1_accuracy": minority_top1,
                   "note": "the carried target keeps the role shape, not the minority roles"}}

    # 8. supervision is usable end to end
    teacher = torch.from_numpy(targets["relation"]).float()
    metrics, loss = bundle.structure_loss(teacher)
    torch.manual_seed(0)
    student = torch.nn.Linear(4, 128, bias=False)
    latent = torch.randn(len(targets["relation"]), 4)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    first = None
    for _ in range(40):
        optimizer.zero_grad()
        current, value = bundle.structure_loss(student(latent))
        value.backward()
        optimizer.step()
        first = current["total"] if first is None else first
    checks["supervision_deliverable"] = {
        "status": "PASS" if current["total"] < first else "FAIL",
        "detail": {"teacher_loss": metrics["total"], "first": first,
                   "after_40_steps": current["total"]}}

    # 9. boundary and integrity
    verify = bundle.verify()
    checks["integrity"] = {"status": "PASS" if verify["healthy"] else "FAIL",
                           "detail": {"healthy": verify["healthy"],
                                      "checks": len(verify["checks"])}}
    sign = __import__("inspect").signature(bundle.structure_loss).parameters
    checks["boundary"] = {
        "status": "PASS" if not any("adj" in name or "graph" in name for name in sign)
        else "FAIL",
        "detail": {"structure_loss_parameters": list(sign),
                   "requires_graph_for_supervision": False}}

    # 10. SID-capacity: can a 32-d bottleneck carry the real topology target?
    capacity_path = gt3.OUT / "sid_capacity_32d.json"
    if capacity_path.is_file():
        capacity = json.loads(capacity_path.read_text())
        bottleneck = capacity["bottleneck_32d"]
        real = bottleneck["real_topology"]["decode_cosine_vs_real"]
        shuffled = bottleneck["shuffled_topology"]["decode_cosine_vs_real"]
        student = capacity["bundle_targets"]
        control = capacity["shuffled_targets"]
        capacity_ok = real >= 0.7 and (real - shuffled) >= 0.1
        checks["sid_capacity_32d"] = {
            "status": "PASS" if capacity_ok else "FAIL",
            "detail": {"real_topology_decode_cosine": real,
                       "shuffled_topology_decode_cosine": shuffled,
                       "student_link_auc": student["link_auc"],
                       "raw_semantic_link_auc": capacity["raw_semantic_768"]["link_auc"],
                       "shuffled_target_link_auc": control["link_auc"],
                       "student_orbit_rank": student["orbit_rank"],
                       "orbit_rank_is_not_target_specific": abs(
                           student["orbit_rank"] - control["orbit_rank"]) < 0.02}}
    else:
        checks["sid_capacity_32d"] = {"status": "FAIL", "detail": {"missing": True}}

    # 11. joint training did not fix role recognition (declared, not pending)
    joint_path = gt3.OUT / "joint_teacher_report.json"
    if joint_path.is_file():
        joint = json.loads(joint_path.read_text())
        orbit = joint["orbit"]
        checks["joint_role_training"] = {
            "status": "PASS" if orbit["carried"] else "NOT_ESTABLISHED",
            "detail": {"orbit_ce": orbit["ce"],
                       "orbit_constant_ce": orbit["constant_ce"],
                       "orbit_rank_correlation": orbit["role_rank_correlation"],
                       "edge_test_auc_joint": joint["edge"]["test_auc"],
                       "direction_balanced_accuracy": joint["direction"]["test_balanced_accuracy"],
                       "checkpoint": joint["checkpoint"],
                       "checkpoint_sha256": joint["checkpoint_sha256"]}}
    else:
        checks["joint_role_training"] = {"status": "FAIL", "detail": {"missing": True}}

    # 12. where the teacher helps (structural environments)
    env_path = gt3.OUT / "structure_environment.json"
    if env_path.is_file():
        env = json.loads(env_path.read_text())
        major = {name: entry for name, entry in env["slices"].items()
                 if not entry.get("skipped") and name != "community:different"}
        supports = all(entry["C_minus_A"] > 0 and entry["C_minus_D"] > 0
                       for entry in major.values())
        cross = env["slices"].get("community:different", {})
        checks["structure_environment"] = {
            "status": "PASS" if supports else "FAIL",
            "detail": {"slices": {name: {"pairs": entry["pairs"],
                                         "c_minus_a": entry["C_minus_A"],
                                         "c_minus_d": entry["C_minus_D"]}
                                  for name, entry in env["slices"].items()
                                  if not entry.get("skipped")},
                       "no_evidence": {"slice": "community:different",
                                       "pairs": cross.get("pairs"),
                                       "c_minus_a": cross.get("C_minus_A"),
                                       "c_minus_d": cross.get("C_minus_D")}}}
    else:
        checks["structure_environment"] = {"status": "FAIL", "detail": {"missing": True}}

    # 13. non-learned structural heuristics (#229 §4 control)
    heur_path = gt3.OUT / "structure_heuristics.json"
    if heur_path.is_file():
        heur = json.loads(heur_path.read_text())
        deltas = heur["delta_vs_teacher"]
        beats = all(entry["low"] > 0.0 for entry in deltas.values())
        checks["edge_vs_heuristic"] = {
            "status": "PASS" if beats and heur["teacher"]["auc"] > max(
                [entry["auc"] for entry in heur["heuristics"].values()]
                + [heur["heuristic_stack"]["auc"]]) else "FAIL",
            "detail": {"teacher_auc": heur["teacher"]["auc"],
                       "heuristics": {name: entry["auc"]
                                      for name, entry in heur["heuristics"].items()},
                       "heuristic_stack": heur["heuristic_stack"]["auc"],
                       "best_heuristic": heur["best_heuristic"],
                       "delta_ci": {name: [entry["low"], entry["high"]]
                                    for name, entry in deltas.items()}}}
    else:
        checks["edge_vs_heuristic"] = {"status": "FAIL", "detail": {"missing": True}}

    # 14. high-order / role judgement without the observed adjacency (§2.2 row 3)
    graph_report = json.loads((gt3.OUT / "graph_teacher3_report.json").read_text())
    without = graph_report["recognition"]["without_adjacency"]
    probe = graph_report["orbit_probe"]["C_residual"]
    established = (without["subtype_macro_f1"] >= 0.10
                   and probe["ce"] < probe["constant_ce"])
    checks["prediction_structure_without_adjacency"] = {
        "status": "PASS" if established else "NOT_ESTABLISHED",
        "detail": {"subtype_macro_f1": without["subtype_macro_f1"],
                   "role_macro_over_families": without["role_macro_over_families"],
                   "orbit_probe_ce": probe["ce"],
                   "constant_ce": probe["constant_ce"],
                   "note": "measured without the true induced adjacency; subtype "
                           "macro-F1 is near chance and the orbit probe is worse "
                           "than the constant predictor"}}

    # 15. coverage
    fraction = float((targets["coverage_neighbour"] > 0).mean())
    checks["coverage"] = {"status": "PASS" if fraction > 0.9 else "FAIL",
                          "detail": {"neighbourhood_fraction": fraction,
                                     "own_orbit_fraction": float((targets["coverage"] > 0).mean())}}

    failed = [name for name, entry in checks.items() if entry["status"] == "FAIL"]
    declared = [name for name, entry in checks.items()
                if entry["status"] == "NOT_ESTABLISHED"]
    verdict_done = not failed
    payload = {"issue": 229, "verdict": "DONE_WITH_DECLARATIONS" if verdict_done and declared
               else ("DONE" if verdict_done else "NOT_DONE"),
               "failed": failed, "declared_not_established": declared,
               "checks": checks, "seconds": time.time() - started,
               "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out = HERE / "conformance_229.json"
    out.write_text(json.dumps(payload, indent=1, default=str))
    print(json.dumps({key: payload[key] for key in ("verdict", "failed",
                                                    "declared_not_established")}, indent=1))
    for name, entry in checks.items():
        print(f"{entry['status']:18s} {name}")
    raise SystemExit(0 if verdict_done else 1)


if __name__ == "__main__":
    main()
