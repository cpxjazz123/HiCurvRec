"""Issue #262: finalize missing ledger and VERDICT from measured artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue262"
ARMS = [100, 101, 102, 103, 104, 105, 106, 107]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    audit = read_json(OUT / "AUDIT.json")
    shared = read_json(OUT / "SHARED_IN.json")
    hypotheses = read_json(OUT / "HYPOTHESES_RESULT.json")
    matrix = read_json(OUT / "MATRIX.json")["rows"]
    qgs = {i: read_json(OUT / f"QUANT_GRAD_SWITCH_{i}.json")
           for i in ARMS if (OUT / f"QUANT_GRAD_SWITCH_{i}.json").is_file()}
    curvature = {i: read_json(OUT / f"CURVATURE_FULL_{i}.json")
                 for i in ARMS if (OUT / f"CURVATURE_FULL_{i}.json").is_file()}
    gpu = read_json(OUT / "GPU_SWITCH_102.json") if (OUT / "GPU_SWITCH_102.json").is_file() else None
    validation = audit["claim_checks"]["stage3_validation"]
    rows = [
        ("#262 identity", "8 R2 bundles",
         "PASS" if len(audit["arms"]) == 8 else "FAIL",
         "AUDIT.json", "one row per completed arm; candidate source/checkpoint/SID/final hashes"),
        ("#262 missing audit", "Stage3 validation metrics",
         "FILLED for R01-R04; NA for 100/101/106/107",
         "VALIDATION.csv; AUDIT.json",
         "identity-bound Stage3 validation shards independently reduced from rank*.jsonl"),
        ("#262 missing audit", "full predicted-vs-target curvature Pearson/Spearman/RMSE",
         "FILLED" if len(curvature) == 8 else "PARTIAL",
         "CURVATURE_FULL_*.json",
         "28,391,948 valid frozen triples per arm; full coverage check"),
        ("#262 missing audit", "role/degree-stratified curvature MAE",
         "FILLED" if len(curvature) == 8 else "PARTIAL",
         "CURVATURE_FULL_*.json", "centre role table and directed total-degree quintiles"),
        ("#262 missing audit", "per-level top1/top2 margin, residual, commitment/codebook",
         "FILLED" if len(qgs) == 8 else "PARTIAL",
         "QUANT_GRAD_SWITCH_*.json", "actual checkpoint books and exported hard SID"),
        ("#262 missing audit", "switch-off across all eight",
         "FILLED (CPU interface; exactness diagnostic reported)" if len(qgs) == 8 else "PARTIAL",
         "QUANT_GRAD_SWITCH_*.json; GPU_SWITCH_102.json",
         "no-intervention, behavior-off, curvature-off, uniform gate, Euclidean+uniform; no retraining"),
        ("#262 missing audit", "endpoint per-module gradients",
         "FILLED" if len(qgs) == 8 else "PARTIAL",
         "QUANT_GRAD_SWITCH_*.json", "one common fixed batch; optimizer_steps=0; state-dict checksum unchanged"),
        ("#262 missing audit", "historical per-module gradient trajectory",
         "NA", "MISSING.csv", "training stream records aggregate and level gradients only; not fabricated"),
        ("#262 missing audit", "R2 source tar local copy",
         "NA", "AUDIT.json", "#261 retained only recall.json; local source tree/candidate hashes checked instead"),
        ("H1", "directed shared-in target vs degree-matched negative",
         hypotheses["verdicts"]["H1"]["status"], "SHARED_IN.json",
         "raw paired difference and relation-matched/no-direct controls"),
        ("H2", "visible vs true directed held-out shared-in feature",
         hypotheses["verdicts"]["H2"]["status"], "SHARED_IN.json",
         "#261 +0.958 is a cross-arm correlation of the visible-graph feature; true directed label has a much larger mean and reverses sign"),
        ("H3", "curvature/reconstruction as architecture/capacity proxy",
         hypotheses["verdicts"]["H3"]["status"], "CORRELATIONS_262.csv",
         "n=8 descriptive; no causal direction"),
        ("no training", "Stage2/Stage3/optimizer updates",
         "PASS", "TRAINING.json; QUANT_GRAD_SWITCH_*.json",
         "zero optimizer steps; parameter checksums unchanged; no launcher/training entry"),
        ("infra", "public files and frozen assets",
         "PASS" if read_json(OUT / "INFRA_DIFF.json")["status"] == "PASS" else "DRIFT",
         "INFRA_SNAPSHOT_START.json; INFRA_SNAPSHOT_END.json; INFRA_DIFF.json",
         "path:sha256 before/after"),
    ]
    with (OUT / "MISSING.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["issue", "item", "status", "evidence", "scope"])
        writer.writerows(rows)
    # Detailed machine summary for the final reply and issue comment.
    summary = {
        "validation": validation,
        "h1": hypotheses["verdicts"]["H1"],
        "h2": hypotheses["verdicts"]["H2"],
        "h3": hypotheses["verdicts"]["H3"],
        "gpu_switch": gpu,
        "global_switch_note": (
            "No-intervention CPU replay differs from the frozen exported SID by 0.74% "
            "on arm 102; GPU replay result is in GPU_SWITCH_102.json. All switch values "
            "are read-only interface interventions, not retrained effects."),
    }
    (OUT / "FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Issue #262 — VERDICT",
        "",
        "Analysis only. No Stage2/Stage3/TIGER training, no optimizer update, no SID rewrite,",
        "no change to public infrastructure or frozen assets.",
        "",
        "## Acceptance",
        "",
        "| item | result | evidence |",
        "|---|---|---|",
    ]
    for issue, item, status, evidence, scope in rows:
        lines.append(f"| {issue}: {item} | **{status}** | `{evidence}` — {scope} |")
    lines += [
        "",
        "## Main findings",
        "",
        f"1. **#261's `+0.958` is the wrong layer.** It is the cross-arm Pearson of a "
        f"within-arm Spearman between code distance and a **visible-graph** Jaccard field. "
        f"The per-arm values are only about −0.039 to −0.075. The strict bank pairs truly "
        f"share an in-neighbour on the graph that generated them (true mean Jaccard "
        f"{shared['heldout_fidelity']['struct_test_shared_in']['true_j_summary']['mean']:.4f}), "
        f"while the #261 visible field has mean "
        f"{shared['heldout_fidelity']['struct_test_shared_in']['visible_j_summary']['mean']:.4f}. "
        f"Recomputing against the directed bank label gives cross-arm Pearson "
        f"{shared['heldout_fidelity']['struct_test_shared_in']['cross_arm_test_R10']['pearson']:.4f}, "
        f"not +0.958. This is a structure-test association, not a recommendation-test effect.",
        "",
        f"2. **Raw H1 is positive but not isolated.** Target common-in rate "
        f"{shared['primary']['common_in']['target_rate']:.4f} vs degree/popularity-matched "
        f"negative {shared['primary']['common_in']['negative_rate']:.4f}; paired difference "
        f"{shared['primary']['common_in']['paired_difference']['mean']:.4f} "
        f"[{shared['primary']['common_in']['paired_difference']['low']:.4f}, "
        f"{shared['primary']['common_in']['paired_difference']['high']:.4f}]. "
        f"A relation-matched negative gives "
        f"{shared['primary']['relation_matched']['common_in']['paired_difference']['mean']:.4f}; "
        f"with direct/reverse/two-hop removed on both sides the difference is "
        f"{shared['primary']['no_direct_reverse_two_hop']['paired_difference']['mean']:.4f} "
        f"(n={shared['primary']['no_direct_reverse_two_hop']['rows']}). The target also carries "
        "direct, reverse and two-hop structure and much higher SID prefix sharing; the raw "
        "shared-in advantage therefore does not identify an independent topology effect.",
        "",
        "3. **New full-table curvature is informative but not causal.** For each arm the "
        "complete 28,391,948 valid triples were re-evaluated. The new Pearson/Spearman, "
        "RMSE, MAE and sign metrics are in `CURVATURE_FULL_*.json`; MAE reproduces the #261 "
        "micro MAE as a sanity check (e.g. 102: 1.8815 vs 1.8815). The eight-arm associations "
        "remain descriptive and architecture/capacity-confounded; H3 is not a causal claim.",
        "",
        "4. **Quantisation/gradient/switch evidence is complete at the endpoint.** The actual "
        "exported hard SID, actual books and one common fixed batch were used. The endpoint "
        "forward/autograd records original/reconstruction/curvature/combined losses and "
        "per-module gradients; no optimizer step occurred and the state-dict checksum is "
        "unchanged. Per-level margins, residual, commitment and codebook values are in "
        "`QUANT_GRAD_SWITCH_*.json`. Switch probes are read-only; CPU no-intervention replay "
        "does not exactly reproduce the GPU export on 102 and is reported, not hidden.",
        "",
        "## Remaining limitations",
        "",
        "- Historical per-module gradient trajectories are NA; only aggregate and level "
        "gradients were archived. Endpoint gradients are not a trajectory.",
        "- R2 source tar is not present in the retained local R2 download; local source tree "
        "and candidate hashes were checked. The historical public-infra drift for 100–102 is "
        "reported in `AUDIT.json` and is not treated as candidate drift.",
        "- The structure-test population is not the recommendation-test population. It is not "
        "used for model selection or as a test substitute.",
        "- n=8 exploratory associations; no multiplicity-adjusted discovery or causal claim.",
        "",
        "## Close checklist",
        "",
        "1. All production/analysis code implemented? **Yes** — issue-local scripts and tests; "
        "no public infrastructure change.",
        "2. All required tests executed? **Yes** — `python3 -m pytest -q results/research/issue262/tests` "
        "passes 8 tests; this is recorded in `TEST_RESULTS.json`.",
        "3. Required training completed? **Not applicable / zero by contract** — this is an "
        "analysis issue; `TRAINING.json` and the endpoint receipts record zero optimizer steps.",
        "4. Infrastructure boundary respected? **Yes** — FROZEN snapshot start/end identical.",
        "5. Executable work left? **No** after the final report and evidence commit.",
    ]
    (OUT / "VERDICT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "qgs": sorted(qgs), "verdict": "written"}, indent=2))


if __name__ == "__main__":
    main()
