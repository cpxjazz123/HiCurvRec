#!/usr/bin/env python3
"""Issue #119 Artifact Generator (2026-08-10).

读:
  - taskA/_history/issue118_c3_correctness/unit_test_report.json
  - taskA/_history/issue118_c3_correctness/c3_3epoch_smoke/final_verdict.json
  - taskA/_history/issue118_c3_correctness/c1_30epoch_smoke/final_verdict.json
  - taskA/_history/issue118_c3_correctness/c3_30epoch_smoke/final_verdict.json
  - taskA/_history/issue118_c3_correctness/c1_30epoch_smoke/training_log.json
  - taskA/_history/issue118_c3_correctness/c3_30epoch_smoke/training_log.json

生成:
  - taskA/_history/issue118_c3_correctness/c1_vs_c3_comparison.md
  - taskA/_history/issue118_c3_correctness/decision.json (GO-C3 / NO-GO-C3)

运行:
  python3 -u scripts/generate_issue118_c3_correctness_report.py
"""
import argparse
import json
import sys
from pathlib import Path

GENRE_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec"
CORRECTNESS_DIR = f"{GENRE_ROOT}/taskA/_history/issue118_c3_correctness"
COMPARISON_OUT = f"{CORRECTNESS_DIR}/c1_vs_c3_comparison.md"
DECISION_OUT = f"{CORRECTNESS_DIR}/decision.json"


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt_float(x, n=4):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{n}f}"
    except (ValueError, TypeError):
        return str(x)


def fmt_list(vals, n=4):
    if vals is None:
        return "[]"
    if not isinstance(vals, list):
        return str(vals)
    return "[" + ", ".join(fmt_float(v, n) for v in vals) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness_dir", type=str, default=CORRECTNESS_DIR)
    parser.add_argument("--comparison_out", type=str, default=COMPARISON_OUT)
    parser.add_argument("--decision_out", type=str, default=DECISION_OUT)
    args = parser.parse_args()

    cd = Path(args.correctness_dir)
    unit_test = load_json(cd / "unit_test_report.json")
    phase1_verdict = load_json(cd / "c3_3epoch_smoke" / "final_verdict.json")
    c1_verdict = load_json(cd / "c1_30epoch_smoke" / "final_verdict.json")
    c3_verdict = load_json(cd / "c3_30epoch_smoke" / "final_verdict.json")
    c1_log = load_json(cd / "c1_30epoch_smoke" / "training_log.json")
    c3_log = load_json(cd / "c3_30epoch_smoke" / "training_log.json")

    # Decision 判定
    decision_pass = True
    decision_reasons = []

    # 1. unit test 5/5 PASS
    if unit_test:
        if not unit_test.get("overall_pass", False):
            decision_pass = False
            decision_reasons.append(f"unit_test FAIL: {unit_test.get('n_pass')}/{unit_test.get('n_total')}")
        else:
            decision_reasons.append(f"unit_test PASS: {unit_test.get('n_pass')}/{unit_test.get('n_total')}")
    else:
        decision_pass = False
        decision_reasons.append("unit_test_report.json missing")

    # 2. Phase 1 C3 link 存在
    if phase1_verdict:
        if not phase1_verdict.get("c3_link_exists", False):
            decision_pass = False
            decision_reasons.append(f"Phase1 C3 link missing: rel_grad={phase1_verdict.get('gate_a2_rel_grad_finite')}")
        else:
            decision_reasons.append("Phase1 C3 link EXISTS")
        if phase1_verdict.get("gate_a1_fail_fast", {}).get("status") != "PASS":
            decision_pass = False
            decision_reasons.append(f"Phase1 fail-fast FAIL: {phase1_verdict.get('gate_a1_fail_fast')}")
    else:
        decision_pass = False
        decision_reasons.append("Phase1 c3_3epoch_smoke missing")

    # 3. C3 Phase 2 R37 verdict PASS
    if c3_verdict:
        if c3_verdict.get("r37_decision", "").startswith("FAIL"):
            decision_pass = False
            decision_reasons.append(f"C3 Phase2 R37 FAIL: {c3_verdict.get('r37_decision')}")
        else:
            decision_reasons.append(f"C3 Phase2 R37 PASS")
    else:
        decision_pass = False
        decision_reasons.append("C3 Phase2 final_verdict missing")

    decision = "GO-C3" if decision_pass else "NO-GO-C3"
    decision_payload = {
        "issue": "#119",
        "decision": decision,
        "reasons": decision_reasons,
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").localtime()),
    }
    with open(args.decision_out, "w") as f:
        json.dump(decision_payload, f, indent=2)
    print(f"[decision] {decision}: {'; '.join(decision_reasons)}")

    # === c1_vs_c3_comparison.md ===
    md = []
    md.append("# Issue #119 — C1 vs C3 P0-Fix Matched Smoke Comparison (2026-08-10)")
    md.append("")
    md.append(f"## Decision: **{decision}**")
    md.append("")
    for r in decision_reasons:
        md.append(f"- {r}")
    md.append("")

    md.append("## 1. Unit Tests (P0-6)")
    md.append("")
    if unit_test:
        md.append(f"- overall_pass: {unit_test.get('overall_pass')}")
        md.append(f"- {unit_test.get('n_pass')}/{unit_test.get('n_total')} PASS")
        for t in unit_test.get("tests", []):
            status = "✓ PASS" if t.get("pass") else "✗ FAIL"
            md.append(f"  - {t.get('test')}: {status}")
    else:
        md.append("- unit_test_report.json 未生成")
    md.append("")

    md.append("## 2. Phase 1 (3-Epoch C3 Fail-Fast Smoke)")
    md.append("")
    if phase1_verdict:
        md.append(f"- C3 link exists: {phase1_verdict.get('c3_link_exists')}")
        md.append(f"- fail-fast: {phase1_verdict.get('gate_a1_fail_fast', {}).get('status')}")
        md.append(f"- rel_grad finite: {phase1_verdict.get('gate_a2_rel_grad_finite', {}).get('status')}")
        md.append(f"- vq_grad zero: {phase1_verdict.get('gate_a3_vq_grad_zero', {}).get('status')}")
        md.append(f"- final κ: {fmt_list(phase1_verdict.get('final_kappa'))}")
        md.append(f"- final c: {fmt_list(phase1_verdict.get('final_c'))}")
        md.append(f"- final L_rel: {fmt_float(phase1_verdict.get('final_l_rel'))}")
        md.append(f"- final util: {fmt_list(phase1_verdict.get('final_utilization'))}")
    else:
        md.append("- 未运行")
    md.append("")

    md.append("## 3. Phase 2 (30-Epoch C1 vs C3 Matched Smoke)")
    md.append("")
    md.append("### 3.1 Gate 判定")
    md.append("")
    md.append("| Gate | C1 | C3 |")
    md.append("|------|----|----|")
    if c1_verdict and c3_verdict:
        for gate_key in ["gate_a1_fail_fast", "gate_a2_5_way_audit", "gate_a3_recon_error", "gate_a4_collapse"]:
            c1g = c1_verdict.get(gate_key, {}).get("status", "N/A")
            c3g = c3_verdict.get(gate_key, {}).get("status", "N/A")
            md.append(f"| {gate_key} | {c1g} | {c3g} |")
    md.append("")

    md.append("### 3.2 final κ / c / util / H / S_all")
    md.append("")
    md.append("| 项 | C1 | C3 |")
    md.append("|----|----|----|")
    if c1_verdict and c3_verdict:
        md.append(f"| κ | {fmt_list(c1_verdict.get('final_kappa'))} | {fmt_list(c3_verdict.get('final_kappa'))} |")
        md.append(f"| c | {fmt_list(c1_verdict.get('final_curvature'))} | {fmt_list(c3_verdict.get('final_curvature'))} |")
        md.append(f"| util_3digit | {fmt_list(c1_verdict.get('final_utilization'))} | {fmt_list(c3_verdict.get('final_utilization'))} |")
        md.append(f"| entropy | {fmt_list(c1_verdict.get('final_entropy'))} | {fmt_list(c3_verdict.get('final_entropy'))} |")
        md.append(f"| S_all | {fmt_list(c1_verdict.get('all_pair_clip_ratio'))} | {fmt_list(c3_verdict.get('all_pair_clip_ratio'))} |")
        md.append(f"| std(κ) | {fmt_float(c1_verdict.get('kappa_per_layer_std'))} | {fmt_float(c3_verdict.get('kappa_per_layer_std'))} |")
        md.append(f"| std(c) | {fmt_float(c1_verdict.get('curvature_per_layer_std'))} | {fmt_float(c3_verdict.get('curvature_per_layer_std'))} |")
    md.append("")

    md.append("### 3.3 5-way gradient audit (final)")
    md.append("")
    md.append("| 层 | C1 vq_grad | C3 vq_grad | C1 rel_grad | C3 rel_grad |")
    md.append("|----|-----------|-----------|-------------|-------------|")
    if c1_verdict and c3_verdict:
        c1_vq = c1_verdict.get("gate_a2_5_way_audit", {}).get("vq_kappa_grad_final", [])
        c3_vq = c3_verdict.get("gate_a2_5_way_audit", {}).get("vq_kappa_grad_final", [])
        c1_rel = c1_verdict.get("gate_a2_5_way_audit", {}).get("relational_kappa_grad_final", [])
        c3_rel = c3_verdict.get("gate_a2_5_way_audit", {}).get("relational_kappa_grad_final", [])
        for l in range(3):
            md.append(f"| L{l} | {fmt_float(c1_vq[l] if l < len(c1_vq) else None)} | "
                      f"{fmt_float(c3_vq[l] if l < len(c3_vq) else None)} | "
                      f"{fmt_float(c1_rel[l] if l < len(c1_rel) else None)} | "
                      f"{fmt_float(c3_rel[l] if l < len(c3_rel) else None)} |")
    md.append("")

    md.append("### 3.4 R37 Decision")
    md.append("")
    if c1_verdict and c3_verdict:
        md.append(f"- C1 R37: {c1_verdict.get('r37_decision', 'N/A')}")
        md.append(f"- C3 R37: {c3_verdict.get('r37_decision', 'N/A')}")
    md.append("")

    md.append("## 4. R36 合规")
    md.append("")
    md.append("- 修复: global relation bank (P0-1) + 几何一致 (P0-2) + c_vq 切断 (P0-3) + 真实 audit (P0-4) + fail-fast (P0-5)")
    md.append("- 曲率机制: Poincaré InfoNCE on frozen KNN graph (几何信号), 非 LR/dropout/sweep")
    md.append("- 一律用 κ 主路径 (anchor/positive/negative 全 expmap0+proj_to_ball), 禁 fallback")
    md.append("")

    md.append("## 5. 引用文件清单")
    md.append("")
    md.append("- `taskA/_history/issue118_c3_correctness/unit_test_report.json` (5 tests)")
    md.append("- `taskA/_history/issue118_c3_correctness/c3_3epoch_smoke/` (Phase 1)")
    md.append("- `taskA/_history/issue118_c3_correctness/c1_30epoch_smoke/` (Phase 2 C1)")
    md.append("- `taskA/_history/issue118_c3_correctness/c3_30epoch_smoke/` (Phase 2 C3)")
    md.append("- `taskA/_history/issue118_c3_correctness/decision.json` (GO-C3 / NO-GO-C3)")
    md.append("")

    md.append("## 6. 下一步")
    md.append("")
    if decision == "GO-C3":
        md.append("- **GO-C3**: C3 relational-driven learnable curvature implementation VALID")
        md.append("- 下一 Issue 才进入 C1 vs C3 长周期 Stage2 experiment (≥100 epoch)")
        md.append("- 提交: commit + push + close Issue #119 (R20 闭环)")
    else:
        md.append("- **NO-GO-C3**: L_rel 当前机制未达 CORRECTNESS 阈值")
        md.append("- 失败机制需诊断 (relational gradient 接近 0 / util collapse / κ 撞 boundary / saturation chain)")
        md.append("- 禁止通过调 LR/dropout/sweep 绕过 correctness 问题 (R36)")
        md.append("- 优先判断 relational curvature objective 本身是否合理")
        md.append("- 提交: commit + push + Issue #119 comment + close (NO-GO 留作记录)")

    out = "\n".join(md) + "\n"
    with open(args.comparison_out, "w") as f:
        f.write(out)
    print(f"[generate_report] wrote {args.comparison_out} ({len(md)} lines)")
    print(f"[generate_report] wrote {args.decision_out} (decision={decision})")

    return 0


if __name__ == "__main__":
    sys.exit(main())