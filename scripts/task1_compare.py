"""task3_compare.py — 汇总 task15 三组（A/B/C）评估结果生成最终对比表

读 task3_eval_A_baseline.json / task3_eval_B_mmq.json / task3_eval_C_gsrq.json
组合成一张三方对比表 + 收敛判定 + Idea 3 故事线建议

输出:
    /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task5/task3_comparison.json
    /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task5/task3_comparison.md
"""

import json
import sys
from pathlib import Path
from datetime import datetime

RESULT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task5")

# group_letter -> json filename
FILE_NAMES = {
    "A": "task3_eval_A_baseline.json",
    "B": "task3_eval_B_mmq.json",
    "C": "task3_eval_C_gsrq.json",
}

# group_letter -> (display_name, color_hex)
NAMES = {
    "A": "标准 RKMeans (normalize=false)",
    "B": "MMQ 纯方向 (normalize=true)",
    "C": "GSRQ gain-shape 融合 (新方法)",
}


def main():
    results = {}
    for g, fn in FILE_NAMES.items():
        p = RESULT_DIR / fn
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        with open(p) as f:
            results[g] = json.load(f)

    if len(results) < 2:
        print(f"[error] at least 2 groups needed for comparison, found {list(results.keys())}")
        sys.exit(1)

    comparison = {
        "task": "task13_gain_shape_fusion_vs_mmq",
        "data": "Amazon Toys (11924 items, 2048-dim flan-t5-xl embedding)",
        "generated_at": datetime.now().isoformat(),
        "groups": results,
    }

    # Delta computation
    if "A" in results and "B" in results:
        comparison["delta_A_minus_B_R10"] = results["A"]["Recall@10"] - results["B"]["Recall@10"]
        comparison["delta_A_minus_B_R5"] = results["A"]["Recall@5"] - results["B"]["Recall@5"]
    if "C" in results and "B" in results:
        comparison["delta_C_minus_B_R10"] = results["C"]["Recall@10"] - results["B"]["Recall@10"]
    if "C" in results and "A" in results:
        comparison["delta_C_minus_A_R10"] = results["C"]["Recall@10"] - results["A"]["Recall@10"]

    # Verdict logic (predetermined per task15.md)
    r_diff_cb = comparison.get("delta_C_minus_B_R10", 0)
    r_diff_ba = comparison.get("delta_A_minus_B_R10", 0)
    if "C" in results:
        if abs(r_diff_cb) < 0.003:
            verdict = (
                f"Group C ≈ Group B (|delta|={abs(r_diff_cb):.4f} < 0.003): "
                "MMQ 纯方向是最优 / GSRQ 无显著增益"
            )
            story = (
                "**审计/诊断型**：系统性检验残差量化假设 "
                "(gain 信息微弱 / 方向主导)，GSRQ gain-shape 融合未带来 Recall 增益"
            )
        elif r_diff_cb > 0.005:
            verdict = (
                f"Group C > Group B +{r_diff_cb:.4f} (>0.005): "
                "GSRQ gain-shape 融合有 Recall 增益价值"
            )
            story = "**新方法型**：提出 GSRQ 残差量化新算法"
        else:
            verdict = (
                f"Group C - Group B = {r_diff_cb:+.4f}: 中间地带，gain-shape 在边界"
            )
            story = (
                "**混合型**：报告纯方向 vs gain-shape 在 Toys 数据上的差异，"
                "需用 task16/19/20 诊断量解释"
            )
    else:
        # Only A vs B without C
        verdict = "Group C 评估未完成（A vs B 先行判定）"
        story = "见 task16/19/20 的诊断量结果"

    if r_diff_ba and r_diff_ba > 0:
        verdict_a_b = (
            f"Group A > Group B (+{r_diff_ba:.4f}): magnitude > direction, "
            "与 MMQ 假设矛盾"
        )
    elif r_diff_ba and r_diff_ba < 0:
        verdict_a_b = (
            f"Group B > Group A (+{-r_diff_ba:.4f}): direction 主导，验证 MMQ"
        )
    else:
        verdict_a_b = "Group A ≈ Group B"

    comparison["verdict"] = verdict
    comparison["verdict_A_vs_B"] = verdict_a_b
    comparison["paper_storyline"] = story

    # Write JSON
    out_json = RESULT_DIR / "task3_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(comparison, indent=2))
    print(f"[saved] {out_json}")

    # Write Markdown table
    md = build_markdown(results, comparison)
    out_md = RESULT_DIR / "task3_comparison.md"
    out_md.write_text(md)
    print(f"[saved] {out_md}")


def build_markdown(results, comp):
    lines = [
        "# Task 17 (Idea 3 收官): 三方对比 — gain-shape 融合 vs MMQ 纯方向",
        "",
        f"**数据集**：{comp['data']}",
        f"**生成时间**：{comp['generated_at']}",
        "",
        "## 三方对比表",
        "",
        "| 组 | 名称 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | 碰撞率 (3-layer) | n_unique_3layer | 评估用户数 |",
        "|----|------|---------|----------|--------|--------|----------------|----------------|-----------|",
    ]

    for g in ["A", "B", "C"]:
        if g not in results:
            continue
        r = results[g]
        lines.append(
            f"| **{g}** | {NAMES[g]} | "
            f"{r.get('Recall@5', 0):.5f} | "
            f"{r.get('Recall@10', 0):.5f} | "
            f"{r.get('NDCG@5', 0):.5f} | "
            f"{r.get('NDCG@10', 0):.5f} | "
            f"{r.get('collision_rate_3layer', 0):.4f} | "
            f"{r.get('n_unique_3layer', 0)} | "
            f"{r.get('n_users_evaluated', 0)} |"
        )

    lines.append("")
    lines.append("## 关键差值")
    lines.append("")
    for k in [
        "delta_A_minus_B_R5",
        "delta_A_minus_B_R10",
        "delta_C_minus_B_R10",
        "delta_C_minus_A_R10",
    ]:
        if k in comp:
            lines.append(f"- **{k}** = `{comp[k]:+.5f}`")
    lines.append("")
    lines.append("## 判定")
    lines.append("")
    lines.append(f"> {comp['verdict']}")
    lines.append("")
    lines.append(f"**A vs B 单独判定**：{comp['verdict_A_vs_B']}")
    lines.append("")
    lines.append("## 论文故事线建议")
    lines.append("")
    lines.append(f"> {comp['paper_storyline']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 与诊断量对应 (task16/19/20 量化结果)")
    lines.append("")
    lines.append("详见 `result/task4_19_20_summary.md`：")
    lines.append(
        "- 量 11 (跨层码本失配 Δ_1)：A=15.9, B=2.29, C=8.39 → "
        "A idea matters (baseline 也付出量化代价)"
    )
    lines.append(
        "- 量 5 (D_l)：B 深层拟合恶化 (0.35→0.46) → 反方向支持 A 是次优"
    )
    lines.append(
        "- 量 4 (f_radial)：C 73-83% > A 47-60% >> B 5-9% → "
        "gain-shape 显式编码最强烈"
    )
    lines.append(
        "- 量 6 (η_l)：C 11→108 (扩张 10×) vs B 22 (停滞) vs A→2048 (白噪声) → "
        "C 残差子空间最丰富"
    )
    lines.append("")
    lines.append("*Generated by task3_compare.py — Idea 3 收官*")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
