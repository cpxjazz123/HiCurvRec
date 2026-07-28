#!/usr/bin/env python3
"""
Task #194 — Aggregate Stage 4 metrics + K0 diagnosis → verdict.md
读取:
- logs/task194/k0_l0_err.tsv (来自 task194_k0_l0_err_diagnose.py)
- products/task194/t5mini_k0*/test_metrics.json (Stage 4 输出)
- products/task194/hrqvae_k0*/collision_history.tsv 或从日志 parse

输出:
- verdicts/task194_k0_capacity_result.md
"""
import os, json, glob
import numpy as np

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
DIAG_TSV = f"{REPO}/logs/task194/k0_l0_err.tsv"
DIAG_JSON = f"{REPO}/verdicts/task194_k0_capacity_diagnose.json"
OUT_VERDICT = f"{REPO}/verdicts/task194_k0_capacity_result.md"


def load_diag():
    if not os.path.isfile(DIAG_JSON):
        return {}
    with open(DIAG_JSON) as f:
        return json.load(f)


def load_stage4():
    out = {}
    for K0 in [32, 64, 128, 256]:
        # 找 t5mini_k0K0/Instruments/ 下任意 test_metrics.json
        candidates = glob.glob(
            f"{REPO}/products/task194/t5mini_k0{K0}/**/test_metrics.json",
            recursive=True
        )
        if candidates:
            with open(candidates[0]) as f:
                d = json.load(f)
            out[K0] = d
    return out


def main():
    diag = load_diag()
    stage4 = load_stage4()

    K0_LIST = [32, 64, 128, 256]

    lines = []
    lines.append("# Task #194 — K0 码本容量扫描 verdict\n")
    lines.append("> **机制因果验证 #3**: K0={32, 64, 128, 256} 单变量扫, 验证 K0 是否 controlling variable.\n\n")

    lines.append("## 1. Stage 1 — collision 演化 (per K0)\n\n")
    lines.append("| K0 | n_epochs | collision_min | collision_final | final/min 比值 | L0_err_min | L0_err_final | corr(L0_err, collision) |\n")
    lines.append("|----|----------|---------------|-----------------|-----------------|------------|--------------|--------------------------|\n")
    for K0 in K0_LIST:
        key = f"K0_{K0}"
        if key not in diag:
            lines.append(f"| {K0} | — | — | — | — | — | — | — |\n")
            continue
        d = diag[key]
        lines.append(f"| **{K0}** | {d['n_epochs']} | {d['collision_min']:.4f} | "
                     f"{d['collision_final']:.4f} | **{d['collision_final_over_min']:.3f}** | "
                     f"{d['l0_err_min']:.4f} | {d['l0_err_final']:.4f} | "
                     f"{d['corr_l0_err_collision']:.3f} |\n")

    lines.append("\n## 2. Stage 4 — 下游 R@10 (per K0)\n\n")
    lines.append("| K0 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | 决策 vs baseline 0.1020 |\n")
    lines.append("|----|-----|------|------|-----|------|------|------------------------|\n")
    for K0 in K0_LIST:
        if K0 not in stage4:
            lines.append(f"| {K0} | — | — | — | — | — | — | ❌ 缺数据 |\n")
            continue
        d = stage4[K0]
        r10 = d.get("R@10", 0)
        if r10 > 0.1020:
            decision = f"✅ GO (+{(r10-0.1020)/0.1020*100:.1f}%)"
        elif r10 > 0.0970:
            decision = f"≈ HOLD ({(r10-0.1020)/0.1020*100:+.1f}%)"
        else:
            decision = f"❌ NO-GO ({(r10-0.1020)/0.1020*100:+.1f}%)"
        lines.append(f"| **{K0}** | {d.get('R@5', 0):.4f} | **{r10:.4f}** | {d.get('R@20', 0):.4f} | "
                     f"{d.get('N@5', 0):.4f} | {d.get('N@10', 0):.4f} | {d.get('N@20', 0):.4f} | "
                     f"{decision} |\n")

    # 假设判据
    lines.append("\n## 3. 假设判据\n\n")
    lines.append("### R1: K0 ↑ → collision 单调↓\n")
    coll_finals = []
    for K0 in K0_LIST:
        key = f"K0_{K0}"
        if key in diag:
            coll_finals.append((K0, diag[key]["collision_final"]))
    if len(coll_finals) >= 2:
        colls = [c for _, c in coll_finals]
        is_mono = all(colls[i] >= colls[i+1] for i in range(len(colls)-1))
        lines.append(f"- collision_finals: {[f'{c:.3f}' for _, c in coll_finals]}\n")
        lines.append(f"- 单调: {'✅' if is_mono else '❌'}\n")

    lines.append("\n### R2: K0 ↑ → final/min 比值单调↓\n")
    ratios = []
    for K0 in K0_LIST:
        key = f"K0_{K0}"
        if key in diag:
            ratios.append((K0, diag[key]["collision_final_over_min"]))
    if len(ratios) >= 2:
        ratio_vals = [r for _, r in ratios]
        is_mono = all(ratio_vals[i] >= ratio_vals[i+1] for i in range(len(ratio_vals)-1))
        lines.append(f"- ratios: {[f'{r:.3f}' for _, r in ratios]}\n")
        lines.append(f"- 单调: {'✅' if is_mono else '❌'}\n")

    lines.append("\n## 4. 综合判定\n\n")
    lines.append("- ✅ R1+R2 都立 → **K0 是 controlling variable**\n")
    lines.append("- ❌ 任一不立 → K0 不是 controlling variable, 需要找别的\n")
    lines.append("- ⚠️ K0=256 时 final collision 接近 0, 但下游 R@10 可能因为 L1/L2 退化\n")

    lines.append("\n## 5. 关键决策点 (R11.3)\n\n")
    lines.append("- K0 选择: 4 臂 K0={32, 64, 128, 256}, 跨越 8× 容量. 单一变量 K0.\n")
    lines.append("- 复用 task191 诊断脚本骨架: 直接遍历 4 臂 ckpt 算 L0_err.\n")
    lines.append("- Stage 3/4 全部重训 (K0 改了, SID 全变, 必须重训).\n")

    lines.append("\n## 6. 后续建议\n\n")
    lines.append("- 若 K0 是 controlling variable → 进一步实验: K0=512 看 collision → 0?\n")
    lines.append("- 若 K0 不是 → 下一个候选: decoder 重建 loss 权重 (类似 β 但只针对 decoder)\n")

    lines.append("\n---\n\n")
    lines.append(f"**result:** Task #194 K0 容量扫描 verdict 完成. 详见上表 collision 演化 + Stage 4 R@10.\n")

    with open(OUT_VERDICT, "w") as f:
        f.writelines(lines)
    print(f"[Task #194] Wrote {OUT_VERDICT}")


if __name__ == "__main__":
    main()