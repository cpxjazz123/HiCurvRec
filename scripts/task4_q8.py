"""task6_q8.py — Successive-refinement gap G_l

定义:
    G_l = D_{l-1} - D_l

其中 D_l 来自 task17 = "用前 l 层码本重建整层 embedding 的拟合距离"。
- G_l > 0: 增量加入第 (l+1) 层能进一步改善拟合 (l → l+1 有信息增益)
- G_l < 0: 第 (l+1) 层 **恶化**了拟合 (successive-refinement 失败 — 重新拟合)

简化实现：直接加载 task17 JSON 的 D_l 数据计算 G_l。

Usage:
    python3 task6_q8.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID/result")
TASK19_DIR = RESULT_DIR / "task17"


def load_task15_D_l(group: str) -> list:
    """加载 task17 量 5 (D_l) JSON，返回 [D_1, D_2, D_3]"""
    candidates = [
        TASK19_DIR / f"task15_{group}.json",
        TASK19_DIR / "task5_all_algorithms.json",
        TASK19_DIR / "task5_batch2.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        with open(p) as f:
            raw = json.load(f)
        # 多种 schema 容错: per-group 文件直接用, all_algorithms 文件用 key
        if "layers" in raw:
            data = raw
        elif group in raw:
            data = raw[group]
        else:
            continue
        # 提取 layers[].mean_dist_l
        if "layers" in data:
            D_l = [L["mean_dist_l"] for L in data["layers"]]
            return D_l
    raise FileNotFoundError(
        f"No task17 D_l data found for group {group}, tried {candidates}"
    )


def compute_G_l(D_l: list, l: int) -> float:
    """G_l = D_{l-1} - D_l (l >= 1)
    D_0 设为嵌入的 L2 范数均值 (占位), G_1 = D_0 - D_1 表示"加入第一层的信息增益"

    注释: D_0 在 task17 没用, 这里用 ||emb|| 的平均作为"无编码时的重建距离"基准
    """
    if l == 0:
        return None  # no G_0
    return D_l[l - 1] - D_l[l]


def main():
    # 三组 (group letter -> file name suffix)
    G_data = {}
    groups = {
        "A_baseline": "A_baseline",
        "B_mmq": "B_mmq",
        "C_gsrq": "C_gsrq",
    }
    for name, letter in groups.items():
        try:
            D_l = load_task15_D_l(letter)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        G_l = []
        for l in range(0, len(D_l)):
            if l == 0:
                G_l.append(None)
            else:
                G_l.append(compute_G_l(D_l, l))
        G_data[name] = {"D_l": D_l, "G_l": G_l}
        print(f"[{name}] D_l = {[round(d, 4) for d in D_l]}")
        print(f"          G_l = {[None if g is None else round(g, 4) for g in G_l]}")

    if not G_data:
        print("[error] no D_l data found in task17")
        return

    # Save JSON
    out_json = RESULT_DIR / "task18" / "task6_q8_G_l.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"task": "task16_q8_G_l", "groups": G_data}, f, indent=2)
    print(f"\n[saved] {out_json}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {
        "A_baseline": "#1f77b4",
        "B_mmq": "#ff7f0e",
        "C_gsrq": "#2ca02c",
    }
    labels = {
        "A_baseline": "A (baseline)",
        "B_mmq": "B (MMQ direction-only)",
        "C_gsrq": "C (GSRQ gain-shape)",
    }
    layers = [1, 2, 3]

    # 左图：D_l
    ax = axes[0]
    for name, data in G_data.items():
        ax.plot(layers, data["D_l"], "-o", color=colors[name], label=labels[name])
    ax.set_xlabel("Layer l")
    ax.set_ylabel("D_l (prefix-reconstruction L2 distance)")
    ax.set_title("Layer-wise fitting distance (task17 D_l)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 右图：G_l
    ax = axes[1]
    for name, data in G_data.items():
        G_plot = [g if g is not None else 0 for g in data["G_l"]]
        ax.plot(layers, G_plot, "-o", color=colors[name], label=labels[name])
        # 标注 G_l < 0
        for l, g in enumerate(data["G_l"]):
            if g is not None and g < 0:
                ax.annotate(f"G_{l+1}<0", xy=(l + 1, g), xytext=(l + 1, g - 0.02),
                            fontsize=8, color=colors[name], ha="center")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("Layer l")
    ax.set_ylabel("G_l = D_{l-1} - D_l  (positive = info gain)")
    ax.set_title("Successive-refinement gap G_l (positive = gain, negative = collapse)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_png = RESULT_DIR / "task18" / "task16_q8_G_l.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")

    # Verdict
    verdict_lines = [
        "# Task 20 量 8 — Successive-Refinement Gap G_l\n",
        "\n## 量化结果（successive refinement gap）\n",
    ]
    for name, data in G_data.items():
        verdict_lines.append(f"\n### {labels[name]}\n")
        verdict_lines.append("\n| Layer l | D_l | G_l = D_{l-1} - D_l |\n|--------|-----|---------------------|\n")
        for l, (d, g) in enumerate(zip(data["D_l"], data["G_l"])):
            g_str = "—" if g is None else f"{g:+.4f}"
            verdict_lines.append(f"| {l+1} | {d:.4f} | {g_str} |\n")

    verdict_lines.append("\n## 判定\n")
    for name, data in G_data.items():
        sign_l1 = data["G_l"][1] if len(data["G_l"]) > 1 else None
        sign_l2 = data["G_l"][2] if len(data["G_l"]) > 2 else None
        if sign_l1 is not None and sign_l1 < 0:
            v_l1 = "❌ G_1<0: 第2层码本**恶化**了拟合"
        elif sign_l1 is not None:
            v_l1 = f"✅ G_1={sign_l1:+.4f}: 第2层码本提供信息增益"
        else:
            v_l1 = "n/a"
        if sign_l2 is not None and sign_l2 < 0:
            v_l2 = "❌ G_2<0: 第3层码本**恶化**了拟合"
        elif sign_l2 is not None:
            v_l2 = f"✅ G_2={sign_l2:+.4f}: 第3层码本提供信息增益"
        else:
            v_l2 = "n/a"
        verdict_lines.append(f"- **{name}**: {v_l1} | {v_l2}\n")

    out_md = RESULT_DIR / "task18" / "task6_q8_verdict.md"
    out_md.write_text("".join(verdict_lines))
    print(f"[saved] {out_md}")


if __name__ == "__main__":
    main()
