"""Issue #221: compose REPORT_DIRECTED_GRAPHLET_ROLE.md from the artifacts."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = ("D0", "D1", "D2", "D3", "D4")
NAMES = {"D0": "D0 / No aux", "D1": "D1 / Directed-z", "D2": "D2 / Directed-q-STE",
         "D3": "D3 / Directed-q-Codebook", "D4": "D4 / Frozen Directed Head"}
FAMILIES = ("K3", "P3", "C4", "P4", "K1_3")


def load(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def main() -> None:
    graph = load(REPO / "results" / "research" / "behavior_graph_directed" / "metadata.json", {})
    pool = load(HERE / "coverage.json", {})
    pretrained = load(HERE / "head_pretrained_directed.json", {})
    evaluation = load(HERE / "evaluation.json", {})
    lines: list[str] = []
    add = lines.append
    add("# 有向行为图上的 Graphlet subtype + Directed Orbit 预测(issue #221)")
    add("")
    add("**Issue #221**。五臂:D0 无辅助 / D1 head 读 z / D2 读硬量化 q 的旧 STE / "
        "D3 读 hard-forward+soft-backward 的 q_ST(梯度进实际码本)/ D4 冻结预训练头。"
        "冻结配置:3000 epoch、global batch 4096、seed=42、三层 256×32、每臂独占一张卡。")
    add("")
    add("## 1. 有向图(取代无向图成为唯一主训练结构)")
    add("")
    counts = graph.get("counts", {})
    degree = graph.get("degree", {})
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 节点 | {counts.get('nodes')} |")
    add(f"| 有向边 | {counts.get('directed_edges')} |")
    add(f"| 转移总数 | {counts.get('total_transitions')} |")
    add(f"| 互惠边 / 互惠对 | {counts.get('reciprocal_edges')} / {counts.get('reciprocal_pairs')} |")
    add(f"| 单向边 | {counts.get('one_way_edges')} |")
    add(f"| 互惠率 | {counts.get('reciprocity_rate'):.4f} |")
    add(f"| 无向参照边数 | {counts.get('undirected_reference_edges')} |")
    add(f"| out/in degree 中位数 | {degree.get('out_median')} / {degree.get('in_median')} |")
    add(f"| out/in degree 最大值 | {degree.get('out_max')} / {degree.get('in_max')} |")
    add(f"| 自环(已剔除) | {counts.get('self_loops_dropped')} |")
    add("")
    add("有向边数 = 单向边 + 互惠边,无向参照边数 = 单向边 + 互惠对:两条恒等式由测试断言。"
        "无向图 `results/research/behavior_graph/` 保留为历史对照,`graph_contract.verify_historical` 仍可校验它。")
    add("")
    add("## 2. 有向 Graphlet 池(同一批 #217 节点集合重新标注)")
    add("")
    if pool:
        add("| 家族 | 实例 | 有向 subtype(见/保留) | other | train / heldout |")
        add("|---|---|---|---|---|")
        for family in FAMILIES:
            row = pool["families"][family]
            add(f"| {family} | {row['instances']} | {row['subtypes_seen']} / "
                f"{row['subtypes_kept']} | {row['other_instances']} | "
                f"{row['train']} / {row['heldout']} |")
        add("")
        add(f"min-support = {pool['min_support']};低于阈值的 subtype 合并为 `other`,并在 coverage.json 记录。"
            "skeleton 不匹配实例数全部为 0:有向图的无向投影与 #217 完全一致,"
            "因此无向/有向两代结果在同一实例集合上可比。")
    add("")
    if pretrained:
        held = pretrained["heldout"]
        add("## 3. 冻结有向头预训练(D4)")
        add("")
        add(f"- 在**冻结的 TIGER_RQ-VAE 基线编码器**的 z 上训练 {pretrained['steps']} 步;")
        add(f"- heldout subtype macro-F1 **{held['subtype']['macro_f1']:.4f}**;"
            f"orbit macro-over-families **{held['role']['macro_over_families']:.4f}**;")
        for family, row in held["role"]["per_family"].items():
            add(f"  - {family}: orbit macro-F1 {row['macro_f1']:.4f}({row['classes']} 类)")
        add("- z 上的可读性不是 q 上的可读性:**E/D4 的价值是排除“可训练头独自吸收监督”**,不是最高分类分。")
        add("")
    add("## 4. 五臂结果")
    add("")
    if evaluation:
        add("| 臂 | subtype F1 | orbit F1 | collision | 有向边 L1 | 有向边 L1(加权) | "
            "role probe | L1 最大桶 | test_R@10 |")
        add("|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            row = evaluation.get(arm)
            if not row:
                continue
            head = row.get("head") or {}
            add(f"| {NAMES[arm]} | {head.get('subtype', {}).get('macro_f1')} | "
                f"{head.get('role', {}).get('macro_over_families')} | {row['collision']:.3f} | "
                f"{row['directed_edges']['edge_L1']:.4f} | "
                f"{row['directed_edges']['edge_L1_weighted']:.4f} | "
                f"{row['role_probe']['mean']} | {row['max_bucket_L1']:.3f} | "
                f"{row['recall'] if row['recall'] is not None else '—'} |")
        add("")
        add("### 各家族前缀保留(all_share,L1,null 已扣除)")
        add("")
        add("| 臂 | " + " | ".join(FAMILIES) + " |")
        add("|---" * (len(FAMILIES) + 1) + "|")
        for arm in ARMS:
            row = evaluation.get(arm)
            if not row:
                continue
            cells = " | ".join(f"{row['families'][family]['L1']['gain']:+.4f}" for family in FAMILIES)
            add(f"| {NAMES[arm]} | {cells} |")
    else:
        add("(等待训练完成)")
    add("")
    add("## 5. 判读")
    add("")
    if evaluation:
        base = evaluation.get("D0")
        for arm in ("D1", "D2", "D3", "D4"):
            row = evaluation.get(arm)
            if not row or not base:
                continue
            delta_edge = row["directed_edges"]["edge_L1"] - base["directed_edges"]["edge_L1"]
            delta_collision = row["collision"] - base["collision"]
            subtype = (row.get("head") or {}).get("subtype", {}).get("macro_f1")
            add(f"- **{NAMES[arm]}** vs D0:subtype F1 {subtype},有向边 L1 {delta_edge:+.4f},"
                f"collision {delta_collision:+.4f},role probe {row['role_probe']['mean']}。")
        add("")
        add("接受条件是联合的:subtype/orbit 可读性提高**且**有向边保留与角色可解码性改善,"
            "且不以碰撞率骤降/真实边丢失换取;D3/D4 的码本梯度必须真的改变硬索引,而不只是 head。")
    add("")
    add("## 6. 局限")
    add("")
    add("- 方向标签在无向骨架上分裂出 6–14 个 subtype,稀有 subtype 已按 min-support 合并;")
    add("- 前缀**相等**本身是对称关系,不能表达箭头;方向可读性由 SID 前缀上的线性角色探针给出,这是唯一可复核的 readout;")
    add("- 有向图的重复转移权重中位数为 1(稀疏信号),方向噪声由互惠率与加权/单向分层指标约束;")
    add("- 单次 seed=42,不提供跨 seed 方差(§19g)。")
    add("")
    add("## 7. 产物")
    add("")
    add("- `analysis_sid_recall/build_directed_behavior_graph.py`、`results/research/behavior_graph_directed/`;")
    add("- `results/research/directed_graphlet_role/{directed_graphlets.py, build_directed_instances.py, "
        "directed_role.py, template/, make_arm.py, pretrain_head.py, evaluate.py, report.py}`;")
    add("- 每臂 `results/stage2_RQ-VAE/TIGER_drq_<arm>/`(checkpoint、SID、trq_metrics.json、日志)。")
    (HERE / "REPORT_DIRECTED_GRAPHLET_ROLE.md").write_text("\n".join(lines) + "\n")
    print(f"report written, {len(lines)} lines")


if __name__ == "__main__":
    main()
