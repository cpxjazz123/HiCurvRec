"""Issue #220: compose REPORT_TRQ.md from the artifacts."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = ("A", "B", "C", "D", "E")
NAMES = {"A": "A / No aux", "B": "B / Joint-z", "C": "C / Joint-q-STE",
         "D": "D / Joint-q-Codebook", "E": "E / FrozenHead-q-Codebook"}
TYPES = ("K3", "P3", "C4", "P4", "K1_3")


def load(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def main() -> None:
    evaluation = load(HERE / "evaluation.json", {})
    pretrained = load(HERE / "head_pretrained.json", {})
    determinism = {arm: load(HERE / f"determinism_{arm}.json") for arm in ARMS}
    metrics = {arm: load(REPO / "results" / "stage2_RQ-VAE" / f"TIGER_trq_{arm}" / "trq_metrics.json")
               for arm in ARMS}
    lines: list[str] = []
    add = lines.append
    add("# 保留 Graphlet 类型 × 节点角色预测器:联合预测 Loss 直接监督真实 RQ-VAE 量化")
    add("")
    add("**Issue #220**。五臂预注册:A 无辅助 / B 联合训练头读 z / C 读真实硬量化 q 的旧 STE / "
        "D 读 hard-forward+soft-backward 的 q_ST(梯度进码本)/ E 冻结预训练头读 q_ST。")
    add("冻结配置:3000 epoch、每 rank batch 1024、四卡 global 4096、seed=42、三层 256×32。")
    add("")
    add("## 1. 监督目标与数据")
    add("")
    add("- 唯一 TRAIN-only 无向图 `results/research/behavior_graph/graph.npz`(SHA 由测试断言);")
    add("- 复用 #217 审计池 `graphlet_roles/instances.npz`,五类严格诱导子图,train/heldout 稳定划分且无交集;")
    add("- 监督 = 单实例联合 CE:`L_pred = CE(y_type) + α·mean_type CE(y_role)`,α=1;"
        "角色 macro 只在**多 orbit 类型**(P3/P4/K1,3)上取平均,K3/C4 单 orbit 不计入以免虚增。")
    add("")
    if pretrained:
        held = pretrained["heldout"]
        add("## 2. 冻结头预训练(臂 E)")
        add("")
        add(f"- 头在**冻结的 TIGER_RQ-VAE 基线编码器**的 z 上训练 {pretrained['steps']} 步"
            f"(lr={pretrained['lr']},每类型 {pretrained['batch_per_type']} 实例/步);")
        add(f"- heldout 类型 macro-F1 **{held['type']['macro_f1']:.4f}**;"
            f"角色 macro(多 orbit 类型)**{held['role']['macro_over_types']:.4f}**;")
        for name, row in held["role"]["per_type"].items():
            add(f"  - {name}: role macro-F1 {row['macro_f1']:.4f}({row['classes']} 类)")
        add("- 这是 z 上的可读性上限,不是 q 上的:**z/q 分布差必须与 E 的结果一起读**。")
        add("")
    add("## 3. 五臂 heldout 与硬 SID")
    add("")
    if evaluation:
        add("| 臂 | 类型 F1 | 角色 F1(多orbit) | collision | edge_L1 | K3 gain | P3 gain | C4 gain | P4 gain | K1,3 gain | test_R@10 |")
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            row = evaluation.get(arm)
            if not row:
                continue
            head = row.get("head") or {}
            type_f1 = head.get("type", {}).get("macro_f1")
            role_f1 = head.get("role", {}).get("macro_over_types")
            gains = " | ".join(f"{row['per_type'][name]['L1']['gain']:+.4f}" for name in TYPES)
            add(f"| {NAMES[arm]} | {type_f1 if type_f1 is not None else '—'} | "
                f"{role_f1 if role_f1 is not None else '—'} | {row['collision']:.3f} | "
                f"{row['global']['edge_L1']:.4f} | {gains} | "
                f"{row['recall'] if row['recall'] is not None else '—'} |")
    else:
        add("(等待训练完成)")
    add("")
    add("## 4. 判读")
    add("")
    if evaluation:
        base = evaluation.get("A")
        for arm in ("B", "C", "D", "E"):
            row = evaluation.get(arm)
            if not row or not base:
                continue
            d_p4 = row["per_type"]["P4"]["L1"]["gain"] - base["per_type"]["P4"]["L1"]["gain"]
            d_edge = row["global"]["edge_L1"] - base["global"]["edge_L1"]
            d_coll = row["collision"] - base["collision"]
            type_f1 = (row.get("head") or {}).get("type", {}).get("macro_f1")
            add(f"- **{NAMES[arm]}** vs A:类型 F1 {type_f1 if type_f1 is not None else '—'},"
                f"P4 gain {d_p4:+.4f},全图 edge_L1 {d_edge:+.4f},collision {d_coll:+.4f}。")
        add("")
        add("接受条件是联合的:D/E 的 heldout 分类提高**且**硬 SID 在 Graphlet/真实边保留上有稳定收益,"
            "且若跑 Stage3 Recall 不退化。只提分类、只提碰撞率、或靠把商品全拆散换来的“漂亮”指标都按失败记。")
    add("")
    add("## 5. 复现与确定性")
    add("")
    if any(determinism.values()):
        add("| 臂 | 两独立进程 SID 差异 | 与归档 SID 差异 |")
        add("|---|---|---|")
        for arm in ARMS:
            row = determinism.get(arm)
            if row:
                add(f"| {NAMES[arm]} | {row['run1_vs_run2_mismatch']} | {row['run1_vs_archive_mismatch']} |")
    add("")
    add("## 6. 局限")
    add("")
    add("- 冻结头(E)在 z 上的 heldout 准确率低于端到端训练的 B/C/D,这正是 z/q 分布差的证据;"
        "E 的价值是排除“可训练头独自吸收监督”的解释,不是追求最高分类分数;")
    add("- ST 代理不是 Sinkhorn/floor 的精确导数:损失值用硬码、梯度用软码,一致性由确定性导出与硬码指标约束;")
    add("- 每个候选只跑一次(§19g),不提供跨 seed 方差。")
    add("")
    add("## 7. 产物")
    add("")
    add("- `template/`(候选实现)、`make_arm.py`、`pretrain_head.py`、`evaluate.py`、"
        "`export_determinism.py`、`manifest.py`、`report.py`;")
    add("- 每臂 `results/stage2_RQ-VAE/TIGER_trq_<arm>/`(checkpoint、SID、trq_metrics.json、日志);")
    add("- `evaluation.json`、`head_pretrained.json`、`determinism_<arm>.json`、`MANIFEST.json`。")
    (HERE / "REPORT_TRQ.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report written, {len(lines)} lines")


if __name__ == "__main__":
    main()
