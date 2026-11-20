"""Issue #222 closing comment, generated from the artefacts on disk."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FAMILIES = ("K3", "P3", "C4", "P4", "K1_3")


def split_audit() -> dict:
    pool = np.load(REPO / "results" / "research" / "directed_graphlet_role" / "instances.npz",
                   allow_pickle=False)
    total_test = 0
    shared_test = 0
    rows = []
    for family in FAMILIES:
        train = pool[f"{family}_nodes"][pool[f"{family}_split"] == 0]
        test = pool[f"{family}_nodes"][pool[f"{family}_split"] == 1]
        pairs = set()
        for row in train:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    pairs.add((min(int(row[i]), int(row[j])), max(int(row[i]), int(row[j]))))
        shared = 0
        for row in test:
            if any((min(int(row[i]), int(row[j])), max(int(row[i]), int(row[j]))) in pairs
                   for i in range(len(row)) for j in range(i + 1, len(row))):
                shared += 1
        rows.append((family, len(train), len(test), shared))
        total_test += len(test)
        shared_test += shared
    return {"rows": rows, "total_test": total_test, "shared": shared_test}


def table(gate: dict) -> str:
    order = ("random", "majority", "degree", "hard_ce", "teacher_kd", "shuffled_kd",
             "teacher_kd_z", "hard_ce_rel", "teacher_kd_rel", "teacher_kd_z_rel",
             "hard_ce_spec", "teacher_kd_spec", "shuffled_kd_spec",
             "teacher_kd_z_spec", "teacher_kd_z_rel_spec")
    lines = ["| 变体 | 头 | 训练 acc | held-out acc | held-out macro-F1 | pair-clean acc | 无条件 orbit F1 |",
             "|---|---|---:|---:|---:|---:|---:|"]

    def fmt(value):
        return "%.4f" % value if isinstance(value, (int, float)) else "-"

    for name in order:
        row = gate["variants"].get(name)
        if row is None:
            continue
        train = row.get("train_mean", {})
        heldout = row.get("heldout_mean", {})
        pair = row.get("pair_disjoint_mean", {})
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
            name, row.get("head", "—"), fmt(train.get("subtype_accuracy")),
            fmt(heldout.get("subtype_accuracy")), fmt(heldout.get("subtype_macro_f1")),
            fmt(pair.get("subtype_accuracy")), fmt(heldout.get("orbit_macro_f1"))))
    return "\n".join(lines)


def main() -> None:
    gate = json.loads((HERE / "gate.json").read_text())
    metadata = json.loads((HERE / "teacher" / "teacher_metadata.json").read_text())
    controls = json.loads((HERE / "teacher_controls.json").read_text())
    audit = split_audit()
    spec = {name: gate["variants"][name]["heldout_mean"]["subtype_macro_f1"]
            for name in ("hard_ce_spec", "teacher_kd_spec", "shuffled_kd_spec",
                         "teacher_kd_z_spec")}
    degree = gate["variants"]["degree"]["heldout_mean"]["subtype_macro_f1"]
    kd_fit = gate["variants"]["teacher_kd_spec"]["train_mean"]["subtype_accuracy"]
    shuffled_fit = gate["variants"]["shuffled_kd_spec"]["train_mean"]["subtype_accuracy"]
    best_learned = max(
        (row.get("heldout_mean", {}).get("subtype_macro_f1", 0.0), name)
        for name, row in gate["variants"].items() if name not in ("majority", "degree"))
    majority = gate["variants"]["majority"]["heldout_mean"]["subtype_macro_f1"]

    text = f"""## 结项:Teacher 可复用、接口可测;结构信号可迁移,方向内容不可迁移

不重开 #221,不跑 Stage2/Stage3。全部证据在 `results/research/directed_graphlet_teacher/`
(`MANIFEST.json` 逐文件 sha256)。

### 1. Teacher checkpoint 已可复用

- `teacher/teacher_directed_v1.pth`,SHA-256 `{metadata['sha256']}`;payload 内含 format、config、词表与 state_dict。
- `load_frozen_teacher` 做 `strict=True` 加载 → `eval()` → 全参数 `requires_grad_(False)`;`save_teacher` 会用固定 probe 批次证明重载前向**逐位相等**,否则拒绝落盘。
- 同口径复跑的三行对照与 #221 完全相同:真实有向邻接 **{controls['directed']['subtype_macro_f1']:.4f} / {controls['directed']['orbit_macro_f1']:.4f}**、方向打乱 {controls['shuffled']['subtype_macro_f1']:.4f} / {controls['shuffled']['orbit_macro_f1']:.4f}、邻接清零 {controls['no_graph']['subtype_macro_f1']:.4f} / {controls['no_graph']['orbit_macro_f1']:.4f}。

### 2. 数据审计:这份池子没有 pair-disjoint 的 held-out

按完整训练池统计,{audit['shared']:,} / {audit['total_test']:,} = **{audit['shared'] / audit['total_test']:.3f}** 的 held-out 实例至少与一条训练实例共享一条无条件节点对(逐族见报告 §3)。#221 的 0.7121 因此是「实例不同、节点对重叠」的识别指标,不是对未见结构的泛化。

### 3. 可迁移性 Gate

同数据、同 seed、同优化预算(每族 2000 个全批量 epoch,训练集 5000/族),只改监督信号:

{table(gate)}

`_spec` 是每族一个专用头(消除共享参数之间的干扰);`_rel` 是带成对项的读出头(其容量正对照在合成关系标签上 >0.95)。

### 4. 判定

- **结构信息能迁移。** 只读冻结基线硬码字 `q` 的专用头:`hard_ce_spec` held-out macro-F1 **{spec['hard_ce_spec']:.4f}**、`teacher_kd_spec`(只有 Teacher 软分布)**{spec['teacher_kd_spec']:.4f}**,多数类 **{majority:.4f}**、度数查表 **{degree:.4f}**;pair-clean 子集排序相同。
- **迁移到的主要是无向骨架。** 方向打乱教师的目标给出 **{spec['shuffled_kd_spec']:.4f}**,与真实方向 **{spec['teacher_kd_spec']:.4f}** 同带;但打乱教师的训练拟合只有 {shuffled_fit:.3f}(真实 {kd_fit:.3f}),所以只能读成「打乱目标已足以填满硬标签天花板」,不能读成「方向无用」。
- **量化不是瓶颈。** 同一专用头在连续 `z` 上 **{spec['teacher_kd_z_spec']:.4f}**,在硬 `q` 上 **{spec['teacher_kd_spec']:.4f}**。
- 共享头(5 族共用一个)held-out 只有 0.02-0.03、训练准确率 0.35-0.59,是欠拟合;判定只使用 `_spec`。
- 按 §C.5 停在接口可行性层,**不启动 Stage2/Stage3**;要继续必须以「无向骨架蒸馏」重新立题,并先补 pair-disjoint 数据。

### 5. 交付

`teacher_model.py` / `frozen_sid.py` / `student_head.py` / `kdistill.py` / `fit_teacher.py` / `gate.py` / `manifest.py`,
`tests/test_teacher_contract.py`、`tests/test_student_interface.py`(15 项,全部通过),
报告 `results/research/directed_graphlet_teacher/REPORT_TEACHER_DISTILL_INTERFACE.md`。
"""
    (HERE / "issue_close_comment.md").write_text(text)
    print(text[:600])


if __name__ == "__main__":
    main()
