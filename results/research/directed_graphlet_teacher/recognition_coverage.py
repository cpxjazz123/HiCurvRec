"""Issue #229: per-shape recognition coverage for the five graphlet families.

#229 names chain, fork, converge, triangle/closure, path and star explicitly.
The subtype vocabulary already contains them, but the delivery only quoted one
macro-F1.  This script decodes every canonical bit key, labels its shape from the
in/out degree multiset, and joins the held-out per-class F1 from the #221 head:

    family | shape | #subtypes | held-out instances | macro-F1

Outputs `teacher/opt/recognition_coverage.json` and
`REPORT_RECOGNITION_COVERAGE.md`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROLE = HERE.parent / "directed_graphlet_role"
sys.path.insert(0, str(ROLE))

from directed_role import FAMILIES, load_vocabulary  # noqa: E402


def shape_of(family: str, key: str) -> str:
    size = int(round(len(key) ** 0.5))
    matrix = np.array([int(bit) for bit in key], dtype=bool).reshape(size, size)
    out_degree = matrix.sum(1)
    in_degree = matrix.sum(0)
    if size == 3:
        if matrix.sum() == 3:
            return "triangle"
        if out_degree.max() == 2:
            return "fork"
        if in_degree.max() == 2:
            return "converge"
        return "chain"
    if size == 4:
        skeleton_degree = (matrix | matrix.T).sum(1)
        if matrix.sum() == 4 and np.all(skeleton_degree == 2):
            return "cycle"
        if skeleton_degree.max() == 3:
            if out_degree.max() == 3:
                return "out-star"
            if in_degree.max() == 3:
                return "in-star"
            return "mixed-star"
        return "path"
    raise ValueError(f"{family}: unexpected pattern size {size}")


def main() -> None:
    head = json.loads((ROLE / "head_pretrained_directed.json").read_text())
    per_class = head["heldout"]["subtype"]["per_class"]
    role_per_family = head["heldout"]["role"]["per_family"]
    vocab = load_vocabulary("directed")
    raw = vocab["raw"]
    rows = []
    for family in FAMILIES:
        for key, index in raw[family].items():
            if key == "__other__":
                continue
            entry = per_class.get(str(index), {"f1": 0.0, "support": 0})
            matrix = np.array([int(bit) for bit in key], dtype=bool).reshape(
                int(round(len(key) ** 0.5)), -1)
            signature = (tuple(matrix.sum(1).tolist()), tuple(matrix.sum(0).tolist()))
            rows.append({"family": family, "subtype": int(index), "key": key,
                         "signature": signature,
                         "f1": float(entry.get("f1", 0.0)),
                         "support": int(entry.get("support", 0))})
    total = len(rows)
    if total != vocab["n_subtypes"]:
        raise RuntimeError(f"subtype count mismatch: {total} vs {vocab['n_subtypes']}")
    families = {}
    for family in FAMILIES:
        members = [row for row in rows if row["family"] == family]
        families[family] = {
            "subtypes": len(members),
            "heldout_instances": int(sum(row["support"] for row in members)),
            "no_adjacency_subtype_macro_f1": float(np.mean([row["f1"] for row in members])),
            "with_adjacency_orbit_macro_f1": float(
                role_per_family.get(family, {}).get("macro_f1", 0.0)),
            "orientation_signatures": len({row["signature"] for row in members})}
    shapes = {"triangle": "K3", "chain/fork/converge": "P3", "4-cycle": "C4",
              "path": "P4", "star": "K1_3"}
    payload = {"source_head": str((ROLE / "head_pretrained_directed.json")
                                  .relative_to(HERE.parents[2])),
               "n_subtypes": total,
               "settings_note": "the per-class F1 and role macro-F1 in this file come "
                                "from the frozen-z predictor (no adjacency); the 0.7121 / "
                                "0.6101 numbers in REPORT_DIRECTED_GRAPHLET_ROLE.md come "
                                "from the with-adjacency experiment and are a different "
                                "capability, not the same measurement",
               "family_metrics": families,
               "shape_mapping": {name: {"family": family,
                                        "subtypes": families[family]["subtypes"],
                                        "present": families[family]["subtypes"] > 0}
                                 for name, family in shapes.items()},
               "orientation_rows": rows,
               "best_subtypes": sorted(rows, key=lambda row: -row["f1"])[:5],
               "worst_subtypes": sorted(rows, key=lambda row: row["f1"])[:5]}
    out = HERE / "teacher" / "opt" / "recognition_coverage.json"
    out.write_text(json.dumps(payload, indent=1))
    lines = ["# 识别覆盖:五类结构与其有向朝向", "",
             "来源:`directed_graphlet_role/head_pretrained_directed.json`。", "",
             "**两个设置不可混读**:本表的 subtype F1 与 role macro-F1 来自 **冻结 z 的无邻接**",
             "预测器;`REPORT_DIRECTED_GRAPHLET_ROLE.md` 的 **0.7121 / 0.6101** 来自",
             "**给定真实诱导邻接**的实验,是另一种能力。", "",
             "## #229 列出的结构 → 家族", "",
             "| 结构 | 家族 | 子类型数 | held-out 实例 |",
             "|---|---|---:|---:|"]
    for name, family in shapes.items():
        entry = families[family]
        lines.append(f"| {name} | {family} | {entry['subtypes']} | "
                     f"{entry['heldout_instances']} |")
    lines += ["", "## 每个家族的两个数字", "",
              "| family | 子类型 | 有向朝向签名 | held-out 实例 | 无邻接 subtype macro-F1 | 无邻接 role macro-F1(按家族) |",
              "|---|---:|---:|---:|---:|---:|"]
    for family in FAMILIES:
        entry = families[family]
        lines.append(f"| {family} | {entry['subtypes']} | "
                     f"{entry['orientation_signatures']} | {entry['heldout_instances']} | "
                     f"{entry['no_adjacency_subtype_macro_f1']:.4f} | "
                     f"{entry['with_adjacency_orbit_macro_f1']:.4f} |")
    lines += ["", f"全部 {total} 个有向子类型都带 held-out 支持与逐类 F1;",
              "链式/分叉/汇聚是 P3 家族内的不同有向朝向,不是另外的家族。"]
    (HERE / "REPORT_RECOGNITION_COVERAGE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("json:", out)


if __name__ == "__main__":
    main()
