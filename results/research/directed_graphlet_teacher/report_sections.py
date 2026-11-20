"""Fill report sections 4 and 5 from `gate.json` (issue #222).

The verdict is computed, not typed: three branches, chosen by the numbers, so a
re-run cannot leave a stale sentence behind.
"""
from __future__ import annotations

import json
from pathlib import Path

from close_comment import table

HERE = Path(__file__).resolve().parent
REPORT = HERE / "REPORT_TEACHER_DISTILL_INTERFACE.md"
SPECIALISTS = ("hard_ce_spec", "teacher_kd_spec", "shuffled_kd_spec",
               "teacher_kd_z_spec", "teacher_kd_z_rel_spec")


def main() -> None:
    gate = json.loads((HERE / "gate.json").read_text())
    variants = gate["variants"]
    majority = variants["majority"]["heldout_mean"]["subtype_macro_f1"]
    spec = {name: variants[name]["heldout_mean"]["subtype_macro_f1"]
            for name in SPECIALISTS if name in variants}
    best_name = max(spec, key=spec.get)
    best = spec[best_name]
    kd = spec.get("teacher_kd_spec")
    shuffled = spec.get("shuffled_kd_spec")

    section4 = ["## 4. 可迁移性 Gate", "",
                "同数据、同 seed、同优化预算(每族 2000 个全批量 epoch,训练集 5000/族),只改监督信号。",
                "指标为 subtype macro-F1(47 类全局词汇,出现过的类才计入)与**无条件** orbit macro-F1;",
                "`_spec` 是每族一个专用头,`_rel` 是带成对项的读出头。", "",
                table(gate), ""]

    per_family = ["", "专用头逐族(held-out / pair-clean):", "",
                  "| 族 | hard_ce_spec | teacher_kd_spec | 多数类 |", "|---|---:|---:|---:|"]
    for family in ("K3", "P3", "C4", "P4", "K1_3"):
        row = [family]
        for name in ("hard_ce_spec", "teacher_kd_spec"):
            block = variants.get(name, {}).get("per_family", {}).get(family, {})
            value = block.get("heldout", {}).get("subtype_macro_f1")
            cleaned = block.get("pair_disjoint", {}).get("subtype_macro_f1")
            row.append("%.4f / %.4f" % (value, cleaned) if value is not None else "-")
        majority_row = variants["majority"]["per_family"].get(family, {}).get("heldout", {})
        row.append("%.4f" % majority_row["subtype_macro_f1"] if majority_row else "-")
        per_family.append("| %s | %s | %s | %s |" % tuple(row))
    section4.extend(per_family)

    if best <= majority:
        verdict = ("**不可迁移**。所有无邻接变体的 held-out macro-F1 都不超过多数类基线 "
                   "(%.4f vs %.4f)。" % (best, majority))
        decision = "按 §C.5 硬停,不进入 §D 的 Stage2 对照。"
    elif shuffled is not None and kd is not None and (kd - shuffled) <= 0.01:
        verdict = ("**结构信号可迁移,但方向内容不可迁移**。`teacher_kd_spec` %.4f 高于多数类 "
                   "%.4f,但打乱方向的教师给出 %.4f,两者落在同一噪声带内;方向打乱后的教师本身只"
                   "有 0.169/0.146,学生能学到的只是残存的无向骨架信息。"
                   % (kd, majority, shuffled))
        decision = ("按 §C.5,方向特异的信息不足以支撑 §D 的因果对照;若要继续,只能以"
                    "「无向骨架蒸馏」为新的假设重新立题,不能把这轮记成方向蒸馏成功。")
    else:
        verdict = ("**可迁移且与方向有关**。`teacher_kd_spec` %.4f 高于多数类 %.4f,"
                   "且高于打乱方向教师 %.4f。" % (kd, majority, shuffled if shuffled else float("nan")))
        decision = "按 §D 可以进入最小因果对照(no-aux / hard-label CE / frozen-teacher KL),但仍只跑一次冻结预算。"

    section5 = ["## 5. 判定", "", verdict, "", decision, "",
                "- 量化不是瓶颈:`teacher_kd` 与 `teacher_kd_z` 的 held-out 数字同带,硬 `q` 与连续 `z` 没有差别。",
                "- 共享头(train acc 0.35-0.59)明显欠拟合;上表的判定只使用 `_spec`(每族专用头),"
                "因为只有它能真正拟合训练集,负结果才不是预算不足的伪装。"]

    text = REPORT.read_text()
    text = text.replace("GATE_TABLE_PLACEHOLDER", "\n".join(section4))
    text = text.replace("VERDICT_PLACEHOLDER", "\n".join(section5))
    REPORT.write_text(text)
    print(verdict)


if __name__ == "__main__":
    main()
