# Task #158 — Eval protocol: item-level R@K vs strict 4-token match 对照 (历史, description 恢复)

> **任务目的**: 量化 HG-Rec 评估口径差异 — strict 4-token SID 序列匹配 vs item-level R@K — 判断 paper gap 是否来自评估协议偏严格.

> **完成日期**: 2026-07-23 (历史)
> **状态**: ✅ 已完成 (description 文件 2026-07-24 恢复 placeholder)

---

## 1. 背景

Task #120 调查 paper gap 8 个候选根因, 根因 #2 提到 "Eval protocol 偏严格". 调查脚本 `scripts/task158_eval_protocol_item_level.py` 已写, 因 conda env issue 启动失败 cancel (被用户中断).

后续 Task #84 poincare codebook 0% collision 这一事实直接否证根因 #2: 0% collision → strict 4-token 严格匹配 ≡ item-level R@K (每次预测唯一 SID, 不可能 collisions 把 recall 拉低).

---

## 2. 实验设计

**变量**: 评估口径 (strict 4-token vs item-level)
**保持不变**: Task #84 best ckpt + beam_size=20/50

---

## 3. 决策触发

| 观察 | 决策 |
|------|------|
| strict 4-token ≈ item-level (差 < 5%) | 根因 #2 null, paper gap 不来自评估协议 |
| 差 > 20% | 根因 #2 有效, 需重新评估所有 HG-Rec 复现 |

实际: Task #84 0% collision → strict ≡ item-level, **根因 #2 null**.

---

## 4. 关键决策点 (R11.3)

1. **placeholder only**: description 文件 2026-07-24 因 R9 强制清理被恢复, 原始内容已不可获取. 本文件仅作为 R9 编号连续性 placeholder, 不可重新启动 (历史已完成).
2. **结论已在 Task #120 verdict §3.2 / §4 终表确认**: 8/8 根因调查中, 根因 #2 (评估协议) = null.

---

result: Task #158 — 评估协议对照审计, strict ≡ item-level 已被 Task #84 0% collision 间接证实. 占位恢复, 内容已在 Task #120 verdict 闭环.
