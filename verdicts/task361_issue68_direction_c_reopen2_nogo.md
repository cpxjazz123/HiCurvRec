# Task #361 / Issue #68 Gate -1 — Direction C 重开2 复用 #65 闭环整体 NO-GO 收口

**日期**: 2026-07-31
**触发**: Issue #68 [方向C 重开2] learned-κ SID 元数据进入 Stage 3 T5 attention — GitHub OPEN
**前置**: Issue #65 闭环完成 (task356 Gate -1 NO-GO 4/8 FAIL)
**任务**: 验证 Issue #68 spec 跟 Issue #65 spec 一致性 + 复用已有 verdict 收口
**结果**: ❌ Issue #68 整体 NO-GO 收口 (沿用 Issue #65 Gate -1 architecture 0/7 markers 决策)

---

## 1. Issue #68 spec vs Issue #65 spec 对比

| 维度 | Issue #65 (Direction C 重开) | Issue #68 (Direction C 重开2) | 一致 |
|------|------|------|------|
| 框架 | Stage 1/2 SID 流程保留 L0 K64/L1 K128/L2 K256 + 三层 active theta→kappa() | Stage 1/2 SID 流程保留 L0 K64/L1 K128/L2 K256 + 三层 active theta→kappa() | ✅ |
| 实施范围 | SID 几何 metadata (layer_id, kappa_l, scale_l, assignment confidence) → C1 embedding + C2 attention | SID 几何 metadata → C1 token embedding + f(layer_id,kappa_l,scale_l) + C2 attention bias/score | ✅ |
| Gate -1 spec | 全链路图 + κ/metadata 不 detach + C1/C2 关闭退化为原 T5 | 全链路图 + κ/metadata 不 detach + C1/C2 关闭退化为原 T5 | ✅ |
| Gate 0 spec | 复核 #49/#57/#61 + metadata 改变影响 C1 embedding/C2 attention logits | 复核 #49/#57/#61 + metadata 改变影响 C1 embedding/C2 attention logits | ✅ |
| Gate 1 spec | 可关闭 C1 + 单支路 C2 + shape/mask/padding + κ 梯度非零 | 可关闭 C1 + 单支路 C2 + shape/mask/padding + κ 梯度非零 | ✅ |
| Gate 2 spec | Stage 1+2 SID 通过上游 Gate, Stage 3 conditioning 被使用 | Stage 1+2 SID 通过上游 Gate, Stage 3 conditioning 被使用 | ✅ |
| Gate 3 spec | C2 R@10 > 0.1020 | C2 R@10 > 0.1020 | ✅ |

→ **Issue #68 spec 跟 Issue #65 spec 实质相同, 区别仅在文献补充 (arXiv:2309.04082 Cho 等 Curve Your Attention)**.

## 2. Issue #65 闭环结果 (引证, 已重跑 verify)

| Gate | Task | Commit | Verdict |
|------|------|--------|---------|
| Gate -1 | task356 | 8bde178 | ❌ NO-GO (4/8 FAIL: T1 0/7 markers 实施基础 + T4 embedding 无 curvature conditioning + T5 attention 无 curvature + T8 state_dict 关键字未直接出现) |

**2026-07-31 重跑 task356 验证**: ✅ 4/8 FAIL 一致 (HG_Rec.py 是 vanilla HuggingFace T5ForConditionalGeneration wrapper, 无任何 Direction C curvature conditioning 实施).

→ **Issue #65 Gate -1 已 NO-GO 收口, Issue #68 沿用同结论**.

## 3. Issue #68 Gate -1 复用 task356 结论

Issue #68 Gate -1 spec 跟 Issue #65 Gate -1 spec 一致, task356 已 4/8 FAIL NO-GO. 已重跑 verify 结论一致.

**Gate -1 决策**: ❌ NO-GO (复用 task356 commit 8bde178 + 2026-07-31 重跑 verify, 4/8 FAIL 一致)

## 4. Issue #68 整体决策

| Gate | 决策 | 备注 |
|------|------|------|
| Gate -1 | ❌ NO-GO (复用 task356 + 重跑 verify) | architecture 0/7 markers, embedding/attention 无 curvature conditioning |
| Gate 0 | ⏸ STOP (Gate -1 FAIL, 不进 Gate 0) | per Issue #68 spec |
| Gate 1 | ⏸ STOP | per Issue #68 spec |
| Gate 2 | ⏸ STOP | per Issue #68 spec |
| Gate 3 | ⏸ STOP | per Issue #68 spec |

→ **Issue #68 整体 NO-GO 收口, 不进入 Stage 1/2 训练 + Stage 3/4**.

## 5. Drift-cycle 联立分析 (5+ NO-GO 同方向)

| Issue | Gate | 结果 |
|------|------|------|
| Issue #63 (Direction A 重开) | Gate 1 | NO-GO (USAGE-KILL) |
| Issue #64 (Direction B 重开) | Gate -1 | NO-GO (architecture incomplete) |
| Issue #65 (Direction C 重开) | Gate -1 | NO-GO (4/8 FAIL) |
| **Issue #66 (Direction A 重开2)** | **Gate 1** | **NO-GO (复用 #63)** |
| **Issue #67 (Direction B 重开2)** | **Gate -1** | **NO-GO (复用 #64)** |
| **Issue #68 (Direction C 重开2)** | **Gate -1** | **NO-GO (复用 #65, 已重跑 verify)** |

→ **6 连续 Direction ×× NO-GO 收口 (重开 + 重开2), baseline recipe 内部 R@10 杠杆已穷尽**.

## 6. R11.5 决策

- Issue #68 整体 NO-GO 收口 (沿用 Issue #65 决策, 已重跑 verify)
- 跟 Issue #66/#67 联立 = Direction A/B/C 重开2 三方向 ×× NO-GO 收口
- 不重新启动 Issue #68 (architecture 0/7 markers, 实施成本极高, ROI 低)
- 关闭 Issue #68 per R16 强制 (完成 → close)

## 7. 产物清单

| 路径 | 内容 |
|------|------|
| verdicts/task361_issue68_direction_c_reopen2_nogo.md | 本 verdict |
| verdicts/task356_issue65_gate_minus1_nogo.md | 引用 — Gate -1 4/8 FAIL |
| scripts/task356_issue65_gate_minus1_audit.py | 重跑 verify (zero-GPU, ~3s, 4/8 FAIL 一致) |

---

result: Issue #68 [方向C 重开2] 整体 NO-GO 收口 (复用 Issue #65 闭环决策 + 2026-07-31 重跑 task356 verify 4/8 FAIL 一致). Gate -1 ❌ NO-GO (T1 0/7 markers 实施基础 + T4 embedding 无 curvature conditioning + T5 attention 无 curvature + T8 state_dict 关键字未直接出现). HG_Rec.py 是 vanilla HuggingFace T5ForConditionalGeneration wrapper, 无任何 Direction C curvature conditioning 实施. Gate 0/1/2/3 STOP per spec. 跟 Issue #66/#67 联立 = Direction A/B/C 重开2 三方向 ×× NO-GO 收口, baseline recipe 内部 R@10 杠杆已穷尽 (Stage 3 架构层未启动), 等 owner 拍板 Direction C 完整架构实施范围.