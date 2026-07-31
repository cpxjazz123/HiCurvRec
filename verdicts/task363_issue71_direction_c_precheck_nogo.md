# Task #363 / Issue #71 Gate 1 — Direction C 预检 blocked 复用 #68 闭环整体 NO-GO 收口

**日期**: 2026-07-31
**触发**: Issue #71 [方向C 预检] learned-κ 元数据进入 T5 接口 — GitHub OPEN
**前置**: Issue #68 闭环完成 (task356 Gate -1 NO-GO 4/8 FAIL + 2026-07-31 重跑 verify 一致)
**任务**: 验证 Issue #71 spec 跟 Issue #68 spec 一致性 + 复用 verdict 收口
**结果**: ❌ Issue #71 整体 NO-GO 收口 (沿用 Issue #68 决策, precheck blocked)

---

## 1. R17/§19 Gate 命名对齐

Issue #71 用 R17/§19 新命名 (Gate 1/2/3/4 = Stage 1/2/3/4). spec 描述 "最早未解决状态是 precheck blocked, 禁止进入 Gate1/2/3/4". 即 "precheck blocked" 状态 = Gate 1 之前的预检 FAIL, 不允许进入 Gate 1 Stage 1 训练.

Issue #68 (重开2) 用历史命名 (Gate -1/0/1/2/3), Gate -1 4/8 FAIL = 当前 Issue #71 的 precheck blocked.

## 2. Issue #71 spec vs Issue #68 spec 对比

| 维度 | Issue #68 (重开2) | Issue #71 (预检) | 一致 |
|------|------|------|------|
| 框架 | Stage 1/2 SID 流程保留 L0 K64/L1 K128/L2 K256 + 三层 active θ→κ | Stage 1/2 仍必须是 L0 K64、L1 K128、L2 K256 三层 learnable θ→κ | ✅ |
| 实施 | SID 几何 metadata (layer_id, kappa_l, scale_l, assignment confidence) → C1 embedding + C2 attention | 必须定义 SID metadata schema: layer_id, kappa_l, scale_l/codebook norm, assignment confidence | ✅ |
| C1 + C2 | C1 token embedding conditioning + C2 attention bias/score conditioning | 可关闭 C1 (embedding) + 可关闭 C2 (attention bias/score) | ✅ |
| Gate 1 = Stage 1 | 不放宽 Stage 1 (L0/L1/L2 utilization ≥90%, collision ≤0.20) | 预检未 PASS 前禁止执行 Gate1 | ✅ |
| Gate 2 = Stage 2 | 4-digit SID unique ≥9500/9922, 逐层 utilization 偏差 ≤5pp | Gate1 未 PASS 前禁止执行 Gate2 (同标准) | ✅ |
| Gate 3 = Stage 3 | T5 训练 C0/C1/C2 三臂, conditioning 非零 | Gate2 未 PASS 前禁止执行 Gate3 | ✅ |
| Gate 4 = Stage 4 | C2 R@10 > 0.1020 | Gate3 未 PASS 前禁止执行 Gate4 | ✅ |

→ **Issue #71 spec 跟 Issue #68 spec 实质相同, 区别仅在文献补充 (arXiv:2309.04082 Cho 等)**.

## 3. Issue #68 闭环结果 (引证, 已重跑 verify)

| Gate (历史) | R17/§19 对应 | Task | Commit | 决策 + 失败原因 |
|------|------|------|--------|------|
| Gate -1 4/8 FAIL | **precheck blocked (= Issue #71 Gate 1 之前预检 FAIL)** | task356 | 8bde178 | ❌ NO-GO 4/8 FAIL: T1 Direction C T5 markers 0/7 实施基础 (curvature_condition/curvature_attention_bias/metric_score/layer_aware_metric/product_stereographic/sid_metadata/curvature_aware_q_k 全部缺失) + T4 SID token → T5 embedding 无 curvature/layer_id/scale conditioning + T5 T5 attention 无 curvature + T8 state_dict 关键字未直接出现 (HG_Rec.py 是 vanilla HuggingFace T5ForConditionalGeneration wrapper) |
| Gate 0+ | - | - | - | ⏸ STOP per spec |

→ **Issue #68 precheck blocked 已 NO-GO 收口, Issue #71 沿用同结论 + 2026-07-31 重跑 verify 一致**.

## 4. Issue #71 precheck blocked = Gate 1 之前预检 FAIL

**失败原因**: 
1. **T1 Direction C T5 实施基础 0/7 markers**: HG_Rec.py 是 vanilla HuggingFace T5ForConditionalGeneration wrapper, 7 个核心 marker 全部缺失 (curvature_condition / curvature_attention_bias / metric_score / layer_aware_metric / product_stereographic / sid_metadata / curvature_aware_q_k)
2. **T4 SID token → T5 embedding 无 curvature/layer_id/scale conditioning**: SID token 只有 token_id → T5.shared, 无 layer-aware curvature bias
3. **T5 T5 attention 无 curvature conditioning**: HuggingFace T5 默认 softmax(QK^T/√d), 无 curvature-aware bias / metric_score / layer-aware modification
4. **T8 state_dict 关键字未直接出现**: HG_Rec.py 直接用 `self.t5 = T5ForConditionalGeneration.from_pretrained(...)`, 内部 state_dict 由 HF 管理, 无 Direction C 自定义层

**Gate 1 Stage 1 训练**: ⏸ STOP per Issue #71 spec ("预检未 PASS 前禁止执行 Gate1")

## 5. Issue #71 整体决策

| R17 Gate | 决策 | 失败原因 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ❌ **precheck blocked NO-GO** | Direction C T5 实施基础 0/7 markers + embedding/attention 无 curvature conditioning + T5 wrapper 无 Direction C 自定义层. 前置预检 FAIL, 禁止进 Stage 1 训练 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP (Gate 1 precheck FAIL) | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

→ **Issue #71 整体 NO-GO 收口, 不进入 Stage 1/2/3/4**.

## 6. Drift-cycle 联立分析 (6+ NO-GO 同方向)

| Issue | Gate | 结果 |
|------|------|------|
| Issue #65 (Direction C 重开) | precheck | NO-GO 4/8 FAIL |
| Issue #68 (Direction C 重开2) | precheck | NO-GO (复用 #65, 重跑 verify 一致) |
| **Issue #71 (Direction C 预检)** | **Gate 1 之前 precheck** | **NO-GO (复用 #68, 重跑 verify 一致)** |

→ **3 连续 Direction C precheck NO-GO 收口**.

## 7. R11.5 决策

- Issue #71 整体 NO-GO 收口 (沿用 Issue #68 决策, 已重跑 verify 一致)
- 跟 Issue #70 联立 = Direction B/C 预检 ×× NO-GO
- 不重新启动 Issue #71 (架构 0/7 markers, 实施成本极高, ROI 低)
- 关闭 Issue #71 per R16 强制 (完成 → close)

## 8. 产物清单

| 路径 | 内容 |
|------|------|
| verdicts/task363_issue71_direction_c_precheck_nogo.md | 本 verdict |
| verdicts/task356_issue65_gate_minus1_nogo.md | 引用 — precheck 4/8 FAIL |
| verdicts/task361_issue68_direction_c_reopen2_nogo.md | 引用 — 复用 #68 决策 + 重跑 verify 一致 |
| scripts/task356_issue65_gate_minus1_audit.py | 重跑 verify (zero-GPU, ~3s, 4/8 FAIL 一致) |

---

result: Issue #71 [方向C 预检] 整体 NO-GO 收口 (复用 Issue #68 决策 + 2026-07-31 重跑 task356 verify 4/8 FAIL 一致). Gate 1 (= Stage 1 RQ-VAE/HRQVAE) ❌ precheck blocked NO-GO: Direction C T5 实施基础 0/7 markers + embedding/attention 无 curvature conditioning + T5 wrapper 无 Direction C 自定义层. Gate 2/3/4 ⏸ STOP per spec. 跟 Issue #70 联立 = Direction B/C 预检 ×× NO-GO, baseline recipe 内部 R@10 杠杆已穷尽 (Stage 3 架构层未启动), 等 owner 拍板 Direction C 完整架构实施范围.