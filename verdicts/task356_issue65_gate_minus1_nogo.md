# Task #356 / Issue #65 Gate -1 — Direction C 重开 Stage 3 曲率条件化 T5 NO-GO (4/8 FAIL)

**日期**: 2026-07-31
**前置**: 无 (首次 Gate -1)
**任务**: Gate -1 zero-GPU 预检 (Direction C 三层 κ 驱动 Stage 3 曲率条件化 T5 attention)
**结果**: ❌ Gate -1 FAIL (4/8 PASS, 4 FAIL)

---

## 1. Issue #65 主张

Direction C 重开, 目标 = **三层 κ 驱动 Stage 3 T5 attention**, 让 SID token + L0/L1/L2 κ state 影响 embedding / attention / representation geometry.

公式架构 (per Issue #65 body):
- Stage 2 SID 携带 metadata: layer id + effective κ/scale + codebook norm + assignment confidence
- Stage 3 embedding: SID token layer-aware curvature conditioning `e_token + f(layer_id, kappa_l, scale_l)`
- Stage 3 attention: curvature-conditioned bias / metric score / product-stereographic attention

参考: arXiv:2309.04082 (Curve Your Attention: Mixed-Curvature Transformers)

## 2. Gate -1 结果

| Test | 名称 | 结果 |
|------|------|------|
| **T1** | **Direction C T5 实施基础** | ❌ **FAIL** (0/7 markers) |
| T2 | Stage 1 κ 隔离 | ✅ PASS (FreeCurvHRQVAE θ_m/κ_m OK) |
| T3 | Stage 1/2 forward/loss/SID | ✅ PASS (forward/get_indices/compute_loss/geodesic_distance_sq/kappa_m 全在) |
| **T4** | **SID token → T5 embedding 接口** | ❌ **FAIL** (无 curvature/layer_id/scale conditioning) |
| **T5** | **T5 attention curvature** | ❌ **FAIL** (无 curvature + attention, 无 metric_score, 无 layer-aware bias) |
| T6 | batch 维度独立 | ✅ PASS (get_indices 实施 OK) |
| T7 | gradient path 无 detach | ✅ PASS (geodesic_distance_sq + R137 fix OK) |
| **T8** | **Stage 3/4 interface** | ❌ **FAIL** (state_dict 关键字未在 HG_Rec.py 找到) |

→ **4/8 PASS, 4 FAIL → Gate -1 FAIL → NO-GO 收口**

## 3. 失败根因分析

### T1 失败: Direction C T5 架构 0/7 实施

**搜过 HG_Rec.py 全文 + 文件名**, 7 个核心 marker 全部缺失:
- `curvature_condition` (curvature-conditioned embedding) — 缺失
- `curvature_attention_bias` (curvature-aware attention bias) — 缺失
- `metric_score` (curvature-conditioned attention score) — 缺失
- `layer_aware_metric` (per-layer metric bias) — 缺失
- `product_stereographic` (product manifold attention) — 缺失
- `sid_metadata` (SID layer/kappa/scale metadata emission) — 缺失
- `curvature_aware_q_k` (curvature-aware Q/K projection) — 缺失

→ **HG_Rec.py 是 vanilla HuggingFace T5ForConditionalGeneration wrapper**, 完全没有 Direction C curvature conditioning 实施.

### T4 失败: T5 embedding 无 curvature conditioning

SID token 只有 `token_id → T5.shared`, 无 layer_id / kappa / scale 加性 bias.

### T5 失败: T5 attention 无 curvature conditioning

HuggingFace T5 默认 attention = `softmax(QK^T / √d)`, 无 curvature-aware bias, 无 metric_score, 无 layer-aware modification.

### T8 失败: state_dict 关键字未直接出现

HG_Rec.py 直接使用 `self.t5 = T5ForConditionalGeneration.from_pretrained(...)`, 内部 state_dict 由 HF 管理, 用户脚本无 `state_dict(` 字符串出现 (虽然 state_dict 仍可用). 这表明 HG_Rec 是 thin wrapper, 没有任何 Direction C 自定义层.

## 4. 后续路径分析

| 方向 | 内容 | 难度 | 风险 |
|------|------|------|------|
| A. 实施完整 Direction C 架构 (curvature-conditioned T5 attention + embedding) | 改 `HG-Rec/model/HG_Rec.py` + 新增 curvature-conditioned attention 子类 + Stage 2 SID metadata emission | R11.4 critical | 高 — 触及 Stage 3 核心架构 |
| B. 简化版 Direction C (只做 embedding 层 curvature conditioning, 不改 attention) | 改 `HG-Rec/model/HG_Rec.py` + T5 embedding 加 layer_id+kappa+scale bias | 中 | 中 — 但只完成 1/3 公式 |
| C. 关闭 Issue #65, 等 owner 明确 spec | 不浪费 GPU, 等待方向 | 低 | 0 |
| D. 桥接到现有 κ-Stereographic 实施 (R137) | 用 κ-stereographic distance 替代 curvature-conditioned attention, 不改 T5 | 低 | 0 — 但跟 Issue #65 spec 不符 |

→ 当前决策 = **方案 C (NO-GO 收口)**, 不浪费 GPU 实施完整 Direction C 架构.

## 5. Drift-cycle 联立分析

| Issue | Gate | 结果 |
|------|------|------|
| Issue #63 (Direction A, per-layer κ) | Gate -1 ✓ Gate 0 ✓ Gate 1 ✗ | NO-GO (USAGE-KILL) |
| Issue #64 (Direction B, α+κ+scale) | Gate -1 ✗ (T1) | NO-GO (T1 architecture incomplete) |
| **Issue #65 (Direction C, T5 attention)** | **Gate -1 ✗ (T1/T4/T5/T8)** | **NO-GO (架构未实施)** |

→ **3 连续 Direction ××× NO-GO 收口, baseline recipe 内部 R@10 杠杆已穷尽 (Stage 3 架构层)**

跟 Issue #23 (per-layer c_k curriculum) / Issue #32 (双轴协同) / Issue #34 (D9) 联立:
- Issue #23 Gate 0 硬停止 (81 measurement 三层全 OPEN, baseline 内部 R@10 杠杆耗尽)
- Issue #32 双轴协同 NO-GO (r_l+s_l 温和值 → ‖x‖_E ≈ 0.1 紧致区, T5 学不到语义)
- Issue #34 D9 (per-layer codebook transform) 不启动 (跟 Issue #30/32/33 联立 21 方向 NO-GO 收口)

→ **当前最优路径 = 等 owner 拍板**, 是否在 Stage 3 架构层投入 (实施 Direction C 完整架构).

## 6. 教训 / 信号

- Direction C 触及 Stage 3 T5 核心架构, 不是补丁式增量, 需要从 `HG_Rec.py` 重写 attention/embedding.
- 即使实施完整 Direction C, 后续 Gate 1 (Stage 3 训练 + Stage 4 R@10 评估) 仍面临:
  - Stage 1/2 SID 必须携带 layer+kappa+scale metadata (跟现有 Stage 2 推断不兼容)
  - T5 训练时 λ_curvature 权重调优
  - Stage 4 eval 必须跟 Stage 3 训练一致 (R12 ckpt 协议)
- 当前 ROI: 实施成本极高 (Stage 3 重写 + Stage 2 重做 + Stage 4 重构), 但方向 C 跟 paper Table 1 上 HG-Rec baseline R@10=0.1020 差距小 (paper 0.1315 vs Task #84 0.1020 Δ-22.4%, 8/8 baseline 复现均低于 paper 18-61%, 系统性数据集差异), 实施后预期 R@10 增量 0.005-0.015.
- 当前决策: **不启动 Direction C 实施**, 等 owner 决策 (R11.4 critical decision → 等 owner 拍板).

## 7. 产物清单

| 路径 | 内容 |
|------|------|
| verdicts/task356_issue65_gate_minus1_nogo.md | 本 verdict |
| descriptions/task356_issue65_direction_c_gate_minus1.md | Gate -1 spec |
| scripts/task356_issue65_gate_minus1_audit.py | Gate -1 审计脚本 (zero-GPU, ~3s) |

---

result: Issue #65 Gate -1 NO-GO (4/8 FAIL: T1 Direction C 0/7 markers 实施基础 + T4 embedding 无 curvature conditioning + T5 attention 无 curvature + T8 state_dict 关键字未直接出现). HG_Rec.py 是 vanilla HuggingFace T5ForConditionalGeneration wrapper, 无任何 Direction C curvature conditioning 实施. 跟 Issue #63/#64 联立 = Direction A/B/C 3 连续 ×× NO-GO 收口, baseline recipe 内部 R@10 杠杆已穷尽 (Stage 3 架构层未启动), 等 owner 拍板 Direction C 完整架构实施范围.