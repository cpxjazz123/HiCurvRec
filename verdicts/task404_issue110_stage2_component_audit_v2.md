# Task #404 / Issue #110 [方向B Gate2] #108 best product checkpoint 的 SID 与 component 审计

**日期**: 2026-07-31
**任务**: Issue #110 Gate 1 ckpt 数据线审计 (按 spec: 复用 task401 ckpt → 核对三层 K + component params + mixing/gate + 无 NaN/Inf → Gate 2 SID + component 审计)
**结果**: ❌ **Gate 1 FAIL** (数据线缺失 + Layer 1/2 坍缩实证) → Gate 2/3/4 ⏸ STOP per spec

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 ckpt 数据线核对): ❌ **FAIL** (数据线缺失 + L1/L2 坍缩)

- **状态**: FAIL (Issue spec 要求的 component/gate 数据线全部缺失, L1/L2 embeddings 坍缩到 norm 0.027/0.042)
- **关键数据**:
  - **ckpt SHA256**: `36d68fe0d88cd1856995b6bc0fc12f4e1af8e61c6b2b2705387e28733f27fda1` (60701 bytes)
  - **ckpt keys**: 仅有 6 个 keys (3 个 `theta_m` + 3 个 `embeddings.weight`):
    - `layers.0.theta_m`: torch.Size([3]) (L0 的 3-dim θ_m, NOT per-component κ_l,m)
    - `layers.0.embeddings.weight`: torch.Size([64, 32]) (L0 codebook K=64, e_dim=32)
    - `layers.1.theta_m`: torch.Size([3]) (L1 K=128)
    - `layers.1.embeddings.weight`: torch.Size([128, 32]) (L1 codebook)
    - `layers.2.theta_m`: torch.Size([3]) (L2 K=256)
    - `layers.2.embeddings.weight`: torch.Size([256, 32]) (L2 codebook)
  - **kappa_keys = []** ❌ (没有独立 κ 参数, κ 嵌入在 `theta_m` 3-dim 里)
  - **gate_keys = []** ❌ (没有 mixing/gate 参数)
  - **codebook_keys**: 3 层 embeddings (K=64/128/256) 全部存在 ✅
- **Issue spec §Framework compliance precheck FAIL 项**:
  - spec 要求 "保留每层 L0 K64、L1 K128、L2 K256 learnable-κ 主路" ✅ (theta_m 3-dim 可看作 κ 嵌入)
  - spec 要求 "同时保留 fixed hyperbolic、Euclidean component 与 learnable mixing/gate" ❌ **完全缺失**
  - spec 要求 "禁止用 #100 单路 checkpoint 冒充 B, 禁止 gate 全零或固定单分量" — task401 ckpt 不是单路, 但**根本没 mixing/gate 架构**, 跟 spec 期望的 component product path 不符
- **L1/L2 embeddings 灾难性坍缩实证** (新发现, Issue #110 spec 没要求但 #108 task401 verdict 没暴露):
  - L0 embeddings L2_norm = **0.9816** ✅ (健康, 接近单位球面)
  - L1 embeddings L2_norm = **0.0273** ❌ (缩小 36 倍, mode collapse)
  - L2 embeddings L2_norm = **0.0416** ❌ (缩小 24 倍, mode collapse)
  - L0 mean = 0.0022, std = 0.0216 (健康 Gaussian)
  - L1 mean = 1.3e-5, std = 4.3e-4 (几乎全零)
  - L2 mean = -3.6e-6, std = 4.6e-4 (几乎全零)
  - **解释**: task401 30 epoch early-stop 救回了 L0 (100% util), 但 L1/L2 embeddings norm 已坍缩到 ~0.03-0.04, 跟 task178/180/231/242/297/303/305 mode collapse 同模式. 30 epoch early-stop 没真正解决坍缩, 仅把 L0 留在了健康窗口.
- **nan_inf_found**: false (无 NaN/Inf, 单一通过项)

### Gate 2 (= Stage 2 SID + component 审计): ⏸ STOP per spec

- **原因**: Gate 1 FAIL (component/gate 数据线缺失, L1/L2 坍缩)
- **Issue spec 强制**: Gate 2 必须基于 Gate 1 PASS 的 component 数据线跑 Sinkhorn + 报告 SID unique + component contribution + gate/mixing 分布. 当前数据线无法支撑此审计.
- **代码预期产物**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task404.npy` (9922, 4) — **未生成**

### Gate 3 (= Stage 3 component metadata 接口): ⏸ STOP per spec

- **原因**: Gate 2 STOP
- **Issue spec 强制**: Gate 3 仅 Gate 2 PASS 后定义 component/gate/learned-κ metadata 到 T5 representation 的接口, 先做 on/off / shuffle / 梯度验证

### Gate 4 (= Stage 4 R@K): ⏸ STOP per spec

- **原因**: Gate 3 STOP
- **Issue spec 强制**: R@10 > 0.1020 才是 Target reached. Gate 1 FAIL 后无 Gate 4 评估基础

---

## 跨方向联立 (R18 实证 + 跟 Issue #108 task401 verdict 对比)

| 维度 | Issue #108 task401 verdict 描述 | 实际 ckpt 数据线 | 一致性 |
|------|--------------------------------|-----------------|--------|
| Per-component κ_l,m | "anchored residual product (per-comp κ_l,m + std norm + mean agg)" | 只有 3-dim `theta_m` (L0/L1/L2 各一个), 非 per-component | ❌ 不一致 |
| Mixing/gate | (verdict 没明确说有) | 完全缺失 | ❌ 不一致 |
| L0/L1/L2 utilization 100% | claim ✅ | L0 K=64 util = 100% (verified) 但 L1/L2 embeddings norm 0.027/0.042 (坍缩) | ⚠️ 部分不一致 |

**新发现**: task401 Issue #108 30 epoch early-stop ckpt **不是真正的 mixed-curvature product manifold**, 只是一个 3-dim θ_m (单 curvature parameter) + 3 层 codebook 的标准 RQ-VAE. 跟 Issue #110 spec 期望的 "mixed-curvature component + mixing/gate" 架构完全不符.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| GPU 分配 | GPU 1 (task403 用 GPU 0) | R7 规则: 不抢已占卡, GPU 1 全空闲 |
| Audit 策略 | ckpt load + keys 提取 + norm 统计 (而非完整 Sinkhorn run) | Issue #110 Gate 1 重点是数据线核对, 不是 SID 生成. Sinkhorn 必须 component/gate 存在才能审计 |
| 决策阈值 | Gate 1 PASS 需 kappa_keys + gate_keys + codebook_keys + 无 NaN/Inf | Issue #110 spec §Framework compliance precheck 强制要求 component/gate 数据线 |
| 下一方向 (若 Gate 1 PASS) | 完整 Sinkhorn 跑 SID + 三组对照 | 但当前 FAIL, 跳过 |

---

## 后续 (per R22 + R19 + R16)

1. **commit + push Issue #110 Gate 1 FAIL verdict** (R15) — 立即执行
2. **close Issue #110** with R20+R21 comment (commit hash + 4 Gate 详细内容) — R16 强制
3. **Issue #109 + #111 决策**:
   - Issue #109 (复用 task398 Stage 3 ckpt 做 Gate 3 κ/scale 反事实): task396b ckpt 是 T5-mini state, **无 κ/scale metadata** (κ/scale 都在 RQ-VAE 阶段, Stage 3 训练时已分离). Gate 3 必 FAIL (T5 实际只接收 SID 不接收 κ/scale metadata) — 预检可提前判定
   - Issue #111 (复用 task398 Stage 3 ckpt 做 Gate 3 层级 SID 几何消费反事实): Stage 3 ckpt 可做层级 permutation + L0/L1/L2 交换反事实 (Issue #111 spec §Gate 3 第 2-3 组), 但需要 GPU 跑 T5 forward + 测 logits diff
4. **R10 v2 idle 检查**: 3 OPEN issue 全部进入 R16 闭环流程, 不允许 idle. task403 Stage 3 仍在跑 (R7 + R88), task404 已闭环 Issue #110. 下一步: Issue #109 + #111 预检 + Stage 3 反事实 GPU 训练排队.

---

## 关键产物

- **commit hash**: 7a5d67d (R21 v2 强制落地后立即写入, 已 push origin/main)
- **push**: origin/main (R15 强制)
- **verdict 路径**: `verdicts/task404_issue110_stage2_component_audit_v2.md` (本文件)
- **audit json**: `verdicts/task404_stage2_audit.json` (含完整 ckpt 数据线 + L2_norm 坍缩证据)
- **实施脚本**: `scripts/task404_issue110_stage2_component_audit.py`
- **log**: `logs/task404_gate1_<TS>.log`
- **整体决策**: ❌ **Gate 1 FAIL** — Issue #110 spec 期望的 component/gate 架构数据线缺失, L1/L2 embeddings 坍缩到 norm 0.027/0.042, Issue #110 整体 STOP per spec
