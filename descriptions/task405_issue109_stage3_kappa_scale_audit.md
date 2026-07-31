# Task #405 / Issue #109 [方向A Gate3] Stage 3 ckpt κ/scale metadata 审计

**日期**: 2026-07-31
**Issue**: #109 [方向A Gate3] — 把 Stage 3 训练过程中 κ/scale metadata 反事实注入 SID embedding 并量化其消费
**任务**: 加载 task396b Stage 3 ckpt + 验证是否存在 κ/scale metadata channel → 若无 → Gate 3 FAIL per spec
**前置**: task395 Stage 1 (Issue #101 K=128 κ-decouple) + task396 Stage 2 Sinkhorn SID + task396b Stage 3 T5-mini 训练
**结果**: ❌ **Gate 3 FAIL per spec** — Stage 3 ckpt (sha256=f592b5821ec36004ed21c77b1c771663b822d76f0a79f4386f8ab1e47d969e75, 22087081 bytes) 仅持有 108 个 T5-mini transformer 权重 (无 κ/scale metadata channel)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE ckpt 可追踪): ✅ PASS per spec
- **状态**: PASS (复用 #101/#102 上游 PASS)
- **Stage 1 ckpt SHA256**: task395 (Issue #101 K=128 κ-decouple)
- **Issue spec §区别于 #90/#93/#104 验证**: 固定同一 checkpoint + 同一 batch + seed=42

### Gate 2 (= Stage 2 SID 可追踪): ✅ PASS per spec
- **状态**: PASS (复用 #102 PASS)
- **SID 文件**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy` (9922, 4)
- **SID unique**: 9922/9922 (100%), collision 0%
- **三层 utilization**: 100%/100%/100%

### Gate 3 (= Stage 3 κ/scale metadata 反事实): ❌ **FAIL per spec 强制判定**
- **状态**: FAIL per Issue #109 spec §Gate3 强制判定: "若 Stage 3 实际只接收 SID 且无 κ/scale 通道, 必须明确判为 Gate3 FAIL"
- **实施**: 同 ckpt (sha256=f592b582...) + 加载并审计 state_dict 全部 key/shape
- **关键数据 (ckpt state_dict 审计)**:
  - **ckpt 文件**: `products/task396_issue99_stage3_t5_train/ckpt/Instruments/Jul-31-2026_19-15-50/HG_Rec_best.pth` (22087081 bytes)
  - **key 总数**: 108
  - **key 类型**: 全部是 T5-mini transformer 权重 (shared.weight, transformer.encoder.block.0.layer.0.SelfAttention.{q,k,v,o}.weight, etc.)
  - **kappa 相关 key**: 0 个 (state_dict 中无 `kappa_m` / `kappa` / `curvature` / `scale` / `c` / `metric` / `manifold` / `geometry`)
  - **结论**: Stage 3 ckpt **架构上根本不持** κ/scale metadata, Issue #109 spec 要求的"反事实注入 κ/scale 到 SID embedding" 在现有 ckpt schema 下不可执行
- **根因**: HG-Rec 现行架构中, Stage 1 → Stage 2 Sinkhorn 转换把 κ/scale 信息丢失. Stage 3 T5-mini 仅接收离散 SID (4 个 digit index), 完全无几何信息输入通道
- **架构意义**: Issue #109 方向 (Stage 3 注入 κ/scale metadata) 在现有 ckpt 上无法验证. 必须重新设计 Stage 1 → Stage 3 的几何信息通道, 这是架构级改造 (Issue #113 范畴), 不是 Stage 3 单点可修复
- **Issue spec §Gate3 FAIL 处理**: 落盘 verdict + close issue (本 task 完成), 后续 Issue #109 必须在新架构 (Issue #113 三分量 product) 落地后再开新 task 验证

### Gate 4 (= Stage 4 R@K): ⏸ STOP per spec
- **原因**: Gate 3 FAIL per spec, Gate 4 禁止执行
- **Issue spec §Gate4 强制**: 只有 Gate3 PASS 才能训练和统一评估

---

## 关键产物

- **commit hash**: 8e0ca7f (R21 v2 强制落地后立即写入, 已 push origin/main)
- **push**: origin/main (R15 强制)
- **verdict 路径**: `verdicts/task405_issue109_stage3_kappa_scale_audit_v2.md`
- **audit json**: `verdicts/task405_issue109_stage3_kappa_scale_audit.json`
- **实施脚本**: `scripts/task405_issue109_stage3_kappa_scale_audit.py`
- **整体决策**: ❌ **Gate 3 FAIL** (Stage 3 ckpt 不持 κ/scale metadata channel, 架构级限制)

---

## 跨方向联立

| Issue | 引用 ckpt | Gate 路径 | 结果 |
|-------|----------|----------|------|
| **#109** | **task396b Stage 3 ckpt** | **κ/scale metadata 反事实** | **❌ Gate 3 FAIL (ckpt 无 κ/scale channel)** |
| #110 | task401 Stage 1 ckpt | component/gate 审计 | ❌ Gate 1 FAIL (ckpt schema 不完整) |
| #111 | task396b Stage 3 ckpt | 5 组层级 SID 几何反事实 | ✅ Gate 3 PASS |

**联立结论**: 
- Issue #109 / #110 共同根因: HG-Rec 现行架构信息丢失 (Stage 1 κ → Stage 2 SID 丢失, Stage 2 → Stage 3 仅传 SID 无 κ)
- Issue #111 PASS 证明 T5 路径可消费层级 SID (但仅 token identity 维度, 不含 κ/scale)
- 架构突破点: Issue #113 (真实三分量 product checkpoint schema) + Issue #114 (product-stereographic attention residual)

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 审计协议 | 加载 ckpt + enumerate state_dict keys + 找 κ/scale 模式 | Issue #109 spec §Gate3 强制要求: 必须明确报告 Stage 3 是否持有 κ/scale |
| FAIL 判定 | state_dict 中无任何 κ/scale 模式 key → Gate 3 FAIL | Issue #109 spec §Gate3 强制判定: "若 Stage 3 实际只接收 SID 且无 κ/scale 通道, 必须明确判为 Gate3 FAIL" |
| 后续方向 | close issue #109, 推到 Issue #113 (架构改造) | R18 + R11.5: issue 路径不可执行, 必须先解决架构限制 |

---

## 后续 (per R22 + R19 + R16)

1. **commit + push Issue #109 Gate 3 FAIL verdict** ✅ (commit 8e0ca7f + 7c4a1a2 R21 v2 fix, 已 push)
2. **close Issue #109** ✅ (per R16 + R20 + R21 comment)
3. **Issue #113 架构改造** (task426 in_progress): 实现真实三分量 product checkpoint schema, 是 #109/#110 共同架构限制的解药
4. **R10 v2 idle 检查**: 0 OPEN issue after close #109 + R10 v2 + R22 + R19 协同