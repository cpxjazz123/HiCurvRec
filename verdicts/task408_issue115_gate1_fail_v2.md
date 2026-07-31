# Task #408 / Issue #115 [方向A Gate3恢复] Gate 1 FAIL 收口 verdict

**日期**: 2026-07-31
**Issue**: #115 [方向A Gate3恢复] — 独立 #100 Stage 1 sidecar (owner 撤销 #112 错误依赖 #113)
**任务**: 训练 K[64,128,256] FreeCurvVQ (R137 fix) 30 epoch + 导出 sidecar + Gate 1 判定

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 FreeCurvVQ K[64,128,256] 30 epoch): ❌ FAIL per spec

**关键数据**:
- ckpt_path: `products/task408_issue115_stage1_freecurv/ckpt/task408_best_epoch_01.pth`
- ckpt_sha256: `782930523889dd84...` (full in sidecar.json)
- ckpt_size_bytes: (in sidecar.json)
- best_epoch: **1** (第 1 epoch util=100% 因 Sinkhorn 一次性激活所有码字, 后续 epoch 退化为 mode collapse)
- best_avg_util: **1.0000** (epoch 1 Sinkhorn-balanced 状态)
- 三层 K 配置: ✅ L0=64 / L1=128 / L2=256 (符合 Issue #115 spec)
- final_epoch_metrics (ep 30):
  - **L0 util=75.0% < 90% ❌ FAIL** (Issue #115 spec §Gate1 要求三层 util≥90%)
  - **L0 max_load=96.9% ≫ 5% ❌ FAIL** (Issue #115 spec §Gate1 要求 max_load<5%)
  - L1 util=96.1% (PASS), L1 max_load=50.8% (FAIL, ≫5%)
  - L2 util=90.2% (PASS), L2 max_load=14.0% (FAIL, >5%)
- κ_m_effective L0: `[-0.0078, -0.0078, -0.0078]` (R137 tanh(θ_m) ≈ 0, 跟 task287/task284 一致 — κ 未学起来)
- codebook_norm_mean L0: **0.0372 ≪ 0.7** (Phase 0 mode collapse 假象, 跟 task178/task180/task231 一致)
- 无 NaN/Inf
- 训练时长: 30 epoch × ~0.6s = ~18s (超快, 验证短训 recipe)

**失败原因**:
1. **L0 K64 容量不足**: 30 epoch 受限短训下, L0 64 个码字无法充分覆盖 9922 items, 单码字 max_load 96.9% (≈ 153 items/码字)
2. **Sinkhorn balanced 假象**: 第 1 epoch Sinkhorn-Knopp 把码字一次均匀分配, util=100%; 但后续 epoch 因 recon_loss 不充分, codebook 退化聚集
3. **κ_m 卡 0**: R137 fix init θ_m=0 → κ_m=0, 但 30 epoch 短训无足够梯度推到非零. 跟 task287/task284 一致 (这是 R137 设计的 init 缺陷, 但已被 task203 REFUTED)
4. **codebook norm 退化**: L0 norm=0.037 ≪ HG-Rec baseline 0.7-1.0 健康区. Phase 0 mode collapse 模式再次出现

**实施脚本**: `scripts/task408_issue115_stage1_freecurv_train.py` (590 lines, 包含 Issue #97 poincare_distance patch + FreeCurvResidualVectorQuantization + Sinkhorn + 30 epoch training + sidecar export)

### Gate 2 (= Stage 2 SID 复用 #102): ⏸ STOP per spec
- **原因**: Gate 1 FAIL, 无 SID 产出可推断
- **Issue spec 强制**: 必须复用 #102 SID (task396, `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy`), shape (9922, 4), collision 0%

### Gate 3 (= Stage 3 T5 adapter): ⏸ STOP per spec
- **原因**: Gate 1 FAIL, 无 sidecar 可供 adapter 加载
- **Issue spec 强制**: 在原 SID embedding 上添加三层独立 adapter/gate, 输入为对应层 κ/scale/norm sidecar. 5 组反事实 (原始/全零/层间置换/#47 一致重标定/故意不一致) 必须基于已训 sidecar.

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 1+2+3 全部 STOP
- **Issue spec 强制**: R@10>0.1020 才 Target reached (vs HG-Rec baseline 0.1020)

---

## 关键产物

- **训练脚本**: `scripts/task408_issue115_stage1_freecurv_train.py` (verified syntax)
- **训练产物**: `products/task408_issue115_stage1_freecurv/` (ckpt + train.out + sidecar.json + train_log.json)
- **训练 PID**: `products/task408_issue115_stage1_freecurv/_TRAINING_PID` (1628006, completed)
- **Sidecar JSON**: `products/task408_issue115_stage1_freecurv/sidecar.json` (含 ckpt SHA256 + 30 epoch metrics + 每层 κ_m_effective + codebook norm)
- **verdict**: `verdicts/task408_issue115_gate1_fail_v2.md` (本文件)

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #112 (已关闭) | Issue #115 (本 task) |
|------|---------------------|----------------------|
| **D1 spec 摘录** | 错误依赖 #113 (owner 撤销) | **撤销依赖, 独立 #100 上游** ✅ |
| **D2 实施核心** | (未执行 adapter) | **FreeCurvVQ K[64,128,256] 30 epoch 训练** ✅ |
| **D3 Gate 失败机制** | (依赖误判) | **L0 K64 容量不足 + κ_m 卡 0 + norm 退化** ✅ 新数据 |
| **D4 引用文献** | arXiv:2405.13979 | arXiv:2405.13979 ✅ |

**R18 v2 强制结论**: Issue #115 路径已实证 (新 ckpt, 新 metrics). 跟 #112 路径**有差异** (撤销依赖 + 独立 K[64,128,256] FreeCurvVQ 训练). 不允许沿用判决.

**跟 #100 (task84) 联立**:
- task84 (200 epoch) HG-Rec baseline R@10=0.1020, 但实际上没用 θ_m (标准 RQ-VAE)
- task408 (30 epoch FreeCurvVQ) 证明 K[64,128,256] 主路 30 epoch 不足以支持 Gate 1 PASS
- **架构级 NO-GO**: K[64,128,256] + FreeCurvVQ + 30 epoch 受限短训, L0 容量 + κ_m 不动 + norm 退化三因素叠加失败.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 训练 K 配置 | K[64,128,256] (跟 Issue #115 spec 强制一致) | spec §Framework compliance precheck 要求三层 K 一致 |
| 训练 epoch | 30 (Issue #115 spec §Gate1 "受限短训") | spec 强制不允许 sweep |
| Sinkhorn | sk_eps=0.003 + sk_iters=3 (跟 HG-Rec baseline 一致) | 跟 task84 recipe 一致 |
| Init | θ_m=0 (R137 fix) + random codebook (no kmeans_init) | 跟 task287/task284 一致 |
| GPU 分配 | GPU 0 (per R7 全部空闲) | R7 + R19 跨 issue 并行 (Issue #116 占 GPU 1) |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |

---

## 整体决策

**❌ NO-GO 收口**

Issue #115 Gate 1 FAIL (L0 K64 util 75% < 90%, max_load 96.9% ≫ 5%, κ_m 卡 0, norm 退化). 跟 #112 联立 = K[64,128,256] learnable variable-curvature 主路架构级 NO-GO.

**Issue #115 closed NO-GO** (per R16 + R20 + R21).

---

## 后续 (per R22 + R19 + R16)

1. ✅ 写 Gate 1 FAIL verdict (本文件)
2. ⏳ commit + push (R15 + R21 v2)
3. ⏳ gh issue close #115 --reason completed (R16)
4. ⏳ gh issue comment #115 含 4 Gate 详细 + commit hash (R20 + R21)