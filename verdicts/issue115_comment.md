## Issue #115 R20+R21 强制 4 Gate 详细内容 + commit hash

### Gate 1 (= Stage 1 FreeCurvVQ K[64,128,256] 30 epoch): ❌ FAIL per spec
- 关键数据: ckpt_sha256=782930523889dd84, best_epoch=1 (Sinkhorn ep1 util=100% 一闪)
- final ep30 metrics: **L0 util=75.0% (< 90% 阈值)**, **L0 max_load=96.9% (≫ 5%)**
- L1 util=96.1% / max_load=50.8% (FAIL), L2 util=90.2% / max_load=14.0% (FAIL)
- κ_m_effective L0: `[-0.0078, -0.0078, -0.0078]` (R137 tanh(θ_m) ≈ 0, κ 未学)
- codebook_norm_mean L0: **0.0372 ≪ 0.7** (Phase 0 mode collapse 假象, 跟 task178/task180/task231 一致)
- 训练时长: 30 epoch × ~0.6s = ~18s
- 失败原因: (1) L0 K64 容量不足, 30 epoch 短训无法覆盖 9922 items; (2) κ_m 卡 0 (R137 init 缺陷); (3) norm 退化
- verdict 路径: `verdicts/task408_issue115_gate1_fail_v2.md`
- commit: 3f79615

### Gate 2 (= Stage 2 SID 复用 #102): ⏸ STOP per spec
- 原因: Gate 1 FAIL, 无 SID 产出可推断
- Issue spec 强制: 必须复用 #102 SID (task396, 9922×4, collision 0%)

### Gate 3 (= Stage 3 T5 adapter): ⏸ STOP per spec
- 原因: Gate 1 FAIL, 无 sidecar 可供 adapter 加载
- Issue spec 强制: 在原 SID embedding 上添加三层独立 adapter/gate, 输入为对应层 κ/scale/norm sidecar. 5 组反事实 (原始/全零/层间置换/#47 一致重标定/故意不一致) 必须基于已训 sidecar.

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 1+2+3 全部 STOP
- Issue spec 强制: R@10>0.1020 才 Target reached (vs HG-Rec baseline 0.1020)

### 关键产物
- commit hash: **3f79615**
- push: origin/main (5aed3c5..3f79615)
- verdict: `verdicts/task408_issue115_gate1_fail_v2.md`
- 实施: `scripts/task408_issue115_stage1_freecurv_train.py` (590 lines, Issue #97 poincare_distance patch + FreeCurvResidualVectorQuantization + Sinkhorn + 30 epoch training + sidecar export)
- 训练产物: `products/task408_issue115_stage1_freecurv/` (ckpt + train.out + sidecar.json + train_log.json)
- 整体决策: **❌ NO-GO 收口** (Issue #115 K[64,128,256] learnable variable-curvature 主路架构级 NO-GO)

### R18 v2 跨方向联立 (跟 #112 路径对比)
| 维度 | Issue #112 (已关闭) | Issue #115 (本 task) |
|------|---------------------|----------------------|
| D1 spec 摘录 | 错误依赖 #113 | **撤销依赖, 独立 #100 上游** ✅ |
| D2 实施核心 | (未执行) | **FreeCurvVQ K[64,128,256] 30 epoch** ✅ 新数据 |
| D3 Gate 失败机制 | (依赖误判) | **L0 K64 + κ_m 卡 0 + norm 退化** ✅ 新根因 |
| D4 引用文献 | arXiv:2405.13979 | arXiv:2405.13979 ✅ |

新数据 + 新路径, 不允许沿用判决 (R18 v2).