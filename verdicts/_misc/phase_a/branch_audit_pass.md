# 新 Issue: Prefix-Conditioned Branch Curvature RQ-VAE — Phase A Structure Audit PASS

日期: 2026-08-08
commit: pending

---

## 1. 总结

**Phase A PASS** — 对 v15 SID (sha=5f8331cc462c867f40a40bf4378760c27c4abb74b5843fb91380f2274f07) 跑 Structure Audit, **同层不同 prefix 之间存在显著异质性**, 满足 Gate A 判据 (CV(B) > 0.10 阈值), 进入 Phase B (3-Expert Branch Curvature RQ-VAE 实现).

**关键证据**:
- **L0 层 (64 个 prefix, utilization=100%)**: n_p ∈ [70, 353] (4.99x 跨度), B(p)=exp(H(p)) ∈ [33.33, 82.86] (2.49x 跨度), **CV(B)=0.180, CV(σ_r)=0.208** — 健康异质性
- **L0-L1 层 (4485 个 prefix, utilization=54.75%)**: n_p ∈ [1, 17] (17x 跨度), B(p) ∈ [1.00, 17.00] (17x 跨度), **CV(B)=0.849, CV(σ_r)=1.940** — **极强异质性**, 大量 leaf prefix (n=1) 与 hub prefix (n=17) 并存
- **结构性证据支持 branch-specific curvature**: 不同 prefix 的 branch complexity 和 residual variance 差异显著, branch-aware 曲率有充分动机

---

## 2. 实验设置

- **输入 SID**: `taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy`, shape (9922, 4), int64
- **L0 codebook 容量**: 64 (utilization 64/64=100%)
- **L1 codebook 容量**: 128 (实际 usage 4485/(64×128)=54.75%)
- **L2 codebook 容量**: 256 (实际 usage 待查, 不影响 L0/L0-L1 分析)
- **Stage1 嵌入**: `HG-Rec/dataset/Instruments/item_emb.parquet` (9922, 768) sentence-t5-base Euclidean
- **统计量**: item count (n_p), child count (n_children), Shannon entropy H(p), effective branching factor B(p)=exp(H), residual variance σ_r (subtree item emb spread)

---

## 3. L0 Prefix 统计

| 指标 | min | max | mean | median | CV |
|---|---|---|---|---|---|
| n_p (item count) | 70 | 353 | 155.0 | 149 | 0.402 |
| n_children (L1 子码字数) | 39 | 109 | 70.1 | — | — |
| H(p) | 3.506 | 4.417 | 3.991 | — | 0.057 |
| B(p)=exp(H) | 33.33 | 82.86 | 55.04 | — | **0.180** |
| σ_r (residual var) | 0.0335 | 0.0845 | 0.0527 | — | **0.208** |

**解读**:
- L0 utilization 100% (64/64), 健康
- 每个 L0 prefix 平均 70 个 L1 子节点 (L1 总容量 128, 利用率 70/128=55%)
- **B(p) CV=0.180**: 不同 L0 prefix 的子树复杂度差异 ~18%, 中等异质性
- **σ_r CV=0.208**: 不同 L0 subtree 在 768d 嵌入空间分散度差异 ~21%, 中等异质性
- L0 σ_r max/min ratio = 0.0845/0.0335 = **2.52x**, 部分 prefix 的残差结构复杂度是其他 prefix 的 2.5 倍

---

## 4. L0-L1 Prefix 统计 (深层细粒度)

| 指标 | min | max | mean | median | CV |
|---|---|---|---|---|---|
| n_p (item count) | 1 | 17 | 2.2 | 2.0 | 0.929 |
| n_children (L2 子码字数) | 1 | 17 | 2.2 | — | — |
| H(p) | -0.000 | 2.833 | 0.559 | — | 1.106 |
| B(p)=exp(H) | 1.00 | 17.00 | 2.20 | — | **0.849** |
| σ_r (residual var) | 0.0000 | 0.0874 | 0.0070 | — | **1.940** |

**解读**:
- L0-L1 utilization 54.75% (4485/8192), 健康
- **大量 leaf prefix (n_p=1, 即单 item leaf)**: median=2.0, mean=2.2 — 绝大多数 L0-L1 prefix 只有 1-2 个 item
- **少量 hub prefix (n_p=17)**: max=17 — 部分 L0-L1 prefix 是 dense hub
- **B(p) CV=0.849**: 极强异质性 — 部分 prefix 退化为 leaf (B=1), 部分 prefix 是 rich hub (B=17)
- **σ_r CV=1.940**: 极强残差异质性 — σ_r max/min ratio = 0.0874/0.0000 → 理论 ∞ (因 min=0 是单 item prefix)
- **L0-L1 层 branch-aware curvature 动机极强**

---

## 5. Gate A 评估

**判据**: 不同 prefix 的 branching factor 有明显分布差异, CV(B) > 0.10 阈值.

| 指标 | L0 | L0-L1 | 阈值 |
|---|---|---|---|
| CV(B) | 0.180 | **0.849** | > 0.10 |
| CV(σ_r) | 0.208 | **1.940** | (支持证据) |

**Gate A 决策: ✅ PASS**.

**理由**: L0 和 L0-L1 两个尺度的 prefix 异质性都超过阈值, 且 L0-L1 层异质性极强 (CV(B)=0.849, CV(σ_r)=1.940), branch-specific curvature 有充分结构动机.

---

## 6. 后续步骤

进入 **Phase B: 3-Expert Branch Curvature RQ-VAE 实现**.

**架构思路** (待 Phase B 详化):
- 每层 (L0/L1/L2) 维护 **3 个曲率专家**: κ_low / κ_mid / κ_high
- **Router** 根据 prefix 结构特征 (B(p), n_p, σ_r) 选择 expert
- 路由策略: 
  - n_p 小 + σ_r 大 → κ_high (high curvature, 更细粒度几何)
  - n_p 中等 + σ_r 中 → κ_mid (中等曲率)
  - n_p 大 + σ_r 小 → κ_low (low curvature, coarse geometric approximation)
- 共享 codebook, expert 仅影响 κ 调度

**风险**:
- 增加 ~2x 参数 (3 个 κ per layer), Stage2 训练时间增加 ~10-20%
- Router 需用 prefix structure features (来自 SID + item_emb), 不能用 raw embedding 否则 leak test info
- 实验预期: Phase B 完成后 → Phase C (kappa-sync Stage2) → Phase D (Stage3 推荐实验)

**严禁**任何 DECOR 机制. 纯曲率路线继续.

---

## 7. 文件清单

- 审计脚本: `/home/wlia0047/.claude/jobs/6ae5ecdb/tmp/phase_a_structure_audit.py`
- 审计输出: `/fs04/ar57/wenyu/GeneRec/verdicts/phase_a_structure_audit.json`
- verdict: 本文件
- log: `/tmp/phase_a_audit.log`
- 输入: `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy`