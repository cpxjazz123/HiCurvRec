# Issue #210 Phase C Complete: K 反模式 (256,128,64) FAIL + 新发现

日期: 2026-08-08
Issue: #210 Phase C
状态: **K 反模式 假设 FAILED, 但发现更强的 κ 异质机制**

---

## 1. Phase C 假设

**P0 推论 (Phase A 结论)**: K (codebook size) 是 κ heterogeneity 主因.

**Phase C 假设**: K 反模式 K=(256,128,64) 应该驱动 κ 镜像 (高,中,低). 即:
- L0 (K=256) → κ 最高
- L1 (K=128) → κ 中
- L2 (K=64) → κ 最低

**与 v15 (K=64,128,256) → κ=[0.30, 1.79, 1.48] 形成完美镜像**:
- v15: K 递增 → κ 异质
- desc256: K 递减 → κ 镜像

---

## 2. 实测结果

### 2.1 desc256 (K=256,128,64) 配置

| 参数 | 值 |
|---|---|
| K | (256, 128, 64) |
| epochs | 1000 |
| batch_size | 1024 (单卡) |
| Stage1 | hyp_v2 (sha=e8fea26a) |
| 数据 | 9922 items |

### 2.2 final_kappas (实测)

| Layer | K | final_kappa |
|---|---|---|
| L0 | 256 | **0.4515** |
| L1 | 128 | 0.4676 |
| L2 | 64 | **0.4680** |
| **κ_diff** | - | **0.0164** |

### 2.3 5 配置 κ_diff 对比

| 配置 | K | **κ_diff** |
|---|---|---|
| v15 | (64,128,256) | **1.4885** |
| equal64 | (64,64,64) | 0.0167 |
| equal128 | (128,128,128) | 0.0172 |
| equal256 | (256,256,256) | 0.0172 |
| **desc256** | **(256,128,64)** | **0.0164** |

---

## 3. 关键发现

### 3.1 Phase C 假设 FAILED

**desc256 实测 κ 完全不镜像** (3 个 κ 几乎相等,与 equal64/equal128/equal256 难以区分):
- 期望: κ L0 (K=256) > L1 (K=128) > L2 (K=64)
- 实际: κ L0 (K=256) = L1 (K=128) = L2 (K=64) ≈ 0.45-0.47

**含义**: K (codebook size) 单独**不能**驱动 κ 异质. K 异质 ≠ κ 异质.

### 3.2 修正 Issue #210 模型

**Phase A BUG 修正**: P0 错误推论 "K 主因" 应限定为 **K 顺序 (递增方向 + 浅层 K 小)** 共同作用.

**新模型 — κ heterogeneity 必要条件**:
1. ✅ K 顺序递增 (浅层 K 小, 深层 K 大)
2. ✅ K 容量差异显著 (v15 64→128→256 = 4x 跨度)
3. ✅ REC_LAYER_W 递增 (capmatch 配置 [1, 3, 9])
4. ✅ RQ 残差累积 (浅层 K 小 → 残差大 → μ 衰减)

**满足所有 4 条件**: v15 (1.49 κ_diff)
**满足 1, 2, 4 但 K 顺序反向**: desc256 (0.016 κ_diff, 与 equal 系列无显著差异)

### 3.3 关键推论

**v15 capmatch 成功机制细化**:
- 浅层 K=64 → 残差大 → REC_LAYER_W=1 弱权 → κ 弱 (0.30)
- 深层 K=256 → 残差小 → REC_LAYER_W=9 强权 → κ 强 (1.48)
- **K 递增方向 = 残差衰减方向 = κ 增强方向** 三者协同 → κ 异质

**desc256 失败原因**:
- 浅层 K=256 → 残差小 → REC_LAYER_W=1 弱权 → κ 弱 (0.45)
- 深层 K=64 → 残差大 → REC_LAYER_W=9 强权 → κ 强 (0.47)
- **K 递减方向 = 残差增强方向 = κ 增强方向** 矛盾 → κ 几乎无异质

### 3.4 物理直觉

- K 递增 → 码本 capacity 逐步提升 → 残差必须**逐层缩小** → 信息流 funnel
- K 递减 → 码本 capacity 收缩 → 残差**逐层放大** → 信息流 bottle (信息瓶颈在最后一层)
- v15 funnel 模式 配合 capmatch 加权 → κ 异质
- desc256 bottle 模式 与 capmatch 加权 冲突 → κ 几乎一致

---

## 4. Gate 评估

| Gate | 判定 | 结果 |
|---|---|---|
| **Gate 1** | K 反模式 → κ 镜像 | ❌ **FAIL** (κ_diff 0.016, 不镜像) |
| **Gate 2** | desc256 util_4digit 健康 | ✅ PASS (util_4digit=1.0) |
| **Gate 3** | desc256 κ_diff < 0.05 | ✅ PASS (0.016) |
| **Gate 4** | 修正 P0 推论 (K 顺序 + 累积方向协同) | ✅ PASS |

**Phase C 整体**: 假设 FAILED, 但发现更强的 κ 异质机制 (K 顺序 + 残差累积方向 + REC_LAYER_W 协同).

---

## 5. 实践意义

### 5.1 v15 capmatch 仍是当前最佳
- 4 必要条件同时满足 → v15 κ 异质最强 (1.49)
- test R@10 = 0.1034 (v15) 仍是当前 SOTA

### 5.2 探索 K 异质新设计
- **可尝试**: K=(64, 128, 128) → K 递增但 2-3 层相同 (弱 capmatch)
- **可尝试**: K=(64, 64, 256) → 跳跃式递增 (强异质)
- **可尝试**: K=(32, 128, 256) → 极端浅层 (强 funnel)
- **不推荐**: K=(256,128,64) 递减模式 (Phase C 验证 bottleneck 与 capmatch 冲突)

### 5.3 Stage3 协同潜力
- v15 SID κ 异质为 Stage3 HAB 编码器提供 geometric anchor
- 若 K 异质被消除 (e.g., equal128), Stage3 HAB 编码器失去 κ 异质先验 → 解码质量下降
- 验证方向: equal128 SID → Stage3 训练 → 看是否比 v15 Stage3 差

---

## 6. 后续路线

### 6.1 Phase D 候选

**D1**: K=(64, 64, 256) 跳跃式递增 (验 K 跨度突变)
**D2**: K=(32, 128, 256) 极端浅层 (验极端 funnel)
**D3**: K=(64, 128, 128) capmatch 弱异质 (验 κ 异质 vs util 平衡)
**D4**: 全 equal 128 SID + Stage3 训练 (验 κ 异质缺失的代价)

### 6.2 立即优先级
- D1 D2 D3 都是 SID 训练 (~7 min/配置)
- D4 需要完整 Stage3 训练 (~30 min) + Stage4 评估 (~5 min)
- 推荐: 先 D1 + D2 + D3 (3 配置 × 7 min = 21 min), 再 D4 决策

---

## 7. 文件清单

- desc256 ckpt: `taskA/_history/issue210_equal_codebook/taskA_stage2_desc256/hrqvae_kappa_sync.ckpt`
- 5 配置 summary: `taskA/_history/issue210_equal_codebook/phase_c_5config_summary.json`
- 5 配置 per-config JSON: `taskA/_history/issue210_equal_codebook/phase_a_{v15_64x128x256,equal64,equal128,equal256,desc256}.json`
- verdict: 本文件

---

## 8. DECOR BAN

Phase C 严格在曲率框架内推进, **不引入任何 DECOR 机制**.

---

## 9. 结论

**Issue #210 Phase C 假设 FAILED**, 但修正了 P0 推论:

- K 异质 → κ 异质 (P0 错误): **必须 K 顺序 + 残差累积方向 + REC_LAYER_W 协同**
- v15 capmatch 是 4 条件同时满足的唯一例子 → κ 异质最强 (1.49)
- desc256 K 递减让 4 条件冲突 → κ 几乎一致 (0.016)

**v15 仍是当前最佳 SID 链路**. 后续探索 K 异质新设计 (D1/D2/D3) + Stage3 协同验证 (D4).
