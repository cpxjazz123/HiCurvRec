# Issue #76 Precheck — 重大发现

**Date**: 2026-08-07
**precheck 状态**: **PASS** (must-have 全部捕获)
**但**: 发现关键事实推翻了 Issue #76 Phase A 的前提假设

---

## 1. Precheck 完整结果

### v77 baseline (已知)
| 指标 | 值 |
|------|-----|
| valid_R10 | 0.1312 (ep104) |
| test_R10 | 0.1080 |
| λ_raw | [-1.37, 0.57, 1.19] |
| λ_eff | [-0.20, 0.20, 0.20] (全饱和) |
| U/V l2_norm | [0.047, 0.063, 0.088] (极小) |

### v77 产物 hash
- v77 ckpt SHA256: `6b65d9890c7332ef9b27bd619e37698458d644d579b9ba9aefeefd1d8b9e7074` (22.5MB)
- Stage2 ckpt SHA256: `0fe023d38e1e6e6a851ee227f79964bedba878b10f9ed602631a6c401c9a3e99` (4.6MB)
- SID SHA256: `d01a89174bce4150d18621b9e3991a2d91aece6c5cd255dc358a8bccd8e82b15` (9922×4)
- SID unique_3digit ratio: **0.8392** (8327/9922)

### Stage2 真实状态 (本次发现)

```
config: num_emb_list=[64, 128, 256], e_dim=32, layers=[512,256,128,64]
final_kappas:      [0.0, 0.0, 0.0]    ← 三层曲率全 0
final_cs:          [1.0, 1.0, 1.0]    ← 三层曲率参数 c 全 1
final_mix_weights: [1.0, 1.0, 1.0]
gate2_decision:    "FAIL" (历史 verdict)
```

**codebook shapes**:
- vq_layers.0.embeddings.weight: [64, 32]
- vq_layers.1.embeddings.weight: [128, 32]
- vq_layers.2.embeddings.weight: [256, 32]

### 用户/交互审计 (leave-one-out 风格)

| split | users | items | interactions |
|-------|-------|-------|--------------|
| train | 24772 | 9913 | 156609 |
| valid | 24772 | 9921 | 181381 |
| test  | 24772 | 9922 | 206153 |

- **用户完全重叠** (train∩valid=test=24772, 同 user 不同时间窗口)
- **物品高度重叠** (train∩test=9913, valid∩test=9921)
- 这是 leave-one-out split, **行为图只能从 train 序列构造** (Issue #76 hard constraint 满足)

---

## 2. 🚨 关键发现: Stage2 实际是 FLAT EUCLIDEAN

**v77 SID 来自 `taskA_stage2_hyp_v2_capmatch_1000ep`**, 该 Stage2 的:

- final_kappas = [0, 0, 0]
- final_cs = [1, 1, 1]  
- gate2_decision = "FAIL"

**含义**: v77 SID 是在**完全欧氏空间**量化的, **没有任何负曲率变形**. v77 HAB 用的 Dbar=[0.4109, 0.2375, 0.2108] (或 v77 原始 [0.5384, 0.2881, 0.2210]) 是**欧氏距离的中位数**, 不是双曲距离.

**v77 test_R10=0.1080 的"双曲"成分**:
- Stage1 per-item radius: **真正生效** (在球面切空间做 radius 缩放)
- Stage2 κ: **DORMANT** (c=1, flat)
- Stage3 HAB: 用**欧氏距离**作为 bias, 残差学习的 U·V^T 范数极小 (0.04-0.09), 学到的是 Dbar 微调

**v77 不是真正的双曲推荐**, 而是 **per-item radius Stage1 + flat Euclidean Stage2 + 残差欧氏 bias Stage3** 的混合架构.

---

## 3. Issue #76 假设验证

Issue #76 Phase A 假设:
> "v77 的瓶颈不是 HAB 强度不足, 而是 Stage2 曲率与 next-item 行为目标未充分对齐"

**precheck 判定**: **假设不成立**

**原因**: Stage2 final_cs=[1,1,1] 即 c=1, **没有曲率信号可供对齐**. "Stage2 曲率与行为未对齐" 这句话的逻辑前提 (Stage2 存在曲率) 不成立.

**Issue #76 的真正正确表述应是**:
> "v77 没有 Stage2 曲率 (c=1 flat). 需要先用行为监督激活 Stage2 曲率, 再让 HAB 使用真正双曲的距离."

---

## 4. 三条路径 (请用户裁决)

### 路径 A: 按 Issue #76 Phase A 完整重训 (推荐)

**操作**: 
1. 接受 v77 SID 的 Stage2 是 flat 的事实
2. 用 Issue #76 Phase A 设计: **行为监督 + 曲率隔离更新**, 从 c=1 开始激活三层独立 κ
3. 重建 SID (新曲率下), 用新 SID + 新 Stage2 ckpt 进 Stage3

**风险**: 
- 重训 SID 破坏 Stage3 兼容性 (类似 v10b 教训, Issue #76 自身 verdict 提到)
- Phase A 实现复杂, 行为图 + 流行度校正 + 曲率隔离更新都要写
- 单 seed 验证, 失败成本高

**预期 ROI**: HIGH (如果成功, 这是真正的双曲推荐)

### 路径 B: 修正 Issue #76 假设, 跳过 Phase A 直接调参

**操作**:
- 接受 v77 已经是 flat Stage2 + 残差 HAB
- Issue #76 Phase B "商品级乘积曲率差分 HAB" 仍然适用, 因为 Dbar 是欧氏距离
- 但 ΔG = log(D_hyp/D_flat) 在 c=1 时退化为 0 (因为 D_hyp=D_flat), 无信号

**结论**: 路径 B 在 v77 flat Stage2 下 **完全无意义**. 必须先激活 Stage2 曲率.

### 路径 C: 回到 v77→v85 调参路线, 认 0.1080 是真实上限

**操作**: 不做架构级改动, 接受 v77 test=0.1080 是 HAB+Stage1+flat Stage2 架构天花板

**结论**: 与 Issue #76 派工冲突. 如果派工 Issue #76, 必须走路径 A.

---

## 5. 我的推荐

**走路径 A**, 但需要修订 Issue #76 的 Phase A 设计, 加上"激活 Stage2 曲率"作为前置步骤:

**A.1**: 用 v10b_relstruct 的方法 (c=exp(κ) + stop-grad c + REL_STRUCT) 让 Stage2 学到非零 κ
**A.2**: 加 Issue #76 设计的 L0/L1/L2 行为图 + 流行度校正 + KL(p_beh||p_geo)
**A.3**: 交替更新 (kappa step 冻 encoder/codebook ↔ RQ step 冻 κ)
**A.4**: 验证 SID 兼容 (Phase A 自己 valid 评估, 不读 test)

**当前任务已完成 precheck 阶段, 等待用户裁决走哪条路径.**

---

## 产物

- precheck script: `issue76_precheck.py` (R30 inline)
- precheck JSON: `/home/wlia0047/.claude/jobs/6ae5ecdb/tmp/issue76_precheck.json`
- 本 verdict: `verdicts/issue76_precheck_critical.md`
