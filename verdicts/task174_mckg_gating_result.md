# Task #174 verdict — κ-Stereographic + M=2 + MCKG 门控融合 D 臂 [⛔ C3 NO-GO]

> **任务目的**: 验证"M=2 多分量 + 门控融合 MLPGate 是否能救 κ-Stereo 框架"假设 (用户 design §5 D 臂). R5 fixed seed=42.
> **完成日期**: 2026-07-25
> **状态**: ⛔ C3 NO-GO — test R@10=0.0993 < baseline 0.1058 (-6.2%)

---

## 1. 关键指标

| 指标 | 测试值 | Baseline #105 | #164 纯 κ-Stereo | #171 dead_code_reset | Δ vs baseline |
|------|-------|-------|--------|--------|-------|
| **Recall@10 (主要)** | **0.0993** | 0.1058 | 0.0964 | 0.1001 | **-6.2%** |
| Recall@5 | 0.0811 | 0.0822 | 0.0777 | 0.0798 | -1.3% |
| Recall@20 | 0.1219 | 0.1214 | 0.1175 | 0.1221 | +0.4% |
| NDCG@10 | 0.0749 | 0.0823 | 0.0718 | 0.0728 | -9.0% |
| NDCG@20 | 0.0806 | 0.0901 | 0.0775 | 0.0783 | -10.6% |

---

## 2. 执行时间线

- **Stage 1 launch**: Phase A/B 200 epoch (Phase A ep1-100 frozen κ=0, Phase B ep101-200 lr_theta=1e-5)
- **Stage 2 codebook inference**: 完成 (12:14-12:22, Sinkhorn 30 iter, 810 dup→0)
- **Stage 3 T5-mini 9.18M training**: 12:23 → 12:56 (~33 min, 200 epoch + early stop @ ep56)
- **Stage 4 test eval**: 13:03 → 13:04 (~1 min, after Stage 4 daemon auto-trigger 失败 → 手动 v3 fix 后成功)
- **总时长**: ~50 min

---

## 3. 关键发现 (Stage 1 + Stage 4)

### 3.1 κ_m 几乎学不到 (跟 A/B/C 臂一致)
- κ_m final = [-0.0095, -0.0086, -0.0074] (L0/L1/L2 各分量)
- 跟 #164 (-0.09) / #169 (-0.09) 相比**10× 更浅**
- best ckpt 保存于 Phase A ep30 (loss=0.9471), pre-collapse 模型
- L0u=100%, L1u=100%, L2u=98.44% Phase A 末全部健康

### 3.2 Gating 退化为单 component [关键 finding]
- L0/L1/L2 w_mean = [1.000, 0.000] (Phase A ep50+, Phase B ep100+, ep150+, ep200+)
- gating network 学到把所有 weight 给 component 0, **component 1 完全不用**
- D 臂**退化为 A 臂** (单 component κ-Stereo)

### 3.3 Test 性能 (Stage 4)
- test R@10 = 0.0993 (-6.2% vs baseline)
- **第二最接近 baseline 但仍 NO-GO** (仅次于 #171 -5.4%)
- val R@10 推断 ~0.10+ (跟 #164/#171 一致 val/test gap -7% to -19%)

---

## 4. 距离公式正确性 (Numerical Verification 已做)

用户问 "cost 是普通 L2 还是 κ-curved" — **已 verify 是真正的 κ-Stereographic**:

| Test | 内容 | 结果 |
|------|------|------|
| κ → 0 退化 Euclidean | rel err | 7.2e-8 ✓ |
| κ_m=0 impl vs Σ w_m·L2 | max diff | 0.000000 ✓ |
| κ_m=-0.5 impl vs Berman-Metzler 2020 reference | **max diff = 0.00000000** ✓ |
| gate_net grad flows (NOT detached) | grad norm | 0.0622 ✓ |
| gate output softmax (Σ w_m = 1) | min/max | 1.0000 / 1.0000 ✓ |

**结论**: distance formula 实现正确 (Berman-Metzler 2020 Eq 3), gate 收到梯度, softmax 正常. **NO-GO 不是公式错, 是 κ_m ≈ 0 + gating 退化双重失效**.

---

## 5. 分析解读

### 5.1 假设证伪
**假设**: "门控融合 w_m(x) 能学出非平凡 component 权重, 跟 κ-Stereo 测地距离协同救场"
**结果**: NO — gating 学到 w=[1,0] (跟 #89 A 臂 18/18 κ_m=0 一致的退化模式). 模型选择走单 component κ-Stereo 路径.

### 5.2 D 臂退化为 A 臂的根因
- M=2 多分量理论提供 2 个独立 κ_m, 但量化器优化只激励一个
- L0u=100% 健康但只用了 component 0 → component 1 κ_m 仍 ≈ 0 (无梯度信号)
- 跟 #89 #164 一致: "Toys 数据本质欧式最优, κ 学不到非零"

### 5.3 D 臂 vs 4 κ-Stereo 变体 ranking
| 变体 | test R@10 | Δ vs baseline | 排名 |
|------|-----------|---------------|------|
| #171 dead_code_reset | 0.1001 | -5.4% | 第 1 (距 baseline 最近) |
| **#174 D 臂 MCKG gating** | **0.0993** | **-6.2%** | **第 2** |
| #172 κ_max=4.0 | 0.0972 | -8.1% | 第 3 |
| #164 纯 κ-Stereo | 0.0964 | -8.9% | 第 4 |
| #169 Sinkhorn L2 only | 0.0863 | -18.4% | 第 5 |
| #170 Sinkhorn all 3 | 0.0836 | -21.0% | 第 6 (最差) |
| Baseline #105 (Euclidean RQ-VAE) | 0.1058 | — | reference |

---

## 6. 决策触发判定

按 descriptions/task174 §3 表:
- 实际 test R@10 = 0.0993 < 0.1058 baseline → **⛔ NO-GO**
- 跟 #163 synthesis verdict 联合: **8 个 κ-Stereo 变体穷尽, 全部 < baseline**
- **Stop hook 条件未满足**

---

## 7. 产物清单

- **Stage 1 best ckpt**: `products/task174/phase_b_mckg_gating/jul-25-2026_12-12-00/best_loss_model.pth` (4.57 MB)
- **Stage 2 SID codebook**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_phase_b_mckg_gating.npy` (int64 (9922, 4), 100% util)
- **Stage 3 best ckpt**: `products/task174/t5mini_mckg_gating/jul-25-2026_12-23-44/Instruments/Jul-25-2026_12-23-54/HG_Rec_best.pth` (30.48 MB)
- **Stage 4 metrics**: `verdicts/task174_mckg_gating_metrics.json` (test_recalls, test_ndcgs)
- **Stage 4 eval script**: `scripts/task174_t5mini_mckg_gating_stage4_eval.sh` (v3 with config dict pattern)
- **D 臂 model**: `HG-Rec/model/hrqvae_mckg_gating.py` (κ-Stereo 公式已 numerical verified)

---

## 8. 后续建议 (R8 + 决策)

1. **§16 cleanup (R8)**: 立即从 §16 删除 Task #174 行 (Stage 4 已 exit 0)
2. **#163 synthesis verdict**: 整合 8 κ-Stereo 变体 NO-GO 报告 (Task #174 是第 8 个, 闭环)
3. **Stop hook 决策**: 8 变体穷尽, κ-Stereo 路径在 Toys 数据上 NO-GO. 用户可选:
   - (a) 接受 Euclidean RQ-VAE 0.1058 作为 final baseline
   - (b) 尝试完全不同方向 (e.g., 完全放弃 κ, 看 baseline 0.1058 是否有 paper Table 1 RQ-VAE Toys 0.034 上限)
   - (c) 转向更大模型 (TIGER 原 paper 220M) 或完全不同的 SID framework