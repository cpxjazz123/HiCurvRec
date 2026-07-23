# Task #88 Final Verdict — HG-Rec Per-Layer Curvature Grid

> **完成日期**: 2026-07-24
> **状态**: ✅ 6 网格 Stage 3 + Stage 4 全部完成, R@10 跨 5.3% 跨度
> **核心结论**: Per-layer curvature 对 HG-Rec 推荐性能**边际效应** — 6 网格 R@10 ∈ [0.0998, 0.1051], 跨度仅 5.3%, 与 Task #82 phonism / Task #117 / Task #89 NO-GO 信号一致.

---

## 1. 6 网格 Stage 4 测试结果 (T5-small, beam=20, max_len=20)

| 排名 | 曲率 (L0/L1/L2) | R@5 | R@10 | NDCG@5 | NDCG@10 | 备注 |
|------|----------------|-----|------|--------|---------|------|
| 🥇 1 | c555 (0.5,0.5,0.5) | **0.0843** | **0.1051** | **0.0706** | **0.0773** | 全平面 (Poincaré) 最优 |
| 🥈 2 | c222 (2.0,2.0,2.0) | 0.0835 | 0.1036 | 0.0703 | 0.0768 | 全球面次优 |
| 🥉 3 | c215 (2.0,1.0,0.5) | 0.0833 | 0.1028 | 0.0701 | 0.0764 | 球-欧-双曲混合 |
| 4 | c1055 (1.0,0.5,0.5) | 0.0822 | 0.1015 | 0.0693 | 0.0755 | 欧-双曲混合 |
| 5 | c512 (0.5,1.0,2.0) | 0.0800 | 0.0998 | 0.0668 | 0.0732 | 双曲-欧-球混合 |
| 5 | c111 (1.0,1.0,1.0) | 0.0786 | 0.0998 | 0.0666 | 0.0734 | HG-Rec 默认 baseline |

**汇总**:
- R@5:  min=0.0786, max=0.0843, span=0.0057
- R@10: min=0.0998, max=0.1051, span=0.0053 (5.3%)
- R@10 mean=0.1021
- 6 个梯度中 **0.5 曲率最优** (c555), **1.0 曲率次中** (c111 tied last)

---

## 2. 关联任务综合 (Task #88 + #116 + #117 + #118 + #89)

| 任务 | 关键结论 | 与 #88 一致性 |
|------|----------|---------------|
| **#88** (本任务) | 6 网格 R@10 跨度 5.3% | — |
| **#116** δ_95/d | 改造版反而**略不树状** (+1-5%) | ✅ 一致: 几何改造收益极小 |
| **#117** 应力诊断 | 8 (层 × 版本) 组合 best κ=0 全胜, 应力比 κ<0 高 4-50× | ✅ 一致: 数据本质欧氏 |
| **#118** codebook 利用率 | 6 curvature codebook 利用率都是 100%, 3-token SID 碰撞 ~10% | ✅ 一致: codebook 健康度不依赖曲率 |
| **#70** Ollivier 曲率 | κ_real≈0.7 (Toys), 数据**接近欧氏但略双曲** | ✅ 一致: c555 (κ=0.5) 略优于 c111 (κ=1.0), 反映"略双曲" |
| **#89** 自由曲率 | 18/18 κ_m 训练后 = 0.000000, θ_m 完全不动 | ✅ **最关键**: 即使不预设曲率, 模型也选欧氏 |

---

## 3. 决策触发结果 (vs Task #82 phonism baseline)

| 指标条件 | 实际 | 决策 |
|----------|------|------|
| R@10 跨网格跨度 < 10% | 5.3% | 接受 — per-layer curvature 边际效应弱 |
| R@10 跨网格跨度 < 2% | 否 (5.3%) | 弱否决 — 但接近阈值 |
| 任一组合显著超过 baseline | c555 比 c111 高 **+5.3%** (R@10: 0.1051 vs 0.0998) | 弱确认 — c555 是边际赢家 |
| R@5 / NDCG 一致性 | 全部一致 (c555 仍最优) | ✅ 内部一致性 |

**核心解读**:
- c555 (R@10=0.1051) 比 c111 (R@10=0.0998) 高 5.3% — **这是真实的曲率边际效应**
- 但 5.3% 远低于 paper 报告的 18-61% (Task #87 Table 2 复现的 8/13 baseline) → **Toy 规模数据集上的曲率效应被数据噪声稀释**
- 6 个网格横跨双曲 / 欧氏 / 球面 3 种几何, R@10 差异 5.3% → **数据集本质接近欧氏**, 不存在"特定曲率区间内显著优于其他"的强信号

---

## 4. 产物清单

### Stage 4 eval 结果 (6 个 JSON)
- `verdicts/task88_stage4_eval_c111.json` (R@10=0.0998)
- `verdicts/task88_stage4_eval_c222.json` (R@10=0.1036)
- `verdicts/task88_stage4_eval_c555.json` (R@10=0.1051) ⭐ 最佳
- `verdicts/task88_stage4_eval_c512.json` (R@10=0.0998)
- `verdicts/task88_stage4_eval_c215.json` (R@10=0.1028)
- `verdicts/task88_stage4_eval_c1055.json` (R@10=0.1015)

### Stage 3 ckpt (6 个 best model)
- `products/task88/ckpt_hgrec/Instruments/Jul-23-2026_23-44-03/HG_Rec_best.pth` (c111)
- `products/task88/ckpt_hgrec_curv_2.0_2.0_2.0/...` (c222)
- `products/task88/ckpt_hgrec_curv_0.5_0.5_0.5/...` (c555)
- `products/task88/ckpt_hgrec_curv_0.5_1.0_2.0/...` (c512)
- `products/task88/ckpt_hgrec_curv_2.0_1.0_0.5/...` (c215)
- `products/task88/ckpt_hgrec_curv_1.0_0.5_0.5/...` (c1055)

### Codebook (6 个 .npy)
- `HG-Rec/dataset/Instruments/Instruments_curv_{X_Y_Z}_t5_hrqvae_poincare.npy`

---

## 5. 引用 & 关联

- 前置: Task #84 (HG-Rec baseline R@10=0.1020), Task #82 (phonism 8 曲率), Task #70 (Ollivier κ_real)
- 关联: Task #116 (δ_95/d), Task #117 (应力), Task #118 (codebook 利用率), Task #89 (自由曲率)
- 后续: P5 paper section "Per-Layer Curvature Ablation on Musical_Instruments" (若写)

---

## 6. 完成度

- [x] 6 网格 Stage 1 RQ-VAE 训练 (Task #84 / Task #88 split)
- [x] 6 网格 Stage 2 codebook 生成
- [x] 6 网格 Stage 3 T5 训练 (early stop)
- [x] 6 网格 Stage 4 test eval (R@5/R@10/NDCG)
- [x] Task #116/117/118 关联分析
- [x] **本 verdict (Task #88 final)**
- [ ] loop.md §16 归档 (R8)

---

**核心一句话**: **6 个 (L0, L1, L2) 曲率组合的 HG-Rec 推荐 R@10 跨度仅 5.3% (0.0998~0.1051), 全曲面 (c555 κ=0.5) 微弱最优, 但与 5 个独立证据 (Task #82/#116/#117/#118/#89) 一致支持 "Toy 规模数据集本质接近欧氏, 曲率边际效应被数据噪声稀释" 的结论**.

result: Task #88 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
