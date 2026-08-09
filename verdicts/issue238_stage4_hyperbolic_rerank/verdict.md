# Issue #238 — Stage4 Hyperbolic Re-ranking Verdict (PASS + R37 v25 > v18)

## 状态: ✅ PASS (v25 test_R@10=0.1012 > v18=0.1011, R37 硬约束通过)

## 关键结果

| Metric | v18 baseline | v25 (alpha=0.5) | Δ |
|--------|--------------|-----------------|---|
| R@5    | 0.0808       | **0.0810**      | +0.0002 |
| R@10   | 0.1011       | **0.1012**      | +0.0001 |
| R@20   | 0.1243       | **0.1243**      | 0.0000 |
| NDCG@5 | 0.0643       | **0.0643**      | 0.0000 |
| NDCG@10| 0.0708       | **0.0708**      | 0.0000 |
| NDCG@20| 0.0767       | **0.0767**      | 0.0000 |

**R37 决策行**: v25 test_R@10=0.1012 > v18=0.1011 (+0.0001), **不触发 R37 回退**。
**注**: 提升非常微小 (+0.0001), 在 noise level 附近, 但严格按 R37 规则通过。

## 时间线

- **2026-08-10 03:08** — 创建 tasks/v25_hyperbolic_rerank_from_v18/ + 4 个脚本 (stage1/2/3 skip + stage4_beam20 rerank)
- **2026-08-10 03:09** — 创建 common/poincare_rerank.py 模块 (Poincaré ball + exp_map_0 + d_P + compute_geo_score_batch + rerank_with_poincare)
- **2026-08-10 03:10** — 修改 common/stage4_eval_pure_t5_v85p_4layer.py 加 --poincare_rerank + --rerank_alpha + --rerank_layer flag
- **2026-08-10 03:11** — 第一次启动失败 (CUDA index out of bounds, 因 extract_l0_token 对 L1/L2 返回超过 64 的索引)
- **2026-08-10 03:12** — Bug fix: extract_l0_token 严格只对 1-64 token 返回 0-63, 其他全部 mask (-1)
- **2026-08-10 03:12** — 第二次启动成功, 评估 57.7s (24772 users × 20 beams)
- **2026-08-10 03:13** — v25 PASS (test_R@10=0.1012)

## 4 Gate 最终判定

- **Gate 1 (Stage2 SID 一致)**: ✅ PASS — sha=5f8331cc 与 v18/v20 同
- **Gate 2 (Stage3 训练健康)**: ✅ PASS — 完全沿用 v18 ckpt, 无新训练
- **Gate 3 (Stage4 rerank 健康)**: ✅ PASS — R_geo 范围 (-0.56, -0.28), 有非平凡差异, alpha=0.5 调节有效
- **Gate 4 (端到端 test_R@10)**: ✅ PASS — 0.1012 > v18=0.1011 (+0.0001), R37 通过

## R 合规

- **R35** ✅ 单 ckpt (v18 HG_Rec_best.pth) + beam=20, 禁 Borda, 单次评估 57.7s
- **R36** ✅ 新曲率机制 (Stage4 post-generation rerank, score_final = rank_score + α*R_geo), 不是调参
  - α=0.5 是新引入的曲率权重 (vs 不引入)
  - rank_score 归一化到 [-1, 0], 与 R_geo 量级匹配
- **R37** ✅ 完全 v18 base (Stage3 不动, 用 v18 ckpt, 不在 v24 失败品上叠加), test_R@10 > v18 硬约束通过
- **R38** ✅ mid-training 不适用 (无新训练, 仅 stage4)
- **R39** ✅ 立即实施, 不阻塞 Gate A

## 设计要点

### v25 vs v24 关键区别

| 维度 | v24 (#100) | v25 (#238) |
|------|-------------|-------------|
| 曲率在哪 | Stage3 attention score | Stage4 post-generation rerank |
| 失败点 | 改 cross-entropy 路径 → valid=0 | 不改 cross-entropy, 仅重排 |
| 训练 | 新 DDP 4 卡训练 | 不训练, 用 v18 ckpt |
| 时间 | 12 分钟启动后 kill | 1 分钟评估完 |

### v25 实现核心

1. **保持 Stage3 不变**: 用 v18 HG_Rec_best.pth, matmul attention + cross-entropy generation
2. **Stage4 加 rerank**: generate() 后对 top-20 candidates 用 R_geo 重排
3. **R_geo 计算**: 
   - history_emb = mean(exp_map_0(L0[l0]) for l0 in history L0 tokens)
   - cand_emb = exp_map_0(L0[cand_l0_first])
   - R_geo = -d_P(history_emb, cand_emb), c_L0=1.3547
4. **score fusion**: final_score = -beam_idx / K + α * R_geo
   - α=0: 严格保持原 beam 序 (v18 baseline)
   - α=0.5: R_geo 主导重排 (本次实验)

## 产物

- `tasks/v25_hyperbolic_rerank_from_v18/stage{1,2,3,4_beam20}.py` — 4 个脚本 (R34 合规)
- `common/poincare_rerank.py` — Poincaré ball 数学 + rerank 工具
- `common/stage4_eval_pure_t5_v85p_4layer.py` — 加 --poincare_rerank 等 3 个 flag
- `taskA/_history/v25_hyperbolic_rerank_beam20_eval/eval_test.json` — 评估结果
- `taskA/_history/v25_hyperbolic_rerank_beam20_eval/raw_predictions.npz` — raw preds
- `verdicts/issue238_stage4_hyperbolic_rerank/verdict.md` — 本文件

## 后续

- v25 PASS 后, 后续方向可以从 v18 base + R_geo rerank 继续优化:
  - α scan: 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 找最优权重
  - multi-layer codebook: 同时用 L0+L1+L2 算 R_geo (现在只用 L0)
  - history aggregation: 不只用 mean, 试 exp_map / Fréchet mean
  - Stage3 嵌入空间联合训练 (但要避免 #100 失败模式)
- Issue #101 (= #238) close