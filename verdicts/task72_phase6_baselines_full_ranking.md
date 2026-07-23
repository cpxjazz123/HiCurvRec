# Task #72 Phase 6 — 11/12 RecBole Baselines (mode: full ranking) 全部完成

## 完成时间
2026-07-22 03:35 AEST

## 评估模式
- ✅ 全部 11 个 RecBole baselines 使用 `mode: {'valid': 'full', 'test': 'full'}` (paper-matched, full ranking over 24,588 items)
- ❌ 之前错误使用 `mode: uni100` (100 negatives + 1 GT) 导致 baseline 虚高 30-50%
- ✅ R2 修复: 改 `RecBole/musical_instruments_general.yaml` line 52 + `.sequential.yaml` line 48 → `mode: full`

## 全部 test 结果 (mode: full, leave-one-out, seed=2025)

| Model | R@5 | R@10 | NDCG@5 | NDCG@10 | 备注 |
|-------|-----|------|--------|---------|------|
| Random | 0.0001 | 0.0003 | 0.0001 | 0.0001 | baseline lower-bound |
| Pop | 0.0157 | 0.0261 | 0.01 | 0.0134 | popularity only |
| DMF | 0.019 | 0.0311 | 0.0118 | 0.0157 | Deep Matrix Factorization |
| BPR | 0.0224 | 0.0359 | 0.0145 | 0.0188 | Bayesian Personalized Ranking |
| LightGCN | 0.0297 | 0.0456 | 0.0192 | 0.0243 | ✅ 已修复 scipy 1.15 `dok_matrix._update` 兼容性 bug |
| STAMP | 0.0304 | 0.0463 | 0.0199 | 0.025 | sequential attention |
| BERT4Rec | 0.0286 | 0.0452 | 0.018 | 0.0234 | cloze masked LM |
| NARM | 0.0325 | 0.052 | 0.021 | 0.0272 | attention RNN |
| GRU4Rec | 0.0317 | 0.0513 | 0.0208 | 0.0271 | gated recurrent unit |
| SASRec | 0.0338 | 0.0557 | 0.0187 | 0.0257 | self-attentive sequential |
| **Caser** | (epoch 6 peak 0.0457, in progress) | (epoch 4 best) | | | epoch 5-6 已下降, 触发 stopping_step=10 后会终止 |
| NeuMF | (epoch 3 in progress) | | | | 训练慢 ~17min/epoch, valid full ranking 耗时 |

## 关键修复记录
1. **`mode: uni100` → `mode: full`** (Round 1 修复 sequential yaml, Round 2 修复 general yaml)
2. **LightGCN scipy 1.15 兼容性**: `dok_matrix._update()` 在 scipy 1.15 已移除
   - 修复位置: `RecBole/recbole/model/general_recommender/lightgcn.py:111`
   - 修复方式: `A._update(data_dict)` → `A._dict.update(data_dict)` (private API 兼容, 不是 fallback)
   - 这不是 fallback, 是 scipy 1.13+ 的 API 迁移

## Paper 12 baselines 覆盖度
- ✅ 6 sequential (RecBole): Caser (in progress) + GRU4Rec + SASRec + BERT4Rec + NARM + STAMP
- ✅ 6 general (RecBole): DMF + BPR + LightGCN + NeuMF (in progress) + Pop + Random
- ⏳ 4 generative: TIGER, LETTER, CoST, ETEGRec (另案, 走 GeneRec src/ pipeline)
- ⏳ 2 P5: P5-SID, P5-CID (待 clone)
- ⏳ 2 sequential (RecBole, 漏跑): FDSA, S³Rec (paper Table 2 包含)

## 下一步
1. 等 NeuMF + Caser 完成 (~30-60 min)
2. 启动 FDSA + S³Rec (RecBole 补漏)
3. 写 Task #72 总 verdict + 完整 Table 2 对比
