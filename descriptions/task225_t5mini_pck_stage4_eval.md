# Task #225 — Stage 4 test eval (Per-Codeword κ 完整流水线)

## 来源
- Task #222 verdict: healthy ckpt 锁定 (ep29, L0=20%, L1=98%, L2=91%)
- Task #223 verdict: Stage 2 SID 输出 6245 unique (62.94%)
- Task #224 (in_progress): T5-mini 200 epoch 训练
- **本任务**: 加载 Task #224 best_ckpt, 在 Instruments test set 上跑 R@5/10/20, NDCG@5/10/20

## 配置 (对齐 #170-#174 stage4 protocol)
- ckpt: products/task224/t5mini_pck/*/Instruments/*/HG_Rec_best.pth (R12 best ckpt)
- code_path: _t5_rqvae_task223_pck.npy (Task #223 SID)
- config dict: T5-mini 9.18M (d_model=256, 4 layers), codebook_size=[32,64,256,1], beam_size=20, infer_size=96

## 决策阈值
| 指标 | GO | NO-GO |
|------|-----|-------|
| **test R@10** | **> 0.1020** (HG-Rec baseline #84) | ≤ 0.1020 |
| test R@5 | > 0.0816 | ≤ 0.0816 |
| test R@20 | > 0.1279 | ≤ 0.1279 |
| test NDCG@10 | > 0.0755 | ≤ 0.0755 |

## 决策矩阵
- **🟢 GO (R@10 > 0.1020)**: Per-Codeword κ 逃法成功, paper 写 §X.Y 第一条 escape route, 进入下一探索
- **🟡 MARGINAL (0.0900-0.1020)**: Per-Codeword κ 改善但未超 baseline, 写"partial improvement + 改进方向"
- **🔴 NO-GO (R@10 ≤ 0.0900)**: SID 健康但 T5 学不出来, 写"healthy SID ≠ T5 benefit"结论

## 产物
- verdicts/task225_pck_metrics.json (recalls + ndcgs + best_ckpt)
- verdicts/task225_pck_stage4_eval_result.md (verdict 模板)
- logs/task225/stage4_eval_${TS}.log

## 相关任务
- #222: Stage 1 训练 (上游)
- #223: Stage 2 SID 推断
- #224: Stage 3 T5-mini 训练 (前置)
- **本任务**: Stage 4 eval (下游)