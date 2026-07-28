# Task #224 — Stage 3 T5-mini 训练 (Per-Codeword κ healthy SID)

## 来源
- Task #223 verdict: 🟢 healthy SID 锁定 (L0=20%, L1=99%, L2=91%, 6245 unique)
- 用户 2026-07-26 提议: 投 Per-Codeword κ 路线, 验证下游 R@10 能否击败 baseline 0.1020

## 配置 (对齐 baseline #84)
| 项 | 值 |
|----|-----|
| T5 | T5-mini 9.18M (d_model=256, d_ff=1024, 4 layers) |
| dataset | Musical_Instruments |
| SID file | _t5_rqvae_task223_pck.npy (Task #223 Stage 2 output) |
| codebook_size | 32 64 256 1 (跟 baseline 一致) |
| num_epochs | 200 |
| batch_size | 256 |
| lr | 1e-4 |
| seed | 42 (R11.5 排除 multi-seed per user 2026-07-23) |
| early_stop | 20 (跟 baseline) |
| beam_size | 20 |
| infer_size | 96 |
| GPU | 0 |

## 启动
- 2026-07-26 23:13 启动 (Task #224 launcher PID 1386703)
- 预期 ~67 分钟完成 (26 it/s, 515 batch/epoch × 200 epoch)
- products/task224/t5mini_pck/jul-26-2026_23-13-16/
- logs/task224/stage3_train_jul-26-2026_23-13-16.log

## 决策阈值
| 指标 | GO | NO-GO |
|------|-----|-------|
| test R@10 | **> 0.1020** (HG-Rec baseline #84) | ≤ 0.1020 |
| test R@5 | > 0.0816 | ≤ 0.0816 |
| test R@20 | > 0.1279 | ≤ 0.1279 |
| test NDCG@10 | > 0.0755 | ≤ 0.0755 |

## 下游
- Task #225 Stage 4 test eval (load best_ckpt from #224)

## 相关任务
- #223: Stage 2 SID 推断 (上游)
- #225 (待登记): Stage 4 test eval (下游)