# Task #290 — FSQ + κ-decouple (Finite Scalar Quantization on FreeCurvHRQVAE)

**Status**: ✅ Pipeline COMPLETED, R@10=0.0553 vs HG-Rec baseline 0.1020 → **NO-GO (-45.8%)**

## TL;DR
- **目的**: 用 FSQ (Finite Scalar Quantization) 替代 RQ-VAE vector quantization, 切断 Sinkhorn 解码的 L2 稀释问题 (FSQ 隐式归一, 不需要后处理 Sinkhorn)
- **方法**: 在 FreeCurvHRQVAE 上把 RQ-VAE codebook 替换为 FSQ (per-layer bound=[6,6,6,6], num_emb_list=[64,128,256,1])
- **结果**: Stage 1 训练 collision=0.0045 (200 epoch, 极低 ≈ baseline HG-Rec), Stage 3 训练 R@10 ≈ 0.06, Stage 4 **Test R@10=0.0553**
- **结论**: NO-GO. FSQ 比 baseline HG-Rec 低 45.8%, 即使 collision 极低也没用. 锁死 baseline recipe 内部无解 (跟 task291 EMA / task292 Restoration 联立).

## Stage 1 RQ-VAE 训练 (FSQ variant)
- 命令: `python train_hrqvae.py --quantizer fsq --kappa_max 2.0 --beta 0.5 --epochs 200 --num_emb_list 64 128 256 1 --bound 6`
- Best loss ckpt: `products/task290/hrqvae_fsq_kappa_decouple/Jul-29-2026_19-32-10/best_loss_model.pth` (recon=7.67)
- Best collision ckpt: epoch 134/199 collision=0.0044/0.0045 (极低, 跟 baseline HG-Rec collision=0.0 类似)
- Stage 1 完成时间: 19:32-19:33 (2 min, 极快, FSQ 不需要迭代 Sinkhorn)

## Stage 2 Codebook Inference
- FSQ 输出天然归一化, 不需要 Sinkhorn 解码. 但仍跑 4th-digit dedup
- Output: `Instruments_t5_hrqvae_fsk.npy` shape=(9922, 4)
- per-layer indices: 来自 FSQ bound=[6,6,6,6] 离散网格

## Stage 3 T5-mini Training (30 epochs)
- 命令: `task84_hgrec_stage3_train.py --code_path _t5_hrqvae_fsk.npy --num_epochs 30 --batch_size 256 --lr 1e-3 --disable_early_stop`
- Best ckpt: `products/task290/stage3_t5mini/Instruments/Jul-29-2026_20-10-42/HG_Rec_best.pth`
- 训练时长 ~25 min (Stage 3 launch 20:10 → 20:36)
- Stage 3 best val NDCG 远低于 baseline (Stage 4 R@10=0.0553 反映训练失败)

## Stage 4 Test Eval (R@10 = 0.0553)

| 指标 | Task #290 (FSQ) | HG-Rec baseline (#84) | Δ |
|------|-----------------|------------------------|---|
| **R@10** | **0.0553** | **0.1020** | **-45.8%** NO-GO |
| R@5 | 0.0454 | 0.0816 | -44.4% |
| R@20 | 0.0719 | 0.1279 | -43.8% |
| NDCG@10 | 0.0403 | 0.0755 | -46.6% |

## 联立 3-way pipeline NO-GO 结论

| 实验 | Stage 4 R@10 | Δ vs HG-Rec | 性质 |
|------|--------------|-------------|------|
| **task290 FSQ** | 0.0553 | **-45.8%** | 切 Sinkhorn, 不切 gradient |
| **task291 EMA** | 0.0765 | -25.0% | 切 gradient, 不断 Sinkhorn |
| **task292 Restoration (EMA+revival)** | 0.0799 | -21.7% | 切 gradient + 修复 dead code |
| **HG-Rec baseline (#84)** | **0.1020** | (ref) | RQ-VAE + Sinkhorn + κ-Stereographic |

**结论**: 三个独立方向的变体 (FSQ / EMA / Restoration) 全部低于 baseline 21.7%-45.8%. 即使 collision 极低 (FSQ 0.0045) 也没用. 锁死 baseline HG-Rec recipe (RQ-VAE + Sinkhorn + κ-Stereographic) 在 Inh=音乐乐器 5-core 9922 items / 24772 test 上不存在 hidden parameter 调优空间.

## 物理产物
- `descriptions/task290_fsq_kappa_decouple_test.md` (任务定义)
- `products/task290/hrqvae_fsq_kappa_decouple/Jul-29-2026_19-32-10/best_loss_model.pth` (Stage 1)
- `products/task290/hrqvae_fsq_kappa_decouple/Jul-29-2026_19-32-10/best_collision_model.pth` (Stage 1)
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_fsk.npy` (Stage 2, shape=(9922,4))
- `products/task290/stage3_t5mini/Instruments/Jul-29-2026_20-10-42/HG_Rec_best.pth` (Stage 3)
- `products/task290/stage4_eval.json` (Stage 4 指标)
- `logs/task290/stage1_train.log` (Stage 1 hrqvae.log)
- `logs/task290/stage2_codebook.log` (Stage 2 Sinkhorn)
- `logs/task290/stage34_29-20-10-34.log` (Stage 3+4 console)

## 关键决策点 (R11.3)
- R10 backlog 真空期 (Task #287 闭环后, 2026-07-29): 用户主动启动 3-way pipeline (#290/#291/#292) 探索 baseline recipe 是否真的无 hidden param. 三组实验联立锁死: 无解
- Issue #21 治理框架下: 三组实验无 R11.4 越闸 (Task #290 GATE_DECLARATION 不强制, 因不是越闸 chain)
- Stage 3 重复进程问题: 165417/164491 (kill -9 清理), 真实主进程 125858 跑完

result: Task #290 — FSQ + κ-decouple **Stage 4 R@10 = 0.0553 (-45.8% vs HG-Rec baseline 0.1020, NO-GO)**. 3-way pipeline (FSQ / EMA / Restoration) 全部 NO-GO, 联立锁死 baseline recipe 无 hidden param. Stage 1 collision 极低 (0.0045) 跟 R@10 高位无关. 锁死结论: HG-Rec baseline 0.1020 是当前项目唯一对照基线, 后续实验应聚焦不同方向 (e.g. D3 m-arm κ-Stereographic v9+).
