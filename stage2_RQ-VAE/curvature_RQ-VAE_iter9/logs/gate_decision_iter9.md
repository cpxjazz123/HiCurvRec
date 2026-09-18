# iter9 Gate Decision: NO-GO

## Mechanism

4-layer RQ-VAE (N_LAYERS=4, CODEBOOK_SIZE=256, MIDPOINT_LAYER_MASK=[F,F,F,F])

## Stage 2 SID Quality Gate (R50 4 metrics)

Source: step10000 partial ckpt (rank 0 NFS hang at ~step10000, no full ckpt available)

| Metric | iter9 step10000 | TIGER baseline | Status |
|--------|-----------------|----------------|--------|
| shape | (24587, 4) | (24587, 3) | — iter9 是 4 层 |
| 4-token unique | 24267/24587 (98.7%) | — | — 4-token |
| L0 unique | 256/256 (100%) | — | PASS |
| L1 unique | 256/256 (100%) | — | PASS |
| L2 unique | 256/256 (100%) | — | PASS |
| L3 unique | 256/256 (100%) | — | PASS |
| 3-token unique | 23574/24587 (95.9%) | — | — iter9 是 4 层 |
| full_gini (3-token) | 0.0553 | 0.0672 | -17.7% PASS |
| per_layer mean Gini (3-token) | 0.227 | 0.257 | -11.7% PASS |

**Gate 1 (SID quality) PASS**: 100% utility on every layer, full_gini -17.7%, per_layer mean -11.7%.

Note: 由于 step 10000 rank 0 hang, 完整 50k-step ckpt 不可用; 我们用 step 10000 partial ckpt 做 gate.

## Stage 3 R37 (downstream test_R@10)

- valid_ndcg@10: 0.02897
- valid_recall@10: 0.053152 (epoch 200)
- **test_recall@10: 0.04732** vs iter8 baseline 0.0552 → **-14.3% REGRESS**

**Gate 3 (R37) FAIL**.

## Stage 3 R37 Ceiling Status

iter9 是连续第 **9** 个 stage2 mechanism 在 Stage 3 T5 ceiling 上 lock (~0.055):
v282 (0.1136) → v316_fixed (0.1161) → v317 (0.1176) → v318 (0.1179) → v337 (0.2740, Instruments 2018) → v339 → v340 → iter6 (R37 FAIL) → iter7 (R50 FAIL) → iter8 (0.0552) → iter9 (0.0473)

**Stage 3 T5 ceiling 在 Amazon-2023 Instruments dataset 上锁定 ~0.055 已超过 9 个连续 stage2 mechanism**.

## Root Cause Analysis

- iter9 训练仅 10000 steps (step budget 仅 20% 用尽), step 10000 时 loss/vq loss 仍单调下降中, ckpt 是 partial
- 4-token SID 输出 shape 正确, 100% utility per layer
- 但 stage 3 valid R@10 (0.0532) 与 test R@10 (0.0473) 比 iter8 (test 0.0552) 显著退化
- 可能的根因: (a) 训练不充分 step 10000 → codebook 还没学到 4 层语义; (b) Stage 3 T5 表征对 stage1 端细节透明, 5-token (vs 4-token) 增加 position decoding 难度而无新信息

## Outcome

**NO-GO**: iter9 not promoted. iter5/iter8 仍为当前 promoted baseline (test_R@10=0.0552).

## Cleanup

iter9 directory 即将按 skill cleanup rule 删除 (保留 logs/ 在 git).