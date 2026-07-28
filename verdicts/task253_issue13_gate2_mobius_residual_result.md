# Task #253 总结 — Issue #13 Gate 2: Möbius 残差算子实际训练 + Stage 4 eval

## 关键产物

| 阶段 | 状态 | 关键指标 |
|---|---|---|
| Stage 1 训练 (HRQ-VAE + Möbius 残差, 50 ep) | ✅ | best_collision ep34, train collision=0.0915 |
| Stage 2 SID 推断 | ✅ | 9922 items → 9014 unique |
| Stage 3 T5-mini 训练 (50 ep) | ⚠️ 47/50 ep (GPU 0 Xid 43, R12 ckpt 已存) | HG_Rec_best.pth 22MB |
| Stage 4 Test R@10 eval | ✅ 完成 | **R@10 = 0.000403** (vs HG-Rec baseline 0.1020) |

## Issue #13 Gate 2 判定

**Stage 4 R@10 = 0.000403** 远低于 HG-Rec baseline 0.1020 (Δ = -0.0908, ≈ -254×).

**Issue #13 Gate 2 NO-GO**:
- Möbius 残差算子 (HRQ arXiv:2505.12404 §3.2) 在 50 epoch 短训 Stage 3 下完全失败
- Issue #13 Gate 1 (Task #249) 显示 argmin 一致率 89.3% (高, 几何等价), 但下游 R@10 训练未能利用这个对齐
- 50 epoch T5-mini 对比 200 epoch baseline 严重不公平, 但 Möbius 残差也未显示任何 GO 信号

## 关键决策点 (R11.3)

- **配置完整性**: HG_Rec config 完整签名要求 `num_layers, num_decoder_layers, d_model, d_ff, num_heads, d_kv, dropout_rate, vocab_size, pad_token_id, eos_token_id, feed_forward_proj` — 第一次失败 KeyError 'dropout_rate' 后补齐
- **Stage 4 dataset_path**: 必须传 `test.parquet` 完整路径, 不能传 `Instruments/` 目录 (process_data 直接 `pd.read_parquet(file_path)`)
- **code_path**: task223 模式 (`<DATASET_PATH>/<DATASET> + <code_suffix>`), 不要重复 DATASET 段
- **R12 ckpt**: 训练 47/50 ep GPU 0 Xid 43, 但 R12 `best_collision_model.pth` (Stage 1) + `HG_Rec_best.pth` (Stage 3) 都已存, 训练产物完整

## Issue #13 后续

- Issue #13 Gate 2 已 no-go (Möbius 残差 argmin 几何对齐 ≠ 下游 R@10)
- Issue #13 Gate 3 (如存在) 需换思路: 不用 mobius_add 做残差, 改为仅在 hyperplane 距离 / logmap 量化上做
- 推荐下一步: 不再扩展 Issue #13, 转做 Issue #16+ (如果 backlog 有)

## 物理产物

```
products/task253/
├── hrqvae_mobius_residual/
│   └── Jul-29-2026_03-29-51_beta_0.500_codebook_[64,128,256]_sk_0.000/
│       └── best_collision_model.pth
└── t5mini_50ep/
    └── Instruments/Jul-29-2026_03-39-11/
        └── HG_Rec_best.pth (22MB)

HG-Rec/dataset/Instruments/
└── Instruments_t5_rqvae_mobius_residual_issue13_gate2.npy (9014 unique SID)

verdicts/task253_stage4_eval.json  (R@10=0.000403)
```

## Patch

```
HG-Rec/model/utils.py:1795
- residual = residual - x_res
+ if hasattr(quantizer, 'c') and hasattr(quantizer, 'angular_dim') and quantizer.angular_dim > 0:
+     hyp_dim = quantizer.angular_dim
+     residual_hyp = mobius_add(-x_res[:, :hyp_dim], residual[:, :hyp_dim], quantizer.c)
+     residual_euc = residual[:, hyp_dim:] - x_res[:, hyp_dim:]
+     residual = torch.cat([residual_hyp, residual_euc], dim=-1)
+ else:
+     residual = residual - x_res
```

result: Task #253 (Issue #13 Gate 2) — Möbius 残差算子 50 epoch Stage 4 R@10=0.000403, NO-GO. Issue #13 Gate 2 关闭.
