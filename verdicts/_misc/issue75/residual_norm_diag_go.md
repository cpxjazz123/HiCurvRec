# Issue #75 (Issue A) 三层残差范数 ‖r_ℓ‖ 层间异质性诊断 — Verdict: **GO**

| Gate | 状态 | 关键数据 / 失败原因 |
|------|------|-------------------|
| **Gate 1 (Stage 2 Vanilla-RQ)** | **PASS** | 50 epoch 训练完成, util_3digit 末 epoch [1.000, 0.984, 0.953], 5922 items 唯一码占比 0.949 |
| **Gate 2 (‖r_ℓ‖ 收集)** | **PASS** | 6 audit epoch (ep0/10/20/30/40/49) + 末 epoch 全量 npz 落盘 (`residual_norms_final.npz` 1390054 B, N=9922) |
| **Gate 3 (KS 检验)** | **PASS** | L0 vs L1: stat=0.9521, p≈0; L1 vs L2: stat=0.4884, p≈0; L0 vs L2: stat=0.9868, p≈0 — 全部 sig@0.001 |
| **Gate 4 (Table + Figure)** | **PASS** | `verdicts_data/issue75_residual_norm_table.json` (3813 B), `issue75_residual_norm_histogram.png` 出图 |

## R18 4 维度对比

| 维度 | Issue #75 (本次) | HG-Rec §4 (论文) | Issue #58 (Stage1+Stage2 径向保留) |
|------|------------------|------------------|-----------------------------------|
| **D1 spec 摘录** | Vanilla-RQ 欧氏基线 + 每 epoch 收集 ‖r_ℓ‖ + KS 检验 | 无 ‖r_ℓ‖ 分布诊断 | 改 LorenzResidualHead 不 F.normalize |
| **D2 实施核心** | `--vanilla_rq` argparse → `poincare_recon_loss → mse_loss`, 每 epoch 末 `train_mm.encoder(item_emb)` + 逐层 `residual = residual - x_q`, 收集 `‖residual‖` mean/std/分位数 + 末 epoch npz | 论文无 Stage 2 残差分布数据 | Stage1 unit ball + Stage2 expmap0 强制 norm<1 |
| **D3 Gate 1 失败机制** | **无失败** (util 健康, 训练收敛, rec_loss 31.5→30.4) | N/A (论文未做此诊断) | Stage1 #56 残差头 + Stage2 RQ-VAE 架构不兼容, util 0.36/0.17/0.16 |
| **D4 引用文献** | HG-Rec §4 ablation, Poincaré Rec (NeurIPS 2022) §4 | 原论文无残差分布数据 | 参 Issue #56-#60 同源 |

## 关键发现 (Vanilla-RQ 欧氏基线, N_ITEMS=9922)

### 每层 ‖r_ℓ‖ 五数概括

| 层 | mean | std | q05 | q50 | q95 | max | ratio vs L0 |
|----|------|-----|-----|-----|-----|-----|------|
| **L0** (K=64) | **0.288** | 0.057 | 0.196 | 0.287 | 0.382 | 0.473 | 1.00 |
| **L1** (K=128) | **0.118** | 0.027 | 0.078 | 0.116 | 0.166 | 0.249 | **0.41** |
| **L2** (K=256) | **0.089** | 0.020 | 0.060 | 0.086 | 0.126 | 0.191 | **0.31** |

### KS 检验 (scipy.stats.ks_2samp)

| Pair | ks_statistic | p_value | sig@0.001 |
|------|--------------|---------|-----------|
| L0 vs L1 | 0.9521 | 0.0 | ✅ |
| L1 vs L2 | 0.4884 | 0.0 | ✅ |
| L0 vs L2 | 0.9868 | 0.0 | ✅ |

KS statistic 越大 = 两分布累计分布函数差异越大. L0 vs L2 的 0.987 接近 1.0, 几乎不重叠; L1 vs L2 的 0.488 是中等差距 (L1/L2 几何粒度差异小于 L0/L2).

### 结论

1. **三层 ‖r_ℓ‖ 分布统计显著不同**: L0 mean 0.288 > L1 mean 0.118 > L2 mean 0.089. 几何粒度逐层递增, 深层残差更小 (信息已被前层吸收).
2. **KS 检验全部 p≈0**: 即使 α=0.001 仍拒绝"同分布"原假设, 残差在统计分布层即非同质.
3. **§1.2 (1) 假设验证**: "各层残差在统计与语义粒度上均非同质" 在 Vanilla-RQ 欧氏基线下**完全成立**, 残差异质性不依赖几何先验 (即不依赖 Poincaré metric).

### 实务意义 (衔接 §1.2 (2)/(3))

- **几何先验剥离后仍异质** → 异质性是 Stage 2 RQ-VAE 量化链路的固有属性, 不是 Poincaré metric 引入
- **深层残差小** (L2 mean 0.089, ratio 0.31) → 深层可学空间小 → per-layer κ / 曲率半径应该有层级差异 (深层需要更小球 / 更高 κ)
- **为 §1.2 (2)/(3) 的"逐层不同 κ" 提供数据基础**: 既然残差层间异质, 编码空间也应层级适配

## 产物清单

```
taskA/_history/issue75_vanilla_rq/
├── hrqvae_kappa_sync.ckpt            (4736745 B, R12 强制保存)
├── residual_norms_final.npz          (1390054 B, N=9922 三层 + z_full)
├── residual_norm_history.json        (6 audit epoch × 3 层 × mean/std/分位数)
├── issue41_audit.json                (6 epoch κ audit)
├── sid_output.npy + sid_metadata.json
└── train_curve.json + verdict.json

verdicts_data/
├── issue75_residual_norm_table.json   (3813 B)
└── issue75_residual_norm_histogram.png
```

## 复现命令

```bash
# 训练 (单卡 ~5 min)
CUDA_VISIBLE_DEVICES=0 python3 -u taskA/stage2/taskA_stage2.py \
    --vanilla_rq --epochs 50 \
    --product_dir taskA/_history/issue75_vanilla_rq

# 分析
python3 common/analysis/issue75_residual_norm_diag.py
```

## Verdict: **GO**

- Gate 1-4 全 PASS
- §1.2 (1) 假设在 Vanilla-RQ 欧氏基线下验证
- 三层 ‖r_ℓ‖ 显著非同质 (KS p≈0)
- 为 §1.2 (2)/(3) per-layer κ 必要性提供 Stage 2 维度证据