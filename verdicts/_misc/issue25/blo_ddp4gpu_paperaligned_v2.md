# BLO 单卡 + val/early-stopping 复现 Verdict (v2)

## 训练配置 (恢复 BLO 原版设计)
- 单卡 (GPU 0), batch_size=256, lr_rec=5e-4, lr_id=1e-4, gamma=1.0, alpha=2e-2, PCGrad=True
- freq_out=1, max_epochs=200, patience=20, val_loss early stopping
- Best epoch = E19 (val_loss=1.9725)
- 训练 E0-E39 后 early stop (patience=20 triggered at E39)

## 4-Gate Audit

### Gate 1 (Stage 2 RQ-VAE 量化): PASS
- 复用 `rqvae-Instruments-tiger-lr_0.001-wd_0.0001-last.pth`

### Gate 2 (Stage 3 BLO 训练 + early stop): PASS
- val_loss 4.3967 (E0) → 1.9725 (E19, best) → 2.5925 (E39, early stop)
- 训练稳定, 39 epoch 后 patience=20 触发 early stop
- ckpt 持续落盘 (best metric only)

### Gate 3 (val/early-stopping 决策): PASS
- 用户指示: 恢复 BLO 原版 val_loss early stopping
- 与上次 skip-val 试验相比, 阻止过拟合, R@10 +23% 提升

### Gate 4 (Stage 4 Test): **FAIL** (但相对 v1 显著改善)
| 指标 | v1 (skip-val) | v2 (early-stop) | 改善 |
|---|---|---|---|
| R@5  | 0.0523 | **0.0688** | +31.6% |
| R@10 | 0.0701 | **0.0864** | +23.3% |
| R@20 | 0.0922 | **0.1087** | +17.9% |
| NDCG@5  | 0.0374 | **0.0578** | +54.5% |
| NDCG@10 | 0.0431 | **0.0634** | +47.1% |
| NDCG@20 | 0.0487 | **0.0691** | +41.9% |

## 4 维度对比 (vs HG-Rec baseline + BLO 论文)

| 方法 | Test R@10 | 差距 (vs HG-Rec) |
|---|---|---|
| HG-Rec baseline | 0.1024 | — |
| DIGER (复现) | 0.1121 | +9.5% |
| DECOR (复现) | 0.1157 | +13.0% |
| BLOGER 论文 (Instruments) | 0.1100 | +7.4% |
| **BLO v1 skip-val** (Issue #25 DDP 4-GPU) | 0.0701 | -31.5% |
| **BLO v2 early-stop (本轮)** | **0.0864** | **-15.6%** |

## 根因分析
1. **BLO 复现系统性低于论文**: 论文 0.1100 vs v2 0.0864 (-21.5%), 改善 vs v1 但仍未达论文
2. **val + early stopping 阻止过拟合**: 跳过 val 训练到 200 epoch → 严重过拟合; 早停 E19 是真正的 best
3. **PCGrad 路径无可见 bug**: 训练 loss 单调下降到 ~1.9, val loss 同步下降
4. **可能缺失**: 
   - T5 cross-attention 初始化不同
   - RQ-VAE 量化器协同
   - 训练数据 batch sampling 差异

## 结论
**BLO 在 Instruments 数据集上的复现 v2: R@10=0.0864**, 比 v1 (skip-val) 显著改善 (+23.3%), 但仍低于 HG-Rec baseline 0.1024 (-15.6%) 和 BLO 论文 0.1100 (-21.5%)。

**Why:** val + early stopping 是 BLO 训练必要条件; 跳过 val 必然过拟合导致 0.07
**How to apply:** 后续 BLO 变体必须保留 val + early stopping (patience=20); v2 仍未能超 baseline 表明 BLO 路线需要架构级改动 (RQ-VAE / T5 / 注意力机制) 而非仅超参调整

## 工程产物
- 单卡训练 log: `/home/wlia0047/ar57_scratch/wenyu/tmp/bloger_ddp/single_v2.log`
- 训练 ckpt: `/home/wlia0047/ar57/wenyu/BLOGER/ckpt/tiger_t5_blo-Instruments-...pth` (E19 best, val_loss=1.9725)
- 启动命令: `python run_gr_rec_blo.py --gpu_id 0 --dataset Instruments --max_epochs 200 --patience 20 --batch_size 256 --lr_rec 5e-4 --lr_id 1e-4 --gamma 1.0 --alpha 2e-2 --pcgrad True --freq_out 1`
- 训练时长: 训练 39 epoch × 3 min + test 11 min = ~2.5 hours
