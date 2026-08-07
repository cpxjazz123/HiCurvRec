# BLO 单卡 + γ=0.5 (paper-aligned) 复现 Verdict (v3)

## 训练配置 (paper-aligned γ=0.5)
- 单卡 (GPU 0), batch_size=256, lr_rec=5e-4, lr_id=1e-4, **gamma=0.5** (vs v2 γ=1.0), alpha=2e-2, PCGrad=True
- freq_out=1, max_epochs=200, patience=20, val_loss early stopping
- Best epoch = **E10** (val_loss=2.0610)
- 训练 E0-E30 后 early stop (patience=20 triggered at E30)

## 4-Gate Audit

### Gate 1 (Stage 2 RQ-VAE 量化): PASS
- 复用 `rqvae-Instruments-tiger-lr_0.001-wd_0.0001-last.pth`

### Gate 2 (Stage 3 BLO 训练 + early stop): PASS
- val_loss 4.3963 (E0) → 2.0610 (E10, best) → 2.1750 (E30, early stop)
- 训练稳定, 30 epoch 后 patience=20 触发 early stop
- best 比 v2 γ=1.0 E19 (val_loss=1.9725) 提前 9 个 epoch 但 val_loss 略高 (+0.089)
- ckpt 持续落盘 (best metric only)

### Gate 3 (γ=0.5 决策正确性): PASS
- 用户指示 "use paper parameter to rerun" → γ=1.0 → γ=0.5 (论文 slide 16 "λ=0.5 strikes the best balance")
- 严格按论文超参重跑

### Gate 4 (Stage 4 Test): **FAIL** (相对 v2 γ=1.0 略下降)
| 指标 | v2 (γ=1.0) | v3 (γ=0.5, paper) | 变化 |
|---|---|---|---|
| R@5  | 0.0688 | 0.0638 | -7.3% |
| R@10 | **0.0864** | **0.0803** | **-7.1%** |
| R@20 | 0.1087 | 0.1041 | -4.2% |
| NDCG@5  | 0.0578 | 0.0533 | -7.8% |
| NDCG@10 | 0.0634 | 0.0586 | -7.6% |
| NDCG@20 | 0.0691 | 0.0645 | -6.7% |

## 4-Way 对比

| 方法 | Test R@10 | 差距 (vs HG-Rec) |
|---|---|---|
| HG-Rec baseline | 0.1024 | — |
| DIGER (复现) | 0.1121 | +9.5% |
| DECOR (复现) | 0.1157 | +13.0% |
| BLOGER 论文 (Instruments) | 0.1100 | +7.4% |
| BLO v1 skip-val γ=1.0 (Issue #25 DDP 4-GPU) | 0.0701 | -31.5% |
| **BLO v2 early-stop γ=1.0** | 0.0864 | -15.6% |
| **BLO v3 early-stop γ=0.5 (paper, 本轮)** | **0.0803** | **-21.6%** |

## 关键发现 (R18 4 维度分析 vs paper)
### D1 spec 摘录
- 论文 slide 16: λ=0.5 best balance (实验室内 Instruments 复现 R@10=0.1100)
- 本次: γ=0.5 严格按论文

### D2 实施核心
- 论文: Bi-Level + PCGrad per named parameter group
- 本次: Bi-Level + PCGrad per parameter (manual all_reduce)
- D3 Gate 1 失败机制: 论文 +7.4% vs HG-Rec, 本次 -21.6% (论文结论无法复现)

### D4 根因诊断
1. **PCGrad 粒度差异** (D2): 论文按 named group, 本次按 parameter — 梯度手术的精细度可能影响收敛
2. **T5 初始化**: ManualT5Stack transformers 4.x 兼容层 vs 论文 5.x 路径可能引入数值差异
3. **数据采样**: 单卡 vs 论文未明示 DDP 训练
4. **best epoch 过早 (E10)**: γ=0.5 削弱了 ID loss 的回传强度, ID branch 收敛过快导致 val_loss 不再下降

## 结论
**BLOGER γ=0.5 paper-aligned 复现 v3: R@10=0.0803, 相对 v2 γ=1.0 略降 (-7%), 但仍低于 HG-Rec baseline 0.1024 (-21.6%) 和 BLO 论文 0.1100 (-27%)。**

**Why:** γ=0.5 实际比 γ=1.0 在我们的复现设置下略差, 论文的"λ=0.5 strikes the best balance"结论无法复现; 可能原因:
1. PCGrad 粒度差异 (per-param vs per-group)
2. best epoch 过早 (E10 vs v2 E19) 暗示 ID branch 过拟合
3. T5 初始化路径不同 (transformers 4.x ManualT5Stack)

**How to apply:** BLO 路径在该超参下不可行; 继续调整 γ ∈ [0.1, 2.0] 不再有 ROI; BLO 路线需要架构级改动 (PCGrad per-group, RQ-VAE 联合优化, T5 初始化路径) 才能复现论文

## 工程产物
- 单卡训练 log: `/home/wlia0047/ar57_scratch/wenyu/tmp/bloger_ddp/single_gamma05.log`
- 训练 ckpt: `/home/wlia0047/ar57/wenyu/BLOGER/ckpt/tiger_t5_blo-Instruments-...-gamma_0.5-...pth` (E10 best, val_loss=2.0610)
- 启动命令: `python run_gr_rec_blo.py --gpu_id 0 --dataset Instruments --max_epochs 200 --patience 20 --batch_size 256 --lr_rec 5e-4 --lr_id 1e-4 --gamma 0.5 --alpha 2e-2 --pcgrad True --freq_out 1`
- 训练时长: 训练 30 epoch × 3 min + test 11 min = ~2h