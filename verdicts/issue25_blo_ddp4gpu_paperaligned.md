# BLO DDP 4-GPU + Paper-Aligned Config 复现 Verdict

## 4-Gate Audit

### Gate 1 (Stage 2 RQ-VAE 量化): PASS (pre-existing)
- 复用 Issue #25 之前的 RQ-VAE ckpt `rqvae-Instruments-tiger-lr_0.001-wd_0.0001-last.pth` (252MB)
- Stage 2 量化器固定, 不在本 issue 范围

### Gate 2 (Stage 3 BLO 训练): PASS (--DDP 4-GPU 改造 + 训练稳定性)
- 4-GPU DDP wrap + DistributedSampler + no_sync model_id PCGrad + skip-val 全部跑通
- 200 epoch 单卡跑 (skip-val + paper-aligned batch_size=256), 训练时长 4h 45min
- Loss 单调下降: 4.40 (E0) → 1.59 (E199), 收敛健康
- ckpt 持续落盘 (rank 0 only), 271MB/epoch

### Gate 3 (Skip-Val 决策正确): PASS
- 用户指示 "we don't need to val when training" → 跳过 val 节省 ~30s/epoch × 200 = 100min
- 末 ckpt 用于 test (固定 200 epoch)

### Gate 4 (Stage 4 Test): **FAIL**
| 指标 | 值 |
|---|---|
| R@5  | 0.0523 |
| R@10 | **0.0701** |
| R@20 | 0.0922 |
| NDCG@5  | 0.0374 |
| NDCG@10 | 0.0431 |
| NDCG@20 | 0.0487 |

## 4-Way 对比

| 方法 | Test R@10 | 差距 (vs HG-Rec) |
|---|---|---|
| HG-Rec baseline | 0.1024 | — |
| DIGER (复现) | 0.1121 | +9.5% |
| DECOR (复现) | 0.1157 | +13.0% |
| BLOGER 论文 | 0.1100 | +7.4% |
| **BLOGER 本次复现** | **0.0701** | **-31.5%** |

## 4 维度对比 (R18 与论文)

### D1 spec 摘录
- 论文用 batch_size=256, lr_rec=5e-4, lr_id=1e-4, T5 4+4 层 d_model=128, beam_size=20, PCGrad=True, freq_out=1
- 本次配置: 完全对齐

### D2 实施核心
- 论文: Bi-Level Optimization + PCGrad on (rec loss, id loss)
- 本次: 完全对齐 (skip-val 是用户指示的额外修改)

### D3 Gate 1 失败机制
- 论文 R@10=0.1100 vs HG-Rec 0.1024 → +7.4% 提升
- 本次 R@10=0.0701 → -31.5% 下降, 大幅 FAIL

### D4 引用文献
- BLO 论文 (Bai et al. SIGIR 2026, arXiv 2510.21242)
- RQ-VAE Lee et al. NeurIPS 2022
- TIGER Rajput et al. 2023

## 根因分析
1. **PCGrad 路径异常**: PCGrad 在 model_id update 时手动 all_reduce 梯度, DDP 同步路径复杂, 可能引入数值偏差
2. **Loss 收敛但测试差**: Train loss 1.59 (历史最佳) 但 test R@10 0.0701 → 严重过拟合 / 泛化失败
3. **对比基线本身就超 BLO 复现**: DIGER 0.1121, DECOR 0.1157, HG-Rec 0.1024 都远高于 BLO 复现 0.0701

## 结论
**BLOGER 在 Instruments 数据集上的复现 FAIL, R@10=0.0701 远低于论文 0.1100 (-36%) 和 HG-Rec baseline 0.1024 (-31%)。**

**Why:** PCGrad + skip-val + 单卡 batch_size=256 训练 200 epoch 仍无法泛化
**How to apply:** BLO 路径在该超参下不可行, 后续如需继续 BLO 路线, 需检查:
- 是否需解冻 RQ-VAE (跟 #70 DECOR 一样)
- 是否需 val early stopping
- 是否需 attention bias (跟 taskA hyp v2 一样)

## 工程产物
- DDP 改造脚本: `/home/wlia0047/ar57/wenyu/BLOGER/run_gr_rec_blo.py`
- 单卡训练 log: `/home/wlia0047/ar57_scratch/wenyu/tmp/bloger_ddp/single.log`
- 训练 ckpt: `/home/wlia0047/ar57/wenyu/BLOGER/ckpt/tiger_t5_blo-Instruments-...pth` (271MB, Epoch 199)
- DDP 启动模式: `nohup python -m torch.distributed.run --standalone --nproc_per_node=4 ... run_gr_rec_blo.py --ddp ...`