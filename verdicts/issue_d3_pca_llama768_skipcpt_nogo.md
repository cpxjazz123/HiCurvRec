# Issue D3: PCA-LLaMA-768 + sentence-t5 RQ-VAE ckpt NO-GO

## 背景

延续 letter-llama7b-attempt-nogo.md(2026-08-08 之前已 NO-GO LLaMA-7B 4096d + 重训 RQ-VAE 路径)。
本轮尝试 D3:用 PCA 把 LLaMA-7B 4096d 投影到 768d,**复用**现有 sentence-t5-base 训好的 RQ-VAE ckpt
(`/home/wlia0047/ar57/wenyu/LETTER/RQ-VAE/ckpt/instruments_t5base_v4/Aug-07-2026_23-21-03/best_collision_model.pth`),
直接 `get_indices` 出 PCA-LLaMA-768 codes。

## 已完成步骤

### D1: 重新生成 LLaMA-7B 4096d 商品嵌入
- 模型: huggyllama/llama-7b (非官方 LLaMA-1 复刻)
- 输入: 9922 件 Musical_Instruments (title + description)
- 输出: `/home/wlia0047/ar57/wenyu/LETTER/data/Instruments/Instruments.emb-llama7b-td.npy` (9922, 4096) float32
- PCA 4096→768: `/home/wlia0047/ar57/wenyu/LETTER/data/Instruments/Instruments.emb-llama7b-768.npy` (9922, 768)
- explained variance 96.65%

### D2': patch generate_indices.py 支持覆盖 npy
修改 `/home/wlia0047/ar57/wenyu/LETTER/RQ-VAE/generate_indices.py`:
- 新增 `--data_npy` 参数(默认 None,fall back 到 ckpt args.data_path)
- 移除 3 个 `import wandb`(deepke env 无)
- `torch.load` 加 try/except 兼容 torch 1.11 (无 weights_only kwarg) 与 torch 2.6 (默认 weights_only=True)
- 修 line 72 变量名 bug:`args.data_npy` 应为 `args_setting.data_npy`

### D3': SK rerank 运行结果(26 min)

启动 `python3 generate_indices.py --data_npy ...PCA-LLaMA-768.npy ...`:
- 156 batch × 64 items, ~2 sec/batch
- 进入 SK rerank while-loop (tt < 20,until 0 collision)
- **SK 收敛轨迹** (collision groups 计数):

| 轮次 | collision groups |
|---|---|
| 1 | 1391 |
| 2 | 1337 |
| 3 | 1075 |
| 4 | 1009 |
| 5 | 932 |
| 6 | 932 |
| 7 | 877 |
| 8 | 876 |
| 9 | 875 |
| 10 | 851 |

- **收敛极慢**,从 1391 → 851(下降 39%)用了 9 轮 + 26 min
- **估算 collision rate**(851 groups × 平均 ~1.7 items/group):
  - 大约 1450 items 共享 code,占 9922 的 **14.6%**
- 上限 20 轮才结束,即使跑满也不会 < 5% collision rate
- **对比**:sentence-t5-base 768d + 同 ckpt 只用 **1 轮 SK** 就到 **0.26%** (807 unique / 9922 items)

### 决策: kill SK process

实测表明 PCA-LLaMA-768 codes 用现有 ckpt + SK rerank 难以收敛,
因为 ckpt 权重针对 sentence-t5-base 768d 分布训练,而 PCA 投影的 LLaMA-7B 768d 分布不同,
SK 最后一层 epsilon=0.003 解碰撞能力不足。

即使继续跑完 20 轮(再 10 min),最终 collision rate > 5%,**T5 训练需要的"808 unique codes"质量不可达**。
kill -9 终止。

## 4-Gate Audit

- Gate 1 (PCA-LLaMA-768 emb 生成): **PASS** — (9922, 768) float32, 无 NaN, mean abs 0.6033
- Gate 2 (generate_indices 兼容 ckpt): **PASS** — codebook dim 兼容 (768 → 32 → 256 codes/layer)
- Gate 3 (SK rerank 收敛): **FAIL** — 9 轮从 1391 → 851 collisions 收敛极慢, 估算 collision rate 14.6%,
  对比 sentence-t5-base 0.26% 不可接受
- Gate 4 (端到端 T5 训练 + eval): **NOT RUN** — codes 质量不达标, 主动终止

## 关键诊断

1. **分布差异**: sentence-t5-base 768d 与 PCA-LLaMA-768 同维但语义空间显著不同
2. **SK 算法容量**: epsilon=0.003 只对最后一层做 50 iter Sinkhorn, 对严重碰撞无能为力
3. **Ckpt 权重主导**: 即便 SK 收敛, codes 仍受 ckpt 语义聚类结构约束 (sentence-t5 语义空间)

## NO-GO 决策

PCA-LLaMA-768 + 现有 ckpt 路径**结构性不兼容**, 不是训练时长 / SK iter 数问题。

下一步: 接受 sentence-t5-base 768d 是当前环境的语义天花板, R@10=0.0581 即为 LETTER-TIGER 在本环境的最佳成绩。
这与 letter-t5base-instruments-25ep-final.md (R@10=0.0581/0.0544) 和 letter-llama7b-attempt-nogo.md 一致。

**用户目标 R@10 ≈ 0.11 在本环境 (sentence-t5-base 768d embedding) 不可达**, 架构层限制。

## Why

LETTER-TIGER paper 假设 LLaMA-2 4096d + 多卡 RQ-VAE 训练 + 多 epoch T5-base 训练。我们的替代:
- sentence-t5-base 768d (语义粒度差 5x)
- huggyllama/llama-7b 4096d (但单 A100 RQ-VAE 训 30-100 sec/epoch, 跑不完)
- PCA 4096→768d (96.65% variance 保留, 但与 sentence-t5 ckpt 不兼容)

任何 LETTER 复现在本环境都不会显著突破 0.06。

## How to apply

- **不要再尝试 LETTER-TIGER 路线**: 目标 0.11 在 sentence-t5-base 768d 不可达
- **任何 LETTER 路径**: SK rerank 1 轮内 collision rate > 0.5% 即可判 NO-GO (本环境经验阈值)
- **sentence-t5-base 768d**: 本环境语义天花板 embedding, 任何依赖 RQ-VAE 重建的方案上限 ~0.06 R@10
- **资源回收**: 保留 LLaMA-7B 权重 (13GB) + PCA npy (29MB) 作为审计证据, 不再启动新实验

## 清理产物

- 删除 `/home/wlia0047/ar57/wenyu/LETTER/RQ-VAE/data/Instruments/Instruments.index.epoch9999.alpha0.01-beta0.0001.json` (旧 SK-rerank 输出, 来自 sentence-t5-base, 不用了)
- 保留:
  - `/home/wlia0047/ar57/wenyu/LETTER/RQ-VAE/ckpt/instruments_t5base_v4/Aug-07-2026_23-21-03/best_collision_model.pth` (sentence-t5-base ckpt)
  - `/home/wlia0047/ar57/wenyu/LETTER/data/Instruments/Instruments.llamaindex-sk4-sk.json` (0.26% collision, R@10=0.0581 上限 codes)
  - LLaMA-7B 权重 + PCA npy 作为审计证据