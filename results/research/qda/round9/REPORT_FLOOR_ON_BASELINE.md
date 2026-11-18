# 防坍缩层在**冻结基线**上的实测:换一套量化器、换一套机制,保证照样成立

第 8 轮的实测都在 F0 派生树(E16×H16 混合量化器、α=1/2)。本轮把同一层装进**冻结基线** `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` —— E32 纯欧氏、单码本形态、L3 原本走 Sinkhorn 的另一套实现 —— 用来检验"对任何机制都成立"这句话。

## 做法

`build_baseline_floor.py`:

1. 复制冻结基线到 `trees/B0_floor_euclid`(不修改基线本体);
2. 结果目录改到 `round9/runs/`,预算改为 100 epoch(坍缩窗口在 epoch 50 前结束,100 足够);
3. 每 epoch 记录三级的活跃码数(`occupancy` 事件,用 `get_indices`,即导出规则);
4. 调用 `python -m anticollapse` 的库函数打补丁(自动识别 `euclid` 家族),并在全新解释器里验证;
5. 修掉一个**在新树上才暴露**的缺陷:基线的 `_repository_root()` 只查 `stage2_RQ-VAE` 一个标记,而 `results/` 里也有这个目录,于是根被解析到 `results/` —— 与第 1 轮在混合树上遇到的完全同一个问题。现在要求同时出现 `stage1_GeneEmbedding/output/sentence_t5.npy`。

## 结果

| 项目 | 值 |
|---|---|
| 训练 epoch | 100 / 100 |
| 每 epoch 活跃码(三级最小值) | **256 / 256 / 256** |
| 首个 epoch | 256 / 256 / 256 |
| 末个 epoch | 256 / 256 / 256 |
| 100 epoch 内出现过的最小值 | **256 / 256 / 256** |
| 导出 SID raw 三元组唯一数 / collision | 22,144 / 24,587 → **0.0994** |

作为对照,**未打补丁**的冻结基线已发布 SID(`results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json`,3000 epoch)是 20,425 / 24,587 → **0.1693**。两者预算不同(100 vs 3000 epoch),所以这只是**上下文**,不是受控比较;可引用的结论是"打补丁后 100 个 epoch 里三级各 256 个码字,一个都没死"。

## 这次证明了什么、没证明什么

**证明**:这一层不依赖 F0 的混合几何、不依赖它的 α 表、不依赖 Sinkhorn。换一套量化器实现、换一套损失与残差运算,同一个补丁(同一条命令)照样给出"每一 epoch、每一级 256/256"。

**没证明**:
- 不提高最终 SID 的行为保真 —— 第 8 轮在 F0 上实测 L3 反而低 0.39–0.54 pp;本轮没有对应的行为标注(基线树没有 QDA 观察器),因此不给结论。
- 不保证"码字有质量" —— 退化编码器仍可满足下限,必须由 `round7` 的认证闸门拒绝。
- 本轮预算 100 epoch,不是 3000;保证是逐步成立的(每一步的分配都满足下限),与 epoch 数无关,但 SID 质量随预算变化。

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round9/build_baseline_floor.py
cd results/research/qda/round9/trees/B0_floor_euclid && <genrec_env_v2>/python3.9 train_rqvae.py
```

产物:`build_baseline_floor.py`、`trees/B0_floor_euclid/`(基线副本 + 三处框架补丁 + 防坍缩补丁)、`runs/B0_floor_euclid/{occupancy.json,item_sids.json}`。
