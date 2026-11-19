# 轮转区间在真实训练器里跑通(K > batch)

上一轮只做了单元验证(8 商品 × 64 码字,8 步覆盖 64/64)和"大 K 下检查可行"的测量。这轮把它放进训练器。

## 设置

冻结配方 batch 是 4096,要触发 `K > batch` 需要 8192 码 —— 那要做 8192 簇的 KMeans,代价与本仓库任何配置的收益不成比例。**触发的是同一个 regime,不是同一个绝对规模**:诊断 batch 256 + 首层 512 码,10 epoch。

配置:`CODEBOOK_SIZE = [512, 256, 64]`,`BATCH_SIZE_PER_RANK = 256`(L1: 256 商品 < 512 码字 → 轮转)。

## 结果:跑通,而且该层码字全用上了

```
epoch=10  raw_unique=7007/24587  collision=0.715012
产物审计: PASS  levels 512/256/64
每级 distinct: [512, 256, 64]
```

- **零拒绝**:轮转路径、间距检查、重复检查在真实前向里都正常。
- **L1 的 512 个码字全部被用过** —— 这正是轮转的保证(⌈512/256⌉ = 2 步;实际跑了 960 步)。
- 碰撞 0.715 是**10 epoch、batch 256 的欠训练**结果,不是质量结论;这里只验证机制路径。

## 仍未做的一件事

冻结配方规模下的 `K > batch`(即 K > 4096)没有跑:初始化要对 24587×32 做 8192 簇 KMeans。若真有机制需要它,那是该跑的一次 —— 检查侧已经证明可行(K=8192:去重 17.9 ms、间距 110 ms,内存 O(K·D) 与 O(block·K))。

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round23/build_rotation_run.py
cd results/research/qda/round23/trees/ROTATION && <genrec_env_v2>/python3.9 -u train_rqvae.py
```

约 40 秒。
