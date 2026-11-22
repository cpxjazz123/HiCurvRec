# Issue #232 — INFRA

每条 `需求 → 命令 → 输出摘要 → sha256 → PASS/FAIL`。

## 数据

| 需求 | 命令 | 输出摘要 | sha256 | 判定 |
|---|---|---|---|---|
| Stage2 训练输入 | `ls results/stage0_build_parquet/*.parquet` | items 24,587 / train 396,958 | 沿用 #231 记录 | PASS |
| 语义输入 | `results/stage1_GeneEmbedding/output/sentence_t5.npy` | 存在,768 维 | 沿用 #231 记录 | PASS |

## 基线(不复训,C1)

| 需求 | 路径 | sha256 | 判定 |
|---|---|---|---|
| TIGER SID | `results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json` | `025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059` | PASS |
| TIGER checkpoint | `results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth` | `af8efc29a8bcbbe5394cc5e358acf7b14c0e78ae99d8a8b9e5faf03fdddd97a8` | PASS |

## 冻结 bank(C45/C46/C48)

复用 #231 全局 pair-disjoint bank:`results/research/issue231/banks/`,
`banks_global.json` 的 `hashes` 覆盖 9 个文件,`seed=42`,`shares=[0.6,0.2,0.2]`。

| 划分 | chain | fork | merge |
|---|---:|---:|---:|
| supervision(train) | 1138 | 1059 | 1189 |
| validation | 192 | 149 | 162 |
| test | 151 | 164 | 159 |

**隔离实测**(商品对级别):

| 对 | 商品对交集 | 商品交集 |
|---|---:|---:|
| train & validation | 0 | 611 |
| train & test | 0 | 581 |
| validation & test | 0 | 221 |

- **C46 PASS**:三个划分之间的商品对交集全为 0(商品级别重叠是允许的,见 C48)。
- **C48 记录**:issue 明确"无须禁止同一训练商品对跨结构出现,只须跨数据划分隔离"。实测训练内
  chain/fork/merge 合并后商品对可跨结构复用,符合要求。

## 解释器

| 角色 | 路径 | 判定 |
|---|---|---|
| Stage2 | `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` | PASS |

## 硬件

| 项 | 输出摘要 | 判定 |
|---|---|---|
| GPU | 上一轮实测可见卡数随会话变化;本轮以 `nvidia-smi` 实时结果为准 | 待测 |

## 树

| 需求 | 判定 |
|---|---|
| 冻结基线 `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` 只读 | PASS(尚未复制) |
| 新候选从基线复制、只改 `ARM_NAME` 之外可解释的机制文件 | 待 §2 |
