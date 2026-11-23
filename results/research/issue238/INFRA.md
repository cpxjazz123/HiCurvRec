# Issue #238 — INFRA

每条：需求 → 命令/来源 → 输出摘要 → sha256 → 判定。
原始输出：`results/research/issue238/infra/files.txt`、`env.txt`。

## 1. 数据（C2/C5/C6 的输入）

| 需求 | 文件 | size | sha256 | 判定 |
|---|---|---:|---|---|
| Stage0 items | `results/stage0_build_parquet/items.parquet` | 10,968,789 | `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3` | PASS |
| Stage0 train | `results/stage0_build_parquet/train.parquet` | 11,152,937 | `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` | PASS |
| Stage0 valid | `results/stage0_build_parquet/valid.parquet` | 2,317,087 | `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9` | PASS |
| Stage0 test | `results/stage0_build_parquet/test.parquet` | 2,546,378 | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` | PASS |
| Stage1 T5 | `stage1_GeneEmbedding/output/sentence_t5.npy` | 75,531,392 | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` | PASS |
| 规范行为图 边 | `results/research/behavior_graph/edges.parquet` | 999,811 | `57a4477331019274e630a32be536931dfa3b189ebb898992e751db2463d0e100` | PASS |
| 规范行为图 结构 | `results/research/behavior_graph/graph.npz` | 4,319,971 | `6dbac4545f23779c26f95e3f128daeb916079f53fad7ec159dd8bfb08e7121e8` | PASS |
| 规范行为图 节点特征 | `node_features.npy` | 75,531,392 | 同 Stage1 T5（即 T5 本身） | PASS |
| 规范行为图 元数据 | `metadata.json` | 1,841 | `786a12813f89aad05a61c0ba48b7aace45435a192c03073161e94429ccc05790` | PASS |

`metadata.json` 的 `definition` 为 `E = {{i_t, i_{t+1}} | (i_1..i_T) in S_train}`，
`consumers` 含 §89b 要求的读取方，`sources` 记录 train.parquet 的 sha256 —— 与上表一致，
因此 C5（行为图只来自 train）成立。

## 2. 冻结基线与对照（C1/C2/C38–C41 的参照）

| 需求 | 文件 | sha256 | 判定 |
|---|---|---|---|
| 冻结基线入口 | `stage2_RQ-VAE/baseline/TIGER_RQ-VAE/train_rqvae.py` | `a06d9f9716d9b9ea616cf6c3b8dcf8e354f6f3a96aea50bb4896a31e3b64cf26` | PASS |
| 冻结基线量化器 | `.../model/layers.py` | `0d5eacf6b3cbfe8874c2b8d54f7c7554de1ade392ae8b54521b53cecec4c994a` | PASS |
| 冻结基线配置 | `.../config/train_config.json` | `27461832258ca93f5bd21ec2f5618ccb69943ac32d5337d6ac57527a0e7638f6` | PASS |
| TIGER SID | `results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json` | `025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059` | PASS |
| TIGER checkpoint | `results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth` | `af8efc29a8bcbbe5394cc5e358acf7b14c0e78ae99d8a8b9e5faf03fdddd97a8` | PASS |
| Stage3 入口 | `stage3_T5Train/train_HG-Rec.py` | `7edf83f88dbe085749418509d02765db3223cde7a9bc9953f187bab41ac14cc1` | PASS |

### 冻结基线超参（C2 的比对基准，逐字抄自 `config/train_config.json`）

```
EPOCHS 3000 · BATCH_SIZE_PER_RANK 1024（四卡 → 全局 4096）· LR … · HIDDEN_SIZES [512,256,128]
CODEBOOK_NUM 3 · CODEBOOK_SIZE [256,256,256] · CODEBOOK_DIM 32 · BETA 0.25 · VQ_TYPE vq
EMA_DECAY 0.99 · SK_EPSILON 0.003 · SK_ITERS 50 · PCA_DIM 0 · XAVIER_INIT true
WARMUP_EPOCHS 50 · GRADIENT_CLIP_NORM 1.0 · OPTIMIZER AdamW · SEED 42
NUM_WORKERS 0 · EVAL_INTERVAL 50
```

### 基线 Stage3 对照（C44 的比较基准）

| 运行 | `test_R@10` |
|---|---:|
| 2026-11-01 | 0.040199 |
| 2026-11-12（记录值） | **0.055450** |
| 2026-11-23 10:18（同 SID 重复） | 0.053483 |
| 2026-11-23 10:57（同 SID 重复） | 0.055206 |
| 2026-10-26 | 0.055537 |

同一 SID `025ad8bd` 的三次观测：均值 **0.054713**、sd **0.001072**（与 issue 引用的 0.00107 一致）。

## 3. 解释器（C24/C43）

| 用途 | 解释器 | 版本 | 判定 |
|---|---|---|---|
| Stage0/1/3 | `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10` | py 3.10.20, torch 2.13.0+cu130, np 1.26.4, scipy 1.15.3, pd 2.3.3 | PASS |
| Stage2 / 机制测试 | `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` | py 3.9.25, torch 2.8.0+cu128, np 2.0.2, scipy 1.13.1, pd 2.3.3 | PASS |

## 4. 硬件与端口（C2/C24）

| 需求 | 输出 | 判定 |
|---|---|---|
| 四卡可见 | `0..3 NVIDIA L40S 46068 MiB` | PASS |
| 端口 50202 空闲（Stage2） | `PASS ports 50201/50202 free` | PASS |
| 端口 50201 空闲（Stage3） | 同上 | PASS |
| 卡上空闲 | `nvidia-smi` 3 MiB ×4，无 compute app | PASS |

## 5. 结论

**infra 无 FAIL。** 数据、基线、对照、解释器、卡与端口全部 PASS，可以进入实现。

一项需要记录的差异：本机实际是 **4 × L40S 46 GB**，不是 A100 40 GB；
这不改变 issue 的协议（四卡、全局 batch 4096），只影响显存上限。
