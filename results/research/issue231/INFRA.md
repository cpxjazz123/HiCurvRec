# Issue #231 — INFRA

## 1. 数据

| 需求 | 命令 | 输出摘要 | sha256 | 判定 |
|---|---|---|---|---|
| embedding | `ls stage1_GeneEmbedding/output/sentence_t5.npy` | 24,587 × 768 | 见 `banks/banks.json` | PASS |
| 训练样本 | `results/stage0_build_parquet/train.parquet` | 396,958 行 | 见 C34 记录 | PASS |
| 行为图(有向) | `results/research/behavior_graph_directed/graph.npz` | 24,587 节点 / 323,633 有向边 | 见 triples 审计 | PASS |

## 2. 冻结 bank:**不满足 C33,必须重建**

`results/research/issue230/banks/` 只做了**各结构内部**隔离(#230 的要求)。#231 的 C33 要求
**三种结构合并后**的全局无序对交集为零,且**含实际进入 loss 的负样本**。

实测现有 bank 的违规:

| 检查 | 结果 | C33 要求 |
|---|---:|---|
| chain ∩ fork(supervision) | **119** | 0 |
| chain ∩ merge(supervision) | **105** | 0 |
| fork ∩ merge(supervision) | **37** | 0 |
| chain supervision ∩ 任意 eval 对 | **122** | 0 |
| fork supervision ∩ 任意 eval 对 | **88** | 0 |
| merge supervision ∩ 任意 eval 对 | **69** | 0 |

结论:**FAIL**,需要在 §2 前重建 bank(跨结构全局划分)。

## 3. 基线

| 需求 | 路径 | 说明 | 判定 |
|---|---|---|---|
| 公平基线 256×3 | `results/stage2_RQ-VAE/TIGER_RQ-VAE` | C39:要作为主基线;已由 #230 C12 核验 | PASS |
| 1024 码本(仅历史) | `results/stage2_RQ-VAE/TIGER_graphlet_wide_l1_rq` | **不得**作公平基线 | 记录 |

## 4. 解释器与硬件

| 需求 | 命令 | 输出摘要 | 判定 |
|---|---|---|---|
| Stage2 解释器 | `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` | 存在,torch 2.8.0+cu128 | PASS |
| GPU | `nvidia-smi` | 4 × A100-40GB,当前全空闲 | PASS |

## 5. 树

| 需求 | 状态 | 判定 |
|---|---|---|
| 冻结基线只读 | `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` 未改动 | PASS |
| 候选树从基线复制,只改 `ARM_NAME` | 待建(§2) | NOT_RUN |

## 阻塞项

C33 驱动 bank 重建;这不是"数据缺失",是本轮必须完成的 infra 工作。
