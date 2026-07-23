# Task #77 result — LETTER t5-base 训练 (backbone 升级)

> **任务名**: Task #77 — LETTER backbone 从 t5-small 升级到 t5-base (沿用 Task #50 RQ-VAE tokenizer 692 tokens)
> **完成日期**: 2026-07-23
> **状态**: ⏸️ **用户主动 SIGTERM 中断 (2026-07-23 12:35)**; **首次启动 hf 网络故障, 第二次成功加载, 第三次跑到 52% step 仍中断**

---

## 1. 任务目标

承接 Task #61 LETTER-TIGER R@10=0.0997 超越 paper (paper 0.0581, +72%) 成果, 把 backbone 从 t5-small 升级到 **t5-base** (~3.7× 参数量), 验证是否进一步突破 R@10 上限. 同时作为并行 4 任务的一部分 (Task #74/75/76/77) 提供横向对比.

## 2. 关键决策 (R11.3)

| 决策 | 选择 | 理由 |
|------|------|------|
| backbone | t5-base | vs task50 t5-small, 增大 ~3.7× 容量 |
| per_device_batch_size | 64 (vs task50 128) | t5-base 显存 ~3.7×, bs 减半 |
| GPU | 3 单卡 | finetune.py 硬编码 CUDA_VISIBLE_DEVICES=3 |
| Tokenizer | 沿用 Task #50 .index.epoch5000.alpha0.01-beta0.0001.json (692 tokens) | 保持 SID 不变 |
| Epoch | 200 (vs task50 100) | backbone 大需要更长收敛 |

## 3. 执行时间线 (3 次尝试)

| 尝试 | 起止 | 结果 |
|------|------|------|
| **第 1 次 (02:05-35)** | 立即 OSError | `OSError: google/t5-base is not a local folder and is not a valid model identifier` — HF 网络问题 |
| **第 2 次 (02:08-05)** | 短暂成功 | 加载 t5-base 权重, 但仅 1 个 step (epoch 2.3/103K, ~0.001%), SIGTERM 中断 |
| **第 3 次 (02:16-34)** | 长达 10h+ 训练 | 跑到 step **106401/206000 (~52%)**, 持续 ~10h+, 最终 SIGTERM 中断 |

> **SIGTERM (2026-07-23 12:35)**: 用户主动 kill 4 个并行任务, Task #77 跑到一半进程退出.

## 4. 关键指标

- **首次启动失败原因**: HF 网络不可用 (`google/t5-base` 无法下载). 需镜像 / 本地缓存 / `hf auth login` 才可继续.
- **第二次训练**: 完全加载 t5-base 模型, 显示 model 结构 (T5Stack + lm_head 32792), 但因同时启动 4 卡并行抢占 dataloader 资源, 实际 step 几乎不前进.
- **第三次训练**: 跑到 step 106401 (~52% 全程), 最终 epoch 数未达 200, training loss 无最终 val 报告.

**val_R@10 数据**: **无 (训练未跑到第一个 eval step)**.

## 5. 分析解读

### 5.1 HF 网络是 blocker
首次启动失败表明在没有 `hf auth login` 或镜像源的情况下, t5-base 权重无法自动下载. 修复方案 (R11.3):
- **方案 A**: `huggingface-cli login` 走官方 (需要 token)
- **方案 B**: 用清华镜像 `HF_ENDPOINT=https://hf-mirror.com`
- **方案 C**: 复用 task50 t5-small 缓存 → 走本地 lfs cache

→ 下次启动前必须先验证 HF 网络可达性, **修复后**才能继续 Task #77.

### 5.2 backbone 升级路径 ROI 高
Task #61 LETTER-TIGER R@10=0.0997 vs paper 0.0581 (+72%) 表明 LETTER 路径比 ETEGRec 强很多. backbone 升级 (t5-small → t5-base) 是 LETTER 框架内最具 ROI 的下一步, 仍是必做实验.

## 6. 产物清单

| 路径 | 内容 |
|------|------|
| `/home/wlia0047/ar57/wenyu/GeneRec/scripts/task77_letter_train_t5_base.sh` | 启动脚本 |
| `/home/wlia0047/ar57/wenyu/GeneRec/logs/task77_letter_t5base_*.log` | 3 次尝试日志 |
| `/home/wlia0047/ar57/wenyu/GeneRec/descriptions/task77_letter_t5_base.md` | 任务定义 |
| NOT EXIST: products/task77/ | 无 ckpt (SIGTERM 前未保存) |

## 7. 后续建议

1. **必须修复 HF 网络问题** 才能继续 Task #77 (方案 B 清华镜像优先)
2. 重启时**单卡独占** GPU 3 (避免与其他任务并行)
3. backbone 升级 ROI 高, 应作为下一步 LETTER 路径的核心实验

result: ⏸️ Task #77 LETTER t5-base 因 HF 网络失败 + SIGTERM 中断, 训练未完成 (52% step, 无 val_R@10); 需修复 HF 网络后重启
