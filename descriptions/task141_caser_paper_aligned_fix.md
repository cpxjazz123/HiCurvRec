# Task #141 — Caser paper-aligned 修复 + 重训 (lr=0.001)

> **任务目的**: 修复 Task #87 Caser paper-aligned 配置混用 bug — 用 lr=0.001 (paper Caser baseline) 重训 Caser, 验证 R@10 ≈ 0.039 (paper 0.0392, ±5%).

> **完成日期**: in progress
> **状态**: 🟡 待启动

---

## 1. 背景

User 2026-07-24 质疑 Task #87 表里 Caser +18.1% (R@10=0.0463 vs paper 0.0392):
- **0.0463** 实际来自 `task72_phase6_full_Caser_gpu3.log`, 用 RecBole 默认 yaml (lr=0.001 + wd=0.0)
- **paper_aligned log** (`task72_paper_aligned_Caser_gpu0.log`) 用 yaml `musical_instruments_sequential_paper.yaml` (lr=0.003 + wd=0.05) — **训练失败**, loss 13416 全程不下降, valid R@10 ≈ 0.0003

Task #87 表格**误标**为 "paper-aligned", 实际两个配置完全不同.

**Root cause**:
- generic sequential yaml `musical_instruments_sequential_paper.yaml` 把 lr 改为 0.003 + wd=0.05 (DECOR paper default for sequential transformer)
- Caser 是简单 CNN, lr=0.003 太大震荡不收敛

**Pre-task 状态**:
- Caser 模型代码 **不引用** class feature (cnn-based, 只 user+item embedding) → 不是 class feature 问题
- DECOR paper Caser baseline 行 = RecBole 默认 yaml 设置 (lr=0.001 + wd=0.0), 这是 paper Caser baseline 真实报告 R@10=0.0392 的来源

## 2. 实验设计

**变量**: `learning_rate=0.001` + `weight_decay=0.0` (paper Caser 真实设置) — 仅 Caser
**保持不变**:
- 数据集: Musical_Instruments
- yaml: musical_instruments_sequential_paper.yaml (Caser 也接受, lr/wd 通过 config_dict 覆盖)
- seed: 2025
- gpu_id: 3 (R7: GPU 3 free 0% util, 不抢 #140 S3Rec GPU 0 / #136 ETEGRec GPU 2)
- epochs=200 (RecBole 默认 30 epoch 就收敛, 但留 margin)
- stopping_step=10 (RecBole 默认; sequential paper yaml 是 20 — 用 RecBole 默认)

**启动命令**:
```bash
python3 scripts/task141_caser_paper_aligned_train.py
```

Launcher 用 `recbole.quick_start.run()` 配 config_dict override:
```python
config_dict = {
    "learning_rate": 0.001,    # paper Caser (vs yaml 0.003)
    "weight_decay": 0.0,       # paper Caser (vs yaml 0.05)
    "stopping_step": 10,       # RecBole default (vs yaml 20)
    "seed": 2025, "gpu_id": 3,
    "show_progress": False,
    "checkpoint_dir": "products/task141/train/",
}
```

## 3. 决策触发 (vs paper)

| 观察 | 决策 |
|------|------|
| R@10 in [0.037, 0.041] (±5% paper 0.0392) | ✅ Caser paper-aligned 闭环, 可信任 |
| R@10 in [0.041, 0.045] (+5% ~ +15%) | ⚠️ paper-aligned 闭环但有 marginal bias, 报告给 user |
| R@10 ≈ 0.000x (跟原 paper-aligned log 一样失败) | ❌ config_dict override 没生效, 重新读 yaml 链 |
| R@10 ≥ 0.046 (类似 RecBole 默认 0.0463, +18%) | ⚠️ 数据偏差或 paper Caser baseline 行有差异, 报告 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 数据加载 + Caser CNN warmup | ~3 min |
| 训练 8-30 epoch (按 RecBole 默认 30 epoch + early stop @ 10) | ~5-30 min |
| Test eval | ~30 sec |
| **总计** | **~10-40 min** |

GPU 3 估算: Caser 是 1.8M parameter CNN, RecBole 默认 speed 4-5 min/epoch, 8 epoch ≈ 30 min.

## 5. 风险与缓解

**风险 1**: config_dict override 对 general yaml key 是否生效 (RecBole Config 优先级: config_dict > config_file). 缓解: launch 后立即查 log 第一行 "Training Hyper Parameters: learning_rate = 0.001" 确认.

**风险 2**: R12 ckpt 自动保存但需要命名不歧义. 缓解: checkpoint_dir 写到 products/task141/train/ 隔离, 避免跟 #72 Caser ckpt 冲突.

**风险 3**: Caser 早期 valid 抖动大, 1-2 epoch 才稳定. 缓解: stopping_step=10 给 10 个 epoch patience.

## 6. 完成度跟踪

- [x] R9 audit (max=140, next=141)
- [x] description 落盘
- [x] scripts/task141_caser_paper_aligned_train.py 写
- [ ] py_compile verify
- [ ] launch GPU 3
- [ ] 验证 config_dict override 生效 (log first line)
- [ ] 训练收敛 + best valid ckpt 落盘
- [ ] Test eval + R@5/10, NDCG@5/10
- [ ] 写 `verdicts/task141_caser_paper_aligned_fix_result.md` 含 `result:` 行
- [ ] 更新 Task #87 verdict (§ "误标" 修正)

## 7. R11.3 自主决策记录

- **Caser-specific yaml vs override config_dict**: 选 config_dict (复用通用 yaml, 不增加 yaml 文件). 备选 Caser-specific yaml (独立文件更隔离). 决策: config_dict 简单, R12 ckpt 自动隔离.
- **stopping_step=10 (vs yaml 20)**: 选 10 跟 RecBole 默认对齐, Caser 在 epoch 6 收敛, 早停保留余地. 备选 yaml 20 (更宽容). 决策: 10, 跟 RecBole 默认.
- **launch 立刻 (不等 S3Rec)**: R7 GPU 3 100% 空闲, 跟 GPU 0 (#140 S3Rec) + GPU 2 (#136 ETEGRec) 完全分离. 决策: 立刻 launch.
- **不并行 FDSA fair-comparison 重训 (24h)**: 单独高代价, user 只 ask Caser fix, FDSA 后续单独决定. 决策: 不在本任务做 FDSA.

## 8. 关联

- [[recbole-grid-toys-env-deps]] — RecBole env 补装 pattern (Task #141 用同套 deps)
- [[s3rec-no-go-pretrain]] — Task #81 S3Rec yaml 错配 NO-GO, Task #140 修复 (Task #141 同一类型错误, 但 Caser 用 config_dict fix 不需新 yaml)
- Task #87 verdict — 待 § "误标" 修正 append (R8 §9.3 retroactive compliance pattern)
