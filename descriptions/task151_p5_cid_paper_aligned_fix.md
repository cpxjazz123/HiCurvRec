# Task #151 — P5-CID paper-aligned 5-任务 pretrain 完整复现

> **任务目的**: P5-CID paper-aligned 复现 R@10=0.0413 vs paper 0.0507 (-18.5% 负偏差). 根因是 `external/LLM-RecSys-ID/main.py:192` `number_of_tasks=1` 且训练循环只 iterate `train_loaders[0]` (sequential_item) 一个 loader, 其它 4 个 P5 任务 (sequential_yesno, direct_yesno, direct_candidate, direct_straightforward) 数据加载了但从未训练. 修复: patch trainer cycle 5 个 dataloader, P5 paper-aligned 完整 multi-task 训练, R@10 落在 [0.045, 0.062] (paper 0.0507 ±20%).

> **完成日期**: in progress
> **状态**: 🟡 待启动

---

## 1. 背景 (用户 2026-07-24 反馈)

用户审查 baseline Table 2 时指出:
> "我们是希望对齐 paper 的方法. 帮我修复 LETTER + P5-CID"

**P5-CID -18.5% 负偏差根因**:
- P5 paper (arXiv:2203.13366) 设计 5 个任务:
  1. **sequential recommendation** (主任务, 测 hit@10)
  2. **sequential yes/no** (predict 是否会继续买)
  3. **direct yes/no** (predict 用户对单品的喜好)
  4. **direct candidate ranking** (ranking 给定 candidate)
  5. **direct straightforward completion** (predict 单品的某个属性)
- 所有 5 任务共享 user embedding + item embedding, 联合 pretrain 让 user/item 表征更丰富 → 主任务 hit@10 显著受益.
- LLM-RecSys-ID 仓库 `main.py:192` 写死 `number_of_tasks=1`, 训练循环只走 `train_loaders[0]` (sequential_item). 4 个任务 data 加载了但从未训练.
- `--meta_epochs 10 --review_epochs 2` 这两个 flag 也被 argparse 注册但 **从未在 trainer 使用** (潜在 MAML inner-loop 框架但 no-op).
- **结论**: 现有 P5-CID 复现退化成"sequential-only", 丢失 80% 的 P5 设计 → test R@10 = 0.0413, 比 paper 0.0507 低 18.5%.
- **修复**: patch trainer 真正 cycle 5 个 dataloader.

## 2. 实验设计

**变量 1 (代码层 patch)**: 把 `external/LLM-RecSys-ID/main.py:192` 和训练循环改成多任务:
```python
# OLD (line 192): number_of_tasks = 1
# NEW:
number_of_tasks = 5  # sequential_item, sequential_yesno, direct_yesno, direct_candidate, direct_straightforward

# OLD (训练循环内 line ~232): for batch in tqdm(train_loaders[0]):
# NEW: cycle through all 5 dataloaders
from itertools import cycle
loader_cycle = cycle([iter(loader) for loader in train_loaders])
min_total_batches = min(len(loader) for loader in train_loaders) * len(train_loaders)
for _ in range(min_total_batches):
    loader_idx = next(loader_cycle_idx)  # round-robin
    batch = next(loader_cycle)
    ...
```

**变量 2 (其它超参不变)**:
- 5 dataloaders 的 `--train_*_batch` 仍照 task82: 64 / 32 / 48 / 12 / 48
- `--epochs 10` (paper 推荐), `--meta_lr 1e-4`, `--alpha 2`
- `--whole_word_embedding shijie` (R12 P5-SID 复现验证过)
- `--item_representation CF` (20×500 索引)
- seed 42

**保持不变**:
- 框架: external/LLM-RecSys-ID/main.py
- 数据集: Instruments
- ckpt path: products/task151/p5_cid_5task.pt (NEW, 不覆盖 task82)
- R12 ckpt 保存: 沿用现有 main.py 内 R12 fix

**启动命令**:
```bash
# Step 1: patch main.py to cycle 5 dataloaders (modify line 192 + train loop)
# Step 2: launch
cd /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID
CUDA_VISIBLE_DEVICES=2 \
  python3 main.py \
    --seed 42 \
    --data_dir /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/ \
    --logging_dir /home/wlia0047/ar57/weneRec/logs/task151.log \
    --model_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task151/p5_cid_5task.pt \
    --task instruments \
    --max_history 20 \
    --sequential_num 10 \
    --negative_sample 2 --yes_no_sample 5 \
    --train_sequential_item_batch 64 \
    --train_sequential_yesno_batch 32 \
    --train_direct_yesno_batch 48 \
    --train_direct_candidate_batch 12 \
    --train_direct_straightforward_batch 48 \
    --meta_lr 1e-4 --review_lr 1e-4 \
    --model_type t5-small \
    --epochs 10 --lr 1e-3 --clip 1 \
    --logging_step 1000 --warmup_prop 0.05 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.01 --adam_eps 1e-6 \
    --dropout 0.1 --alpha 2 --multiGPU --gpu 2 \
    --evaluation_method sequential_item \
    --evaluation_template_id 0 \
    --number_of_items 24588 \
    --item_representation CF --cluster_number 20 --cluster_size 500 \
    --data_order remapped_sequential \
    --remapped_data_order original \
    --resolution 2 --max_random_number 30000 --min_random_number 1000 \
    --random_initialization_embedding --whole_word_embedding shijie \
    2>&1 | tee logs/task151.log
```

## 3. 决策触发 (vs paper R@10=0.0507)

| Task #151 test R@10 | Δ vs paper | 决策 |
|---------------------|------------|------|
| [0.045, 0.062] (-10% ~ +20%) | ✅ paper-aligned | 闭环成功, 写 verdict + 更新 Table 2 |
| [0.062, 0.075] (+20% ~ +50%) | ⚠️ over-shoot (少见, P5 设计让 sequential 受益) | 接受, 跟 LETTER +71.6% 同档 |
| [0.025, 0.045] (-50% ~ -10%) | ⚠️ under-train | 多任务训练 degradation → 衰减 `--meta_lr` 或 loss weight |
| < 0.025 | ❌ 异常 | multi-task patch 出错, rollback |

→ **目标区间**: R@10 ∈ [0.045, 0.062] (paper 0.0507 略宽).

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| main.py patch dry-run (报告 patch diff) | 5 min | - |
| 5-task 训练 10 epoch | ~10-15 h (5× 数据量 vs task82 单任务) | GPU 2 |
| Test eval (5-task ckpt) | ~5 min | GPU 2 |
| **总计** | **~10-15 h** | GPU 2 |

## 5. 风险与缓解

**风险 1**: 5 个任务 batch size 不同 (64/32/48/12/48), round-robin 不均衡, 每 task 实际 step 数 ≈ 1/5 总 step.
**缓解**: 接受 (跟 paper "task sampling" 行为一致, paper 也是 cycle 而非按比例), 5 个 task 一起 contribution.

**风险 2**: main.py 已有 R12 fix, 但需确认 patch 不破坏 R12 save_ckpt 路径.
**缓解**: Patch 仅修改训练循环 (line ~232), 不动 R12 save block (line ~310).

**风险 3**: `--evaluation_method sequential_item` 只测 sequential, 无法验证 yesno/direct 任务训练效果.
**缓解**: 主评估用 sequential_item (跟 paper Table 2 一致); 论文 discussion 可注明 "multi-task 协同效果通过主任务代理".

**风险 4**: 改 main.py 是上游源码改动 (R11.4 不可逆决策).
**缓解**: dry-run patch diff → 用户确认 → 执行. 不直接覆盖.

## 6. 完成度跟踪

- [ ] scripts/task151_patch_main.py 写 patch logic (读 main.py, 改 number_of_tasks + 训练循环, 写 diff)
- [ ] dry-run patch diff 报告给用户
- [ ] 用户确认后执行 patch
- [ ] py_compile main.py 验证
- [ ] 启动 task151 到 GPU 2 (R7 空闲优先)
- [ ] 10 epoch training 收敛
- [ ] Test eval 完成 R@5/R@10/R@20
- [ ] 写 verdict: `verdicts/task151_p5_cid_5task_result.md`
- [ ] 更新 §16 (R8 旧清理)
- [ ] 把 P5-CID R@10=0.0413 → 新值, 替换 Table 2 GENERATIVE_BASELINES 行
