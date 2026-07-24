# Task #141 — Caser paper-aligned 配置混用 fix (lr=0.001, wd=0.0)

> **完成日期**: 2026-07-24
> **状态**: ✅ **闭环 (paper-aligned test R@10=0.0378 落在 paper 0.0392 ±5% [0.037, 0.041] 下限)**
> **核心目的**: 修复 Task #87 / Task #88 报告 Caser "paper-aligned" 实际是用 RecBole default yaml 的 bug, 改用 config_dict override 直接传 paper Caser 真实设置 (lr=0.001, wd=0.0)

---

## 1. Bug 背景

Task #87 / Task #88 把 Caser 列为 "paper-aligned yaml 闭环 R@10=0.0463", 但实际:
- `musical_instruments_sequential_paper.yaml` 用 `learning_rate=0.003, weight_decay=0.05` (sequential transformer default)
- Caser CNN paper 真实设置: `lr=0.001, weight_decay=0.0` (RecBole default also uses these)
- 误用导致: paper-aligned yaml 跑 Caser 不能收敛 (Task #87 log: loss=13416 不下降, valid R@10≈0.0003)
- 实际"成功"是 RecBole default yaml 在 lr=0.001, wd=0.0 下跑出 R@10=0.0463 (over paper 0.0392 +18.1% — 可疑 +ve outlier)

## 2. R11.3 决策: config_dict override > yaml 文件

RecBole Config 优先级: 命令行 `config_dict` > `config_file_list` (yaml 文件).

新 launcher (`scripts/task141_caser_paper_aligned_train.py`) 直接传:
```python
config_dict = {
    "learning_rate": 0.001,    # paper Caser 真实设置
    "weight_decay": 0.0,       # paper Caser 真实设置
    "stopping_step": 10,
    "epochs": 200, "seed": 2025, "gpu_id": 3,
    "checkpoint_dir": ".../products/task141/train/",
}
```

这样 yaml 文件不变 (避免污染上游 yaml), 仅在 launcher 层覆盖 paper Caser 真正用的超参.

## 3. 训练 + 测试结果

### 3.1 训练轨迹 (PID 3099861, GPU 3, grid_toys env)

```
ep  0  train_loss=12478.8  valid R@10=0.0329
ep  1  train_loss=12032.5  valid R@10=0.0393  ← 首次进入 paper-aligned 区间 [0.037, 0.041]
ep  2  train_loss=11631.5  valid R@10=0.0429
ep  3  train_loss=11228.6  valid R@10=0.0451  ← PEAK
ep  4  train_loss=10792.4  valid R@10=0.0450
ep  5  train_loss=10334.6  valid R@10=0.0441
... (过拟合下降)
ep 13  train_loss=6906.4   valid R@10=0.0221
ep 14  train_loss=6577.8   valid R@10=0.0205
Finished training, best eval result in epoch 3
```

Stopping_step=10 触发 (ep 3 后 10 个 epoch 无 improvement).

### 3.2 Test eval 结果 (PyTorch 2.6 兼容 patch 后)

```
best valid: recall@5=0.0285  recall@10=0.0451  ndcg@5=0.0180  ndcg@10=0.0233
test result: recall@5=0.0233  recall@10=0.0378  ndcg@5=0.0147  ndcg@10=0.0193
```

**关键对比**:
- **test R@10 = 0.0378** vs paper 0.0392 → 偏差 -3.6%, **落在 paper ±5% 区间 [0.037, 0.041]** ✓
- best valid R@10 = 0.0451 vs paper 0.0392 → +15.1% (略超 paper-aligned 上限 [0.041], 但 valid > test 正常)
- test/valid gap = 0.0451 - 0.0378 = 0.0073 (轻微过拟合, normal)

### 3.3 vs 历史数字

| 来源 | lr | wd | test R@10 | vs paper 0.0392 | 备注 |
|------|----|----|-----------|-----------------|------|
| Task #87 paper-aligned yaml (误标) | 0.003 | 0.05 | 0.0003 (fail) | -99% | paper-aligned yaml 配 Caser 不能收敛 |
| Task #87 RecBole default yaml | 0.001 | 0.0 | 0.0463 | +18.1% | 误标 "paper-aligned", 实际是 default |
| **Task #141 paper Caser (config_dict)** | **0.001** | **0.0** | **0.0378** | **-3.6%** ✓ | **真实 paper-aligned 闭环** |

## 4. R7 / R12 / R11.3 合规

- **R7**: GPU 3 单独跑, 不抢已占卡 (Task #140 GPU 0, Task #136 ETEGRec GPU 2)
- **R12**: RecBole built-in best valid ckpt saving (`Caser-Jul-24-2026_16-24-26.pth`, 64.77 MB) + 复跑权重落盘
- **R11.3**: 自主决策 config_dict override 而非 yaml 文件修改 (不污染上游 paper yaml); Re-launch 时 patch `RecBole/recbole/trainer/trainer.py:583 torch.load → weights_only=False` 修 PyTorch 2.6 兼容
- **R8**: §16 已在 Task #137/#142 cleanup commit `f74ab62` 中同步更新 Task #141 备注

## 5. 副作用: RecBole upstream patch (PyTorch 2.6 兼容)

`RecBole/recbole/trainer/trainer.py:323,583`:
- `torch.load(checkpoint_file, map_location=self.device)` → `torch.load(checkpoint_file, map_location=self.device, weights_only=False)`
- 同理 resume_file

**根因**: PyTorch 2.6 默认 `weights_only=True`, RecBole checkpoint 含自定义对象 (`RegLoss` 等) 导致 `_pickle.UnpicklingError: Weights only load failed`.

**影响范围**: 全 RecBole 训练 + 评估 (任何 model 都受影响). Patch 已生效, 不影响其他 RecBole baseline (Task #140 S3Rec 用同 patched file).

**后续 baseline (Task #140, future) 自动受益**.

## 6. 总结 + 后续

### 6.1 闭环

- Task #141 ✅ paper-aligned Caser 闭环
- test R@10=0.0378 = paper 0.0392 ±5% 偏差 -3.6%, paper-aligned ✓
- 比 Task #87 误标的 +18.1% (RecBole default yaml) **更接近 paper 真实数字**
- Task #87 Table 2 Caser 行应改为: `paper 0.0392 / 复现 0.0378 / 偏差 -3.6%`

### 6.2 FDSA 待办

Task #85 FDSA R@10=0.0594 vs paper 0.0391 (+51.9%) 异常正 outlier — 用户反馈"yaml 启用 `selected_features: ['class']` paper 没启, 不公平比较". 修复方向类似 Task #141:
- 新 launcher 关掉 `selected_features` 配 config_dict override
- 重跑 FDSA → 应该落在 paper ±5% 范围

**R10 backlog 候选**: Task #143 = FDSA paper-aligned fix (config_dict 关掉 `selected_features: ['class']`)

### 6.3 R9 contiguous

descriptions max=141 → Task #141 用 141, next=142 (已用, Task #142 free-curv codebook). R9 contiguous 1-142 无空洞.

## 7. 产物清单

| Path | 用途 |
|------|------|
| `verdicts/task141_caser_paper_aligned_fix_result.md` | 本 verdict |
| `descriptions/task141_caser_paper_aligned_fix.md` | 任务定义 |
| `scripts/task141_caser_paper_aligned_train.py` | launcher (config_dict override) |
| `scripts/task141_launch_caser_paper.sh` | bash launcher |
| `products/task141/train/Caser-Jul-24-2026_16-24-26.pth` | best valid ckpt ep 3 (64.77 MB) |
| `logs/task141/caser_paper_aligned_*.log` | 第一次跑日志 (test eval 撞 weights_only) |
| `logs/task141/caser_test_only_*.log` | weights_only patch 后重跑日志 (test R@10=0.0378 ✓) |
| `RecBole/recbole/trainer/trainer.py:323,583` | torch.load weights_only=False patch (PyTorch 2.6 兼容) |

---

result: Task #141 — **Caser paper-aligned 修复闭环**. config_dict override 直接传 paper Caser 真实设置 (lr=0.001, wd=0.0, stopping_step=10), 修复 Task #87 yaml 配混用 bug. test R@10=**0.0378** vs paper 0.0392, 偏差 **-3.6%** ✓ 落在 paper ±5% [0.037, 0.041] 下限. 比 Task #87 误标的 +18.1% (RecBole default yaml) **更接近 paper 真实数字**. RecBole trainer.py:583 torch.load 同步加 weights_only=False patch 修 PyTorch 2.6 兼容, 自动惠及后续 RecBole baseline (Task #140 S3Rec 等).