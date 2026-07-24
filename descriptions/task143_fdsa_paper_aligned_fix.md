# Task #143 — FDSA paper-aligned fix (关掉 selected_features=['class'])

> **任务目的**: 修复 Task #85 FDSA R@10=0.0594 vs paper 0.0391 (+51.9%) 异常正 outlier. 根因: RecBole `recbole/properties/model/FDSA.yaml:10` 默认启用 `selected_features: ['class']` (使用 item class token 做 feature embedding), paper FDSA 没启. 用 `config_dict override selected_features=[]` 重训, 验证落在 paper ±5% 范围 [0.037, 0.041].

> **完成日期**: in progress
> **状态**: 🟡 待启动

---

## 1. 背景

Task #85 FDSA test R@10=**0.0594** vs paper 0.0391 (+51.9% 异常正 outlier). Task #85 verdict 自查列 3 个可能原因:
1. **Paper class token feature 未启用** (我们的 yaml 用了 `selected_features: ['class']`) ← 验证为本任务根因
2. valid_metric (NDCG@10 vs paper Recall@10) 选择偏差
3. dataset split / full ranking 协议差异

经核查 `RecBole/recbole/properties/model/FDSA.yaml:10`:
```yaml
selected_features: ['class']    # (list of str) The list of selected item features.
```

这是 **FDSA model 自己的 default**, **不是 yaml 文件设置**. 当 `load_col.item: [item_id, class]` (我们有) + model 有 `selected_features: ['class']` default (FDSA 启用了) → RecBole 自动 apply, 给 FDSA 提供额外 class token 语义信号 → 显著 boost performance.

**Paper FDSA 实际只用 item_id**, 没启 class feature. 当前复现"成功"是不公平的 +51.9% positive outlier.

## 2. 实验设计

**变量**: 仅 `selected_features` (config_dict override `[]`)
**保持不变**:
- `musical_instruments_sequential_paper.yaml` (lr=0.003, wd=0.05, MAX_ITEM_LIST_LENGTH=20, valid_metric=NDCG@10, stopping_step=20)
- Musical_Instruments 5-core 数据集
- seed=2025, epochs=200
- RecBole trainer.py weights_only=False patch (Task #141 已修, 沿用)

**启动命令** (类似 Task #141 config_dict override 模式):
```python
config_dict = {
    "selected_features": [],       # 关掉 FDSA class token (paper 没启)
    "learning_rate": 0.003,        # paper FDSA default (跟 yaml 一致)
    "weight_decay": 0.05,
    "stopping_step": 20,
    "epochs": 200, "seed": 2025, "gpu_id": 1,
    "checkpoint_dir": ".../products/task143/train/",
}
run(model="FDSA", dataset="Musical_Instruments", config_file_list=[yaml_path], config_dict=config_dict)
```

GPU 1 100% 空闲 (R7 确认 0% util, 0 MiB).

## 3. 决策触发 (vs paper R@10=0.0391)

| 观察条件 | R@10 区间 | 决策 |
|---------|-----------|------|
| **FDSA 关掉 selected_features 后 test R@10 落在 paper ±5% [0.037, 0.041]** | 0.037-0.041 | ✅ **paper-aligned 闭环**, 替换 Task #85 数字 |
| test R@10 仍在 +30-50% (>0.051) | >0.051 | ⚠️ 部分公平 (selected_features 不是唯一原因), 保留 Task #85 数字 + 标注 |
| test R@10 跌到 paper-aligned 区间外 (-10% 以内) | 0.035-0.037 | ⚠️ paper-aligned 略偏差, 接受, 仍替换 |
| test R@10 跌到 <0.030 | <0.030 | ❌ paper FDSA 在我们数据集上不能复现, 报结果 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 数据加载 + RecBole init | ~3 min |
| FDSA 训练 200 epoch (paper stopping_step=20) | ~30-60 min (FDSA 比 Caser 慢) |
| Test eval | ~1 min |
| **总计** | **~35-65 min** |

RecBole FDSA 训练 ep 时间约 60-100s (transformer 双塔架构, 比 Caser CNN 慢约 30%).

## 5. 风险与缓解

**风险 1**: 关掉 selected_features 后 FDSA 可能 loss 不下降 (paper FDSA 在 Musical_Instruments 小数据集上本就用 item_id-only). 缓解: stopping_step=20 触发后会自然停止, 不浪费 GPU.

**风险 2**: paper FDSA 真设置跟 RecBole FDSA 还有别的差异 (e.g. loss type, embedding dim). 缓解: 仅改 selected_features 这一项, 其它沿用 yaml paper-aligned.

**风险 3**: 修改 RecBole properties/model/FDSA.yaml 会污染上游. 缓解: 不改 yaml, 仅在 launcher 层 config_dict override (跟 Task #141 Caser 同模式).

## 6. 完成度跟踪

- [x] R9 audit (max=142, next=143)
- [x] Task #143 description 落盘
- [x] 基础设施 audit (selected_features 在 FDSA.yaml:10 已确认)
- [x] launcher 写 (config_dict override selected_features=[])
- [x] py_compile + launch GPU 1
- [ ] FDSA 训练收敛 + stopping
- [ ] test eval 拿到 R@10
- [ ] 写 `verdicts/task143_fdsa_paper_aligned_fix_result.md` 含 `result:` 行
- [ ] loop.md §16 cleanup (Task #143 闭环后删除)

## 7. R11.3 自主决策

- **不并行多 FDSA 变体**: 仅跑 1 个 config_dict override fix. 多变体 (e.g. selected_features=['class'] vs [] vs [{class: {dim: 10}}]) 需要更多 GPU 时间.
- **沿用 paper yaml 其它设置**: 不动 learning_rate/weight_decay (paper FDSA 实际设置 paper-aligned yaml 是正确的). 仅 selected_features 一项.
- **launch 立刻不 S3Rec 等**: S3Rec 在 GPU 0 (5h ETA), Caser fix 已闭环, ETEGRec 在 GPU 2 (10h ETA). GPU 1 100% 空闲. 立刻 launch.

## 8. 关联

- Task #85 verdict — `verdicts/task85_fdsa_test_eval_result.md` (FDSA R@10=0.0594 +51.9% 异常正 outlier)
- Task #141 verdict — `verdicts/task141_caser_paper_aligned_fix_result.md` (config_dict override 模式模板)
- Task #87 Table 2 baseline ranking — FDSA 行应改为 paper-aligned fix 后数字
- `RecBole/recbole/properties/model/FDSA.yaml:10` — selected_features=['class'] 默认配置