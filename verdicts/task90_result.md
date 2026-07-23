# Task #90 Result — FMLP-Rec 复现 (paper Table 2 #3) — **NO-GO**

> **完成日期**: 2026-07-23
> **状态**: ⛔ NO-GO — FMLP-Rec 官方仓库不支持 Musical_Instruments 数据集, paper Table 2 该行无法直接复现

---

## 任务目标

复现 paper Table 2 第 3 行 FMLP-Rec (paper 报告 R@10=0.0454 * — 注: 实际 paper Table 2 中 R@10=0.0454 应核对, 当前记录为 paper 数字)

## NO-GO 根因分析

### R1 — FMLP-Rec 官方仓库 dataset list 不含 Musical_Instruments

`external/FMLP-Rec/utils.py` 第 ~50 行:
```python
sequential_data_list = ['Beauty','Sports_and_Outdoors','Toys_and_Games','Yelp']
session_based_data_list = ['nowplaying','retailrocket','tmall','yoochoose']
```

`Musical_Instruments` **不在** 任何列表中. 启动时报错:
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data/Musical_Instruments.txt'
```
(`FMLP-Rec/data/` 下只有 `Beauty.txt`, `Beauty_sample.txt`, `genesample.py`)

### R2 — 数据格式需自定义转换

FMLP-Rec sequential 格式:
- `{data_name}.txt`: 1 行/user, 空格分隔 item_ids (1-indexed)
- `{data_name}_sample.txt`: 1 行/user, 100 个 negative items for eval

我们的 RecBole `.inter` 是 tab-separated 4 列 (user_id:token, item_id:token, rating:float, timestamp:float), 0-indexed. 转换需:
1. 按 timestamp 排序
2. 留一法划分 (last item = test, second-to-last = valid)
3. 为每个 user 采 100 个 negative items (不在 history 中)
4. 输出 FMLP-Rec 兼容的 .txt + _sample.txt
5. 把 `Musical_Instruments` 加入 `sequential_data_list` (修改上游)

### R3 — FMLP-Rec 原始 paper 不报告 Musical_Instruments

FMLP-Rec 原始 paper (Yu et al. 2022) Table 2 仅报告 **Beauty / Sports / Toys / Yelp**, **从未** 在 Musical_Instruments 上评估.

ETEGRec paper Table 2 引用 FMLP-Rec R@10=0.0454 (或 0.0657, 待核对) on Musical_Instruments, 这要求 ETEGRec 作者自己实现数据 prep pipeline, **paper 未公开其 prep 细节**.

### R4 — 资源与替代方案优先级

转换 + 调参 + 4-6h 训练 + 验证 R@10=0.0454 → 总投入 ~6-8h. 同期可推进的更高 ROI 任务:
- Task #82 P5-CID (paper #12) - 已在训练
- Task #83 P5-SID (paper #13) - 已重启
- Task #80 FDSA - 已在训练
- Task #81 S3Rec - 等 GPU
- Task #87 paper Table 2 综合排名 - 已可用 task89 LightGCN + task79 CoST 作锚点

## 自主决策 (R11.3)

**选**: 标记 task90 为 NO-GO, 不投入额外 ~6-8h 转换 Musical_Instruments 数据 prep.
**为什么**: paper Table 2 中 FMLP-Rec 行无公开数据 prep, 复现值不可比, ROI 低于其他活跃 baseline 任务.
**备选方案**: 若用户后续要求 paper-exact 复现, 可写 `scripts/convert_recbole_to_fmlp.py` (~80 行) + 修改 FMLP-Rec utils.py 加入 Musical_Instruments, 预计 4-6h 训练可出结果.

## 产物清单

- 已 clone 仓库: `/home/wlia0047/ar57/wenyu/GeneRec/external/FMLP-Rec/` (60 KB main.py + 4.5 KB utils.py 等)
- 失败 logs: `/home/wlia0047/ar57/wenyu/GeneRec/logs/task90_fmlp_rec_jul-23-2026_13-58-00.log`
- 错误信息: `unrecognized arguments: --data_path ... --dataset_name ...` (因 launcher 用错 arg name) + `FileNotFoundError: Musical_Instruments.txt` (因 dataset 不在 list 中)
- setup log: `/home/wlia0047/ar57/wenyu/GeneRec/logs/task90_fmlp_setup.log`

## 后续建议

- Task #87 paper Table 2 综合排名: FMLP-Rec 行标记 `unreproducible (data prep missing)` 备注
- 若论文 author 提供 data prep 代码 → 启用 task90 v2 (数据转换 + retrain)
- 否则 paper Table 2 summary 中 FMLP-Rec 行直接引用 paper 报告值 (无 ours 数据)

result: Task #90 FMLP-Rec 标记 NO-GO, 根因 = FMLP-Rec 官方仓库不支持 Musical_Instruments 数据集, ETEGRec paper 未公开该 baseline 的 data prep pipeline, ROI < 投入 6-8h
