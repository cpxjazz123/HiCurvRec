## 结项:Teacher 可复用、接口可测;结构信号可迁移,方向内容不可迁移

不重开 #221,不跑 Stage2/Stage3。全部证据在 `results/research/directed_graphlet_teacher/`
(`MANIFEST.json` 逐文件 sha256)。

### 1. Teacher checkpoint 已可复用

- `teacher/teacher_directed_v1.pth`,SHA-256 `25addae9f7ff98f05148016e95d8f87383b128c013564ea60518e37dd5547e4b`;payload 内含 format、config、词表与 state_dict。
- `load_frozen_teacher` 做 `strict=True` 加载 → `eval()` → 全参数 `requires_grad_(False)`;`save_teacher` 会用固定 probe 批次证明重载前向**逐位相等**,否则拒绝落盘。
- 同口径复跑的三行对照与 #221 完全相同:真实有向邻接 **0.7121 / 0.6101**、方向打乱 0.1694 / 0.1462、邻接清零 0.0297 / 0.0117。

### 2. 数据审计:这份池子没有 pair-disjoint 的 held-out

按完整训练池统计,242,765 / 243,530 = **0.997** 的 held-out 实例至少与一条训练实例共享一条无条件节点对(逐族见报告 §3)。#221 的 0.7121 因此是「实例不同、节点对重叠」的识别指标,不是对未见结构的泛化。

### 3. 可迁移性 Gate

同数据、同 seed、同优化预算(每族 2000 个全批量 epoch,训练集 5000/族),只改监督信号:

| 变体 | 头 | 训练 acc | held-out acc | held-out macro-F1 | pair-clean acc | 无条件 orbit F1 |
|---|---|---:|---:|---:|---:|---:|
| `random` | — | 0.0088 | 0.0090 | 0.0043 | 0.0095 | 0.0005 |
| `majority` | — | - | - | 0.0777 | - | 0.0131 |
| `degree` | — | - | - | 0.0182 | - | - |
| `hard_ce` | pooled | 0.3669 | 0.2210 | 0.0208 | 0.2075 | 0.0078 |
| `teacher_kd` | pooled | 0.3483 | 0.2392 | 0.0277 | 0.2249 | 0.0087 |
| `shuffled_kd` | pooled | 0.2774 | 0.2331 | 0.0285 | 0.2209 | 0.0087 |
| `teacher_kd_z` | pooled | 0.3441 | 0.2499 | 0.0280 | 0.2370 | 0.0097 |
| `hard_ce_rel` | relational | 0.5924 | 0.1730 | 0.0213 | 0.1615 | 0.0083 |
| `teacher_kd_rel` | relational | 0.5204 | 0.2000 | 0.0268 | 0.1890 | 0.0085 |
| `teacher_kd_z_rel` | relational | 0.4968 | 0.2226 | 0.0295 | 0.2027 | 0.0090 |
| `hard_ce_spec` | specialist | 0.9199 | 0.3403 | 0.1356 | 0.3373 | 0.0562 |
| `teacher_kd_spec` | specialist | 0.9283 | 0.3475 | 0.1136 | 0.3468 | 0.0461 |
| `shuffled_kd_spec` | specialist | 0.3482 | 0.3243 | 0.1166 | 0.3338 | 0.0437 |
| `teacher_kd_z_spec` | specialist | 0.9290 | 0.3471 | 0.1215 | 0.3422 | 0.0501 |
| `teacher_kd_z_rel_spec` | specialist | 0.9604 | 0.3449 | 0.0844 | 0.3406 | 0.0492 |

`_spec` 是每族一个专用头(消除共享参数之间的干扰);`_rel` 是带成对项的读出头(其容量正对照在合成关系标签上 >0.95)。

### 4. 判定

- **结构信息能迁移。** 只读冻结基线硬码字 `q` 的专用头:`hard_ce_spec` held-out macro-F1 **0.1356**、`teacher_kd_spec`(只有 Teacher 软分布)**0.1136**,多数类 **0.0777**、度数查表 **0.0182**;pair-clean 子集排序相同。
- **迁移到的主要是无向骨架。** 方向打乱教师的目标给出 **0.1166**,与真实方向 **0.1136** 同带;但打乱教师的训练拟合只有 0.348(真实 0.928),所以只能读成「打乱目标已足以填满硬标签天花板」,不能读成「方向无用」。
- **量化不是瓶颈。** 同一专用头在连续 `z` 上 **0.1215**,在硬 `q` 上 **0.1136**。
- 共享头(5 族共用一个)held-out 只有 0.02-0.03、训练准确率 0.35-0.59,是欠拟合;判定只使用 `_spec`。
- 按 §C.5 停在接口可行性层,**不启动 Stage2/Stage3**;要继续必须以「无向骨架蒸馏」重新立题,并先补 pair-disjoint 数据。

### 5. 交付

`teacher_model.py` / `frozen_sid.py` / `student_head.py` / `kdistill.py` / `fit_teacher.py` / `gate.py` / `manifest.py`,
`tests/test_teacher_contract.py`、`tests/test_student_interface.py`(15 项,全部通过),
报告 `results/research/directed_graphlet_teacher/REPORT_TEACHER_DISTILL_INTERFACE.md`。
