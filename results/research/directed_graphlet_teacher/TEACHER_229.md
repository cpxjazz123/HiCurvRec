# Teacher 模型:入口、能力与证据(#229)

本页是 #229 交付的**唯一入口**。所有数字都有机器可读产物;所有“未成立”都逐条声明。

## 0. 一个类拿到全部能力

```python
from behavior_graph_teacher import BehaviorGraphTeacher

teacher = BehaviorGraphTeacher.load()      # bundle + 训练好的图模型
teacher.verify()                           # 两件产物的 SHA/参数摘要
teacher.recognise(matrix)                  # 识别
state = teacher.encode(features, adjacency)
teacher.score_pairs(state, positives, negatives)   # 预测
teacher.targets_for(item_ids)              # 商品级监督行
teacher.structure_loss(student_vectors, item_ids=batch)
teacher.describe()                         # 每项能力的 PASS / NOT_ESTABLISHED
```

## 1. 加载与使用

```python
from structure_teacher_bundle import load_structure_teacher

teacher = load_structure_teacher(
    "results/research/directed_graphlet_teacher/teacher/structure/bundle")

teacher.verify()                          # 源文件 SHA + 载体参数摘要
teacher.recognise(matrix)                 # 诱导邻接 -> {family, key, orbits}
teacher.targets_for([12, 900, 24001])     # 图导出的商品级监督行
teacher.predict_targets(embeddings)       # 768 维载体(无图回退)
metrics, loss = teacher.structure_loss(student_vectors, role_logits=logits)
```

`structure_loss` 既接受**全体商品**的 `(24,587, d)` 表示,也接受**一个 mini-batch** 的
`(len(item_ids), d)` 表示(端点不在批内的 pair 自动丢弃);权重来自 v3 契约:
`relation 1.0 / neighbour_relation 0.5 / neighbour_orbit 0.5 / neighbour_family 0.0 /
own_orbit 0.0 / own_subtype 0.0`。

## 2. 能力矩阵(全部可复核)

| 能力 | 状态 | 数字 | 证据 |
|---|---|---|---|
| 识别:给诱导邻接的 family/type/orbit | **成立** | 五 family 全对;subtype 0.7121 / orbit 0.6101 | `REPORT_STRUCTURE_TEACHER.md` §3.1、`head_pretrained_directed.json` |
| 多角色 | **成立** | 2.08 角色/商品,46.1% 多角色 | `structure_targets_v3.npz` |
| 预测:边存在性 | **成立** | C−A **+0.0395** [0.0372,0.0418];C−D **+0.0343** [0.0322,0.0363],双 seed | `REPORT_GRAPH_TEACHER_OPTIMIZATION.md` |
| 有用环境 / 无证据环境 | **成立** | 闭合/开放、四个度数段 C 均高于 A 与 D;跨连通分量(126 对)无证据 | `REPORT_STRUCTURE_ENVIRONMENT.md` |
| 非学习启发式对照 | **成立** | CN 0.5455 / AA 0.5488 / RA 0.5525 / Jaccard 0.5555 / 启发式栈 0.5734,Teacher **0.7473**;C−栈 **+0.1739** [0.1701,0.1781] | `REPORT_STRUCTURE_HEURISTICS.md` |
| 商品级可承载(32 维) | **成立** | 真实拓扑解码 **0.889** vs 重接线 **0.398** | `REPORT_STRUCTURE_TEACHER_SID_CAPACITY.md` |
| 邻域多角色可承载 | **成立** | CE −8.2%,秩相关 0.593,3,581 少数角色商品 | `REPORT_STRUCTURE_TEACHER_V3.md` |
| 监督可交付 | **成立** | 4 维 student 40 步损失下降;bundle 测试 | `tests/test_structure_teacher_bundle.py` |
| 方向 | **未成立,已声明** | 0.5145 vs 多数类 0.5016 / 度数规则 0.5088 | `verify_conformance_229.py` |
| 少数角色识别 | **未成立,已声明** | 少数角色 top-1 ≈ 0.03 | 同上 |
| 联合训练能救角色识别吗 | **否** | orbit CE 4.3015 vs 常数 3.5031 | `REPORT_STRUCTURE_TEACHER_JOINT.md` |
| SID / Recall 改善 | **不在范围** | 契约不训练 RQ-VAE/Stage2/Stage3 | 契约 `not_established` |

## 3. 产物

| 产物 | 路径 |
|---|---|
| 冻结 bundle | `teacher/structure/bundle/`(`.pth` + contract + manifest) |
| 商品级目标表 v3 | `teacher/structure/v3/structure_targets_v3.npz` |
| 目标契约 v3 | `teacher/structure/v3/teacher_structure_contract_v3.json` |
| 联合训练模型(可加载,`joint_teacher_model.py`) | `teacher/structure/joint/teacher_joint_v1.pth`(+ manifest 参数摘要) |
| 边预测 Teacher(#228) | `teacher/opt/`(报告与 checkpoint) |
| 符合性审计 | `verify_conformance_229.py` → `conformance_229.json` |
| 全部报告 | `REPORT_STRUCTURE_TEACHER*.md`、`REPORT_GRAPH_TEACHER_OPTIMIZATION.md` |

## 4. 边界(硬)

- RQ-VAE Encoder 输入仍是 **768 维 Stage1 向量**;不含邻接、Teacher hidden、实例标签。
- 冻结 Teacher 只在 **训练损失**中使用;SID 推理协议不变。
- 邻接只在**构建目标表**时读取;推理阶段不需要图。
- 本交付不训练 RQ-VAE/Stage2/Stage3,不宣称任何 SID 或 Recall 效果。

## 5. 复现与验收

```bash
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
cd results/research/directed_graphlet_teacher

$PY verify_conformance_229.py        # 退出码 0 + conformance_229.json
$PY -m pytest tests/ -q              # 94 passed
$PY sid_capacity_32d.py              # 32 维容量(约 15 秒)
$PY train_teacher.py                 # 联合训练对照(约 30 秒)
$PY structure_teacher_bundle.py      # 重建 bundle(如目标表更新)
$PY structure_heuristics.py           # 非学习启发式对照(约 40 秒)
```

## 验收

```bash
$PY accept_teacher_229.py   # 9 项检查,退出码 0
```
