# #181 现状：AF3 结构教师已实现（toy 图验收中）

## 已完成

- 新树 `stage2_RQ-VAE/TIGER_af3_structdistill_rq`（`ARM_NAME="TIGER_af3_structdistill_rq"`、`ARM_VARIANT="AF3SD"`），自 #180 树复制（其一级几何选码与数值验收已完成，符合 Issue "待 #180 可用后以其代码为只读起点"）。
- `model/af3_teacher.py`：
  - `af3_values`：`AF3(i,j) = 4 − deg(i) − deg(j) + 3m_ij`；
  - `quantiles`：TRAIN 图 AF3 的 q05/q95（冻结进 manifest）；
  - `weights`：`W_ij = 1 + clip((AF3−q05)/(q95−q05), 0, 1)`，**观测边权重恒 ≥1、桥梁边绝不为 0**（不把负 AF3 当负样本、不删边），对称化，未观测边为 0；稀疏 COO；
  - `normalized`：`P = D^{-1/2} W D^{-1/2}`，**零度节点保留全零行**（Issue 要求的显式退化规则）；
  - `teacher`：`T = (P + P²)/2`；
  - `build`：返回教师与要绑进 checkpoint 的 manifest（物品数、边数、q05/q95、规则原文）。
- 测试 `tests/test_af3_teacher.py`（6 项）：4-环上 AF3 手算恒等、q05/q95、桥梁边权重严格 ≥1 且未观测边为 0 且对称、教师对称且与定义式手算一致、孤立节点零行、确定性。

## 尚待

学生侧 STE 读出 `z_hat = z + stopgrad(q − z)`、候选集合采样（16 一跳 + 16 两跳 + 32 全局）、逐行 zscore+softmax 的 KL、λ* 探针标定、SD_OFF/GEOMETRY_OFF 两个开关的逐位一致、梯度与 DDP 检查、一 epoch 冒烟，然后 Stage2/Stage3 与结构保真报告。
