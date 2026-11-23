## 节点 1：冻结验收清单（30 条）+ 需要训练

- 正文快照 `results/research/issue243/ISSUE_BODY.md`，sha256 **`b78cf4d31d2894e162c9e0977c5d2def4d168fb99412f9949bd379940bec82df`**
- 清单 30 条（C1–C30）见 `results/research/issue243/RUN_LOG.md`
- 单 Agent 执行，按 skill 的七项强制机制：数学契约 → 参考实现与已知答案 → 生产代码 → 变异测试 → 分模块验证 → 训练 → 结论复核

**训练判定：需要。** 依据：步骤 3「真入口多步 smoke 通过后一次正式 Teacher 训练」、
步骤 6「通过后一次 3000 epoch Stage2」、步骤 7「达准入后 Stage3 一次」。基线 TIGER 不重训。

### #242 的两个已确认错误（本 issue 要修的）

1. **双曲距离多乘 2**：`dist_hyperbolic` 用了 `(2/√c)·arcosh(...)`。标准 Poincaré 球（曲率 −c、半径 1/√c）
   的距离是 `(1/√c)·arcosh(1 + 2c‖u−v‖²/((1−c‖u‖²)(1−c‖v‖²)))`。
2. **球面专家不是 κ-GCN**：`Expert` 是普通欧氏 GNN + 末端投影，`project_spherical` 用 `normalize`
   会把零向量静默变成「某个球面点」，而零向量不在球面上；`π/2` 的来源需要逐点核查而非均值断言。

第 2 条正是 #242 门控塌缩的候选根因，本轮先修实现再谈机制。

*（`results/research/issue243/COMMENT_01.md`）*
