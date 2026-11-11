# #176 DUL-S 派发记录

- 实现 `a8eb4e39`：`L_unique` 分母由 `|P|` 改为 **batch 大小 B**；`β` 不再用论文的 2.0，而是 **epoch 1 的 mean(L_RQ)/mean(L_unique)**，结束后冻结并写日志（无质量时记 `BETA_UNMEASURABLE` 并保留论文值）；`m=0.9` 不变；逐 epoch 诊断含 `|P|`、`cos>m` 对数、`β`、`β·L_unique/L_RQ` 比值。
- 修复 `3c415649`：`unique_beta` 关键字被两个补丁各插一次 → `SyntaxError`，冒烟在 import 阶段即拒绝；去重后通过。
- 测试 `tests/test_duls.py` **4 passed**（归一化分母是 B 而非 |P|、零对退化、β 初值为论文值且一旦设定即冻结、梯度仍到 z）。
- 派发：worker 1375739；真实入口一 epoch 冒烟 **SMOKE_PASS**（`dead_terms=[]`、`nonfinite_terms=[]`、梯度范数全正、导出唯一），四卡已启动。配方 seed 42、3000 epoch、全局 batch 4096、四卡 50202。
