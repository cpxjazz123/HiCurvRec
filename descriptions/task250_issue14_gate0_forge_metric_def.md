# Task #250 — Issue #14 Gate 0: 取回 FORGE 官方源码 + 写出 embedding hitrate 可执行定义 (STOP)

## 来源
- GitHub Issue #14 (2026-07-28): [Lit-triggered] FORGE 免 GR 训练的 SID 质量预测指标
- 主文献: Fu K. et al. *FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets*, arXiv:2509.20904, KDD 2026
- 官方代码仓: https://github.com/selous123/al_sid (MIT)

## 任务目的

Issue #14 §阶段闸门 Gate 0 (零 GPU, 分钟级代码核查):
- 从 https://github.com/selous123/al_sid 取回 embedding hitrate + Gini 实现源码
- 写出与本仓库数据结构对齐的可执行定义
- 核实 FORGE 给出的相关性实测支撑强度
- 通过条件: 能写出可执行定义 + FORGE 给明确相关性实测
- 硬停止: 取不到实现源码 / 仅定性描述 → STOP

## 决策 (R11.3 自主决策)
- 拉取 al_sid 全代码仓关键目录 (SID_generation/, algr/, configs/)
- grep "hitrate|hit_rate|hr@|gini|recall|collision|nearest" 找指标实现
- 比对 README (中英文) 是否声明这两个指标
- 结论: 直接写 STOP verdict

## 产物
- verdicts/task250_issue14_gate0_forge_metric_def_result.md
- Issue #14 GitHub 评论 (Gate 0 FAIL → STOP)

## 状态
零 GPU, ~10 min wall.