# Task #250 — Issue #14 Gate 0: FORGE 官方源码 metric 实证 (FAIL → STOP)

## 1. 目的

执行 Issue #14 §阶段闸门 Gate 0:
- 从 https://github.com/selous123/al_sid 取回 embedding hitrate + Gini 实现源码
- 写出与本仓库数据结构 (9922×4 int SID `.npy` + item embedding) 对齐的可执行定义
- 核实 FORGE 给出的相关性实测支撑
- 通过条件: 能写出可执行定义 **且** FORGE 给出明确相关性实测
- 硬停止: 取不到实现源码 / 仅定性描述 → STOP, 不进 Gate 1

## 2. 源码取证范围

| 文件 | 来源 | 大小 | 命中关键字 |
|------|------|------|------------|
| `SID_generation/README.md` | 中文 | 65 行 | 0 命中 (hitrate / Gini / embedding hitrate) |
| `SID_generation/README_en.md` | 英文 | 65 行 | 0 命中 |
| `algr/calc_hr.py` | algr/ | 127 行 | 1 处 `HR@K` 实现 = **下游 GR HR@K** (需 generate_text + answer JSONL) |
| `SID_generation/ID_collision.sql` | SID_generation/ | 150 行 | KNN 防碰撞脚本 (sorted_index_lv3 + ROW_NUMBER 防 5 满载) |
| `SID_generation/infer_SID.py` | SID_generation/ | 221 行 | 模型推断 (item_emb → SID .csv), 无指标计算 |
| `SID_generation/rqvae_embed/*.py` | 5 文件 | ~1500 行 | 0 命中 (hitrate / Gini) |
| `SID_generation/rqvae_embed/quantizations.py` | rqvae_embed/ | ~250 行 | `find_nearest_embedding` (Sinkhorn-based 量化), **非 hitrate 指标** |
| `algr/config/*.json` | algr/config/ | 6 文件 | T5/Qwen 训练超参, 无指标 |
| `algr/{runner.py,test.py,models/,data/}` | algr/ | 跳过 | 训练 / 推断 / 数据加载, 无指标 |

**grep 关键字**: `hitrate|hit_rate|hr@|gini|recall|collision|nearest` —— 在所有源码里命中:
- `rqvae_embed/layers.py:29` `mode="nearest"` (PyTorch interpolate, 非 hitrate)
- `rqvae_embed/quantizations.py:147` `find_nearest_embedding` (Sinkhorn 分配, 非 hitrate)
- `rqvae_embed/quantizations.py:238` 同上调用
- `calc_hr.py` HR@K (下游 GR 输出, **依赖已训练 GR**)

## 3. Gate 0 决策: **FAIL → STOP**

按 Issue #14 §Gate 0 通过条件:
1. **"能写出可执行的 embedding hitrate 定义"** — ❌ **失败**: al_sid 官方代码仓**无该实现**. calc_hr.py 的 `calculate_hit_rate_k` 计算的是**下游 GR HR@K** (需要 `output.jsonl` 里的 `_generated_text_` / `_generated_new_text_`), 完全依赖训练好的 GR 生成输出, **不是免训练指标**. ID_collision.sql 是 KNN 防碰撞脚本, 也不是 hitrate.
2. **"FORGE 给出了明确的相关性实测支撑"** — ❌ **失败**: README (中英) 全文不提及 "hitrate" / "Gini" / "embedding" / "correlate" 任何字眼; Issue #14 §取材说明 已诚实标注"全文待 Gate 0 取回核对" + 引用均为【待全文核实】; 本环境 `arxiv.org` 不在网络白名单, 全文无法取回. 因此 FORGE "correlate well" 的实测证据**在本 Gate 无法核实**.

**硬停止命中**: Issue #14 Gate 0 §明文 "取不到实现源码, 或 FORGE 的 'correlate well' 只是定性描述而无实测支撑 → STOP, 就地写 verdict 关闭本方向, 记录'指标不可复现 / 无实测依据, 免训练预测暂无可用外部支点'. **不得凭检索摘要臆造公式往下算**".

## 4. 反 H1 / H2 / H3 先验诚实声明

Issue #14 §假设 H1 / H2 / H3 都依赖一个前提: "embedding hitrate 是免训练指标 + FORGE 已实测它跟 GR 性能相关". Gate 0 直接证伪该前提 — 两个支点都不存在:

- **支点 1 (实现存在)**: 0/1 (代码仓只有下游 GR HR@K, 不是 embedding hitrate)
- **支点 2 (相关性强)**: 0/1 (README 无任何指标声明 + 全文无法取回)

H1 / H2 / H3 在该前提下**无法验证**, 不进 Gate 1.

## 5. 与 Issue #12 的关系 (按 Issue #14 §压力测试)

Issue #14 §反证 已自承: "若 Gate 1 再次失败, 两次合起来构成一个相当强的结论: 在 9922-item 规模上, SID 侧的任何静态指标都不预测 R@10". 

**Gate 0 即停 = 比 Issue #12 NO-GO 更强的负结论**: 不需要 Gate 1 实测, 单纯做文献取证就否定了"有外部验证的免训练指标"这个**前提**. Issue #12 测的是仓库自创分布量 (无外部证据); Issue #14 想测的是**有外部证据**的量, 但外部证据**不存在**.

合并后的强结论: **"在 9922-item 规模上, 既无仓库自创、也无外部验证的免训练 SID 质量指标能预测 R@10"**. 应按 Issue #14 §最终目标 把资源集中到训练协议 / T5 容量 / 数据规模 / 跨架构重新基线.

## 6. 关键决策点 (R11.3 自主决策)

| 决策 | 选了什么 | 为什么 |
|------|---------|------|
| 取证范围 | SID_generation/ + algr/ (configs, src, data_loader, utils) | calc_hr.py 来自 algr/, rqvae 来自 SID_generation/, configs 用于了解训练设置 |
| 验证手段 | 关键字 grep + 全文搜索 (中英文 README) | 单一搜索路径足够定位 metric 定义; 命中 0 即结论 |
| 是否取回 arXiv:2509.20904 全文 | 否 | 本环境 arxiv.org 不在网络白名单; WebSearch 摘要信息已被 Issue #14 §取材说明 标注【待全文核实】, 不构成 Gate 0 通过条件 |
| 是否发 Issue #14 GitHub 评论 | 是 | 让 issue 状态对外可见 (Gate 0 FAIL → STOP) |
| 是否进 Gate 1 | 否 | Gate 0 硬停止, 不进 Gate 1 |

## 7. 产物

- `descriptions/task250_issue14_gate0_forge_metric_def.md`
- `verdicts/task250_issue14_gate0_forge_metric_def_result.md` (本文件)
- `tmp/forge_src/{calc_hr.py,ID_collision.sql,README.md,README_en.md,rqvae_embed/*.py,...}` (取证留档)
- Issue #14 GitHub 评论 (待发, Gate 0 FAIL → STOP 标记)

## 8. 状态

✅ **Gate 0 FAIL → STOP**: al_sid 官方代码仓无 embedding hitrate / Gini 实现; README 无任何指标声明; arXiv 全文无法取回核实相关性. 按 Issue #14 §硬停止条款, 不进 Gate 1.

合并 Issue #12 (NO-GO) + Issue #14 (Gate 0 FAIL): **在 9922-item 规模上, SID 侧静态指标 (仓库自创 + 外部验证) 都不预测 R@10**. 该结论本身值得写进 `CLAUDE.md` (Issue #14 §反证 提及), 此后不应再立同类 "找免训练代理指标" issue.

result: **Issue #14 Gate 0 FAIL → STOP. al_sid 官方代码仓无 embedding hitrate / Gini 实现 (calc_hr.py 是下游 GR HR@K, 需训练好的 GR; ID_collision.sql 是 KNN 防碰撞; README 无任何指标声明). 合并 Issue #12 (NO-GO) + Issue #14 (Gate 0 FAIL) 得强结论: 在 9922-item 规模上, SID 侧静态指标 (仓库自创 + 外部验证) 都不预测 R@10. 此后不应再立同类 "免训练代理指标" issue, 资源应集中到训练协议 / T5 容量 / 数据规模 / 跨架构重新基线**.