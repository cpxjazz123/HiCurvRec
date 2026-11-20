## 结论:之前不是"图里没有信号",是 **40 次真实更新**

`graph_teacher2.py::train_variant` 在 24 个子批次累积后只调用一次 `optimizer.step()`,外层 40 次
⇒ 40 次参数更新。本轮把一步定义为 `zero_grad → (fwd,bwd)×accum → clip → step`,并分开统计
epoch / microbatch / step(见 `graph_teacher3.py`,13 项回归测试)。

### 三个假设的判决

| 假设 | 判定 | 证据 |
|---|---|---|
| 优化不足 | **主因** | 同架构 40 步 test AUC **0.5039**;600–725 步 **0.678 / 0.670** |
| 表达受限 | 否 | 固定 1,024 对过拟合:B **1.000**、C **0.9934** |
| 监督有错 | 部分(方向) | #227 的对称 head 无法表达 `s(i→j)≠s(j→i)`;联合损失下 B 崩到 0.508,C 不受影响 |
| 无泛化信号 | 否(边) | C−A **+0.0395 [0.0372,0.0418]**、C−D **+0.0343 [0.0322,0.0363]** |
| 方向可学 | **未成立** | 全部变体 balanced acc 0.497–0.511,多数类 0.5016 |

### 关键数字(46,586 测试对,双 seed)

| 变体 | test AUC(226/227) |
|---|---|
| A 语义投影 | 0.7108 / 0.7089 |
| B #227 GNN(修正步数) | 0.6780 / 0.6699 |
| **C 残差 GNN** | **0.7473 / 0.7475** |
| D 度数保持重接线 | 0.7141 / 0.7120 |
| E 同架构无消息 | 0.7201 / 0.7191 |
| 40-step 对照 | 0.5039 |

重接线:192,528 边、1,474,892 次交换、in/out 度 L1 = 0、自环 0;目标对拦截 37,661 次。
数据契约四类违规全 0;split SHA `734fe513…`。

**未成立**:方向;orbital role 仅凭商品表示预测(探针 CE 3.549 vs 常数 3.516)。
**未做**:Stage2/Stage3/RQ-VAE 蒸馏。

产物:`REPORT_GRAPH_TEACHER_OPTIMIZATION.md`、`teacher/opt/graph_teacher3_report.json`、
`tests/test_teacher_optimization.py`、`finalize_teacher3.py`。全部 SHA 在报告与 MANIFEST 中。
