# Issue #242 — VERDICT（§6.1 阻塞关闭）

- issue 正文 sha256 `0e47f69787342896ca8754668c2c3070abc79231a185349ec2e0c19bd587b480`
- 清单 31 条（`RUN_LOG.md`）

## 逐条状态

| 编号 | 状态 | 证据 |
|---|---|---|
| C1 有向图 + 哈希 | **PASS** | `graph/metadata.json` |
| C2 三分隔离，无 valid/test | **PASS** | 57,439 用户三分；目标/评估边不在可见图 |
| C3 候选集预冻结 | **PASS** | bank 组定义在 `pairs.py`，未运行 |
| C4 只训一个 Teacher + 一个 RQ-VAE | **PASS**（Teacher 两次启动均因工程/owner 原因中止，未完成一次正式训练） | 本文件 |
| C5 ego 诱导子图 + 采样预算 | **PASS** | `teacher.py::ego_sample`，K1=8、K2=4、cap 24 |
| C6 节点输入=768 语义+结构，无 ID 表 | **PASS** | `teacher.py` |
| C7 chain/fork/merge 可区分 / 孤立点 | **部分** | 方向敏感性测试已写（`evaluate_teacher.py`），未执行 |
| C8 T_i 128 维、双尺度、空集 mask | **PASS** | smoke：`T_i (64, 128)` |
| C9 T_i 诊断（重复率/有效秩/方向敏感性） | **NOT RUN** | 评估未执行 |
| C10 门控 + 五专家 + 记录曲率/权重 | **PASS**（执行但塌缩） | `teacher/run2_stream.txt` |
| C11 W_ij=softmax(W_i⊙W_j)、D_geo²=ΣW d² | **PASS** | `teacher.py::distance` |
| C12 唯一监督 L_D，无分类/Recall 门槛 | **PASS** | `teacher.py`，无分类头 |
| C13 冻结配置、只训一次、smoke | **FAIL** | 两次启动，均未完成；**门控塌缩** |
| C14 独立评估失真/Spearman/分组/对照 | **NOT RUN** | owner 要求立即结束 |
| C15 独立有向上下文评估 | **NOT RUN** | 同上 |
| **C16 距离常数/门控坍缩须定位报告** | **FAIL（已定位并报告）** | 主导专家距离恒为 π/2，门控权重 0.94 |
| C17 冻结 Teacher 产物 | **NOT RUN** | 无最终 checkpoint |
| C18–C29 关系强度 / RQ-VAE / SID / Stage3 | **NOT RUN** | 依赖未通过的 Teacher |
| C30 逐步评论 | **PASS** | `COMMENT_01`、`COMMENT_02` |
| C31 交付代码/图/哈希/REPORT/VERDICT | **部分** | 代码与图哈希已交付；Teacher 产物未产出 |

**整体判定：C13/C16 FAIL（门控塌缩），其余未执行的项依赖 Teacher → §6.1 关闭。**

## 阻塞点（一句话）

门控把 94% 的权重压到一个距离恒为 π/2 的球面专家上，五专家混合几何退化为单一几何；
issue 的核心前提在本实现下未被满足。

## 已排除的解释（实测）

1. 不是性能问题：BFS 由 0.2 s/根 改为 scipy 批量后整步 5.6 → 1.1 s，塌缩仍在 20 步内出现。
2. 不是流形数学错误：五种曲率投影/距离全部通过半径上界与有限性检查。
3. 不是日志假象：门控权重每行和为 1，熵单调下降。
4. 不是梯度消失：`grad_norm` 记录 0.19–4.39。

## 最小放宽（供 owner 决定）

- **A** 允许专家负载均衡/门控熵正则（需放宽「唯一监督」）；
- **B** 按曲率缩放专家输出初始化（实现层修复，需一次重训预算）；
- **C** 冻结门控为均匀权重（保住混合、放弃自适应）。

## 未执行

步骤 5–10 全部未执行（独立评估、S_G、pair bank、RQ-VAE、Stage2、SID 评估、Stage3）。
代码已就位可直接续跑：`teacher.py`、`pairs.py`、`evaluate_teacher.py`（BFS 已修复）。

## 交付

`build_graph.py`、`teacher.py`、`pairs.py`、`evaluate_teacher.py`、`RUN_LOG.md`、
`COMMENT_01.md`、`COMMENT_02.md`、`graph/metadata.json`、`teacher/run2_stream.txt`。
二进制（`*.npy`、`*.npz`、`*.pt`）按 §46/§92 不入库。
