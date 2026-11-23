# Issue #241 — VERDICT（§6.1 阻塞关闭）

- issue 正文 sha256 `ba50e87a914e212d894b6ca0a6f728b222d6af3a81e9569dec77d94920035c55`
- 清单 23 条（`RUN_LOG.md`）

## 逐条状态

| 编号 | 状态 | 证据 |
|---|---|---|
| C1 有向图（train.parquet、自环、映射、哈希） | **PASS** | `graph/metadata.json`：57,439 用户、293,202 可见有向边、自环 0、源 sha256 `80597e38…` |
| C2 三方隔离（ego / 目标边 / 独立评估边） | **PASS** | 目标边 29,536 不在 ego 图；评估边来自 5,744 个独立用户，其中 30,675 条不在可见图 |
| C3 1/2 跳有向 ego + 入出分离 GNN + 池化 + 32 维 | **PASS** | `teacher.py`，K1=8/8、K2=4、cap 32、T=32 |
| C4 显式训练目标（解码前驱/后继分布） | **PASS** | sampled softmax，29,536 目标边，`teacher/train_log.json` |
| C5 辅助验证（chain/fork/merge、degree-matched） | **部分** | 独立评估 + 两跳结构基线已做；角色分层未单独做，因为 C6 已判定停止 |
| **C6 超越度数/简单结构统计** | **FAIL** | 独立评估边 Recall@10 **0.0** vs 结构基线 **0.0245**；MRR 0.00058 vs 0.0127 |
| C7 冻结 Teacher ckpt / T / 分布 | **NOT RUN** | 门槛未过，按 issue 指令不产出 |
| C8–C12 对关系强度 S_G | **NOT RUN** | 依赖 Teacher 输出 |
| C13–C17 新 RQ-VAE | **NOT RUN** | 依赖 S_G |
| C18–C23 评估与 Stage3 | **NOT RUN** | 同上 |

**整体判定：C6 FAIL → 按 issue「先报告并停止正式 Stage2」执行，§6.1 关闭。**

## 阻塞点

Teacher 在**独立评估边**上 Recall@10 = 0.0000、MRR = 0.00058、中位排名 5,544/24,587，
而无训练的**两跳路径结构基线**在同批商品对上 Recall@10 = 0.0245、MRR = 0.0127 —— 差约 20 倍。
训练损失全程 5.61–5.77，而 65 类均匀分布的 CE 是 **4.17**，即始终劣于随机。

## 已排除的解释（均实测）

| 解释 | 实测证据 |
|---|---|
| 输出退化/塌缩 | 5 个随机根的 P_out 余弦 0.067、top-5 零重叠、熵 7.35–8.15（均匀 10.11）→ 分布是根特异的 |
| 训练不足 | 40 epoch / 4,640 步，损失从未低于均匀基线；瓶颈是信息而非步数 |
| 评估实现错误 | 结构基线用同一 `rank_of`、同一商品对、同一全词表排名，且它胜出 |
| 数据泄漏造成虚高 | 目标是留出边（不在 ego 图），评估边不在可见图中 |
| 特征标准化/设备 bug | 已修复三处（CPU/GPU 索引、池化宽度、设备）；修复后损失轨迹不变 |

## 根因

节点特征只有结构量（根标记、hop、入/出方向、入/出度），**度序列相同的两个根在模型里完全等价**，
因此 T_i 只能编码度数指纹，无法编码「我是哪个商品」；而下一跳购买主要由具体身份决定。
两跳结构基线赢，正是因为它直接使用了真实图路径。

## 若要继续，最小需要放宽什么（选项，供 owner 决定）

- **A（推荐）**：允许 Teacher 读取根节点的 **T5 语义嵌入**（与 RQ-VAE 同一 768 维输入）。
  不是纯 ID lookup，但把 Teacher 变成「结构 + 内容」模型，需要 owner 认可口径变化。
- **B**：允许可学习 per-item embedding 作为节点特征（真正的 ID lookup）。能预测后继，
  但与本 issue「避免纯 ID lookup 当结构表示」冲突。
- **C**：把 C6 判据从「预测具体后继」改为「上下文结构恢复」（共同前驱/后继角色），
  不再要求 Teacher 分辨具体商品。

## 本轮未做的事

第二阶段 S_G、RQ-VAE 候选、Stage3 **均未执行**（依赖未通过的 Teacher）。
Stage3 候选树 `TIGER_topo_distill_rq` 曾在本地建立（仅改 ARM_NAME + 加 `topo_loss.py` 与 4 处接线），
因门槛未过已删除，未进入 git，也未产生任何训练。

## 交付

`build_graph.py`、`teacher.py`、`evaluate_teacher.py`、`pairs.py`（第二阶段实现，未运行）、
`RUN_LOG.md`、`COMMENT_01.md`、`COMMENT_02.md`、`teacher/{train_log.json, teacher_eval.json}`、
`graph/metadata.json`（图哈希）。二进制（`teacher.pt`、`*.npy`、`*.npz`）按 §46/§92 不入库。
