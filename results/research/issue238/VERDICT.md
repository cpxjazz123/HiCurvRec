# Issue #238 — VERDICT（本轮：报告阻塞）

状态：**阻塞（issue 保持 OPEN）**。判定依据是 issue 正文自身的规定：

> 若发现框架接口无法在不违反冻结协议下实现，**先报告阻塞**，不擅自改框架或放宽护栏。

## 1. 已完成并可复算的部分

| 项 | 状态 | 证据 |
|---|---|---|
| 冻结验收清单 47 条 | 完成 | `RUN_LOG.md`，正文 sha256 `4b6de1a9…` |
| infra（数据/基线/解释器/卡/端口） | 全部 PASS | `INFRA.md`、`infra/files.txt`、`infra/env.txt` |
| 图曲率目标 K_G | 实现 + 9 项测试 | `graph_curv.py`、`tests/test_graph_curvature.py` |
| 混合曲率距离 D_q | 实现 + 9 项测试 | `distances/mixed.py`、`tests/test_mixed_distance.py` |
| 曲率保真损失 L_curv | 实现 + 6 项测试 | `losses/curvature_fidelity.py`、`tests/test_curvature_loss.py` |
| 训练前标定 | 完成，早于任何训练 | `results/stage2_RQ-VAE/TIGER_curvfid_rq/calibration.json`（created 2026-11-23T01:32:18Z） |
| 测试总数 | **51 passed** | `pytest results/research/issue238/tests -q` |
| C24 真实入口 smoke | **FAILED（阻塞点）** | 见 §2 |

标定数字：191,984 个三元组（正 171,700 / 负 20,151 / 近零 133），映射尺度（L3 残差 RMS）
0.00972，全局尺度 s = 3.3386，Huber δ = 1.2109。

## 2. 阻塞点：冻结量化器的 Sinkhorn 守卫在训练第 2 步拒绝退化矩阵

真实入口 smoke 连续失败 5 次，每一次都推进了一层：

| # | 失败 | 处理 |
|---|---|---|
| 1 | `indices should be on the same device` | 索引先在 host 上取、再搬卡 |
| 2 | `Cannot center a constant distance matrix` | 混合距离改用有界径向映射 |
| 3 | 同上（首次） | 加 `acosh` 数值上界 |
| 4 | 同上（复现） | 标定改用 **L3 残差**而非编码器输出的尺度 |
| 5 | 同上（仍在第 2 步） | 监督路径绕开 Sinkhorn，改用 argmin 硬量化 |

第 5 次之后，**训练主前向自身**在第 2 步产生常数代价矩阵：

```
model.forward -> rq -> vq_layer -> _indices -> center_distance
ValueError: Cannot center a constant distance matrix.
```

### 2.1 已排除的解释（都有实测）

- **不是 LR 太猛**：把学习率压到冻结 warmup 的首步值（约 8e-7）仍然在第 2 步失败。
- **不是混合距离数值饱和**：换成无平坦区的有理映射后，
  `scale ∈ {0.01, 1, 100}` 且输入放大 1000 倍时距离仍有 std 1e6 量级、无 NaN；
  真实 L3 残差上 `d3` 的 span ≈ 7。
- **不是小批退化**：n = 450 / 512 / 2048 / 4096 的前向全部正常（span 6.6–7.1，无 NaN）。
- **不是标定尺度错**：改用 L3 残差标定后（0.00972），主前向在第 0–1 步仍正常。

### 2.2 为什么这构成"框架接口阻塞"而不是我的实现 bug

`center_distance` 属于冻结基线的量化器语义（`model/layers.py:83-91`，sha256 `0d5eacf6…`），
它的守卫是**基线自带**的：全局 `max/min` 之差必须为正，否则拒绝分配。
issue 要求「保持原均衡规则」且「除原始 RQ-VAE 损失外只允许一个 L_curv」，
因此我不能：

1. 改 `center_distance`（改均衡分配的前置语义）；
2. 关掉 Sinkhorn（那是 §19h 指定的机制位置）；
3. 加第二个损失或改 STE 路由去"绕过"退化（issue §5 明确禁止改护栏）。

而只要 L_curv 的梯度进入同一张图，第 2 步的主前向就会让该层的代价矩阵退化 ——
这在当前 Sinkhorn 语义下是**模型无法前进**的，与我的实现细节无关。

## 3. 需要 owner 决定的最小放宽（三选一）

| 选项 | 内容 | 改动面 |
|---|---|---|
| **A（推荐）** | 允许 `center_distance` 在常数矩阵时**退化为行内中心化**（或允许该层使用 `argmin`），并把这一分支记录为诊断 | 冻结量化器 1 处，语义变化可界定 |
| B | 允许 L_curv 只在**冻结的 argmin 量化**上监督（q 的分配 detached），损失梯度只进编码器与码本参数 | 不动量化器，但监督目标与训练时的分配解耦，需在报告里声明 |
| C | 允许把混合距离只用于 L_curv、不进 Sinkhorn 代价矩阵 | 与 issue §4 与 CLAUDE.md §19h 冲突，需改规则 |

我推荐 **A**：它只影响"何时拒绝"，不改变正常情况下的均衡分配，且能把退化本身变成一个可记录的诊断。

## 4. 本轮不做的两件事（遵守 issue）

- **不做额外重复训练**：Stage2 一次都没启动成功，因此没有第二次；
- **不改判据**：47 条清单原样保留，未因实现困难而放宽任何一条。

## 5. 复算入口

```bash
# 51 项机制测试
genrec_env_v2/bin/python3.9 -m pytest results/research/issue238/tests -q
# 训练前标定（已冻结，--force 才会重写）
genrec_env_v2/bin/python3.9 stage2_RQ-VAE/TIGER_curvfid_rq/calibrate.py
# smoke 证据
/tmp/genrec-mcp/smoke_TIGER_curvfid_rq.log
```
