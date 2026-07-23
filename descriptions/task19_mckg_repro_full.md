# Task #77 — MCKG 主实验完整复现方案（Phase 0-5）

> **任务目的**: 复现 MCKG 论文（RecSys '23, arXiv:2308.15244）从评估协议对齐 → 三数据集准备 → κ-Stereographic 多空间模型实现 → 主实验 → 消融的完整方案；验证上一轮 25x/2.7x 差距主要源于评估协议错配，并达到论文 Table 3-6 的目标数值

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #75（kgat_tf216 env + 100 epoch 训练，Recall@20=0.03271，Hit@20=0.23050）和 Task #76（leave-one-out 协议核实 + 三层 GCN 重跑，**正在跑 GPU 0/1 + watchdog 部署**）的发现：

- **关键洞见**：Task #75 用 KGAT 仓库自带的 full-ranking + multi-test 评估协议（test_user_dict 全物品排序，Recall = hit_count/用户测试物品总数），与 MCKG 论文 4.1.3 节规定的 **1+100 leave-one-out 协议**（HR = 测试正样本在 Top-K 的用户比例）口径不一致
- **预期**（用户在 /loop 反馈中已确认）：协议修正后 KGAT HR@20 应从 0.23 上升至 0.55-0.65 区间，对齐 MCKG 论文 Table 3 KGAT 行 HR@20=0.614
- **后续目标**：在协议对齐基础上，复现 MCKG 论文的多几何空间（κ-Stereographic）模型在 LastFM/MovieLens-1M/Book-Crossing 三数据集上的主实验与消融

---

## 2. 实验设计（按论文 Section 4 + Section 3）

### Phase 0 ★ 最优先：评估协议对齐（解释此前 25x/2.7x 差距）

**变量**: KGAT 评估函数从 full-ranking + multi-test 改成 1+100 leave-one-out
**保持不变**: Task #75 已训好的模型 embedding + 数据切分（KGAT 仓库自带 last-fm 切分）

**验收**:
- ✅ 协议修正后，KGAT LastFM HR@20 落入 **0.614±0.05** → Phase 0 通过（论文 Table 3 KGAT 行）
- ⚠️ 0.40-0.55：部分对齐，检查 KGAT 论文其他差异（epoch 数、batch_size 等）
- ❌ <0.40：仍有根本性问题，回退查 KGAT 复现深度

**现状**: Task #76 已在 GPU 0/1 跑 1 层 / 3 层 GCN 训练 + watchdog 跑 leave-one-out eval，等结果出来直接判定 Phase 0

### Phase 1：数据准备（三数据集统计核验）

**变量**: 数据集 from KGAT/RippleNet 仓库 → 转换到统一格式
**保持不变**: KGAT 仓库自带的 last-fm 数据（已 OK），新增 MovieLens-1M + Book-Crossing

**目标统计量**（论文 Table 2）:

| 指标 | ML-1M | LastFM | Book-Crossing |
|---|---|---|---|
| Users / Items | 6,036 / 2,347 | 1,872 / 3,846 | 17,860 / 14,910 |
| Interactions | 753,772 | 42,346 | 139,746 |
| KG Entities / Relations | 16,954 / 32 | 9,366 / 60 | 25,787 / 18 |
| KG triples | 20,195 | 15,518 | 60,787 |

**判定**:
- ✅ 误差 <1% 通过
- ⚠️ 1-15% 记录差异继续
- ❌ >15% 停止排查数据源

**优先顺序**: LastFM（最小先跑通，KGAT 仓库自带）→ ML-1M → Book-Crossing

### Phase 2：MCKG 模型实现（按论文 Section 3 逐模块）

#### 2.1 κ-Stereographic 统一空间（公式 5-7 + Table 1）
- 实现 U^n_κ 基础算子：⊕κ（加法）、⊗κ（乘法）、⊖κ（拼接）、⊙κ（点积）、d_κ（测地距离）、exp/log map
- tan_κ / tan⁻¹_κ 按公式 6-7 分 κ>0 / κ=0 / κ<0 三段实现
- **κ 为可训练参数**，每个子空间独立一个 κ
- 单元测试对照 geoopt 的 Stereographic 流形验证自实现算子

#### 2.2 高阶信息提取（公式 8-14）
- 初始 embedding 在欧氏空间，经 exp^κ_o 映射入统一空间（公式 8）
- 关系注意力 c^r_u = e_u ⊙κ e_r（公式 9），softmax 归一后聚合邻居（公式 10）
- **聚合器选 GCN Aggregator（公式 11）**——消融 Table 4 显示 GCN-g 全面最优
- 层组合：e*_v = e^(0) ⊕κ e^(1) ⊕κ ...（公式 14）；用户只用 e^(0)

#### 2.3 多空间融合（公式 15-19）
- M=3 个子空间，各自独立参数与独立 κ
- 融合：子空间均值 e*g → 与各子空间 embedding ⊖κ 拼接更新（公式 15）
- 全局距离 = 注意力加权的各子空间测地距离之和（公式 16-19）

#### 2.4 几何感知 margin 损失（公式 20-21）
- margin ranking loss：L = max(dist²(u,i) − dist²(u,j) + mg, 0)
- **mg = σ( dist(u,i) / (dist(u,o)+dist(i,o)) ) + c**（公式 21）
- 注意：HICF 公式 22 方向相反且效果更差

### Phase 3：训练配置（论文 4.1.4）

| 配置项 | 值 |
|---|---|
| train/test 切分 | 70% / 30% 随机 |
| 邻居采样 size / hop | ML-1M: 8/3; Book-Crossing: 8/3; LastFM: 4/3 |
| 子空间数 M | 3（默认） |
| 聚合深度（实际最优） | ML-1M: 2; LastFM: 1; Book-Crossing: 2（Table 5） |
| 早停 | HR/NDCG 连续 20 epoch 不升 |
| latent dim d | 扫 {8, 16, 32}（Figure 3） |

⚠️ 论文自身矛盾点：4.1.4 说 hop=3，但 Table 5 显示 depth=3 时"rapid collapse"、最优为 1-2。复现时以 Table 5 的最优深度为准，hop=3 作对照记录。

### Phase 4：主实验目标数值（论文 Table 3 完整对照）

**LastFM**:
| 方法 | H@10 | H@20 | N@10 | N@20 |
|---|---|---|---|---|
| KGAT | 0.571 | 0.614 | 0.364 | 0.377 |
| **MCKG（目标）** | **0.635** | **0.691** | **0.405** | **0.432** |

**MovieLens-1M**:
| KGAT | 0.615 | 0.778 | 0.394 | 0.407 |
| **MCKG（目标）** | **0.645** | **0.802** | **0.401** | **0.418** |

**Book-Crossing**:
| KGAT | 0.379 | 0.551 | 0.309 | 0.356 |
| **MCKG（目标）** | **0.402** | **0.603** | **0.348** | **0.392** |

**验收**:
- ✅ 每项绝对差 ≤0.03
- ⚠️ 0.03-0.08：检查聚合器/margin/深度配置
- ❌ >0.08 或低于 KGAT baseline：回退检查 κ 训练稳定性

### Phase 5：消融实验（验证实现正确性的第二道保险）

**5.1 margin 策略（Table 4, GCN 聚合器）**: 必须复现 **-g > -c > -h** 顺序
**5.2 聚合深度（Table 5）**: depth ∈ {1, 2, 3}，复现最优 depth 对齐
**5.3 子空间数量（Table 6）**: M ∈ {1, 2, 3, 4}，复现 M=1→2 明显提升、M≥2 趋平

---

## 3. 决策触发（Phase 0 协议对齐部分）

| Phase 0 协议修正结果 | KGAT HR@20 | 决策 |
|---|---|---|
| HR@20 ∈ [0.55, 0.68] | 对齐 MCKG 论文 Table 3 KGAT=0.614 ±0.05 | ✅ 协议对齐，进入 Phase 1 数据准备 |
| HR@20 ∈ [0.40, 0.55] | 部分对齐 | ⚠️ 查 KGAT 论文其他差异（epoch/batch） |
| HR@20 < 0.40 | 仍有根本性问题 | ❌ 回退查 KGAT 复现深度 |

---

## 4. 预算（按论文 + 工程估算）

| 阶段 | 内容 | 估算 |
|---|---|---|
| Phase 0 | Task #76 watchdog 已自动跑 leave-one-out eval | 已进行中（~5 min after 训练完成） |
| Phase 1 | 三数据集统计核验 | 0.5-1 天 |
| Phase 2 | MCKG 模型实现（算子→聚合→融合→margin） | 3-5 天 |
| Phase 3-4 | LastFM 主实验对齐 | 1-2 天 |
| Phase 4 扩展 | ML-1M + Book-Crossing 主实验 | 1-2 天 |
| Phase 5 | 三组消融 | 1-2 天 |
| **合计** |  | **约 7-12 天** |

---

## 5. 风险与缓解

**风险 1**: κ-Stereographic 算子数值稳定性（κ 过零点、边界裁剪）→ **缓解**: 每个算子写单元测试对照 geoopt 包验证
**风险 2**: latent dim 默认值 + 优化器细节论文未明说 → **缓解**: 小规模扫参 {8, 16, 32}，记录偏差来源
**风险 3**: Phase 0 协议不对齐导致后续所有数值对照无意义 → **缓解**: Phase 0 必须最先做（Task #76 已在跑）
**风险 4**: Task #76 后台训练若与新任务 GPU 资源冲突 → **缓解**: GPU 2/3 空闲足够；Task #76 跑完自然归档

---

## 6. 完成度跟踪

### Phase 0: 协议对齐
- [ ] Task #76 watchdog 完成 GPU 0（1 层 GCN）leave-one-out eval
- [ ] Task #76 watchdog 完成 GPU 1（3 层 GCN）leave-one-out eval
- [ ] 对比 full-ranking Hit@20 vs leave-one-out HR@20 vs MCKG 论文 HR@20=0.614
- [ ] 写 `verdicts/task76_result.md` + Phase 0 判定
- [ ] 若通过 → 进入 Phase 1；若不通过 → 调整评估函数后重评

### Phase 1: 数据准备
- [ ] LastFM 统计核验（KGAT 仓库自带，已对齐）
- [ ] MovieLens-1M 下载 + RippleNet 格式转换
- [ ] Book-Crossing 下载 + RippleNet 格式转换
- [ ] 三数据集统一转换脚本（可复用）

### Phase 2: MCKG 模型实现
- [ ] 2.1 κ-Stereographic 算子实现（⊕/⊗/⊖/⊙/d + exp/log/tan/tan⁻¹）
- [ ] 单元测试对照 geoopt
- [ ] 2.2 高阶信息提取（exp^κ_o 映射 + 关系注意力 + GCN 聚合器）
- [ ] 2.3 多空间融合（公式 15-19）
- [ ] 2.4 几何感知 margin 损失（公式 20-21）

### Phase 3-4: 主实验
- [ ] LastFM MCKG 训练 → HR@20=0.691±0.03
- [ ] ML-1M MCKG 训练 → HR@20=0.802±0.03
- [ ] Book-Crossing MCKG 训练 → HR@20=0.603±0.03

### Phase 5: 消融
- [ ] 5.1 margin 策略：-g > -c > -h 顺序
- [ ] 5.2 聚合深度：最优 depth 对齐
- [ ] 5.3 子空间数量：M=1→2 提升、M≥2 趋平

### 最终交付
- [ ] 写 `verdicts/task77_result.md`
- [ ] 更新 loop.md §16 + CLAUDE.md（如有新 pipeline / env）

---

**注**: Task #77 的核心起点是 Task #76 的 leave-one-out eval 结果。若 Phase 0 不通过（HR@20 < 0.40），需重新核对协议后回退；若通过，则 Phase 1-5 按上述清单推进。
