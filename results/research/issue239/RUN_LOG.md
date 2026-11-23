# Issue #239 — RUN_LOG

- issue: #239 Stage2: 保留 TIGER 欧氏量化，仅在全局 Loss 中引入混合曲率保真监督
- state at freeze: OPEN, created/updated 2026-11-23T01:49:38Z
- body sha256: `7a1ee963ef2b0432888b7587b31cd5ac42a603ed3136d98c4b7894065abebfc1`
- 前序: #238（commit `d4d3fc15`）按 §6.1 以阻塞关闭；本 issue 采用其**方案 C**

## 是否需要训练 — 判定

**需要训练，且只训一次。** 依据 issue「评估与验收」：

> 同样的冻结评估样本及固定标定口径比较量化前 z、最终硬 q、从导出 SID 重建的 q …
> 评估实际 SID：三码碰撞率、各层利用率/熵 … 通过准入后按原协议运行一次 Stage3（150 epochs）

`K_q`、硬 q 的曲率保真、三码碰撞与 Stage3 Recall 都只能由一个新训练的 Stage2 权重产生。
训练预算：3000 epochs、四卡、全局 batch 4096、seed 42，其他超参与冻结 TIGER 相同，**一次**，不做参数扫描与多种子（C6）。

## 冻结验收清单（C1..C16）

| # | 判据 | 阈值 | 数据 | 口径 |
|---|---|---|---|---|
| C1 | 从冻结 `baseline/TIGER_RQ-VAE` 派生独立候选；复用 #238 的图曲率目标/采样/混合空间实现；不依赖 #237 输入 | 候选树唯一；`diff -r` 只显示 `ARM_NAME` + 机制文件 | 冻结基线 | `diff -r` 输出入库 |
| C2 | 所有层的码字选择、残差更新、欧氏代价、Sinkhorn 所在层及均衡/退化保护逻辑**与 TIGER 逐字节相同** | `model/layers.py`、`distances/distance.py`、`config/train_config.json` 与基线 sha256 相同 | 候选树 vs 基线 | `sha256sum` 对比 |
| C3 | 硬 `q=e1+e2+e3`；只在该向量用固定混合距离算 `K_q`；沿用 #238 三元组与同一参考集；统一尺度；正负/度数分层；不可达不填零；中点条件不成立则称四点代理 | 复用 #238 `calibration.json` 的离散量与参考集；`K_q` 计算路径不含 `z` | 规范行为图 + 候选硬 q | 代码路径 + 测试 |
| C4 | `L_total = L_RQVAE + λ·L_curv`；无 InfoNCE/独立距离/对比/分层监督；混合距离**不进** residual→codeword 代价 | 新增 loss 恰好 1 个；`_distances` 仍用欧氏 | 候选 `losses/`、`model/layers.py` | `grep` + 测试 |
| C5 | 图仅由训练行为构建；监督/标定/评估隔离；验证/测试边不参与目标、参考或标定；重新核验冻结样本哈希；方向未建模须声明 | 图 metadata 的 train.parquet 哈希一致；无验证边读取 | `behavior_graph/` | 哈希比对 + 代码 |
| C6 | 训练前冻结 λ、尺度、梯度预算、Huber 参数；不扫参不多种子 | `calibration.json` 的 mtime 早于 Stage2 启动 | 结果目录 | mtime + git log |
| C7 | 复核 #238 失败：第 0/1/2 步的 encoder、残差、码本、混合距离与梯度的有限性、动态范围、参数更新量；给具体证据；不得用常数矩阵回退掩盖 | 给出逐量数值表，并指出退化的那个量 | #238 候选 + 冻结基线权重 | 诊断脚本输出入库 |
| C8 | 新候选中以真实入口验证欧氏代价与冻结 TIGER 相同 | 与基线 `distances()` 数值一致（rtol 1e-6） | 候选 init 后的模型 | 测试 + smoke |
| C9 | 硬 q 前向与 STE 路由正确；无软硬不一致；encoder/各受监督码本梯度有限且非零；无 NaN/Inf；禁用结构项时与原协议等价 | `dead_terms=[]`、梯度范数 > 0、NaN 计数 0 | 四卡 smoke | smoke summary |
| C10 | 单元测试 + 真实四卡 smoke + 现有准入通过后才运行一次正式 Stage2；若仍失败交付根因证据，不放宽门槛 | smoke `passed=true` | 候选 | MCP 输出 |
| C11 | 唯一基线为冻结 TIGER（不重训）；768→32、三层 256、语义重建不变 | 与基线 config 逐键相同 | 候选 config | 脚本比对 |
| C12 | 同冻结样本与固定口径比较 `z`、硬 `q`、SID 重建 `q`：MAE/RMSE、Spearman、正负符号一致率、各曲率/度数组；中点偏差与真实边排序仅诊断 | 三路齐全、按分层报告 | 冻结评估样本 | 评估脚本 |
| C13 | 实际 SID：三码碰撞、各层利用率/熵、前缀共享与行为关系、链/分叉/汇合满足率、重建 MSE、重导一致性 | 实测并报告 | `item_sids.json` + checkpoint | 评估脚本 |
| C14 | 成功判据：相对 TIGER 硬 q 曲率保真改善、报告权衡、SID 结构/重建/碰撞不不可接受退化、遵循冻结准入；不预设成功 | 如实判定 | C12/C13 | 报告 |
| C15 | 通过准入后按原协议跑一次 Stage3（150 epoch），报告 Recall/NDCG、SID 哈希，遵守 §25；不把单次小幅差值称显著（0.00107 波动） | 实测 + 与 0.0547±0.0011 比较 | Stage3 `test_final.json` | 报告 |
| C16 | 交付独立实现/配置/测试/smoke 证据/日志/报告/哈希；不满足如实报告；不覆盖冻结资产；不创建额外 Issue | 全部存在 | `results/research/issue239/` | 文件清单 |

## 关键约束（与 #238 的差别）

| 项 | #238 | #239 |
|---|---|---|
| 量化器代价 | **混合距离**（进 Sinkhorn） | **欧氏**（与 TIGER 逐字节相同） |
| 混合距离用途 | Sinkhorn 代价 + L_curv | **仅 L_curv** |
| §19h | 机制必须进均衡代价矩阵 | 本 issue 明确写出「针对 §19h 的本轮限定例外」 |

因此本 issue 的实现可以完全避开 #238 的阻塞点（`center_distance` 守卫），但仍须按 C7 交付根因证据。

## 当前阶段

- [x] issue 正文冻结
- [ ] infra（复核 #238 的哈希仍有效）
- [ ] 根因诊断（C7）
- [ ] 实现（TDD）
- [ ] 检查循环（smoke）
- [ ] Stage2 训练
- [ ] 评估表 + Stage3
- [ ] 结论对账与关闭
