# Issue #238 — RUN_LOG

- issue: #238 Stage2: 仅截面曲率保真监督（混合曲率 + 全局硬量化表示）
- state at freeze: OPEN, created 2026-11-23T00:52:21Z, updated 2026-11-23T00:55:21Z
- body sha256: `4b6de1a92811bf45d3cd0692d69c83db8f5f3f21517057df667c27f5dcd0459d`
- frozen at: 2026-11-23 (this run)

## 是否需要训练 — 判定

**需要训练。** 依据 issue 的「验收与交付」第 3–4 条：
> 在冻结评估样本上比较 TIGER：K_G 与 K_q 的 MAE/RMSE … 对实际三层 SID 报告 … 三码碰撞
> 按现有三码碰撞准入与冻结 Stage3 协议运行一次合格候选，记录实际读取 SID 的 SHA256、Recall/NDCG

`K_q`、硬 q 的几何代理误差、三码碰撞与 Stage3 Recall 都只能由一个**新训练的 Stage2 权重**产生，
无法由已有产物复算。因此本 issue 必须走 Stage2（3000 epoch）+ Stage3（150 epoch）各一次。

## 冻结验收清单（C1..C44）

「阈值」一栏写 `实测并报告` 的条目是 issue 明示的**诊断项**（issue 原文：「这是诊断而非擅自增加准入门槛」），
判定 PASS 的条件是**按冻结口径报告了数值**，不是数值达到某个门限。判据来自 issue，不得解释性放宽。

### A. 固定协议（C1–C5）

| # | 判据 | 阈值 | 数据 | 口径 |
|---|---|---|---|---|
| C1 | 仅从 `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` 派生一个独立候选，除 `ARM_NAME` 外源码逐文件可解释 | 候选树唯一；`diff -r` 只显示 `ARM_NAME` 与机制改动 | 冻结基线树 | `diff -r` 输出入库 |
| C2 | 768→32 瓶颈、三个 256 码字、原始重建目标、Sinkhorn 均衡、3000 epoch、四卡/全局 batch 4096/seed 42 不变 | 与基线 `config/train_config.json` 逐键相同 | 候选 `config/train_config.json` | 脚本比对 + 输出入库 |
| C3 | 无 L1/L2/L3 分层结构 loss；结构监督只在硬 `q = e1+e2+e3` 上算一次 | 源码中 `L_curv` 调用点恰好 1 处，输入为硬 q | 候选 `losses/`、`train_rqvae.py` | `grep -c` + 代码路径说明 |
| C4 | 除原始 RQ-VAE 损失外只新增一个 `L_curv`；无 InfoNCE/对比/独立距离 loss | 新增 loss 模块恰好 1 个 | 候选 `losses/__init__.py` 注册表 | 注册表条目 + `grep` |
| C5 | 训练行为图只来自 `train.parquet` | 不存在验证/测试边读取 | 图构建脚本 | 脚本读的文件清单 + sha256 |

### B. 方法规格（C6–C15）

| # | 判据 | 阈值 | 数据 | 口径 |
|---|---|---|---|---|
| C6 | `K_G` 用测地三元组 `(m;b,c)`，`m` 与 `b,c` 相连且 `d_G(b,c)=2`，同一参考节点 `a`，公式 `K_G=mean_a[d(a,m)²+d(b,c)²/4−(d(a,b)²+d(a,c)²)/2]` | 公式逐字实现；覆盖/分层计数落盘 | 规范行为图（train 边） | 单元测试 + 覆盖统计文件 |
| C7 | `D_q` 在 32 维统一坐标分块（双曲/欧氏/球面），exp/log 映射，固定正权重 `Σw=1`，维度总和 32，曲率/尺度/权重训练前标定并记录 | `sum(dims)==32`、`sum(w)==1`、`w>=0` 由测试断言 | 标定脚本 + 冻结常量 | 测试 + `CALIBRATION.md`（在训练前生成） |
| C8 | `K_q` 用同一参考集与 `D_q`；全局尺度可复现；不按三元组归一；不用零缩放 | 尺度由训练前标定确定，记录数值 | 同 C6 参考集 | 代码常量 + 测试 |
| C9 | `L_curv = mean_t robust_loss(K_q(t) − K_G(t))`；正/负/度数分层聚合规则预先冻结 | 聚合规则写在 `CALIBRATION.md` 且实现一致 | 三元组表 | 代码 + 测试 |
| C10 | 中点偏差 `(d_q(b,m),d_q(m,c),d_q(b,c))` 仅诊断；不成立时只能称四点距离几何代理 | 不存在把中点项加入 loss 的代码 | 候选 `losses/` | `grep` + 报告声明 |
| C11 | 唯一新增目标 `L_total = L_RQVAE + λ_curv·L_curv` | 总损失构成只有这两项 | 候选 loss 组合处 | 代码 + smoke `probed_terms` |
| C12 | 同一混合距离用于现有 Sinkhorn 层 residual→codeword 代价；均衡规则不变；其他层原样 | Sinkhorn 层 `distance` 是 `D_q`；L1/L2 仍是基线距离 | 候选 `model/layers.py` | 测试断言层对象身份 |
| C13 | 三个开关只做廉价数学/接线测试，不训练额外臂；curve-off→统一欧氏；behavior-off→关闭真实图目标 | 只存在一个训练臂 | `results/stage2_RQ-VAE/<ARM>` 唯一 | 目录清单 |
| C14 | 梯度真正来自硬 q；显式 STE 路由；检查 encoder 与实际受监督码本参数 | 两者梯度范数 > 0 且有限 | smoke | smoke `probed_terms` + 测试 |
| C15 | 损失权重与梯度预算标定规则冻结，不做结果导向调参 | `λ_curv` 由冻结规则推出并落盘 | `CALIBRATION.md` | 文档 + 代码常量 |

### C. TDD（C16–C24）

| # | 判据 | 阈值 | 数据 | 口径 |
|---|---|---|---|---|
| C16 | 路径/星形/环/格点 + 真实小批量上的图曲率公式测试 | 全部通过 | 合成图 + train 子图 | `pytest` |
| C17 | 固定参考节点测试 | 通过 | 合成图 | `pytest` |
| C18 | 非欧氏距离测试（三个分量各自测地距离） | 通过 | 合成坐标 | `pytest` |
| C19 | 统一尺度标定测试 | 通过 | 标定表 | `pytest` |
| C20 | 断连测试（不可达点不得当零） | 通过 | 合成图 | `pytest` |
| C21 | 中点偏差诊断测试 | 通过 | 合成坐标 | `pytest` |
| C22 | 零距离/对称/有限梯度/硬 q 前向一致 | 通过 | 合成张量 | `pytest` |
| C23 | 两个开关 + Sinkhorn 代价接线 | 通过 | 候选树 | `pytest` |
| C24 | 真实入口梯度 + DDP 检查，且**通过后才训练** | 通过 | 候选 `train_rqvae.py` | smoke `SMOKE_PASS` |

### D. Stage2 评估表（C25–C37）

| # | 判据 | 阈值 | 数据 | 口径 |
|---|---|---|---|---|
| C25 | `K_G` vs `K_q` 的 MAE / RMSE | 实测并报告 | 冻结评估样本 | 评估脚本 |
| C26 | 符号一致率（正/负分别） | 实测并报告 | 同上 | 评估脚本 |
| C27 | Spearman 相关 | 实测并报告 | 同上 | 评估脚本 |
| C28 | 各度数 / 曲率组表现 | 实测并报告 | 同上 | 评估脚本 |
| C29 | 局部距离比例与 midpoint 偏差（仅评估） | 实测并报告 | 同上 | 评估脚本 |
| C30 | 同口径报告 量化前 z / 硬 q / 由实际 SID 重建 q | 三路齐全 | 同一 SID 表 + checkpoint | 评估脚本 |
| C31 | 真实行为边 vs 负候选的距离排序（仅评估） | 实测并报告 | train 边 + 采负 | 评估脚本 |
| C32 | 三层 SID 前缀共享的正/负样本差异 | 实测并报告 | item_sids.json | 评估脚本 |
| C33 | 链/分叉/汇合规则满足率；方向无辨识能力须声明 | 实测并声明 | item_sids.json + 序列 | 评估脚本 |
| C34 | 三码碰撞 | 实测并报告 | item_sids.json | `sid_quality` 同口径 |
| C35 | 各层码本利用率 / 熵 | 实测并报告 | item_sids.json | 评估脚本 |
| C36 | 重建 MSE | 实测并报告 | 评估样本 | 评估脚本 |
| C37 | 重导漂移 | 实测并报告 | `export_check` manifest | `export_check` |

### E. 结构假设与 Stage3（C38–C44）

| # | 判据 | 阈值 | 数据 | 口径 |
|---|---|---|---|---|
| C38 | 硬 q 几何代理误差相对冻结 TIGER 降低 | 实测；未改善则照实判定 | C25 表 | 对照基线同口径值 |
| C39 | 正/负曲率组均改善，或明确报告权衡 | 实测；二选一如实写 | C26/C28 | 同上 |
| C40 | 直接行为关联保留（C31 排序） | 实测并对照基线 | C31 | 同上 |
| C41 | SID 区分性与语义重建不发生不可接受退化（C34/C35/C36） | 实测并对照基线 | C34–C36 | 同上 |
| C42 | 报告口径在训练前冻结 | `CALIBRATION.md` 早于 Stage2 启动时间 | 文件 mtime + commit | git log |
| C43 | 按三码碰撞准入 + 冻结 Stage3 跑一次合格候选 | `stage3_start` 返回 started 或闸门拒绝（拒绝即记录） | 候选 SID | MCP 输出 |
| C44 | 记录实际读取 SID 的 SHA256 + Recall/NDCG 与历史 TIGER 比较；§25；单次小差值不得宣称显著 | 报告 SID sha256 与差值，并按 1.07e-3 波动解读 | Stage3 `test_final.json` | 报告 |

### F. 交付（C45–C47）

| # | 判据 | 阈值 | 数据 | 口径 |
|---|---|---|---|---|
| C45 | 独立代码、配置、训练日志、冻结图/样本/源码/checkpoint/SID 哈希、测试证据、完整结果表、`REPORT.md` | 全部存在且路径可点 | `results/research/issue238/` + 候选树 | 文件清单 |
| C46 | 遵循归档规则（§6b：四件齐全才上传 R2） | 四件齐全时上传并记 ledger | 候选产物 | 上传日志 |
| C47 | 不创建额外 Issue，不覆盖冻结基线 | `git status` 中基线树无改动；issue 数仍为 1 | 仓库 | `git status` + `gh issue list` |

## 与既有探索的关系（必须记录）

`results/research/issue238/REPORT.md` 记录了本 issue 之前的探索（7 个机制臂 + 3 次 Stage3 校准），
其中两条结论被本 issue 直接引用：

1. Stage3 同一 SID 重复评估的波动 **sd ≈ 0.00107**（n=3：0.055450 / 0.053483 / 0.055206）；
2. 因此单次 Stage3 的 ±0.001 差值不可宣称显著。

issue 又明确「不做额外重复训练」，所以本轮**只训一个候选**，不重复、不扫参。
既有 7 个臂是 issue 之前的历史产物，不作为本 issue 的候选，也不重复使用其结论作为证据。

## 当前阶段

- [x] issue 正文冻结（sha256 已记）
- [ ] infra
- [ ] 实现（TDD）
- [ ] 检查循环
- [ ] Stage2 训练
- [ ] Stage3
- [ ] 结论对账
- [ ] 关闭
