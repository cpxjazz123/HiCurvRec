## #271 准入：根因、竞争假设与冻结门槛（INFRA_MODE=FROZEN）

**模式判定**：Issue 原文 "INFRA_MODE=FROZEN，复用已有全量训练图距离、三元组/参考点、#269冻结κ缓存、
训练/评估及R2流程。只允许候选机制和专属测试、分析，不修改公共Infra或重建公共资产。" → 无 BUILD 授权，
公共树与 `assets/` 只读。Infra 快照 ① 已写入 `results/research/issue271/INFRA_SNAPSHOT_START.json`
（protected path:sha256，结束时重算）。

**是否需要训练**：需要。验收的 J、碰撞/量化与 Stage3 R@10 只能由新权重产生，Issue §3/§4 要求一次
3000-epoch Stage2 与一次 150-epoch Stage3。已写入 `RUN_LOG.md`。

**根因诊断**（`DIAGNOSIS.md`）：#269 局部 A.1 改善（Σ 标尺 −0.249）但全局图距离 J 从 0.183 升到 0.326，
L2 前缀 170→113（127 拆分/93 合并），R@10 0.052595→0.045231；#270 42 臂观察性分析无稳定正相关
（Spearman −0.068，CI [−0.51,+0.35]），协议/族方差主导。候选根因：
H1 局部 κ 收益以全局距离场退化为代价，SID 压缩是中间量；H2 局部代理与推荐无关（标尺不匹配）；
H3 单卡/协议混淆；H4 量化/SID 层决定 recall。竞争测量见 DIAGNOSIS §2。

**准入六问**：①问题=固定全局距离场下局部 κ 能否改善 SID/推荐；②证据=J 0.183→0.326、L2 压缩、
ΔR −0.00736；③通路=μ(J−J_max) 的非负对偶项只在违反时施压，改变 encoder/码本的更新方向；
④区别=#269 只有局部项，本候选加入显式全局不等式与全图验收；⑤指标=全图 J≤J_max、独立共同尺子
局部 A.1<TIGER、碰撞<0.40、Stage3 R@10 vs 0.055206 与 0.065；⑥失败知识=区分 H1/H2/H4，并定位 μ 轨迹。

**已冻结的判据**（`CONTRACT.md`）：J_TIGER 与 ε 在训练前用冻结 TIGER checkpoint 标定；候选唯一；
3000 epoch/seed42/原 global batch；Stage3 150 epoch/seed42/beam20，test 只测一次；不重训 TIGER。

**低成本探针**：共同尺子 κ=0 极限（因子 1，非 2）、全图/分层 J、固定 pair 表的覆盖与权重、
真实 batch 的局部/全局梯度范数、夹角、码本/encoder 梯度与常量无梯度，位于
`tests/test_global_constraint.py`、`tests/test_global_mutation.py` 与 `results/research/issue271/module_checks.py`。
