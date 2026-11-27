## #272 round 1 结果：FAIL（无 Stage3；冻结判据未满足，未放宽）

**候选**：`TIGER_active_j_rq`，run `stage2_4dffd8b4528440b3aca334b380749f01`，单卡 3000 epoch，
seed42，batch 4096，wall 约 1900 s（含末尾全图 J）。唯一主改动：训练 J 上界 0.781941 → **0.320**
（验收 full J ≤ 0.338336，扣除固定 pair 估计偏置 0.018336）。

**PASS 表（逐条，阈值训练前冻结）**

| 判据 | 阈值 | 实测 | 判定 |
|---|---:|---:|---|
| full J | ≤0.338336 | **0.256476** | PASS |
| 1-hop J | ≤0.798295 | **1.244779** | **FAIL** |
| 2-hop J | ≤0.278106 | 0.257521 | PASS |
| 局部 MAE | ≤1.893506 | **2.343824** | **FAIL** |
| 局部 Spearman | >0.015851 | **−0.098838** | **FAIL** |
| 局部 sign | >0.432453 | **0.348062** | **FAIL** |
| var ratio / slope | ≥0.5 / ≥0.1 | 0.6227 / **−0.0361** | 方差 PASS，斜率 FAIL |
| L2 前缀 | ≥10,918 | **6,848** | **FAIL** |
| effective rank | ≥8 | **2.774** | **FAIL** |
| 码本占用 | 256×3 | 256/256/256 | PASS |
| L3 gain / collision | 记录 | 2.662 bits / 0.119779 | 记录（碰撞仅记录） |
| 重建（同口径 #265 8.18e-5 ×1.05） | ≤8.59e-5 | **9.13e-5** | **FAIL** |
| Stage3 | 禁止 | 未运行 | PASS（无 Recall） |

**约束活跃性**：μ>0 仅 80/3000 epoch（2.7%，首次 epoch 9，max μ 0.021894）；J 下降主要不是持续全局
梯度贡献。受控真实 batch 探针证明约束可激活：J_est 0.351831>0.32，μ 0.1→0.10318，全局梯度到
encoder 0.1154/码本 0.0934、decoder 0、hard q 有梯度；满足侧用受控估计 0.30→μ 0.098（该 pair 表上
最近的真实缩放 q 为 0.3242，未找到 ≤0.32 的既有 q，已如实标注）。

**#271 失败归因（C4）**：MAE 改善部分来自欠离散（var ratio 0.335、slope 0.041）；紧约束后 mean J
从 0.3514 降到 0.2565，但 **1-hop 从 0.6420 升到 1.2448**、MAE 2.344、Spearman −0.099、sign 0.348，
即“总体均值改善、近邻与局部代理恶化”，被预冻结的一跳/局部判据捕获。

**前沿与不可行性**：五个冻结臂中 L2≥10,918 与 rank≥8 从未共存 —— TIGER（λ=0）7,374/11.48；
#269 6,246/4.34；#271 6,096/4.50；本轮 6,848/2.77；唯一 L2≥10,918 的 #265 是 **Euclidean 局部监督**
（本 Issue 机制边界外，rank 3.05）。λ=0 是 TIGER 端点，λ>0 观测到 L2/rank 同时下降；J 上界只能
小幅抬 L2、压低 rank。MAE≤1.8935 与 Spearman>0.01585 也从未共存（TIGER 是唯一过 Spearman/sign 的臂，
MAE 7.31）。按 Issue §7 记录为**机制边界内的经验不可行性**，未改判据。

**结论**：本轮不满足关闭条件；Issue 保持 **OPEN**，等 owner 选择最小替代：
A) 允许不同局部目标/显式 rank-capacity 项（新机制，需新 Issue 与 §121 闸门）；
B) 按观测前沿重注册阈值（明确不是 #272 PASS）；
C) 接受仅 mean J + collision 的候选（记录为未满足冻结判据）。

**证据**：`results/research/issue272/{VERDICT,DIAGNOSIS,CONTRACT,RUN_LOG}.md`、
`ruler_inventory.json`、`sid_inventory.json`、`local_diagnostics.json`、
`candidate_layer_metrics.json`、`MODULE_CHECKS.json`、`INFRA_SNAPSHOT_{START,END}.json`（对比 ok，
零漂移）、run 结果目录的 `geometry_final.json`/`item_sids.json`/`rqvae_best.pth`。
**未启动 Stage3，未生成任何新 Recall。**
