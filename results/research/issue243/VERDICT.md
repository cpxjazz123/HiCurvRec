# Issue #243 — VERDICT

- issue 正文 sha256 `b78cf4d31d2894e162c9e0977c5d2def4d168fb99412f9949bd379940bec82df`
- 清单 30 条（`RUN_LOG.md`），七项强制机制全部落文件

## 逐条状态

| 编号 | 判定 | 证据 |
|---|---|---|
| C1 复用 #242 图与哈希 | **PASS** | `graph` 与 #242 同一份，`ego_csr.npz` sha256 `b17c2754…` 一致 |
| C2 契约 + 论文/官方实现对照 | **PASS** | `CONTRACT.md`（Eq.1–10、`geoopt.Stereographic`、`kappaGCNConv` 逐模块对照 + 本项目差异） |
| C3 配置/seed/采样/指标冻结 | **PASS** | `CONTRACT.md` §4/§5 |
| C4 参考实现与已知答案先行 | **PASS** | `reference.py` + `tests/test_known_answers.py` **14/14** |
| C5 变异测试 ≥5 且全被抓到 | **PASS** | `MUTATIONS.md`：**6/6 CAUGHT**（其中一个暴露测试太弱，已补强） |
| C6 球面修复 | **PASS** | 弃用 `normalize+arccos`，改用统一氏重写；`MODULE_CHECKS.json` 中 k=+1 距离 mean 0.680 / std 0.205（非恒定） |
| C7 双曲系数修复 | **PASS** | `tests/test_reference.py::test_distance_is_not_double_scaled` 钉住 k=−3 手算值 |
| C8 κ-GCN 替代末端投影 | **PASS（层）** | `kappa_gcn.py`：logmap→加权聚合→expmap；入/出各一套权重；`test_kappa_layer_identity_weights_matches_reference` 手算一致 |
| C9 诱导子图 | **部分** | 参考实现 `induced_edges/ego_nodes` 有手算测试；生产 teacher 仍沿用 #242 的采样器，未改为"先定节点集再取全部诱导边" |
| C10 pair bank 类型 | **NOT RUN** | 依赖修正后的 teacher 与 bank 重写 |
| C11 评估（Jaccard 真交并、Spearman ties、方向反转） | **部分** | `reference.jaccard/spearman_with_ties` 有手算与变异覆盖；`test_kappa_layer_direction_sensitivity` 证明方向被使用；T 维度断言未加 |
| C12 分模块验证 | **PASS** | `MODULE_CHECKS.json`：5 曲率 × 128 真实商品，零向量率 0、距离方差非平凡、全部参数有非零梯度且 AdamW 真实位移 |
| C13–C16 Teacher 正式训练 | **部分** | 60 步 smoke 完成（`teacher/train_log.json`）；**400 步正式训练未跑** |
| C17–C20 重叠子图分析 | **NOT RUN** | — |
| C21–C22 S_G 与冻结 | **NOT RUN** | — |
| C23–C26 RQ-VAE 与 Stage2 | **NOT RUN** | — |
| C27–C29 SID 评估与 Stage3 | **NOT RUN** | — |
| C30 每步评论与交付 | **PASS** | `COMMENT_01`、本文件、全部证据文件 |

## 已确证的结果

**#242 的两个数学错误已修复并有独立证据**：

1. 双曲距离多乘 2 —— 现在由手算值（k=−3, r=0.1）钉死，变异测试证明测试能抓到这个错误。
2. 球面专家用 `normalize` + `arccos` 冒充 κ-GCN —— 改为统一 κ-stereographic（Möbius 加法 + `tan_k^{-1}`），
   且用「球面点到模型点的逆球极投影」做独立交叉验证（`test_stereographic_matches_sphere_geometry`）。

**修正后 60 步 smoke 与 #242 的对比（同一图、同一采样、同一种子族）**：

| 指标 | #242（错误数学） | #243（修正后） |
|---|---|---|
| 门控熵 | 1.61 → **0.26**（塌缩） | 1.60 → **1.594**（最大 ln5 = 1.609） |
| 门控权重 | 单一专家 **0.94** | [0.159, 0.219, 0.239, 0.224, 0.159] |
| 各专家平均距离 | 一个恒为 **π/2** | [2.695, 0.224, 2.299, 0.061, 0.023] |
| Spearman(D_geo, g) | 0.44–0.67（在塌缩状态下） | 0.485 |

因此 issue 的判断成立：**#242 的门控塌缩与 π/2 常数来自数学实现，而不是数据或机制本身**。

## 仍然存在的、已定位的问题

五个专家的距离尺度相差约 **100 倍**（0.023 – 2.695）。混合权重再均衡也无法让尺度小的专家参与，
这与 issue 步骤 3 要求的「预先用合法训练探针确定各空间输入尺度与初始化」直接相关，**尚未做**。
在完成尺度标定之前，任何"混合几何优于单几何"的结论都不成立。

## 阻塞点（按 skill §4 的 `resource` 分类）

剩余工作是一个完整的多日程序：步骤 3 的 400 步正式训练 + 独立评估 → 步骤 4 的重叠子图兼容性 →
步骤 5 的 S_G 与 bank → 步骤 6 的 RQ-VAE（TDD + λ 标定 + smoke）→ 步骤 6 的 3000 epoch Stage2 →
步骤 7 的 SID 评估与 150 epoch Stage3。本轮预算内无法完成，且 issue 明确禁止用放宽判据或跳过检查来压缩。

## 恢复计划（按顺序，无需重新决策）

1. **专家尺度标定**（步骤 3 前置）：对每个曲率在合法训练探针上测 `d_e` 的分位数，用可逆的尺度因子
   （`expmap0(α_e · logmap0(h))`，α_e 由探针中位数对齐到同一目标尺度）使五个专家的距离中位数同量级；
   规则与 α_e 冻结进 `CONTRACT.md`。
2. **正式 Teacher 训练** 400 步 + 步骤 5 的独立评估（未见评估对上 `Spearman(D_geo,g)` 必须同时胜过
   语义欧氏与 Jaccard 两个非训练对照）。
3. 步骤 4 的重叠子图兼容性（`E_t`、最差子图、同时保真率、度数匹配对照）。
4. 步骤 5 的 S_G 与修正后的 bank（含 C10 的类型逐条验证）。
5. 步骤 6 的 RQ-VAE：`kappa_gcn`/`manifold` 可作为既有基础设施；λ_max 用冻结探针标定。
6. 步骤 7 的 SID 评估与 Stage3。

## 交付

`CONTRACT.md`、`reference.py`、`manifold.py`、`kappa_gcn.py`、`teacher.py`（κ-GCN 版）、
`module_checks.py` + `MODULE_CHECKS.json`、`tests/test_known_answers.py`（14）、
`tests/test_reference.py`（9）、`tests/test_mutation.py` + `MUTATIONS.md`（6/6）、
`teacher/train_log.json`（60 步 smoke）、`RUN_LOG.md`、`COMMENT_01.md`、本文件。
二进制与 `_mutation/` 工作目录不入库。
