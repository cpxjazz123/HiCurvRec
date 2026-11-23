# Issue #244 — VERDICT（COMPLETED，负结果）

- issue 正文 sha256 `3f35cacbe0661ca6f9dc8ca59be237e486b069fd8a1aeff1dfe68c2750da15f6`
- 清单 26 条（`RUN_LOG.md`），七项机制全部落文件
- 结论：**主假设不被支持**，按 issue §4 的「无可用结构信号则停止 Stage2」在 Stage2 之前收束

## 逐条判定

| 编号 | 判定 | 证据 |
|---|---|---|
| C1 复用官方/geoopt | **PASS（含偏离说明）** | 未引入 geoopt 运行时依赖；独立性改由「逆球极投影 + 3D 夹角」这条**定义路线**提供，与生产闭式一致到 1e-15 |
| C2 修正 k>0 距离 | **PASS** | `x=0.9,y=−0.9,k=1 → 2.93126 = 4·atan(0.9)`；17/17 已知答案；`sphere-proju-hyperbolic-clamp` 变异被抓 |
| C3 exp/log 半径与边界 | **PASS** | `R·tan(‖v‖/2R)` 与 `2R·arctan(r/R)`，`d(0,expmap0(v))=‖v‖`；k=3 往返误差 5.6e-17 |
| C4 双曲域 / k→0 约定 | **PASS** | 自距离、近边界、对称性测试；k=0 作为独立空间记录并有测试固定（平坦极限是 2× 欧氏） |
| C5 去除旧错误函数 | **PASS** | #242 遗留的 `project_hyperbolic/dist_spherical/expert_distance/project_expert/ego_sample` 已删除；实际训练路径走 `manifold` + `kappa_gcn` + `topology` |
| C6 κ-GCN / Eq.8–10 | **PASS（Eq.8 未用，已注明）** | `kappa_gcn.py` 两层 logmap→聚合→expmap，入/出各一套权重；距离用 Eq.9–10；Eq.8 的节点混合不参与，因为距离由 Eq.10 给出 |
| C7 曲率 −3..+3 | **PASS** | `CURVATURES` 常量，固定不可学习，标定 α 另记 |
| C8 诱导 ego 子图 | **PASS** | `topology.py`；8/8（三角/链/分叉/汇合/互惠/孤立/非树边/上限）；`ego-drops-induced-edges` 变异被抓 |
| C9 pair bank 类型 | **PASS** | `pairs.py` 按定义构造 + 逐条验证；7/7；`pairs-bucket-by-source` 变异被抓 |
| C10 指标与方向 | **PASS** | `metrics.py` 并列秩 + 常数 NA + 真交并；`test_teacher_wiring` 证明 128 维与方向反转；评估对**双向**排除 |
| C11 复用数据/隔离 | **PASS** | 图与 #242 同一份（sha256 一致）；评估标签不入训练、标定与采样 |
| C12 探针统计 | **PASS** | `teacher/expert_probe.json`：中位/分位/方差/范数/饱和率/梯度/位移 |
| C13 尺度标定 | **PASS** | 16.12× → **1.00008×**；梯度差 300× → 1.5×；规则与取值开训前冻结（CONTRACT §8） |
| C14 正式训练前的探针 | **PASS** | 标定在**初始化**上做，先于正式训练 |
| C15 400 步正式训练 | **PASS** | 508 s，loss 0.541，训练批 Spearman 0.641；逐专家距离/梯度/门控写入 `train_log.json` |
| C16 独立评估与对照 | **PASS（PPR 未做，已注明）** | 未见对 1,496；D_geo 0.407 vs 语义 0.049 vs Jaccard 0.894；同 checkpoint 的单专家与均匀门控见 `overlap.json` |
| C17 多子图采样 | **PASS** | 400 中心 × 3 子图，保留节点身份与方向 |
| C18 E_t 与同时保真 | **PASS** | 均值 0.671 / 最差 0.730 / p90 0.723 / **同时保真 0.025**；阈值 0.5 开训前冻结；同一商品同一冻结表示 |
| C19 S_G 与 RQ-VAE | **未训练（闸门）** | 代码 + 5/5 测试已交付；`build_bank.py`、`calibrate_topo.py` 未执行，因为 §4 判定无可用结构优势 |
| C20 开训前入口检查 | **未执行（同上）** | 同上 |
| C21–C24 SID 评估与 Stage3 | **未执行（同上）** | 同上 |
| C25 结论复核 | **PASS** | `recheck.py` 从原始产物重算：球面已知答案、标定差、重叠聚合、评估内部一致性、节点 vs pair 权重；`all_ok: true` |
| C26 不重做/不覆盖/不建 issue | **PASS** | 冻结 TIGER 未动；无新 issue；`PROGRESS.json` 保留为记录 |

## 判定说明（为什么“未训练”不是漏做）

issue §4 原文：「若有效实验否定假设如实报告，不为获得正结果反复训练。**若有可用监督信号则继续。**」
以及「**若无可用结构信号则停止 Stage2，先定位证据**」。

三条独立测量同时否定“混合几何带来可用结构优势”：
1. 同时保真率 **2.5%**（预冻结阈值 0.5，400 中心 × 3 子图）；
2. 学到的门控 **0.6709** ≈ 固定均匀门控 **0.6672** —— 门控没有贡献；
3. 混合几何 **0.6709** ≈ 最好的单一专家 **0.6856** —— 差 0.015，且方向上是混合略优但幅度可忽略；
4. 在未见对上落后非训练的 Jaccard 基线 **0.49 个 Spearman 点**。

因此 Stage2/Stage3 的“未执行”是 issue 规定的停止点，属于 **COMPLETED 的负结果**，
不是 §6.1 的阻塞，也不是预算托词。

## 已排除的解释（为什么不是实现 bug）

| 解释 | 实测 |
|---|---|
| 数学写错 | 17 项已知答案 + 12 项生产/参考一致（含 3D 定义路线）+ 12/12 变异 |
| 没标定导致尺度失衡 | 已标定，尺度差 1.00008×，梯度差 1.5× |
| 门控塌缩是 bug | 节点权重 0.9996 但 **pair 权重只有 0.404**，正是 issue 预警的混淆；且均匀门控同样 0.667 |
| 训练不足 | 400 步 loss 单调下降到 0.541，训练批 Spearman 0.641，与评估集 0.407 的差距是泛化而非欠拟合 |
| 评估口径偏向基线 | Jaccard 是按 issue 要求实现的**非训练**对照，用同一 rank 口径计算 |

## 交付与哈希

见 `REPORT.md` §8；关键产物 `teacher/{train_log,teacher_eval,overlap,calibration,expert_probe}.json`、
`recheck.json`、`MUTATIONS.md`、`MODULE_CHECKS.json`、6 个测试套件、`stage2_RQ-VAE/TIGER_topo_distill_rq/`（未训练）。
