# Task #244 — Issue #12 Gate 0: SID 分布画像 (零 GPU 纯分析)

## 来源

- GitHub Issue #12 (2026-07-28 lit-triggered): [Lit-triggered] SID 沙漏效应与路径稀疏 — 码本健康↔召回解耦的分布层解释 (Kuai et al., arXiv:2407.21488)
- 主文献: Kuai et al. *Breaking the Hourglass Phenomenon of Residual Quantization.* arXiv:2407.21488v2 [cs.IR], 2024-10-31
- Issue #9/#10/#11 全部 NO-GO, Issue #12 是当前 open issue, Gate 0 零 GPU 可立即启动

## 背景

Issue #12 指出仓库现有 SID 诊断**只测两个量**(utilization + collision_rate),**完全不测"用得多不均匀"**。即便 100% utilization 也可能 9000 个 item 挤在 3 个码字上(Gini 极高)。按主文献的沙漏效应机制,这正是码本健康但 R@10 不提升的真因。

Issue #12 Gate 0 主张: 把**逐层 token 分布画像**(Shannon 熵 / Gini 系数 / 标准差 / top-1 占比 / top-10 累计占比 / 路径稀疏度)纳入仓库指标体系,验证 H1 (本数据集存在沙漏式集中)。

## 假设

**H1**: 至少一层 token 分布 Gini ≥ 0.5 且该层 utilization ≥ 90% — 证明"高利用率掩盖了高度不均匀"。

## 输入

- Arm A SID: `/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy` (Task #84 baseline)
- Arm B SID: `/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB.npy` (Task #237 partial Sinkhorn)
- Arm C SID (phonism): TBD — 需查找 phonism 项目的 stage 2 SID 输出

## 输出

两 arm (A/B) × 三层 (L0/L1/L2) 的指标对照表:
| Layer | Shannon 熵 | Gini | std | top-1 占比 | top-10 累计占比 | 路径稀疏度 | utilization |
|-------|-----------|------|-----|------------|------------------|------------|-------------|

**指标定义** (照主文献 §4.1):
- **Shannon 熵**: H = -Σ p_i log p_i, p_i = token_i 的频次 / 总 token 数
- **Gini 系数**: G = (Σ_i Σ_j |x_i - x_j|) / (2 n Σ x_i)
- **std**: token 频次的标准差 (主文献 Figure 3 用)
- **top-1 占比**: 频次最高 token / 总频次
- **top-10 累计占比**: 前 10 高频 token 累计 / 总频次
- **路径稀疏度**: 实际 unique 路径数 / 理论最大路径数 (K0 × K1 × K2 × K3)

**补充比较**: 9922 个 item 在 K 个码字均匀分布的**理想 Gini** (作为平凡 baseline 排除), 只有显著偏离理想 Gini 才计入 H1 通过。

## Gate 0 通过条件

存在至少一层满足 `Gini ≥ 0.5` 且该层 `utilization ≥ 90%`.

## 硬停止

三层 Gini 均 < 0.5 → **H1 证伪**, STOP, 就地写 verdict 关闭方向, 明确记录"沙漏效应在本数据集不成立". **不得**因为"反正 Gate 0 便宜"就顺手往下跑.

## 步骤

1. **写 description** (本任务, R9 max+1 = #244)
2. **写分析脚本** `scripts/task244_sid_distribution_profile.py` (纯 numpy, 零 GPU)
3. **跑 Arm A** (T84 baseline SID)
4. **跑 Arm B** (T237 partial Sinkhorn SID)
5. **跑 Arm C** if SID 文件存在 (phonism)
6. **写 verdict** `verdicts/task244_issue12_gate0_sid_distribution_result.md` 包含:
   - 三层 × 三 arm 指标对照表
   - 与理想均匀分布 Gini 对照
   - Gate 0 PASS / FAIL 判定
   - 若 FAIL: 关闭方向; 若 PASS: 进入 Gate 1 准备

## 资源

- 零 GPU
- CPU only, 9922 × 4 SID 数组, ~40 KB
- 跑完 < 1 min

## 依赖

- `verdicts/task237_issue10_arm_b_result.md` (Arm B 数据来源)
- `verdicts/task223_stage2_sid_inference_result.md` (Arm A 数据来源)
- `verdicts/task236_collision_metric_unification_result.md` (utilization 权威定义)
- 主文献 arXiv:2407.21488v2 §4.1 指标定义

## 产物

- `descriptions/task244_issue12_gate0_sid_distribution.md` (本文件)
- `scripts/task244_sid_distribution_profile.py` (纯 numpy)
- `verdicts/task244_issue12_gate0_sid_distribution_result.md`
- 候选: `verdicts/task244_sid_distribution_profile.json` (机器可读)

## Status

R10 主动推进. Issue #12 Gate 0 零 GPU 可立即启动, 不阻塞 Task #243 (Stage 3 训练在 GPU 1/2 跑).
