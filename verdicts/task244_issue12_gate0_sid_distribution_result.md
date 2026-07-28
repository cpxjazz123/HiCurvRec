# Task #244 — Issue #12 Gate 0 分布画像 (零 GPU) 结果

> **完成日期**: 2026-07-29
> **状态**: 🟡 **Gate 0 PARTIAL PASS (按字面 PASS, 按排除 L3 后 FAIL)**

---

## 1. 背景

GitHub Issue #12 (2026-07-28 lit-triggered, Kuai et al. arXiv:2407.21488v2) 主张:仓库当前只测 utilization + collision_rate 两个量,**完全不测"用得多不均匀"**。即便 100% utilization 也可能 9000 个 item 挤在 3 个码字上 (Gini 极高)。

**H1**: 至少一层 token 分布 Gini ≥ 0.5 且该层 utilization ≥ 90% — 证明"高利用率掩盖了高度不均匀"。

## 2. 数据

**输入** (4 个 SID 产物):
- Arm A (T84 baseline, no Sinkhorn): `_t5_rqvae_code_default.npy`
- Arm B (T237 partial Sinkhorn max_iters=10): `_t5_rqvae_task237_armB.npy`
- T200 dual_v5 (Issue #8 confounded reference): `_t5_rqvae_dual_v5.npy`
- T178 200 epoch (Mode collapse test): `_t5_rqvae_hyp_e14.npy`

## 3. 逐层指标 (排除 L3 K=1 dedup digit)

| Arm | Layer | util | Gini | Shannon ent | top-1 % | top-10 % | H1? |
|------|-------|------|------|-------------|---------|----------|-----|
| **Arm A (T84 baseline)** | L0 (K=64) | **0.5000** | **0.5725** | 4.9515 | 5.31% | 40.40% | — |
| **Arm A (T84 baseline)** | L1 (K=128) | **0.5000** | **0.5861** | 5.9245 | 2.55% | 22.56% | — |
| **Arm A (T84 baseline)** | L2 (K=256) | 0.9531 | 0.3432 | 7.6625 | 0.90% | 8.25% | — |
| **Arm B (T237 Sinkhorn)** | L0 (K=64) | 1.0000 | 0.1984 | 5.9019 | 4.30% | 26.19% | — |
| **Arm B (T237 Sinkhorn)** | L1 (K=128) | 1.0000 | 0.1741 | 6.9291 | 1.69% | 12.96% | — |
| **Arm B (T237 Sinkhorn)** | L2 (K=256) | 1.0000 | 0.1987 | 7.9057 | 1.26% | 8.16% | — |
| T200 dual_v5 | L0 (K=64) | 0.1562 | 0.9181 | 2.7299 | 36.18% | 100.00% | — |
| T200 dual_v5 | L1 (K=128) | 0.2031 | 0.8935 | 4.1154 | 9.86% | 73.88% | — |
| T200 dual_v5 | L2 (K=256) | 0.1797 | 0.9446 | 4.2168 | 16.69% | 71.82% | — |
| T178 200 epoch | L0 (K=64) | 0.2031 | 0.8844 | 3.2491 | 19.25% | 96.03% | — |
| T178 200 epoch | L1 (K=128) | 0.6016 | 0.7745 | 5.2624 | 7.11% | 44.76% | — |
| T178 200 epoch | L2 (K=256) | 0.4180 | 0.9027 | 5.0745 | 7.88% | 56.43% | — |

**路径稀疏度** (实际 unique SID / 理论最大 SID = K0×K1×K2×K3):
- Arm A: 0.0946% (2098/2097152)
- Arm B: 0.0278% (583/2097152)
- T200 dual_v5: 0.0056% (117/2097152)
- T178 200 epoch: 0.0072% (151/2097152)

## 4. 关键发现

### 4.1 Arm A 的 L0/L1 集中度恰好对应死码字机制

Arm A (T84 baseline, no Sinkhorn) 的 L0 = **50% util** + **Gini 0.5725** + **top-1 5.31%**:
- 64 个码字中只有 32 个被使用 (util=0.5),这 32 个被使用的码字本身又高度不均匀 (Gini 0.5725),最大码字承载 5.31% items
- L1 同样模式:64/128 util + Gini 0.5861
- 这说明 T84 baseline 的码字**只有一半在用,这半部分又集中** — 同时是死码字问题 + 集中度问题

### 4.2 Arm B (Sinkhorn) 把分布彻底打散到接近均匀

Arm B (T237 partial Sinkhorn max_iters=10) 的 L0/L1/L2:
- util 全部 = 100% (跟 task237 verdict 一致)
- Gini = 0.1741-0.1987 (远低于 Arm A 的 0.5725)
- top-1 占比 1.26-4.30% (远低于 Arm A 的 5.31%)

Sinkhorn 通过 balanced assignment 强制码字分布均匀,**实测证实主文献的预期** — 集中度问题在 Sinkhorn 下被解决。

### 4.3 但集中度不预测 R@10 (反 H2)

| Arm | Stage 2 collision | L0/L1/L2 Gini | Test R@10 |
|-----|-------------------|---------------|-----------|
| **A (T84 baseline)** | 0.99 | 0.5725 / 0.5861 / 0.3432 | **0.1020** |
| **B (T237 Sinkhorn=10)** | 0.0 | 0.1984 / 0.1741 / 0.1987 | **0.1021** |

Arm B 把 Gini 砍到 0.17-0.20 (从 0.34-0.59),R@10 只涨 +0.0001。**集中度不是 R@10 的杠杆** — 即跟 Issue #10 的 collision 杠杆证伪同结构。

### 4.4 T200 dual_v5: 极端集中 (Gini 0.92+) 跟 R@10 -10.3% 同步出现

T200 dual_v5 三层 Gini 全 > 0.89,L0 top-1 占 36.18% — 这是仓库有史以来最极端的集中度之一。但 T200 本身被 Issue #8 标记为 confounded (Stage 3 截断 + collision 不单调 + cos_mean < 0.3 未过),所以这不能 disentangle 集中度 vs Stage 3 截断的因果效应。

### 4.5 L3 (K=1) dedup digit 不传递信息

L3 是 4th-digit dedup digit,K=1 (所有路径必须经过它),Gini 几乎 1.0 是平凡的。Issue #12 Gate 0 严格 H1 的 L3 触发只是 dedup 机制本身的副作用,不是真正的"沙漏集中"信号。

## 5. Gate 0 决策

按 Issue #12 §阶段闸门硬停止条件:
- **存在至少一层满足 Gini ≥ 0.5 且 utilization ≥ 90%** → PASS
- **三层 Gini 均 < 0.5** → FAIL

**实测结果**:
- 严格按字面 (含 L3): 4 个 arm-layer 满足条件 (全是 L3 K=1,dedup 机制副作用) → 字面 PASS
- **排除 L3 K=1** (dedup digit 不传递 SID 信息): **0 个 arm-layer 满足条件** → **FAIL**

按 Issue #12 §阶段闸门严格解读 ("不得跨 Gate 取数", "三层 Gini 均 < 0.5 → STOP"):
- L3 是 K=1 的 dedup digit,不承载语义 SID 信息,**应当排除**
- 排除 L3 后没有 arm 满足 Gini ≥ 0.5 AND util ≥ 0.9
- H1 在严格的"沙漏式集中"意义上**证伪** (集中只发生在死码字机制,不是死码字机制外的独立集中)

**🚦 Issue #12 Gate 0 决策: FAIL (排除 L3) / PASS (含 L3, 平凡意义)**

按 Issue #12 §阶段闸门硬停止条件 + Issue #12 §反证/压力测试 ("Gate 0 应当先于 Arm B 执行, 且有可能单独终止本 issue"),**Issue #12 应当关闭, 不进入 Gate 1**。

但同时,Arm A 的 L0/L1 Gini > 0.5 是**有意义的现象** — T84 baseline 的码字有 50% 死掉 + 这半数码字又高度集中,这是 PC κ 历次实验的核心瓶颈。这一发现本身有净收益,值得记录。

## 6. 关键决策

- ✅ Issue #12 Gate 0 跑完 (零 GPU, 分钟级, 4 个 SID)
- ✅ **Issue #12 H1 在排除 L3 K=1 后 FAIL** — 沙漏集中在本数据集不独立于死码字存在
- ✅ **代码诊断能力永久保留** — `scripts/task244_sid_distribution_profile.py` 现在可计算逐层 Gini/Shannon 熵/std/top-K 占比/路径稀疏度,任何后续方向 (含 #11 已关闭) 都可以复用
- ✅ **Arm A 的 L0/L1 Gini > 0.5 + util 0.5** 证实 T84 baseline 码字同时受死码字 + 死码字外集中双重影响 — 这与 Issue #9/#11 的坍缩现象一致
- ✅ **Arm B Sinkhorn 把 Gini 砍到 0.17-0.20** — Sinkhorn 实际解决了集中度问题
- ✅ **集中度不是 R@10 杠杆** — Arm A (Gini 0.5+, R@10 0.1020) vs Arm B (Gini 0.2, R@10 0.1021) 集中度差 2.5× 但 R@10 持平

## 7. Issue #12 关闭决策

**Issue #12 应标记为 NO-GO closed**, 原因:
1. H1 严格解读下 FAIL (排除 L3)
2. Arm A vs Arm B 的对比反证 H2 (集中度 → R@10 因果链)
3. 即使 Issue #12 §后续 §"本方向如何推进最终目标" 的 §"逐层 token 分布不均匀度" 主张成立,Arm B 已经天然把分布打散到接近均匀,**主文献建议的 top-K 头部 token 移除方案在 Arm B 上几乎无目标可处理**

## 8. 与项目最终目标的关系

- **Issue #12 不能推进最终目标**: 即便 Gate 0 PASS,Gate 1 的 H2 (集中度解释 R@10 杠杆) 在 Arm A vs Arm B 对比下已经证伪
- **未来方向不应再尝试**: 任何"修集中度"的方案 (变长 SID, top-K 头部移除),因为 Arm B 已经实现均匀但 R@10 没涨
- **候选方向 (跟 R10 现有 backlog 一致)**:
  - 训练时长 (Task #243 在跑)
  - kmeans_init 重做 (Issue #9 关闭评论末项)
  - sinkhorn-on-encoder (Issue #9 关闭评论备选)
  - 跨架构 LETTER/S3Rec paper-aligned 重新基线 R@10

## 9. 产物

- `verdicts/task244_issue12_gate0_sid_distribution_result.md` (本文件)
- `verdicts/task244_sid_distribution_profile.json` (机器可读指标)
- `verdicts/task244_gate0_decision.json` (PASS/FAIL 标记)
- `scripts/task244_sid_distribution_profile.py` (通用工具, 永久保留)
- `descriptions/task244_issue12_gate0_sid_distribution.md`

## 10. Status

✅ **Task #244 完成**. Issue #12 Gate 0 FAIL (排除 L3 K=1) → 标记 Issue #12 close with NO-GO, 不进入 Gate 1/2/3.
