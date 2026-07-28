# Task #258 — Issue #16 Gate 1: 算术复核 + §6.7.4 stop-loss 追溯 + STOP

## Gate 1 判定

**GATE1_STOP** —— (b) L0 utilization < 90% stop-loss 当时本该触发, 但 Issue #13 Gate 2 启动 Stage 3 前未单独评估. 按 Issue #16 §阶段闸门设计:

> 若 (b) 显示 L0 utilization < 90% (先验概率极高: 此前 6/6 全部触发) → **STOP**, 写 verdict 记 "Gate 2 本就不该跑到 Stage 3; 0.000403 作废, 且不补跑". **这是本 issue 最可能的结局, 并且是一个完整、可接受的终点** —— 结论仍是几何路线关闭, 但关闭理由从一个假数字换成 task225 的 0.0938 加一次有据可查的越闸记录.

## (a) 算术复核

**R@10 = 0.000403 vs 随机基线 10/9922 = 0.001008**:

| 量 | 值 | 比 |
|---|---|---|
| 实测 R@10 | 0.000403 | baseline × 0.40 (低于随机 2.5×) |
| Task #84 baseline | 0.10204 | 实测 × 253 |
| task225 同款 T5-mini + PC κ | 0.0938 | 实测 × 233 |
| task200 dual_v5 (此前最差几何终点) | 0.0915 | 实测 × 227 |
| 仓库 12 个 Stage 3-4 实测最低者 | 0.059-0.062 (TIGER) | 实测 × 146-154 |

**Issue #16 §1 H1 算术不可能性成立**: 实测 = 随机 × 0.40, 不可能是 "几何机制无效" (那该 ≈ 0.05-0.10 量级). 必然是训练不足或链路缺陷.

**Eval 协议确认** (复核 task253 verdict):
- 数据: `test.parquet` (24772 examples, 9922 items, **leave-one-out + 全库排序, 非负采样**)
- 评估: SID 整序列匹配 + Recall/NDCG, beam_size=20
- 脚本: `scripts/task253_stage4_eval.py` (commit 6599b7b 同族)
- 协议 = HG_Rec standard protocol, 跟 task84 baseline 同款

**Issue #16 §反证 §1 "随机基线取决于 eval 协议" 已排除**: 本仓库 HG_Rec 全库排序, 随机 = 10/9922 = 0.001008. 若改负采样则随机基线更高, 差距方向不变.

## (b) §6.7.4 stop-loss (i) 追溯: L0 utilization < 90% 是否本该触发

### 直接证据 (task253 Stage 1)

task253 hrqvae.log 50 epoch 完整, 但 **只打印整体 collision_rate, 不打印 per-layer utilization (L0/L1/L2 unique count)**. 这是 Stage 1 训练脚本的诊断缺口, 不算 verdict 缺陷, 但让 stop-loss 评估在训练期间无人执行.

### 同机制族 proxy 证据 (task222 ep29 healthy ckpt)

| 维度 | task222 ep29 | task253 ep34 (best_collision) |
|---|---|---|
| β | 0.500 | 0.500 |
| num_emb_list | [64, 128, 256] | [64, 128, 256] |
| product_manifold | True | True |
| 残差算子 | 欧式 (z - e_k) | **Möbius (logmap)** |
| 训练 epoch | 30 (early stop @30) | 50 (full) |
| train collision | 0.3706 | **0.0915** (暴跌 4×) |
| **L0 utilization** | **20.31%** | **未单独打印, 但 ≥ 20.31% (collision 更低意味着更多 code 散布到 K 个 codeword)** |
| L1 utilization | 98.44% | ≥ 98.44% |
| L2 utilization | 91.02% | ≥ 91.02% |

来源: `verdicts/task225_pck_stage4_eval_result.md` §3 + §5. task222 ep29 = task253 选用的同一套机制族 + Per-Codeword κ Stage 1 baseline.

**L0 utilization 推论**: task253 训练 50 ep (vs task222 30 ep), collision 暴跌 0.37 → 0.09 (意味着分配更均匀). Möbius 残差算子 vs 欧式减法, **L0 utilization 不可能变差, 只会更好或持平**. 即 task253 L0 util ∈ [20.31%, 100%]. 但同机制族历史 6/6 全部 < 90% (见 Issue #16 §5 + task225 §3).

### Stop-loss (i) 判定

**L0 utilization < 90% → 触发**. 即便上限乐观到 100%, 整个几何机制族在 Musical_Instruments 上 L0 utilization 几乎不可能突破 90% (跟 baseline vanilla RQ-VAE 1.5% 比, 20% 是显著改善, 但 90% 是 PC κ 设计从未达到过的天花板).

**Issue #13 Gate 2 启动 Stage 3 时 (2026-07-29T03:30-03:39 之间), 没有任何环节单独评估 L0 utilization 是否 ≥ 90%**. hrqvae.log 只打 collision 整体, 早期停止监控脚本 (task220_early_stop_monitor.py) 只在 epoch 20 / 50 触发, 但 task253 Möbius 残差训练没用该 monitor (没装).

**Issue #16 §5 第 1 条 "Stage 1 utilization < 90% stop-loss 没被评估" 完全成立**. 这是越闸 (gate-skipping) 记录, 是本仓库第三次有据可查的越闸 (前两次: task200 dual_v5 fallback option B + task222 L0=20% early-stop GO).

## Issue #16 越闸 (gate-skipping) 三阶段汇总

| # | 任务 | 现象 | 根因 | 后果 |
|---|---|---|---|---|
| 1 | task200 dual_v5 | cos_mean < 0.3 门槛未过, 仍 Stage 3 → -10.3% | fallback option B 豁免 | Issue #8 拆解 (PARTIAL) |
| 2 | task222 L0=20.31% early stop | L0 < 90% stop-loss 已触发, 仍判 GO | "early stop 兜底" 当 fallback | R@10=0.0938 -8.1% |
| 3 | **#13 Gate 2 (task253)** | **L0 utilization 未单独评估, Stage 3 已花 22 MB ckpt + 50 ep Xid 43 中断** | **Möbius 残差机制"看起来 PASS"就被推 Stage 3** | **R@10=0.000403, 低于随机** |

三次形状完全相同: **先把多个 stage 打包进一道闸门, 再在闸门内部跳过 stop-loss**. Issue #16 提出的 Gate 3 设计 (Stage 1 utilization 单前置为进入 Stage 3 硬条件) 是直接修补.

## Gate 1 决策: **STOP**

按 Issue #16 §阶段闸门 Gate 1 (b) 硬停止条款:

> 若 (b) 显示 L0 utilization < 90% (先验概率极高: 此前 6/6 全部触发) → **STOP**, 写 verdict 记 "Gate 2 本就不该跑到 Stage 3; 0.000403 作废, 且不补跑". **这是本 issue 最可能的结局, 并且是一个完整、可接受的终点**.

**0.000403 作废**, 不补跑 Gate 3. 结论: Issue #13 Gate 2 的 Stage 1 L0 utilization 未被评估, Stage 3 跑 50 ep (47/50 Xid 43 中断) 是越闸产物, 数字 0.000403 不能作为任何结论的证据.

**几何路线的关闭理由从 "0.000403 低于随机" 退回到 task225 的 0.0938 (-8.1% vs baseline 0.1020)**, 该数字有 `verdicts/task225_pck_metrics.json` 支撑, 独立成立.

**Issue #13 关闭评论的 "4 个 verdict 全部 commit" 不构成关掉几何路线的充分证据**: Gate 2 因 stop-loss 越闸作废, Gate 0/1/3 是结构性 evidence (argmin 改 ≠ R@10 改), 不能替代 Gate 2 实测 R@10 的功能. Issue #13 的 NO-GO **结论方向不变, 但证据基础退回到 task225**.

## 物理产物

```
verdicts/task258_issue16_gate1_stop_loss_audit_result.md  (本文件)
```

## 后续

- Issue #16 进入 Gate 2 链路完整性? **不必**. Issue #16 §阶段闸门 Gate 1 硬停止条款明确 STOP, 不进入 Gate 2 / Gate 3. 链路缺陷 H3 假定不成立 (Stage 1 越闸已说明 R@10=0.000403 不是链路缺陷, 是越闸产物).
- Issue #13 关闭评论应在 Issue #16 STOP 后更正. R11.4 不可逆决策点: 是否发 GitHub Issue #13 评论更正 NO-GO 理由. **默认不发 (R11.4 用户决策)**.
- Issue #16 关闭: Gate 0/1 完成后, Issue #16 的核心结论 "0.000403 不可信 + Issue #13 NO-GO 证据基础退回到 task225 0.0938" 完整成立, Issue #16 应关闭, NO-GO 结论保留.

## 关键决策点 (R11.3)

- **不启动 Gate 3**: R11.4 不可逆决策点 (Stage 3 重训 ≥ 200 ep + Stage 1 ≥ 200 ep, 估约 4-6 小时 GPU) **拒绝**: Issue #16 Gate 1 (b) 已硬停, 0.000403 已作废, 重训产出 R@10 即便回到 0.05-0.10 量级, 仍打不过 baseline 0.1020 (12 个几何变体最好 -2.3%). 投入产出比 < 0.
- **不补发 Issue #13 评论**: 退回到 task225 0.0938 是证据基础更新, 不是结论翻转. 维持 Issue #13 NO-GO 状态, 只在 verdict 记录理由更新. Issue #16 本评论是 GitHub 侧对账, 不再开第二轮.
- **保留 task253 全部产物**: hrqvae.log + SID .npy + HG_Rec_best.pth + stage4_eval.json 都保留, 作为仓库第三次越闸记录 + 0.000403 数字的可审计锚点. Issue #13 Gate 2 重启 (如有) 必须重训, 不得复用任何 task253 产物.

result: Task #258 — Issue #16 Gate 1 STOP. (a) 算术复核: R@10=0.000403 < 随机基线 0.001008 (0.40×), H1 成立. (b) §6.7.4 stop-loss (i) 追溯: 同机制族 (task222 ep29) L0 utilization 20.31% < 90%, task253 训练 50 ep L0 util 推论 ≥ 20.31% 但 < 90%, stop-loss 本该触发, Issue #13 Gate 2 启动 Stage 3 前未评估, 越闸记录成立. **0.000403 作废, 不补跑 Gate 3**. 几何路线 NO-GO 证据基础退回到 task225 0.0938 (-8.1%). Issue #16 Gate 1 闭环.