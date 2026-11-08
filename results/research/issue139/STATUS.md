# Issue #139 — status (2026-11-08)

## 0. The goal, restated by the owner

Complete the Stage 2 of all five mechanisms, record the structural distortion of
the five resulting SIDs, and run the corresponding Stage 3 for each of them.
That widens the earlier Stage-2-only framing, and it has two consequences that
are recorded rather than assumed:

* **The learned collision gate no longer gates these five arms.**
  `tools/genrec-mcp/stage2_quality_gate.py` carries an explicit
  `ISSUE139_STAGE3_EXCEPTION` naming the five arm directories; the collision is
  still measured and recorded as a diagnostic, the exception is written into the
  admission evidence, and nothing else is relaxed. A test pins both halves: the
  excepted arms pass and an unrelated arm with the same collision is refused.
  Recorded in `CLAUDE.md` as the owner's instruction of 2026-11-08.
* **The sequence is now five Stage 2 runs, then five measurements of the
  structural distortion, then five Stage 3 runs**, each about two and a half
  hours on four cards, so the Stage 3 module is the long pole.

### Where the five arms stand

| arm | Stage 2 | distortion recorded | Stage 3 |
|---|---|---|---|
| F0 | done | yes (0.296474) | pending the exception, which now exists |
| G | done | yes (0.295375) | pending |
| E | smoke running, then training | — | — |
| A | not started (admitted) | — | — |
| L | not started (admitted) | — | — |

**State: in progress.** Two of five arms are complete on the reviewed source, one
is training, two have not run. The core hypothesis has been tested directly and
holds, with a small margin. Nothing here is a Stage 3 result.

## 1. The source being measured

Everything below is the reviewed source, commit `3ee9374c` (the review changed the
mechanism path of the trained arms by 52 files, +7498/−1238, and fixed defects on
it: the repeated squared cost, the tangent-to-ball mapping, the incremental degree
dependencies, the rank-zero codebook init and the alpha rollback).

Three regressions that review introduced had to be repaired before any arm could
be dispatched, each with a regression test:

| regression | symptom | fix |
|---|---|---|
| the geometry files lost their "inherited" exemption | all five arms failed `geometry_contract`; the baseline's own `expmap0` read as candidate geometry | restored the baseline's bytes (the tooling had rewritten `utils.py` CRLF→LF and added a newline to `poincare_contract.py`) |
| a pre-review checkpoint probed against the reviewed model | smoke died on `Missing key(s): "rq._extra_state"` | compatibility probe before the smoke; an incompatible checkpoint counts as absent |
| `PYTHONUNBUFFERED` dropped | a healthy run left a zero-byte log | restored, with `tools/research/stage2/tests/test_launch_env.py` |
| `bootstrap.py` never committed | a checkout deleted the only fork-free launch path; the smoke passed and training failed | rewritten and committed |

The NCCL overrides were also dropped by the review. The earlier two-way
measurement says they were never the cause (the hang was the fork), so they are
not restored; the test records that instead of pinning them.

## 2. The arms

| arm | state | wall | final collision | L3 collision | L1 positive hit | L1 hard ER | L2 prefixes | valid SID |
|---|---|---|---|---|---|---|---|---|
| F0 | **done** | 552.7 s | 0.2579 | 0.3885 | 0.1429 | 4.586 | 4,256 | no (0.3885 > 0.35) |
| G | **done** | 975.8 s | 0.2330 | 0.3572 | 0.1315 | 5.017 | 4,631 | no (0.3572 > 0.35) |
| E | **crashed at its first round** | — | — | — | — | — | — | — |
| A | not started | — | — | — | — | — | — | — |
| L | blocked | — | — | — | — | — | — | — |

**E died at its epoch-1500 round, and the reason is a measured cost, not a bug
in the mechanism.** Its last epoch (1499) was recorded at 01:15:48 and the
collective watchdog fired at 01:25:49 — exactly the default NCCL timeout of 600
seconds — with `Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=48043, OpType=BROADCAST)`, the weight-table broadcast at the
round's boundary. G's rounds finish in 108 seconds; E's first round did not
finish in ten minutes. The difference is the arm: with the curvature at zero the
codes are coarser, so the qualifying prefixes are bigger, so each per-prefix
candidate evaluation updates far more pairs. The review removed the wall-clock
ceiling this work had introduced as an unapproved policy, and there is now
nothing between that cost and the collective timeout.

Three ways forward, and only one of them is mine to take: raise the
process-group timeout so a long round is not aborted (an infrastructure limit,
not a scientific one); build the exact incremental path for large prefixes; or
have the owner approve a declared per-round budget. E is idle until one of them
is chosen, and L would hit the same wall for the same reason.

G's outer loop accepted 2 of its 12 candidate weights: one at epoch 2000
(D 0.29688 → 0.29686) and one at 2500 (D 0.29693 → 0.29656); rounds at 1500 and
3000 accepted nothing. About 108 s per round.

The frozen shared readout was written by F0 at its 1500th epoch with the
reviewed schema and identity digest: tau 0.507976, intercept −0.041207, mean
predicted probability equal to the true edge density to seven digits. The
pre-review one is archived under `premove/`.

## 3. The core comparison (this is the hypothesis)

Same graph, same 940,498-pair support, same frozen readout, same Huber scale:

| quantity | F0 | G | G − F0 |
|---|---|---|---|
| **D_AF3 of the final hard SID** | **0.296474** | **0.295375** | **−0.001100** |
| D_AF3 of the continuous encoder output | 0.298475 | 0.297905 | −0.000570 |
| quantization cost | −0.002001 | −0.002531 | −0.000529 |
| adjacency Brier | 0.2519 | 0.2513 | −0.0006 |
| neighbour retention | 0.0108 | 0.0114 | +0.0005 |

`D(G) < D(F0)` **holds**, by 0.37 % relative, and the three independent
quantities agree in direction. Read it as "the weight selection moves the
quantization the right way in this run", not as a strong effect.

The decomposition matters: for both arms the discrete representation has a
*lower* distortion than the continuous one, so the codebook here acts as a
smoother and what remains is mostly the encoder's own error. Absolute neighbour
retention is ~1 % for both — only the comparison between arms is meaningful.

An earlier run of this measurement reported the opposite sign because the
readout was compared without gathering it through the training-item ids; the
graph's node ids are training indices, the readout is rowed by item id. Both
numbers are recorded in `STRUCTURAL_FIDELITY.md`, because the wrong one is the
more instructive.

## 4. What is not done

- **A** has not run; **E** has not finished; **L** fails closed pending an
  approved fallback-transition specification.
- The differential check across three independent derivations was not run.
- The frozen-snapshot diagnostics of the issue's §3.1 (Forman instead of AF3,
  shuffled AF3, separated degree/triangle errors, shuffled local table) were not
  run.
- No arm entered Stage 3, so nothing here says anything about recall.
- The pre-review artifacts of F0 and G are archived under
  `results/stage2_RQ-VAE/_prereview_*` and are *not* results of this source.

## 5. Files

| file | what it holds |
|---|---|
| `REPORT.md` | the earlier report (pre-review state); superseded for the numbers by this file |
| `STRUCTURAL_FIDELITY.md` | the core comparison, its decomposition and the alignment caveat |
| `structural_fidelity.json` | the raw measurement |
| `measure_structure.py` | the script that produces it, on one footing for every arm |
| `reviewed_arms_so_far.json` | the reviewed arms' layered metrics, selected by the live SID digest |
| `arms_table.md`, `arms_summary.json` | the arm table from `collect_arms.py` (pre-review numbers, kept for contrast) |
| `precheck.json` | the measured graph, support, readout cost, engine timings, active-prefix counts |
| `premove/` | the pre-review shared readout |

## 6. 单卡主机与共享校准（2026-11-08 晚）

**主机只剩一张 A100**，四卡配方不可用。七个训练器（含冻结基线 `TIGER_RQ-VAE`）都加了单卡分支：可见卡 ≤1 时**不 fork torchrun**，单进程运行，**per-process batch 4096 = 四卡全局 batch**（每 epoch 6 个优化步、同一批样本，仅求和顺序不同）。实测单卡 F0：**collision 0.2572**，四卡版 0.2579 ⇒ 该配方复现四卡结果。

**共享校准（τ、b）** 是把 SID 读出换算成"边概率"的标尺，由 F0 在 epoch 1500 冻结、其余四臂共用（否则各用自己的尺子量自己，D_AF3 不可比）。当前在用：**τ = 0.374305，b = −0.044043**，来源 F0@1500。

**溯源摘要改为"记录、不拦截"（owner 2026-11-08 指令）**：此前六项 SHA-256 不匹配即拒绝运行，导致"改一行代码 ⇒ 校准作废 ⇒ 重训 F0"的循环，重复三次。现在摘要照旧写入校准与各臂记录（报告可核对是否共用同一溯源），但只有尺子本身的性质 fail-closed：schema、来源必须是 F0@1500、τ 有限为正、平均预测概率等于真实边密度。

**Stage2 可并行（owner 2026-11-08 规则，CLAUDE.md §19d0b）**：多个 Stage2 可同时运行；为此每个 launch 取自己的 rendezvous 端口（首个仍用 50202）。

**已修的相关缺陷**（每条都有对应记录）：fork-after-import 死锁（改由无 torch 的 `bootstrap.py` fork）、`bootstrap.py` 曾未入库、MCP 无条件 `--world-size 4`、单卡分支缺失、端口复用、归档目录按秒命名、并发准入共用临时目录、支撑集顺序不可复现、冒烟探针的模型签名。

## 7. 五臂单卡结果（2026-11-08 18:27，全部 3000 epoch 完成）

同口径：同一训练图、940,498 对支撑、共享校准 τ=0.374305 / b=−0.044043、同一 Huber 尺度。

| 臂 | D_AF3（硬 SID） | 连续表示 | 量化代价 | Brier | 邻居保持 | collision | 外循环接受 |
|---|---|---|---|---|---|---|---|
| A | 0.289100 | 0.291957 | -0.002857 | 0.2480 | 0.0109 | 0.3137 | — |
| G | 0.291589 | 0.294850 | -0.003261 | 0.2497 | 0.0110 | 0.2807 | 2 |
| F0 | 0.295281 | 0.298137 | -0.002856 | 0.2516 | 0.0113 | 0.2572 | — |
| L | 0.297422 | 0.300854 | -0.003431 | 0.2520 | 0.0109 | 0.2975 | 370 |
| E | 0.298023 | 0.300694 | -0.002671 | 0.2532 | 0.0112 | 0.2161 | 346 |

**核心假设**：$D(G)-D(F0) = -0.003692$ ⇒ **成立**（四卡版为 −0.0011，单卡更大）。

**三条结论**：
1. A（F0 + AF3 辅助损失）依旧最优 ⇒ "加损失"优于"选度量"；
2. G 胜过 F0 ⇒ AF3 选出的全局权重确实让 SID 更贴近原图；
3. **E 与 L 均劣于 F0**：两臂各接受了 346 / 370 次权重改动、每轮都把失真压低 2–3×10⁻⁴，但训练结束时反而更高 ⇒ 轮内改善未在训练中留存（issue 预警的 structure-only / tradeoff 情形）。

**权衡**：L 的失真更好（0.297422 < 0.298023），E 的码本更均匀（collision 0.2161 < 0.2975）⇒ 双曲与欧氏的局部选择各胜一项，不存在"非欧几何必要"的单一答案。

**未完成**：五个 Stage3（单卡约 8–10 h/臂，合计 40–55 h）；报告因此仍为 INCOMPLETE。

## 8. 五臂同口径结果（单卡，2026-11-08 18:27）

同一张训练图、同一 940,498 对支撑、同一共享校准（τ = 0.374305、b = −0.044043）、同一 Huber 尺度（图自身的 κ IQR）：

| 臂 | D_AF3（硬 SID） | 连续表示 | 量化代价 | 邻接 Brier | 邻居保持 | collision | 外循环接受 |
|---|---|---|---|---|---|---|---|
| **A** | **0.289100** | 0.291957 | −0.002857 | **0.2480** | 0.0109 | 0.3137 | — |
| **G** | **0.291589** | 0.294850 | −0.003261 | 0.2497 | 0.0110 | 0.2807 | 2 |
| F0 | 0.295281 | 0.298137 | −0.002856 | 0.2516 | 0.0113 | 0.2572 | — |
| L | 0.297422 | 0.300854 | −0.003431 | 0.2520 | 0.0109 | 0.2975 | 370 |
| E | 0.298023 | 0.300694 | −0.002671 | 0.2532 | 0.0112 | **0.2161** | 346 |

**核心假设** `D_AF3(G) < D_AF3(F0)`：**成立**，差距 −0.003692（四卡版为 −0.001100）。

**三条结论**

1. **A（F0 + AF3 辅助损失）结构保真最好**（0.2891），与四卡版一致 ⇒ 就这条判据而言，"把 AF3 当作损失加进去"优于"用 AF3 去选几何度量"。
2. **G 胜过 F0**（0.2916 vs 0.2953）⇒ 由 AF3 选择的全局几何权重确实让最终 SID 更贴近原图，效应量小但方向稳定（单卡与四卡同向）。
3. **E 与 L 都劣于 F0**（0.2980 / 0.2974 vs 0.2953）⇒ 两个局部臂各接受 346 / 370 次权重改动、每轮都把 D 压低 2–3×10⁻⁴，但**训练结束时的失真反而更高** ⇒ 轮内改善没有被训练留存，属 issue 明示的 structure-only / tradeoff 情形，不能称成功。

**E 与 L 是一组真实的权衡**：L 的失真更低（0.2974 < 0.2980），E 的码本更均匀（collision 0.2161 < 0.2975）⇒ 双曲与欧氏的局部选择各有胜负，"非欧几何必要"得不到支持，但也不能说它有害。

**注意口径**：单卡的全局 batch 与四卡相同（4096），但样本到优化步的分配不同（单进程整批 vs 四卡各 1024 再平均），因此轨迹分叉；F0 几乎完全重合（0.2572 vs 0.2579），G/A 有可见偏移。两套数字并列保留，不混用。

**仍未完成**：五个 Stage3（单卡每臂 8–10 h，合计 40–55 h）；differential 检查与 §3.1 冻结快照诊断未做；因此本文件不含任何召回结论。

## 9. 冻结基线 TIGER 的同口径测量（2026-11-08 18:56）

此前五臂只互相比较，缺少"相对原始 TIGER 如何"的答案。基线从未导出 `item_sids.json`，因此用它的 `sids_for_hgrec.npy` 与 `rqvae_best.pth` 重建同样的 32 维读出（三层码字的**欧氏和**代替混合读出），其余口径完全不变（同图、同 940,498 对支撑、τ = 0.374305、b = −0.044043、同 Huber 尺度）。脚本：`measure_baseline.py`。

| 臂 | D_AF3（硬 SID） | 连续 | 量化代价 | Brier | 邻居保持 | 相对 TIGER |
|---|---|---|---|---|---|---|
| **TIGER** | **0.309848** | 0.314043 | −0.004195 | 0.2593 | **0.0123** | — |
| A | **0.289100** | 0.291957 | −0.002857 | **0.2480** | 0.0109 | **−6.70%** |
| G | 0.291589 | 0.294850 | −0.003261 | 0.2497 | 0.0110 | −5.89% |
| F0 | 0.295281 | 0.298137 | −0.002856 | 0.2516 | 0.0113 | −4.70% |
| L | 0.297422 | 0.300854 | −0.003431 | 0.2520 | 0.0109 | −4.01% |
| E | 0.298023 | 0.300694 | −0.002671 | 0.2532 | 0.0112 | −3.82% |

**结论一（回答"是否比 TIGER 更好"）**：**是**。五个臂的 AF3 结构失真全部低于冻结基线，A 最好（−6.70%）；仅混合空间本身（F0）就已比 TIGER 低 4.70%，AF3 辅助损失再贡献约 2 个百分点。

**结论二（必须同时披露的反向指标）**：**TIGER 的邻居保持率最高**（0.0123 vs 0.0109–0.0113），且量化代价最负（−0.0042）⇒ 基线码本"平滑"更强，绝对近邻结构反而更好；但它的邻接 Brier 最差（0.2593）。三个结构量并不一致，因此正确表述是：**在 AF3 目标本身（度数 + 三角形）上，五个臂都比基线更贴近原图**，不能扩展成"在所有结构意义上都优于 TIGER"。
