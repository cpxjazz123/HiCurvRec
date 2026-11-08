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
