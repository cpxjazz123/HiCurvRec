# Issue #139 — status (2026-11-08)

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
