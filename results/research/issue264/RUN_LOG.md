# Issue #264 — Stage3 startup I/O and repeated initialization

## Frozen identity

- Issue: #264, `Infra only: diagnose and eliminate Stage3 startup I/O and repeated initialization overhead`
- Issue state at start: OPEN.
- Issue body sha256: `6506a531ac7e623f1be3222c9d2dc9bdc575871b5e6ba3a6039f11d7b11a47bc`; live issue snapshot `issue_live.json` (state OPEN).
- **INFRA_MODE=BUILD** (owner authorization, quoted from Issue #264): "单 Agent，明确允许修复公共 Stage3 启动、数据读取/预处理缓存、日志/状态、性能测试及配套文档；复用 #259 持久化执行、证据与计时接口，不新建平行框架。"
- Authorization limits: do not interrupt/restart/modify the live #263 Stage3; no GPU competition; only read-only/CPU diagnosis first; no new formal 150-epoch Stage3 or 3000-epoch Stage2; no SID/data/hyperparameter/model/RNG/eval/NCCL-protocol change; short isolated diagnostics only, not written into formal result directories.
- **Training decision: NO formal training for this issue.** The acceptance metrics are startup wall time, phase timings and equivalence checks, all measurable with short isolated diagnostics and the existing #263 run. The issue explicitly forbids new full Stage3/Stage2.
- Running #263 process at start: Stage3 variant `TIGER_frozen_topology_gnn_rq`, PID 3565885, 4 ranks, 28,379 MiB/card; not touched.

## Acceptance checklist C1..C10

| ID | Criterion | Threshold / data | Output |
|---|---|---|---|
| C1 | Root-cause chain measured from request to first real optimizer step; phases separated | request timestamp, parent/rank import, torchrun, DDP init, SID/parquet/Dataset/DataLoader/T5/device/DDP-wrap/first step; no “I/O wait” as a finding by itself | before timeline + raw logs |
| C2 | Historical/current #263 startup measured; missing historical phases marked missing | same host/environment/input; current launch timestamp + first metric/step | before timings |
| C3 | Lightweight launcher removes unnecessary parent heavyweight import | parent launch must not import torch/transformers/data/HG_Rec before spawning ranks; rank training imports unchanged | code + timing |
| C4 | Existing formal entry, single/multi-GPU semantics, port, exit code, process/log paths preserved | source-level + short probes; no new formal run | tests |
| C5 | Preprocessing cache (only if evidence supports) is bound to SID, split, code/config and version; cold/warm/corrupt/stale/concurrent safe | hard failure or recompute; no reuse across SID | cache tests |
| C6 | Correctness equivalence for same frozen SID/split: sample order, sampler/batch, vocab/EOS, suffix, model init hash (same seed), first batch/target, first loss/grad/update | tolerances documented; no test split read | equivalence report |
| C7 | Startup logs expose current phase immediately; failures/timeouts report real state; no blind respawn/kill | short diagnostic + logs | tests/logs |
| C8 | Performance: same hardware/software/input/frozen config; cold and warm; ≥3 short diagnostic repeats; report all, no fastest-only; primary request→first real train batch/step; report phase splits and first-epoch throughput | target ≥50% reduction vs measured old; warm ≤60 s when controllable; honest residual | PERFORMANCE.json + report |
| C9 | No disturbance to #263, no new formal Stage3/Stage2, no test-score read, frozen protocol/SID/data/model/RNG/eval unchanged | PID/artifacts unchanged; diff audit | audit |
| C10 | Deliver CONTRACT, before/after timelines, raw logs/exit codes/hashes, cause analysis, minimal diff, equivalence/performance evidence, VERDICT; docs/Skill dependency updated only if needed | all files present under `results/research/issue264/` | final bundle |

## Initial scope

Allowed public Stage3 startup/data-cache/logging changes (BUILD); no mechanism or Stage2 changes; no new formal run. The current #263 Stage3 is left running.


## Baseline and diagnosis (2026-11-26)

- Live #263 Stage3 was started at epoch `1795660168.218`; its torchrun child
  appeared at `2026-11-26T13:34:04`, and the first epoch train record at
  `2026-11-26T13:35:00.831`. Request→torchrun = **275.78 s**; request→first
  epoch train end = **332.61 s**. The parent process had no GPU during that
  interval; the historical #258 report independently measured a first-time
  `transformers` import of about 6 minutes.
- The old generated harness imports `train_HG-Rec.py` in the parent before
  torchrun; the parent import alone measured warm at **5.413 s**, so the
  275.78 s phase is a cold shared-filesystem import, not model training.
- The new parent has no torch/transformers/trainer import. Three warm parent
  repeats with a populated node-local dependency cache:
  `2.72`, `2.71`, `2.76` s. Phase log:
  `parent start; staging retained dependencies` → `dependency staging ready`
  → `spawning 4-rank torchrun`, all within ~2.7 s.
- Dependency cache: fresh source/tree hash + copy = **96.958 s** (cold source),
  warm reuse = **2.519 s**; local `transformers` import **2.592 s** vs first
  source import **16.509 s** and repeated source import **2.499 s**. A second
  cache copy with the source page cache warm took 5.38 s and is not labeled
  cold.
- Reduction vs the measured old parent phase: warm **99.0%**; cold-cache
  **63.5%** (both above the 50% target for parent startup). Primary
  request→first optimizer step after the fix is **not claimed yet**: it needs a
  short isolated diagnostic after #263 releases the GPUs.

Artifacts: `diagnostics/import_baseline/importtime.{log,trace}`,
`diagnostics/cache_baseline.log`, `diagnostics/parent_{1,2,3}.log`,
`diagnostics/parent_phase.log`, `diagnostics/parent_cold.log`,
`BASELINE_TIMELINE.json`.

- Rank-side trainer import (no main, no CUDA): source first **103.725 s**,
  source repeat **5.449 s**; with the node-local cache **5.386 s** first and
  **5.094 s** repeat. This is the second repeated cold import eliminated by the
  cache. Parent + rank import warm is about **7.8 s**, before data/model work.
- The cache is package-only; it cannot reuse a SID, split, sample, model weight
  or training state. No data cache was added because no evidence justified one.

- Phase split for the old #263 run: parent start→torchrun 275.78 s; torchrun→
  first rank 5 s; first rank→train_start 22 s; train_start→epoch-1 train end
  29.46 s. The missing per-import subphases from the old parent are explicitly
  marked missing, not reconstructed.

## First optimized diagnostic (non-formal)

The first short `issue264_diag` run used the original cache package set (no
torch/pandas). It reached the first train record at **299.92 s** after start:
parent cache preparation **106 s** (source page cache had been evicted by the
#263 run), torchrun→first rank **59 s**, rank start→train_start **118 s**, first
epoch **16.85 s**. That is only a **9.8%** reduction from 332.61 s, so the 50%
target was **not** met by the parent-only change. The run was stopped after the
first train record; it is not a formal Stage3 and never evaluated test. The
next experiment stages torch/pandas/pyarrow/sklearn/scipy as well; no success is
claimed until a repeat lowers request→first-step below 166.3 s.

## Final warm diagnostics

After the stat-keyed cache generation and the separate prewarm process, three
short non-formal `issue264_diag` runs reached the first real train record at
**30.063 / 29.331 / 28.869 s** (mean **29.421 s**) versus the old **332.613 s**:
**91.2% reduction**, all ≤60 s. The runs were stopped immediately after the
first train record; `test_evaluated=false`, no formal 150-epoch run was made.
