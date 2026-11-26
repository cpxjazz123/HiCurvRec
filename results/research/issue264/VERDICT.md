# Issue #264 verdict (in progress)

## Status

**Completed.** The optimized warm start reached the first real train record in
**30.063 / 29.331 / 28.869 s** (mean **29.421 s**) versus the old **332.613 s**:
**91.2% reduction**, all ≤60 s. The diagnostic was stopped after the first train
record; no test was evaluated and no formal 150-epoch run was started.

## Root cause

The measured old launch spent **275.78 s before torchrun** and **332.61 s before
the first epoch train record**. The old generated harness imported
`torch/transformers/train_HG-Rec` in its parent before spawning ranks. The cold
shared-filesystem `transformers` import is independently reported in #258 as
about six minutes; a warm parent import in this environment is 5.413 s. Therefore
the delay is the eager parent import, not a training or NCCL failure. The old
parent also serialized that cold import before every rank could start; the rank
side independently repeated it.

## Implemented change

- The generated parent is stdlib-only: it stages the small dependency package
  set to a node-local content cache and starts torchrun/one child without
  importing the trainer.
- `tools/stage3/startup_cache.py` stores only installed package files, with a
  SHA-256 manifest, source-tree fingerprint, atomic rename and `flock`.
- Preserved: trainer source, SID, split, frozen 150-epoch protocol, beam/seed,
  model init, NCCL variables, port 50201, single/4-rank branch, exit code and
  formal log/result paths. No data cache and no SID/state cache were added.

## Measured evidence

| item | value |
|---|---|
| Old request→torchrun | 275.78 s |
| Old request→first epoch train end | 332.61 s |
| New parent, 3 warm diagnostics | 2.72 / 2.71 / 2.76 s |
| Cold source cache preparation | 96.958 s |
| Warm cache preparation | 2.519 s |
| Rank import, source first | 103.725 s |
| Rank import, local cache | 5.386 s |
| Rank import, local cache repeat | 5.094 s |

Parent-startup reduction: warm **99.0%**; cold-cache **63.5%**.

The final non-formal diagnostics measured request→first real train record
**30.06, 29.33, 28.87 s** (mean **29.42 s**) against the old **332.61 s**, a
**91.2% reduction**; all three are below 60 s. The first epoch loss is 4.612525
versus the old 4.612519 (float nondeterminism), with the same frozen SID,
trainer, split, seed and protocol. The diagnostic variant was stopped after the
first train record and never evaluated test.

## Tests

`tools/stage3/tests/test_startup_cache.py`: **7 passed** — reuse, source
invalidation, corrupt repair, missing package, concurrent atomic publication,
parent no-torch import/exit propagation, single-GPU env preservation. The full
suite was run with the Stage3 Python.

## Acceptance

| ID | Status | Evidence |
|---|---|---|
| C1 request→first step phases | PASS | before `275.78/332.61 s`; final `30.063/29.331/28.869 s` |
| C2 historical/current baseline | PASS | #263 timestamps; #258 historical cold import |
| C3 lightweight parent | PASS | parent 2.72/2.71/2.76 s; no trainer import at parent |
| C4 entry/single/multi/exit/log preserved | PASS | 7 tests; source diff only harness/cache |
| C5 cache correctness | PASS | hash manifest, stat+content validity, cold/warm, corrupt/missing/concurrent tests |
| C6 equivalence | PASS with float tolerance | first epoch loss new 4.612519-4.612525 vs old 4.612519; same SID/trainer/split/seed/protocol; no test read |
| C7 phase logs | PASS | `_stage3_launcher.log` parent start/deps/prewarm/spawn |
| C8 performance | PASS | 91.2% reduction; 3 repeats; warm ≤60 s; cold cache reported |
| C9 no #263 disturbance | PASS | #263 closed separately; its artifacts unchanged by #264 |
| C10 contract/evidence/diff/verdict | PASS | this directory, `PERFORMANCE.json`, `INFRA_COMPARE.json` |

## Limits

- The optimized diagnostic is a non-formal short run, not a 150-epoch result.
- No new full Stage3/Stage2 was run; #263 was not modified.
- Cache is node-local and ephemeral by design; cold construction cost is
  measured and reported, not hidden.
- The first epoch loss difference is sub-float nondeterminism (4.612519 vs
  4.612525), not an intentional training change.
