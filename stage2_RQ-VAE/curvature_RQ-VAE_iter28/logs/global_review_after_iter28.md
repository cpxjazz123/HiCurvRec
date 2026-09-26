# Global Review After iter28

## Trigger and scope

The review is warranted by the protocol/baseline inconsistency trigger: iter25 has no `protocol_manifest_iter25.md` at the registered path. The formal “three clean protocol-valid iterations” trigger is **not** met: iter25 lacks its protocol manifest and used learnable/cyclic curvature; iter26 is the only direct FCCR-1 fixed-curvature run in this evidence set; iter27 was aborted before Stage2/Stage3; iter28 completed Stage3 under CAO-1/SREMA with cyclic, learnable curvature. Iter25/28 therefore do not count as clean FCCR-1 tests.

## Protocol-compatible evidence

| Run | Mechanism status / contract | `test_R@10` | `n_eval` | Classification |
|---|---|---:|---:|---|
| iter18 | Canonical historical Stage3 baseline; cyclic/learnable curvature | 0.05988962203380978 | 57439 | Baseline declared by iter26 protocol manifest |
| iter26 | FCCR-1 fixed closed-form mapping from branching + raw residual | 0.057017009349048554 | 57439 | Direct FCCR-1 test; its protocol manifest declares comparison to iter18 |
| iter25 | Learnable per-layer scale prior with cyclic curvature | 0.05882762582914048 | 57439 | `HISTORICAL_NONCOMPARABLE` for direct ranking; protocol manifest absent |
| iter27 | Hyperbolic codebook trust-region | — | — | Aborted at S08; not a completed negative or review-count trial |
| iter28 | CAO-1 SREMA codebook update with cyclic/learnable curvature | 0.05649471613363742 | 57439 | Stage3 result is comparable to iter18 under its declared inherited Stage3 protocol, but is not FCCR-1 evidence |

Evidence paths: iter18 `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`; iter26 `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/protocol_manifest_iter26.md`, `logs/mechanism_contract_iter26.json`, `logs/hypothesis_iter26.md`, and `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`; iter25 `logs/failure_attribution_iter25.md` and its exact `test_final.json`; iter27 `logs/iteration_abort_iter27.md`; iter28 `logs/protocol_manifest_iter28.md`, `logs/mechanism_contract_iter28.json`, and exact `test_final.json` under `Sep-27-2026_00-15-26/`.

The iter26 protocol declares Amazon_2023_Instruments / 2026-09 Refactor, Stage2 seed 42 and 100,000 steps, three 256-entry codebooks, Stage3 seed 42 and 150 epochs, beam 20, `n_eval=57439`, and iter18 as canonical baseline. Stage3 source-tree diffs across the declared iter26 and iter28 commits show no tracked Stage3 source change. This supports the declared Stage3 comparison; it does not make iter28 a fixed-curvature experiment. No complete iter18 protocol manifest was found, so the new S01 protocol lock must independently reconcile exact baseline/protocol bookkeeping before claiming a new direct delta.

## Cleanly tested, confounded, and unresolved

- Iter26 is a clean implementation-level FCCR-1 test of one fixed mapping. Its precomputed values were `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`; its registered attribution reports curvature invariance and mechanism activation checks passed. The run scored `0.057017009349048554`, which is `0.002872612684761226` below iter18 and `0.007982990650951446` below the user's `0.065` threshold.
- Iter25's learned/cyclic prior and iter28's SREMA with cyclic/learnable curvature are not clean tests of FCCR-1. Their observed scores do not establish that fixed closed-form mappings fail.
- Iter27 is an infeasibility abort, not a scientific negative.
- The single iter26 result does not establish a robust family-level negative; it has no matched-seed replication. The reported ~0.003 historical noise band is not independently estimated by these runs. Conversely, the result does not support promotion or guarantee that a modified mapping will reach the target.
- Stage2 Gini, collision, utilization, entropy, and other SID metrics remain descriptive; they do not explain or replace Stage3 evidence.

## Strongest evidence-supported direction

Keep FCCR-1 active. Autonomously register **one bounded alternative closed-form mapping** of the same behavior-branching and raw-residual-median inputs to precomputed immutable per-layer curvature. Use iter26's fixed-curvature implementation and run configuration as the immediate parent; keep Stage1 inputs, initialization/warm start, data, Stage2/Stage3 settings, evaluation, and seed policy unchanged so the only conceptual change is the preregistered mapping. Explicitly reconcile canonical-baseline bookkeeping in the new protocol manifest. Do not retune iter26 after registration or introduce any optimizer, Sinkhorn, behavior-loss, Stage1, or Stage3 mechanism.

## Paused directions and remaining question

Pause learnable/cyclic curvature, curvature regularization, curvature-conditioned optimizer/Sinkhorn/behavior changes, codebook optimizer replacement, and Stage1/Stage3 modifications; these are outside FCCR-1. The unresolved core question is whether a different allowed fixed mapping of branching and raw residual structure can improve the quantization representation enough to increase downstream recommendation recall. The next iteration must register one precise mapping and evaluate it end-to-end; no user choice is required.

## Success criterion

The user's explicit task criterion is `test_recall@10 >= 0.065`. The repository/skill's strict `>0.065` wording is preserved as a lower-priority policy conflict; the user-level threshold governs this task. Only a protocol-valid Stage3 result can establish success.
