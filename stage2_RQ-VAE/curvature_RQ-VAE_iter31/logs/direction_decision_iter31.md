# Iter31 direction decision — S12 canonical

```text
ITERATION=31
RUN_VALIDITY=VALID_CORRECTED_STAGE3_RUN
MECHANISM_STATUS=ACTIVE_NEUTRAL
PROMOTION_STATUS=PROMOTION_FAIL
NEXT_ACTION=S13_GIT_ARTIFACT_CLOSURE
USER_INPUT_REQUIRED=NO
```

## Decision

Iter31's registered HRA Step6 mechanism was computationally active under the fixed FCCR-1 contract (S08 round-5 Judge `logs/deliberation/S08_MVG/round_5/judge.md`, SHA-256 `b3030748270d3711198d0d22c9a901f700073be1e55c20bd4832208bf913c8dc`; S10 canonical geometry report `logs/sid_geometry_iter31.md`, SHA-256 `4bd7d18b10ea88ed896c7f77dce234bda555b5e7374375dd56c0e9e953685da2`). The corrected full-epoch Stage3 run is valid; its final-test hash is `d446c6b622e51c140cfaab1be5a76e3cb6ee56ec318ecb0dd5c4b3ca8c8f3957` and training-metrics hash is `0f706eb1c7daefd1bf9eac04a64d26424ded1c8b6f237bf06c3cb4009af499dd`. Against protocol-compatible Iter29 (`test_final.json` SHA-256 `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`), the four observed deltas are negative but modest; no run-noise estimate exists. Apply the small/ambiguous-delta rule: `ACTIVE_NEUTRAL`, not causal negative. Independently, strict R@10 > 0.065 is not met, so `PROMOTION_FAIL`.

The earlier abort record `logs/iteration_abort_iter31.md` (SHA-256 `24e18d2752de439791eccfd3d243ee07decee9b3c9328993e1e2c97790848712`) remains preserved history. S08 round-5 `REPAIR_PASS` invalidated rounds 1–4 incomplete provenance for the current gate, and S09 prelaunch record `logs/stage2_preflight_iter31.md` (SHA-256 `bfc8d96b76a7664a1f7874ef8a2d9735dc55eb1fb11a26995f81a317cf75a0c1`) documents the gate pass naming both S08 round 5 and S09 round 2 repair passes. This later evidence resolves the old abort for the current completed Stage2/Stage3 path without altering the abort history.

## Autonomous next action

Proceed to S13 Git/artifact-closure deliberation. Audit required Iter31 code and Stage2/Stage3 artifacts, all canonical decision files, ignore state, and GitHub-only `main` synchronization under root `CLAUDE.md` §13 before any iteration-closure claim. This result alone authorizes no new experiment and no performance direction: do not rerun, replicate seeds, sweep, ablate, estimate noise, or investigate root cause. S13 reads this canonical direction decision and S12 Judge; the A/B candidates are audit-only.

**S12 canonical Judge:** `logs/deliberation/S12_RESULT_CLASSIFICATION/round_1/judge.md` (verdict `ACCEPT_B`, overall confidence MEDIUM). No user decision is required.
