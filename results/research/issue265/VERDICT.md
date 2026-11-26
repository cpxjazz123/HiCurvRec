# Issue #265 verdict

## Headline

The mechanism is implemented and ran end to end. It is **not a recommendation
success**: `test_R@10 = 0.0525949`, below the 0.065 target and below the frozen
TIGER reference 0.055537, although slightly above the #263 parent 0.0518985.
It is a valid negative result; no retry, seed search, or test-guided adjustment.

## Root-cause evidence

- Frozen #263 mixed selection versus Euclidean selection: **44/24587 hard SIDs
  differ (0.179%)** on the full catalogue; this supports selection-order
  insensitivity, not a guarantee about behavior sufficiency.
- Candidate Stage2 final collision **0.0412006** versus #263 **0.0607**; the SID
  clearly changed. L1 Hard ER 8.656 vs #263 8.568; L1 positive 0.06241 vs 0.06389.
  So the structural supervision did change the discrete representation, but not
  in the direction of recommendation gain.
- Gate-only scale reduction is not used as evidence: the fixed equal-weight
  `fixed_squared` path remains in the candidate tests/evaluator; the trainable
  objective is reported separately.

## Mechanism

Euclidean squared selection for every level; gate absent from the selection cost
and soft surrogate. Final hard `q = sum_l c_l`. The structural loss uses the
center gates (equal mean over three levels) for a local mixed distance
`D_i(j,m)^2 = sum_k w_i,k d_k(q_j,q_m)^2`, on the actual final q. The parent floor,
last-level Sinkhorn, residual update, codebook/commitment terms, frozen token/GNN
cache and Stage3 protocol are unchanged.

## Formal results

| item | value |
|---|---|
| Stage2 | 3000 epochs, four ranks, seed42; run `stage2_4891fa8...` |
| Stage2 final collision | **0.04120063448163669** (<0.40) |
| Stage2 wall / mean epoch | 813.357 s / 0.24937 s |
| Export gate | VERIFIED; 216 live items rewritten from checkpoint |
| §63 layer record | L2 prefixes 10918; L3 collision 0.0412413; L1 Hard ER 8.6560; conditional G3 MISSING |
| Stage3 | 150 epochs, beam20, seed42, n_eval 57439 |
| test R@10 / R@5 | **0.0525949 / 0.0340883** |
| test NDCG@10 / @5 | 0.0282505 / 0.0223063 |
| R2 | **`r2:genrec/109/`** (113 files, 23 recall keys) |

## Hypotheses

- **H265-1 (selection insensitivity):** SUPPORTED by the 44/24587 selection diff.
- **H265-2 (effect is only gate scaling):** REFUTED — hard SID collision and layer
  metrics changed materially, with a nonzero structural gradient to gate.
- **H265-3 (frozen structural targets lack sufficient behavior information):**
  INCONCLUSIVE, but the small R@10 change (+0.0007 vs #263) and below-reference
  result are consistent with it. No causal claim beyond one frozen run.
- **C9 limitation:** the fixed equal-weight evaluation is represented by the
  independent `fixed_squared` tests and the separate §63 layer record; a full
  all-triple fixed curvature error was not re-run after Stage3 (recorded as
  NOT_AVAILABLE rather than fabricated). This does not change the R@10 result.

## Acceptance

C1 frozen token/GNN inherited and no online GNN: PASS. C2 diagnosis: PASS.
C3 contract: PASS. C4 tests: new/adapted focused tests pass; final full GPU suite **131 passed / 0 failed**. C5 gradient
route: PASS (gate/encoder/codebook path; formal loss finite). C6 Stage2: PASS.
C7 export/layer: PASS. C8 Stage3: PASS. C9 fixed evaluation: PASS for the pre-registered fixed equal-weight probe;
`FIXED_EVAL.json` reports fixed loss 1.6491 vs local 1.5811 on one frozen
training sample (256 items, 28 triples); a full all-triple replay is not claimed. C10 performance: PASS
(0.24937 mean, <0.3197 target and <0.8 hard). C11 R2: PASS (`109`). C12 no
public infra/data changes: PASS.

## Closing

Production code implemented; required Stage2/Stage3 completed; negative result
recorded without retry. Local/remote commit and R2 are recorded in the closing
comment.
