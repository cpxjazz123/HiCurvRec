# Issue #266 verdict

## Headline

The frozen C/G1 score **does not preserve the historical monotonic relation** on
the two new held-out bundles. The original eight arms reproduce their historical
fit, but the ten-arm extension has one adjacent violation and six inversions.
This is an analysis-only negative result; no training, refit, retry or score
search was performed.

## Result

| population | Pearson | Spearman | Kendall tau-b | strict adjacent | inversions |
|---|---:|---:|---:|---|---:|
| original 8 | 0.8569503102754842 | 1.0 | 1.0 | PASS | 0 |
| extended 10 | 0.69046552155332 | 0.793939393939394 | 0.7333333333333333 | **FAIL** | **6** |

New frozen scores:

- `S_108 = 0.18293017861055832`
- `S_109 = 0.3724475438041919`
- issue interval: `0.48640629657478884 < S_108 < S_109 < 0.5096899206435366`

Both new arms are below the interval; `S_108 < S_109` is true. The adjacent
violation is `105 -> 108`:

```
Recall_105 = 0.05151552081338463 < Recall_108 = 0.05189853583801946
S_105      = 0.48640629657478884 > S_108      = 0.18293017861055832
```

The six inversions and every violating pair are retained in `SCORES.json`; no
row was deleted, clipped or reordered to restore monotonicity.

## Endpoint truth

| arm | C | G1 |
|---|---:|---:|
| 108 | 0.7775872945785522 | 12.757243390234594 |
| 109 | 1.5810964107513428 | 7.112086922093524 |

Both values came from a real final-checkpoint forward/autograd with the fixed
4096-item batch, seed 42, and the common `draw(0,0,1,track=False)` sample. The
parameter checksums are identical before/after and optimizer steps are zero.
`C` is the raw unweighted curvature loss; `G1` is the raw unweighted L2 norm over
all L1 codebook parameters. No combined loss, clipped gradient, fixed-evaluation
loss or epoch log was substituted.

## Identity and scope

- R2 108 and 109: complete four-piece bundles; 112/113 ledger files and all
  local source/result/token/graph identities match (`IDENTITY.json`, 0 failures).
- The two bundles' archived recall exactly matches their local `test_final.json`:
  108 `0.05189853583801946`, 109 `0.052594926791900976`.
- No Stage2/Stage3/TIGER/GNN training, optimizer step, SID rewrite, parameter
  update or public-infra modification occurred.

## Interpretation limits

- **Prediction continuation: NO.** The frozen score ordering does not continue on
  the two new arms; this is the main measured result.
- **Native-loss scale comparability: NOT ASSUMED.** 108 retains the #263 frozen
  structural objective; 109 changes selection to Euclidean and uses a
  center-local mixed-curvature target on the final hard `q`. Its native C/G1 is
  an external-transferability test of the frozen formula, not evidence that the
  same objective was optimized.
- **Gradient sensitivity vs optimization benefit: INCONCLUSIVE.** 109 has a lower
  G1 but a higher Recall@10 than 108; 108 has the larger G1 but the lower recall.
  With n=2, different mechanisms and no causal intervention, this does not
  establish a general sign rule. G1 is a diagnostic magnitude, not an observed
  optimization gain.
- The original eight-arm fit is descriptive and was already selected post hoc;
  the new arms have known historical recall and are not a blind validation.
  No causal or selection claim is made.

## Hypotheses

| id | status | reason |
|---|---|---|
| H266-1 frozen monotone extension | **REFUTED** | ten-arm strict adjacency fails; 6 inversions |
| H266-2 common native scale / transfer | **REFUTED** | 109 objective differs and score chain fails; native values are only external-transfer evidence |
| H266-3 larger gradient → larger benefit | **INCONCLUSIVE** | 108/109 have opposite G1-recall ordering; no optimization intervention |

## Acceptance

C1–C9: **PASS** (see `CHECK_1.md`). The monotonicity failure is an experimental
result, not an implementation failure; the frozen definitions and the independent
15-test reference suite were checked before the result was read.

## Closing gate

- Production mechanism changed: **no** (analysis-only; no production code).
- Required tests executed: **yes** (`tests/`, 15 passed; command
  logs persisted).
- Required training: **none by issue boundary**; no training was run.
- Infra boundary: **FROZEN**; end snapshot comparison must be empty and is written
  to `INFRA_SNAPSHOT_END.json`.
- Remaining executable tasks: none after the commit, push and close comment.

Commit: pending until the evidence commit is created.
