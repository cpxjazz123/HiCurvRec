# Issue #263 — frozen mechanism contract

Frozen before implementation. Baseline/pretraining assets are identified in
`INFRA_SNAPSHOT_START.json`; the exact issue body is `ISSUE.json`.

## Root cause targeted

The parent R01 GNN is trained jointly with the RQ-VAE quantizer and gate. It
receives no explicit multi-role graphlet/orbit distribution or curvature
distribution target, and the complete ego computation is online during training.
The candidate tests whether a directed ego GNN pretrained once on TRAIN-only
distribution targets, then frozen as four 32-dimensional tokens, gives the
trainable gate a more stable structural signal and removes the online GNN from
Stage2.

This is not a claim that freezing alone will raise Recall; the negative result is
an accepted outcome.

## Architecture and input

- Parent actual GNN: `SubgraphTokens` from the R01 source. It takes the frozen
  six-feature node asset (not the 768-d semantic vector); 2 directed message
  layers, hidden/token dimension 32; four token types in fixed order
  `one_hop, two_hop, in_only, out_only`, shape `[24587,4,32]`; mask `[24587,4]`.
- The four input populations are the frozen full one-hop ego, complete two-hop
  ego, centre+in-neighbours, centre+out-neighbours; direction is preserved as
  required by the parent contract.
- No topology token is concatenated into the 768-d RQ-VAE semantic input and the
  32-d bottleneck, three 256-code semantic SID and three {-1,0,+1} experts stay
  unchanged.

## Distribution targets (TRAIN graph only)

- **Own orbit 376**: `own_orbit` rows from the existing v3 structure target table;
  row sums are normalised; rows with `coverage=0` are masked and never imputed.
  This is a multi-role distribution, not a single label.
- **DGDV 30**: frozen directed triad orbit counts `directed_counts`; normalise
  where count sum > 0; empty rows masked. This gives an all-node directed orbit
  distribution.
- **Curvature histogram 12 bins**: for each centre item, read every valid row
  from frozen `assets/v8_runtime_layout/targets.npy` using `target_offsets`;
  fixed training-side edges
  `[-∞,-4,-3,-2,-1,0,1,2,3,4,8,∞]`; missing centres are masked. No single mean and
  no test-derived bin edge is used.
- `own` and `DGDV` are distribution matching; the curvature term is distribution
  matching, not the Stage2 Huber scalar loss.

## Pretraining objective and budget

For a batch `B`, four token rows `T[b,t,:]` and mask `m[b,t]`:

```
logits_r = Head_r(T, m),  r in {own376, dgdv30, curve12}
Head_r(T,m) = sum_t m[b,t] * Linear_r(T[b,t]) / max(sum_t m[b,t], 1)
L_r = KL(p_r || softmax(logits_r))  over rows with m_r=1 and p_r sum>0
L_role = 0.5 * L_own + 0.5 * L_dgdv
L = L_role + lambda * L_curve
```

- Fixed seed 42, 1,500 steps, batch 256, AdamW lr 1e-3, no weight decay,
  gradient clipping 1.0. One run only; final step is the selected checkpoint.
- 80/20 **item-ID split** frozen before training: 80% are fit items, 20% are
  structural diagnostic items. All labels come from the TRAIN graph; no
  validation/test recommendation row is read.
- `lambda` is set once from the gradient norm ratio of the two GNN objectives on
  one fixed calibration batch, clipped to `[1e-3,1e3]`; it is not tuned on a
  validation/test metric.

## Freeze and token cache

After the final step:

1. remove prediction heads;
2. `model.eval()`, all GNN parameters `requires_grad=False`;
3. one full item pass yields `tokens[24587,4,32]` float32 and
   `mask[24587,4]` bool in item-ID order;
4. save a checkpoint and cache manifest with source/asset/input SHA;
5. reload the checkpoint into a fresh GNN and repeat full inference; exact
   checkpoint replay and deterministic re-export must match within 0 tolerance.

The cache is a persisted dependency, not sampled.

## Stage2 contract

- The frozen cache is loaded by item ID; the trainable `LayerExpertAttention`
  gate sees only tokens/mask plus its own parameters.
- The GNN module is not instantiated in the Stage2 forward path; GNN forward and
  backward call counts must be 0. The GNN parameters are not in the optimizer.
- Encoder, decoder, codebooks, gate, assignment and the parent curvature loss
  remain trainable/active exactly as in R01. The candidate's only intended
  mechanism change is the frozen topology token source.
- `ARM_NAME = TIGER_frozen_topology_gnn_rq`; result/checkpoint/SID path follows
  the framework contract. 3000 epochs, four-card DDP, global batch 4096,
  seed 42, final checkpoint, no early stop.
- Stage3 uses the frozen 150-epoch protocol, beam 20; validation is evaluated
  before one final test; no test access for parameter choice.

## Falsifiable predictions

- **P1:** on 20% held-out item IDs, token predictions beat the degree-binned
  frequency baseline on role/curvature KL and are not a constant-class solution.
- **P2:** the frozen token hash is unchanged before/after Stage2; GNN call count
  is zero; gate gradients are non-zero.
- **P3:** Stage2 train time is lower than the parent online-GNN cost, and the
  frozen token path keeps L3 collision <0.40.
- **P4:** Stage3 `test_R@10` is the final decision; it must be reported against
  the frozen TIGER and 0.065 line, not inferred from structural metrics.

## Numeric tolerances / edge cases

- token export: exact float32 equality for checkpoint reload; deterministic
  re-export exact.
- role/curvature KL: finite; empty/missing rows are skipped, never zero-filled.
- item IDs: integer 0..24586; no permutation; mask false means unavailable.
- token norm/rank: finite; empty ego has no valid token and is represented by
  the existing gate null token path.
