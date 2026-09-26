# Iteration bridge — Iter24 → Iter25

## Mechanism execution

- **Implementation: PASS.** Iter24 `mvg_check_iter24.log` records `MVG PASS`; the Stage2 run completed 100,000 steps and exported a validated four-token SID file.
- **Activation: PASS.** The balanced-minus-unbalanced behavior-loss delta had nonzero gradient norm `0.00834073`; same-seed 200-step ON/OFF total losses differed by `0.02317619`.
- **Geometry alignment: FAIL relative to Iter18 on observed proxies; no separate Iter24 Agent-D report exists.** Both runs used all 256 codes per layer, but Iter24 had lower `H(L1|L0)` and `H(L2|L0)`, more prefix and three-token collisions, and fewer unique SIDs. These are descriptive comparisons, not a gate or causal result.

## Difference from best completed Stage3 baseline (Iter18)

| Fingerprint | Iter18 | Iter24 | Δ Iter24 − Iter18 |
|---|---:|---:|---:|
| L0 utilization | 1.0000 | 1.0000 | 0.0000 |
| H(L1\|L0), bits | 5.590675 | 5.394702 | -0.195973 |
| H(L2\|L0), bits | 5.934625 | 5.742606 | -0.192019 |
| Three-token collision rate | 6.6539% | 8.9885% | +2.3346 pp |
| L0 oracle | unavailable | unavailable | unavailable |
| Stage3 test_R@10 | 0.05988962 | 0.05619875 | -0.00369087 |

`L0 oracle` is unavailable: the registered `compute_oracle_topk` implementation returns `None`. SID geometry values above were computed from the existing raw SID arrays. The Stage3 metric comes from the completed `test_final.json` (`n_eval=57439`).

## Dominant bottleneck

**Iter24's added behavior-loss reweighting did not improve downstream recall and its completed run had weaker hierarchical SID proxies than Iter18, so Iter25 should test the unresolved raw-branching/raw-residual-to-learnable-cyclic-prior mapping rather than add another behavior-loss change.** This is an evidence-based next hypothesis, not proof that the Iter24 mechanism caused the regression.

## Forbidden next directions

- Do not add or reweight another behavior-loss term in Iter25.
- Do not freeze layer curvature or remove the cyclic schedule (Iter16 regressed).
- Do not modify Stage1 embeddings, input data, or the Stage3 trainer/configuration.
- Do not use Stage2 SID statistics as a progression gate.

## Falsifiable Iter25 objective

Use only the bounded prior replacement
`h_l = log(B_l)/max_j(log(B_j))`, `r_l = m_l_raw/max_j(m_j_raw)`,
`u_l_prior = 0.01 + 0.98*(h_l+r_l)/2`, with the locked inputs
`B=[19.32,1.46,1.015]`, `m_raw=[1.000,0.10941,0.09331]`.
The preregistered prior is `[0.990000,0.126233,0.058186]`; it initializes and anchors the existing learnable layer-scale parameters. Preserve Iter11's cyclic formula, loss/commitment settings, Sinkhorn, optimizer, warm-start, and Stage3 protocol. Before training, require four-layer MVG (including five-step updates and a same-seed 200-step ON/OFF loss difference >1e-6). Run Stage2 to its configured maximum; all SID geometry remains descriptive. Only a complete Stage3 run with `test_R@10 > 0.065` meets the adoption target; compare against canonical Iter11 `0.0597677536` and best completed Iter18 `0.0598896220`.

## Prior provenance caveat

The Iter11 training log says its `layer_norms.json` was missing and the actual initialized prior was `[0.001,0.932889,1.0]`, derived from raw medians `[1.0,0.10941,0.09331]`. Iter25 therefore replaces Iter11's applied prior mapping; it is not a pure add-branching ablation with the old residual mapping held fixed.
