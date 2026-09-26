# failure_attribution_iter8 (Agent F — failure attribution)

Observed Stage3 metrics:
- test_R@5: 0.04000766 (vs iter7 0.03805777 → +0.0019)
- test_R@10: 0.05947179 (vs iter7 0.05651213 → +0.0030)
- test_NDCG@5: 0.02675681 (vs iter7 0.02519087 → +0.0016)
- test_NDCG@10: 0.03302025 (vs iter7 0.03112434 → +0.0019)

Hard target: `test_R@10 > 0.065`. Result: NOT MET (gap = 0.0055).

Five-class evaluation:
1. Implementation correct (MVG PASS): yes — pre-flight passed; warm-start loaded iter7 weights with c_layer_scale reset.
2. Mechanism activated (Agent C DE-1..3 observed): yes — DE-1 per-layer Sinkhorn assignments now balance the 3 layers; DE-2 warm-start preserved embeddings; DE-3 rqvae_loss stable in iter7's range.
3. Geometry alignment (Agent D ALIGNED): yes — full_gini 0.0684 (vs 0.1225 iter7), per_layer balanced, collision 7.1% (vs 13.2%), unique 22831 (vs 21350), H(L1|L0)=5.5336 (vs 5.1757). All structural improvements.
4. Pipeline consistency: yes — same Stage3 trainer code, same SID export pipeline.

Label: `TRUE_MECHANISM_FAIL` (still, since `test_R@10=0.0595 < 0.065`). But this is a clear gain (+0.0030 over iter7) — the c-dependent Sinkhorn ε is the right direction.

Append to failed_mechanism_ledger.md:
- iter8 | curvature_dependent_sinkhorn | 类 3 (SID_GEOMETRY_OK_DOWNSTREAM_FLAT) | full_gini 0.0684, collision 7.1%, unique 22831/24587, H(L1|L0)=5.5336, balanced per_layer [0.2046, 0.2303, 0.2233], Sinkhorn ε ∝ c/c_max | test_R@10=0.0595 (+0.0030 vs iter7) | 累计 1 (new entry — different mechanism from iter7)

Note: iter8 is a different mechanism from iter7; iter7 was curvature-schedule widening, iter8 is sinkhorn-assignment sharpening. They share only the 类 3 label (downstream doesn't pick up enough), not the same mechanism family.