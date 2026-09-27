# failure_attribution_iter7 (Agent F — failure attribution)

Observed Stage3 metrics:
- test_R@5: 0.03805777 (vs iter4 0.03774439 → +0.0003)
- test_R@10: 0.05651213 (vs iter4 0.05619875 → +0.0003)
- test_NDCG@5: 0.02519087 (vs iter4 0.02527648 → -0.0001)
- test_NDCG@10: 0.03112434 (vs iter4 0.03121687 → -0.0001)

Hard target: `test_R@10 > 0.065`. Result: NOT MET.

Five-class evaluation:
1. Implementation correct (MVG PASS): yes — pre-flight passed on 640-pair batch with iter4 warm-start + c_layer_scale reset.
2. Mechanism activated (Agent C DE-1..3 observed): yes — DE-1 wider curvature range exercised; DE-2 warm-start preserved iter4 embeddings; DE-3 sharper behavior loss active.
3. Geometry alignment (Agent D ALIGNED): yes — full_gini 0.1225 (improved), collision rate 13.2% (improved), H(L1|L0) 5.1757 (improved), unique 21,350 (improved); L2 utility regressed but only mildly.
4. Pipeline consistency (Stage3 input SID / export / config same as iter4): yes — same Stage3 trainer code, same SID export pipeline, only item_sids.json source differs.

Label: `TRUE_MECHANISM_FAIL` — wider cyclic-curvature range + warm-start + sharper behavior loss are all properly activated, the SID geometry improved on most axes, yet downstream `test_R@10` only improved by +0.0003 over iter4. The mechanism is exhausted at the Stage-1 ceiling.

Append to failed_mechanism_ledger.md:
- `iter7 | wide_curriculum_warmstart | 类3 (SID_GEOMETRY_OK_DOWNSTREAM_FLAT) | full_gini 0.1225, collision 13.2%, unique 21350/24587, H(L1|L0)=5.1757 | test_R@10=0.0565 | 累计 1`