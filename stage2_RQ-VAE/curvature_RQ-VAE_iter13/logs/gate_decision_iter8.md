# gate_decision_iter8 (final)

Decision: NO-GO. `test_R@10=0.0595` < 0.065 target.

Outcome vs iter7:
- test_R@5: 0.0400 vs 0.0381 → +0.0019
- test_R@10: 0.0595 vs 0.0565 → +0.0030
- test_NDCG@5: 0.0268 vs 0.0252 → +0.0016
- test_NDCG@10: 0.0330 vs 0.0311 → +0.0019

Outcome vs iter4 (baseline):
- test_R@5: 0.0400 vs 0.0377 → +0.0023
- test_R@10: 0.0595 vs 0.0562 → +0.0033
- test_NDCG@5: 0.0268 vs 0.0253 → +0.0015
- test_NDCG@10: 0.0330 vs 0.0312 → +0.0018

Failure attribution: TRUE_MECHANISM_FAIL but with positive delta. Logged in failed_mechanism_ledger.md (different mechanism family from iter7). See `failure_attribution_iter8.md`.

Iter8 stays archived (not promoted), but Stage-2 ceiling has now been meaningfully broken. Iter9 should continue the Sinkhorn ε refinement.

- Audit commit: pending (commit after iter9 finalization; iteration cap not exhausted).