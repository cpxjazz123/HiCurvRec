# gate_decision_iter7 (final)

Decision: NO-GO. `test_R@10=0.0565` < 0.065 target.

Outcome vs iter4 (baseline):
- test_R@5: 0.0381 vs 0.0377 → +0.0003
- test_R@10: 0.0565 vs 0.0562 → +0.0003
- test_NDCG@5: 0.0252 vs 0.0253 → -0.0001
- test_NDCG@10: 0.0311 vs 0.0312 → -0.0001

Failure attribution: TRUE_MECHANISM_FAIL (Agent F). See `failure_attribution_iter7.md`.

Iter7 stays archived (not promoted). Iteration 8 follows Agent G's iteration_bridge: try a qualitatively different Stage-2 mechanism (manifold replacement or Riemannian Adam), forbidden to re-iterate within curvature-schedule family.

- Audit commit: pending (iteration cap not exhausted; commit after iteration 8 finalization).