# Shared-path verifier repair — no production rule change

Actual four-rank validation failed before dispatch. The old auxiliary_ties check assumed that the frozen floor allocator returned a global minimum, then applied an optimal-value perturbation bound. The production allocator is explicitly greedy repair. R02 does not change that inherited allocation protocol.

The exact2x2 hand counterexample in test_shared_assignment_oracle.py uses costs [[0,2],[1,2.0009765625]] and [[0,2],[1,1.9990234375]]. Frozen greedy IDs are respectively[1,0] and[0,1], both floor-feasible. Objectives3 and1.9990234375 differ by1.0009765625, whereas the sum of rowwise maximum cost perturbations is0.001953125. The left greedy objective is also worse than the hand-enumerated alternative2.0009765625. Thus the old verifier's premise is false; increasing its tolerance or replacing the prescribed allocation algorithm would not be a valid fix.

The repaired verifier separately checks:
1. Both real populations and all three levels are captured from the actual shared production forward. Every native recorded assignment, including an independently executed two-branch GNN result, is replayed exactly on its own original cost tensor through the unchanged floor/Sinkhorn rule. Wrong hard codes fail the new negative regression.
2. Native branch IDs, cost perturbations and objectives are reported even when different. No unconditional bitwise equivalence of differently batched float32 native assignments is claimed.
3. Under the same recorded discrete branch, a fresh independent two-branch GNN computation must match continuous gate/cost/output/loss values and every parameter VJP using the unchanged existing tolerances. The temporary branch control exists only on the disposable diagnostic model and is restored; it never changes actual Trainer execution or formal parameters.
4. Actual real-loss/DDP updates, actual cumulative-prefix hard decoding and exact whole-catalogue checkpoint/SID export remain separate mandatory checks. A fixed-branch reference alone cannot establish any of these.

Two CPU regressions passed: the valid greedy counterexample is accurately reported, and wrong recorded IDs are rejected. Full current-source four-rank replay and admission remain required; the superseded137+64+11 tests and4-rank profile are preserved in pre_shared_oracle/ and are not reused as current credentials.

Unit-suite CUDA kernels now also execute only beneath the sanctioned four-card launcher. Every collected case is assigned exactly once, each suite uses fresh interpreters, and four per-rank case lists must reconcile to the complete collection with no skipped/duplicate/missing cases. This is unit execution, not a claim of DDP Trainer verification; the actual four-rank Trainer diagnostic remains separate. No frozen training hyperparameter or data population changes.
