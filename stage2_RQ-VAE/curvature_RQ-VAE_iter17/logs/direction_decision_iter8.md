# direction_decision_iter8 (Agent B — direction judge)

Inputs:
- Bottleneck (Agent G iter7): Stage-3 locked onto iter4 SID fingerprint; previous 5 Stage-2 variations gave +0.0003 — Stage-1 ceiling exhausted in the curvature-schedule family.
- Failed-mechanism ledger: iter7 wide_curriculum_warmstart (类 3, count 1).
- Mechanism pool: P5 (Sinkhorn c-dependent ε), P6 (Riemannian Adam), P8 (warm-start).

Scoring (a–f):

| Candidate | (a) fixes bottleneck | (b) paper support | (c) curriculum compat | (d) Stage-1 ceiling risk | (e) Gap-closing | (f) Curvature-relevant | Total |
|---|---|---|---|---|---|---|---|
| P5 (c-dependent ε) | yes — changes quantization assignment, not schedule | yes (Geneva 2022) | high — local change inside Quantize | medium (high) | yes — directly explains "structural change to assignment step" | yes — manifold/assignment | 6/6 |
| P6 (Riemannian Adam) | yes — but custom optimizer is a structural overhaul | yes (Bécigneul 2019) | medium | medium | partial | yes — Riemannian | 4/6 |
| P8 (warm-start only) | no — used in iter7 already | weak | high | high | partial | low | 2/6 |

Recommendation: **P5** (Sinkhorn c-dependent ε).

Reasoning: P5 changes the Sinkhorn assignment step directly while keeping the curvature schedule and embeddings untouched. Iter7 demonstrated that curvature-schedule variation does not break the Stage-3 ceiling (test_R@10 = 0.0565, only +0.0003 over iter4). A different Stage-2 mechanism family — quantization assignment — is the only remaining lever within the project rules.

Backup: P6 (Riemannian Adam) if P5 also fails.

Mechanism: `effective_eps = sk_eps * (c / c_max).clamp_min(1e-4)`. At c = c_max the original `sk_eps` is preserved (high-c sharpen). At c = c_min the eps shrinks, softening assignments (avoids degenerate hard assignments at the boundary).

Decision: iter8 is `curvature_dependent_sinkhorn`, single mechanism P5.