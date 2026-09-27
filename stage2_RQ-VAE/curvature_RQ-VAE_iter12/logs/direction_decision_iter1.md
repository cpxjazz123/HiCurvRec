# Iter1 decision

## Outcome

Do not promote the residual-scale-derived cyclic-curvature variant. Stage3 `test_recall@10` was `0.05522380264280367`, below the required strict threshold `> 0.065`.

## Evidence

- Stage2 reached 100000 steps; descriptive final metrics and the baseline comparison are recorded in `hypothesis_iter1.md`.
- Stage3 test ran over 57439 examples. `test_recall@5=0.03638642734030885`, `test_recall@10=0.05522380264280367`, `test_ndcg@5=0.02378050997806901`, `test_ndcg@10=0.029818537549432242`.
- Relative `test_recall@10` differences: +0.0001741 vs TIGER and +0.0014624 vs repaired curvature. These are small, and the absolute adoption gate failed.

No literature-based claim is made: no literature search was completed. Earlier generated candidate rankings and citation claims were unsupported and are withdrawn.
