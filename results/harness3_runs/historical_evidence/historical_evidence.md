# Historical evidence baseline (issue #109, section 4)

Offline, read-only. No cold run, no frozen evaluation, no probe rerun.

Historical models: 38 in 5 families; complete-case models: 38.
The ER-only Recall@10 out-of-fold MAE is 0.003753 (historical reference 0.003753; reproduces: True).

| Target | Feature set | n | MAE | Delta vs ER (same rows) | 95% CI over families | Counterexamples |
|---|---|---:|---:|---:|---|---:|
| recall | ER | 38 | 0.003753 | +0.000000 | [+0.000000, +0.000000] | 0 |
| recall | ER_plus_G_H1 | 38 | 0.002835 | +0.000918 | [-0.001460, +0.001721] | 18 |
| recall | ER_plus_D2 | 38 | 0.003169 | +0.000584 | [-0.000807, +0.000844] | 17 |
| recall | ER_plus_G_H1_plus_D2 | 38 | 0.002897 | +0.000856 | [-0.001455, +0.001657] | 19 |
| recall | G_H1_only | 38 | 0.005421 | -0.001669 | [-0.003884, -0.000186] | 26 |
| recall | D2_only | 38 | 0.005957 | -0.002204 | [-0.005066, -0.001185] | 27 |
| ndcg | ER | 38 | 0.002101 | +0.000000 | [+0.000000, +0.000000] | 0 |
| ndcg | ER_plus_G_H1 | 38 | 0.001638 | +0.000463 | [-0.000780, +0.000871] | 17 |
| ndcg | ER_plus_D2 | 38 | 0.001804 | +0.000297 | [-0.000432, +0.000440] | 17 |
| ndcg | ER_plus_G_H1_plus_D2 | 38 | 0.001685 | +0.000417 | [-0.000760, +0.000831] | 19 |
| ndcg | G_H1_only | 38 | 0.002941 | -0.000840 | [-0.001987, -0.000066] | 27 |
| ndcg | D2_only | 38 | 0.003126 | -0.001024 | [-0.002569, -0.000511] | 28 |

**Not certified for screening.** The frozen ER stays the only pre-Stage-3 gate.

## Limitations

- #48 is invalidated; only the corrected #49 archive is read
- #50's prose-versus-CSV conflict is unresolved, so none of #50's evidence is used
- the historical G_H1 is the seed-42 rows only and is never mixed with the published three-seed means
- the historical cohort is retrospective and already inspected; it is not a pristine prospective confirmatory test
- no predictor here is certified to reject an arm or to declare success
