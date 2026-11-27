## Close-condition proof (executable): P2/P3 empty

`python3 results/research/issue272/check_close_condition.py` (exit 0; log
`results/research/issue272/close_condition_check.txt`) reads the frozen geometry artifacts
(28,391,948 rows) and the SID metrics:

| arm | MAE | Spearman | sign | L2 | rank | J |
|---|---|---|---|---|---|---|
| TIGER (λ=0 baseline) | 7.309217 | +0.015851 | 0.432453 | 7,374 | 11.5 | 0.739635 |
| #271 `TIGER_global_j_rq` | 1.893506 | −0.005483 | 0.392646 | 6,097 | 4.501 | 0.351361 |
| #269 `TIGER_calibrated_kappa_rq` | 1.997261 | −0.070083 | 0.363704 | 6,246 | — | 0.338336 |
| issue272 round-1 `TIGER_active_j_rq` | 2.343824 | −0.098838 | 0.348062 | 6,848 | 2.77 | 0.789 |
| #265 `TIGER_euclid_local_curve_supervision_rq` | 2.245950 | −0.071041 | 0.332350 | 10,918 | 3.05 | 0.189476 |

- P1 `J<=0.338336`: {#269, #265}; P4 `recon<=8.589e-5`: {round1, #265}; P5 `mu>0`: {#269, round1}.
- **P2 empty**: minimum MAE = **1.893506311** (#271) > 1.893506; maximum Spearman =
  **0.015850898** (TIGER) < 0.015851; the sign maximum is the TIGER's 0.432453349 and every local
  arm is lower. The MAE target is the #271's own value, which fails the P2 non-degeneracy
  (`var_ratio` 0.335, `slope` 0.041); the Spearman target is above the TIGER and no arm exceeds it.
- **P3 empty**: the only L2>=10,918 arm is the #265 (L2 10,918) with rank 3.05; the only rank>=8
  arm is the TIGER with L2 7,374; no arm has both. The κ_t base's L2 maximum is 7,374 (λ=0 TIGER).

The script asserts these emptiness claims; exit 0. The P1/P4/P5 sets are non-empty, but their
intersection with P2/P3 is empty. Issue §3 says report the infeasibility; skill §7 keeps the issue
**OPEN**; no hard training on a known-empty target.

Options: (A) amend P2 to a non-degenerate MAE and TIGER-level (not strictly above) Spearman/sign;
(B) amend P3 to the κ_t base's ceiling, or authorize a new issue with the Euclidean-local base plus
a separate rank/reconstruction repair; (C) close as BLOCKED only with explicit owner authorization.
