## Close-condition infeasibility: P2 and P3 are empty in the frozen arm set

The issue's close condition requires one full 3000-epoch Stage2 candidate to pass all frozen
non-collision criteria. The frozen arms show the P2 and P3 conjunctions are **empty** under the
issue's fixed mechanism (κ_t A.1 local + λ + μ(J-J_max); architecture/selection/Sinkhorn/
reconstruction fixed; only λ/dual/schedule/gradient budget may change).

| arm | local MAE | Spearman | sign | L2 | rank | J | recon |
|---|---|---|---|---|---|---|---|
| TIGER (λ=0 baseline) | 7.309217 | 0.015850898 | 0.432453349 | 7,374 | 11.5 | 0.739635 | 6.15e-4 |
| #271 `TIGER_global_j_rq` | 1.893506 | −0.005483 | 0.392646 | 6,097 | 4.501 | 0.351361 | n/a |
| #269 `TIGER_calibrated_kappa_rq` | 1.997261 | −0.070083 | 0.363704 | 6,246 | n/a | 0.338336 | n/a |
| issue272 round-1 `TIGER_active_j_rq` | 2.343824 | −0.098838 | 0.348062 | 6,848 | 2.77 | 0.789 | 8.57e-5 |
| #265 `TIGER_euclid_local_curve_supervision_rq` | 2.245950 | −0.071041 | 0.332350 | 10,918 | 3.05 | 0.189476 | 8.18e-5 |

Sources: `results/research/issue271/assets/geometry/*.json` (MAE/Spearman/sign, 28,391,948 rows,
frozen evaluator), `results/stage2_RQ-VAE/*/stage2_metrics/*.json` (L2/rank/J), the issue272
`VERDICT.md` (round-1), `results/research/issue272/ruler_inventory.json` (common-ruler J).

**P2 is empty.** The MAE target 1.893506 is the #271's own value, and the same arm fails the P2
non-degeneracy (`var_ratio` 0.335, `slope` 0.041): the MAE improvement is the shrink the P2
excludes. The Spearman target `> 0.015851` is **above** the TIGER's 0.015850898 — no arm,
including the TIGER, exceeds the TIGER; the sign target `> 0.432453` is met only by the TIGER
(0.432453349), and every local arm is below it (0.392646, 0.363704, 0.348062, 0.332350). The
local term — the mechanism's core — **lowers both ordering metrics**; the issue's own #271
observation says exactly this (`MAE down, Spearman/sign worse`).

**P3 is empty.** L2 ≥10,918 is met only by the #265 (the **Euclidean** local, not the frozen κ_t
base), which has rank 3.05. Rank 11.5 exists only at the TIGER (L2 7,374). The κ_t base's L2
maximum is 7,374 (λ=0 TIGER); every λ>0/μ>0 point lowers it (6,097–6,848). No arm has
L2 ≥10,918 and rank ≥8.

P1/P4/P5 are individually passable (#269 P1 0.338336/P5; round-1 P4 8.57e-5/P5), but the
conjunction is empty; the #265's L2 10,918 comes with P2 MAE 2.245950, rank 3.05, and P2
Spearman/sign below the TIGER.

The issue's §3 says report mathematical/empirical infeasibility rather than fake a pass; skill §7
keeps the issue **OPEN**; no hard training on a known-empty target. Owner options: (A) amend P2
to a non-degenerate MAE and TIGER-level (not strictly above) Spearman/sign; (B) amend P3 to the
κ_t base's ceiling, or authorize a **new issue** with the Euclidean-local base plus a separate
rank/reconstruction repair (the #265's L2 with rank ≥8 is not observed); (C) close as BLOCKED
only with explicit owner authorization and the evidence above.
