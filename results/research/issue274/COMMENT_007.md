### Round 3 result: the coverage lever reproduces the frontier — C3, C5 and C8 remain out of reach (final, not a PASS)

Run `stage2_6fe2faf7fc204330b129dfa4900ae641`, the dense 1-closed/1-open matching index (38,871
groups against 26,583; the only change from round 2). Full table:
`results/research/issue274/round3_pass_table.json`.

| criterion | r1 | r2 | r3 | threshold | verdict |
|---|---|---|---|---|---|
| C1 hold-out ranking accuracy | 0.6279 | 0.8064 | **0.8034** | > 0.5386 | PASS |
| C2 fixed-margin satisfaction | 0.0048 | 0.4685 | **0.4642** | > 0.0815 | PASS |
| C3 L2 distinct prefixes | 7,929 | 7,923 | 7,911 | >= 10,918 | FAIL |
| C4 effective rank | 15.61 | 12.35 | 11.20 | >= 8 | PASS |
| C5 prefix discrimination | 2/3 | 2/3 | 2/3 | — | FAIL |
| C6 neighbourhood fidelity | 0.08197 | 0.07925 | **0.08097** | >= 0.07932 | PASS |
| C7 full J | 0.2261 | 0.3037 | 0.3152 | <= 0.3383 | PASS |
| C8 1-hop / 2-hop J | 2.806 / 0.443 | 9.657 / 1.098 | 9.567 / 1.158 | <= 0.838 / <= 0.292 | FAIL |
| C9 reconstruction | 6.70e-05 | 6.70e-05 | 6.71e-05 | <= 8.589e-05 | PASS |
| C10 SID identity | pass | pass | pass | — | PASS |

`all_pass = false` (7 pass, 3 fail). **Not a PASS; the issue stays open.**

**What the three rounds establish together.** Three structurally different levers — the codebook
channel, the encoder channel, and the matching-coverage channel — end at the same place: the L2
count stays at 7,911-7,929 and the SID collision rises monotonically with training (0.071 at epoch
50 -> 0.136 at 750 -> 0.1543 final, against round 2's 0.1573). The C3 ceiling (~7,900 against
10,918) is not a sampling artifact; the threshold 10,918 is the #265 Euclidean local-curvature
value. C5 fails 2/3 in every round (the mechanism's SID prefix structure sits slightly below the
frozen TIGER values). C8's thresholds (0.838 / 0.292) are the #269 pointwise-curvature values; the
pair-ranking channel trades the graph-distance strata (9.57 / 1.16) for the ranking accuracy
(0.803), while the codebook-only round kept the strata better (2.81 / 0.44) but broke the scale.

The predictable half of the contract is met in every encoder-channel round: the curvature-selected
ranking (C1 0.803-0.806, C2 0.464-0.468, C7 0.304-0.315 on a 0.338 budget, C9 6.7e-05, C4 11-12.4).
The three unreachable criteria all carry thresholds from the two supervision families this issue's
fixed mechanism forbids (the #265 Euclidean local-curvature supervision and the #269 pointwise
curvature supervision). Closing the issue would require amending the contract — that decision is the
owner's, not the agent's; this issue therefore stays open with the full evidence recorded.

Artifacts: `results/research/issue274/{CONTRACT,DIAGNOSIS,RUN_LOG,VERDICT}.md`,
`round{1,2,3}_pass_table.json`, `round3_ranking.json`, `round3_within_centre.json`; the frozen
evaluator scripts under the same directory. Round-3 candidate: `TIGER_pairrank_cover_rq`.
