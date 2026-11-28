# Issue #288 — REPORT (cumulative)

All numbers are the frozen #277 evaluation of the round's **own checkpoint-bound export** (57,439
validation rows; the last-history anchor vs the true target; the legal uniform negative population).
Thresholds: P > 0.08097285816257246, N < 0.008752316503966327, C < 0.1511367795989751 on one export.

| round | arm | the single main change | P | N | C | thresholds |
|---|---|---|---|---|---|---|
| R01 | `TIGER_sameanchor_finalq_rq` | the pair construction: the #274 outer pairs → the **same-anchor** pairs, keeping the final hard `q` | **0.08194780549800658** | 0.009817290481916674 | 0.16207752064098913 | P |
| ref | R2 (#274) | — | 0.0783613920855168 | 0.008752316503966327 | 0.1511367795989751 | — |
| ref | R3 (#274 cover) | — | 0.08097285816257246 | 0.00901661164455637 | 0.1543498596819457 | — |
| ref | TIGER (no ranking) | — | 0.09253294799700552 | 0.011456922353416539 | 0.16927644690283483 | P |

## R01 — the same-anchor pairs on the final hard q

*The diagnosis that motivated it* (`DIAGNOSIS.md`): the in-bucket behavioural purity is 0.333–0.354
across the five frozen exports (the measured ceiling on P), and the curvature's within-stratum
AUC (0.7291) is below the degree-sum control (0.8707) — the curvature screen is a scale proxy.

*The measured result*: P 0.081948 (pass), N 0.009817 (fail), C 0.162078 (fail); the in-bucket purity
0.34958, the unshared related pairs 95.46 %, the L1 bucket mean/median/max 231.9/216/548.  The
training path verified the intent: **the pair gradient reaches the three codebooks**
(0.080/0.085/0.081) and not the encoder; the reconstruction↔pair cosine 0.079; λ = 2.112e-06
(the 2 % encoder-free budget).

**Round verdict: 1 of 3.** The same-anchor construction does raise P (R2 +0.0036, R3 +0.0010) —
H73 stays SUPPORTED — but the final-`q` form did **not** preserve R2's N/C when the pairs changed
(H74 REFUTED): the C (0.162078) is worse than even #287's same-anchor + L1 arm (0.158376). The pair
construction therefore owns the P/N/C *shape*: the #274 outer pairs give the best N/C, the
same-anchor pairs the best P. This is the knowledge R02 uses.

## R02 — the gradient budget (pre-registered)

*The change*: the #274 strict index + the **L1 selected codeword** supervision (the #287 R1 form,
the only arm that passed P **and** C simultaneously: P 0.083532, C 0.150933, N 0.009392) with the
**gradient budget raised 2 % → 10 %**.  Everything else frozen (the #274 margin 0.2529585361480713,
the 256 groups, the 200-step ramp, the final-q... no: the L1 codeword, the protocol).

*The evidence*: the measured L1 assignment change rate after the first step was 0.85 % (a weak
force), and the in-bucket purity has stayed ~0.35 under five mechanisms — the pair term is too weak
to reshape the partition; the budget is the one legal knob never varied in #287.

*The falsifiable prediction*: N must fall below 0.008752 while P stays above 0.080973 and C below
0.151137; the mechanism diagnostics must show an L1 assignment change rate materially above 0.85 %
and the purity above 0.354; if N instead rises with C, the budget strengthens the coarsening and R03
must change the schedule or the margin selectivity.
