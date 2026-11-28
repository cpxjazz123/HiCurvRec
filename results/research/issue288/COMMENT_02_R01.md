## #288 R01 result — the same-anchor pairs on the final hard q: P passes, N/C fail (1 of 3)

The R01 candidate (`stage2_RQ-VAE/TIGER_sameanchor_finalq_rq`, 3000 epochs, single card, global
batch 4096, seed 42) is judged on its **own checkpoint-bound export** with the frozen #277 evaluator:

| metric | R01 | R2 | R3 | threshold | verdict |
|---|---|---|---|---|---|
| P | **0.08194780549800658** | 0.0783613920855168 | 0.08097285816257246 | > 0.08097285816257246 | **PASS** |
| N | 0.009817290481916674 | 0.008752316503966327 | 0.00901661164455637 | < 0.008752316503966327 | FAIL |
| C | 0.16207752064098913 | 0.1511367795989751 | 0.1543498596819457 | < 0.1511367795989751 | FAIL |

SID sha256 `59c5567e0eef910742ef05887892989075b214b1f74fb869bfc5fc8a7d8f48b8`; the record is
`results/research/issue288/results/R01.json`; the training path is verified: the pair gradient
reaches the **three codebooks** (0.080/0.085/0.081) and not the encoder, the reconstruction↔pair
cosine 0.079, λ = 2.112e-06 (the 2 % budget), margin 0.00470653947787214 (`MARGIN_R01.json`).

**Why this change did not break the P/N linkage** (the issue's required explanation): the pair
construction moved P up exactly as predicted (H73: the same-anchor pairs raise P by +0.0036 against
R2) but the **in-bucket purity stayed at 0.34958**, below the 0.3542 #287 maximum, and the L1 bucket
sizes stayed large (mean 231.9, max 548). The final-`q` supervision restored the all-levels gradient
path but the same-anchor pairs displaced the L2/L3 shaping that R2's outer pairs provided (C
0.162078 vs R2's 0.151137).  The measured knowledge: **the pair construction owns the P/N/C shape** —
the #274 outer pairs give the best N/C, the same-anchor pairs the best P; the purity ceiling is not
moved by either at the 2 % budget (the measured L1 assignment change is 0.85 % per first step).

**H74 REFUTED** (the combination did not keep R2's N/C); H73 stays SUPPORTED; the diagnosis
hypotheses H70/H72 stand.

## R02 — pre-registered: the gradient budget (2 % → 10 %)

Single main change: the **gradient budget** on the #287 R1 form (the #274 strict index + the **L1
selected codeword** supervision — the only arm that has passed P *and* C simultaneously: P 0.083532,
C 0.150933, N 0.009392).  Evidence: the measured L1 assignment change rate after the first step was
0.85 % and the in-bucket purity has stayed ~0.35 across five mechanisms — the pair term is too weak
to reshape the partition, and the budget is the one legal knob never varied in #287.  Frozen: the
#274 margin 0.2529585361480713, 256 groups, the 200-step ramp, the protocol, seed 42, batch 4096.
Falsifiable prediction: N < 0.008752 with P > 0.080973 and C < 0.151137, the L1 assignment change
rate materially above 0.85 % and the purity above 0.354; if N rises with C instead, the budget
strengthens the coarsening and R03 changes the schedule/margin selectivity.
