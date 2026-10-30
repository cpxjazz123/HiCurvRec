# CONCMID-GEO-SID -- round two

Selected by `round_20261031_054914_600` (two first places, mean rank 1.67,
IN_SCOPE). The baseline is the round-one mechanism tree; the round framework
resolved the FROZEN baseline instead (the round-one Stage 3 was run by hand and
its round state never recorded `stage3: done`), so the round-one model
(`curved_chain.py`, `model.py`), its tests and its trainer were merged into this
tree, as §19b intends.

## The mechanism, as implemented

`prepare_behaviour.py` builds, from the training split only:

    C[i_t, i_{t+1}] += 1.0 ; C[i_t, i_{t+2}] += 0.5
    p_ij = C_ij / row_sum ; a_ij = p_ij / sqrt(p_j) ; w_ij = a_ij / sum a
    eta_i = sum_j w_ij^2                                 (Herfindahl concentration)
    z_i   = exp_0(kappa t_i)                             (kappa = 0.9 / max||t||)
    v_i   = sum_j w_ij log_{z_i}(z_j)
    b_i   = exp_{z_i}(eta_i v_i)

and the moved points feed the frozen-geometry curved chain unchanged. Switches:
`BEHAV_SMOOTH` (eta = 0 when off) and `CURV_C` (0 makes the chain flat).

Precompute measurements: **24,474 items with transitions**, 113 without; eta
median 0.179, p90 recorded in the manifest; **99.1% of the items move** (median
displacement 0.0207, p90 in the manifest). The smoothing is therefore not a
no-op: the partition is genuinely re-aimed by behaviour.

## Stage 2 (four GPUs, 3000 epochs, 592.9 s)

| level | distinct prefixes | largest group |
| --- | --- | --- |
| L1 | 230 | 684 |
| L2 | 1,251 | 628 |
| L3 | 1,298 | 628 |

Every item covered, full SID unique. The partition is COARSER than round one's
(251 / 1,769 / 2,201), which is the second risk the proposal named: pulling items
toward a common successor hub merges prefixes. Technical validity passed (L1 Hard
ER 5.20, better than round one's 6.09; L1 positive hit rate 0.0537, negative
0.0103; L2 distinct prefixes 1,251).

## Stage 3

Launched under the frozen protocol (150 epochs, seed 42, beam 20, no mid-run
eval, test on), four GPUs, pid 710068. The recall reading is pending.

## The result

Stage 3 finished all 150 epochs (1975 s) and wrote the test record at epoch 150:

| metric | value |
| --- | --- |
| test_recall@5 | 0.03500 |
| **test_recall@10** | **0.05165** |
| test_ndcg@5 | 0.02294 |
| test_ndcg@10 | 0.02830 |
| queries | 57,439 |

**The target of test_recall@10 > 0.065 was NOT reached, and the mechanism is
worse than round one** (0.05165 against 0.05439). The partition it produced is
coarser at every level (230 / 1,251 / 1,298 against 251 / 1,769 / 2,201), which is
the second risk the proposal itself named: pulling items toward a common
successor hub merges prefixes, and the merged prefixes cost more than the
behaviour alignment gains.

## Two rounds, two negative results

| round | mechanism | L1 / L2 / L3 prefixes | test_recall@10 |
| --- | --- | --- | --- |
| one | curved chain on the frozen text geometry | 251 / 1,769 / 2,201 | 0.05439 |
| two | concentration-scaled geodesic step toward the successor barycentre | 230 / 1,251 / 1,298 | 0.05165 |
| gate | -- | -- | **> 0.065** |
| frozen baseline | -- | -- | ~0.0679 |

Both mechanisms run the full frozen chain (Stage 2 3000 epochs at 0.145 to 0.16 s
per epoch, Stage 3 150 epochs) and both fall short. The consistent signal across
them is that coarsening the deeper prefixes hurts: round one kept 2,201 L3
prefixes and scored 0.0544, round two kept 1,298 and scored 0.0517.
