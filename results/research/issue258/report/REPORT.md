# Round report: TIGER_role_conditioned_l2_rq

Run: `issue256_r05_formal`
Stage3: `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/TIGER_role_conditioned_l2_rq/logs/Amazon_2023_Instruments/Nov-25-2026_23-03-28`

## SID structure

| quantity | value |
|---|---|
| items | 24587 |
| H_L1 | 7.396808467342224 |
| H_L2_given_L1 | 5.218217225330692 |
| H_L3_given_L1L2 | 1.8515703497702383 |
| distinct_L1 | 256 |
| distinct_L2 | 9609 |
| distinct_triples | 23228 |
| items_per_L2 | 2.558747008013321 |
| items_per_triple | 1.0585069743413122 |
| collision | 0.05527311180705252 |

## Role conditioning

| quantity | mean | median | p90 |
|---|---|---|---|
| roles_per_prefix | 1.6771 | 1 | 3 |
| items_per_prefix | 2.5587 | 1 | 5 |
| items_per_cell | 1.5257 | 1 | 3 |
| distinct_L3_per_cell | 1.497 | 1 | 3 |

## Stage3 (complete population)

| metric | value |
|---|---|
| n_eval | 57439 |
| test_recall@5 | 0.033270077821689094 |
| test_recall@10 | 0.05034906596563311 |
| test_ndcg@5 | 0.021846481598707893 |
| test_ndcg@10 | 0.02732772020207201 |

## Validation

reused: 6, rerun: 0

## Timing

merged wall: 8951.906s, GPU wall: 4581.704s

## Verdict

status: **COMPLETED_NEGATIVE**
- C1: PASS
- C2: PASS
- C3: PASS
- C4: PASS
- C5: PASS
- C6: PASS
- C7: PASS
- C8: PASS
- C9: PASS
- C10: PASS
- C11: IN_PROGRESS
- C12: IN_PROGRESS
- C13: IN_PROGRESS
