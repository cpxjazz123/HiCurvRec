# Issue #267 verdict

## Root cause

The parent #265 hard-q curvature representation is weak on the complete target
population: full 28,391,948-triple q Spearman `-0.09342`, z `-0.09208`; q
prediction/target variance `0.88857/2.87297`, mean offset `+1.7986`. Within
centres the q median Spearman is `0.08695`; the gate probe improves a sample but
does not explain the whole failure. Root-cause evidence is in `DIAGNOSIS.md`,
`analysis/CURVATURE_FULL_109_{Q,Z}.json`, `analysis/CENTER_ORDER_109.json` and
`analysis/GATE_PROBE.json`.

## Mechanism

Only the relative term was added. Same-centre target pairs use
`softplus(-sign(Δy) Δp / tau)`, `tau=1.0516836255159168`, `|Δy|>1e-6`, upper
triangle mean; `alpha = 0.2*||grad L_point||/||grad L_rel||` was calibrated before
the first optimizer step (`alpha=0.3075788718799832`). The point term, hard q,
Euclidean selection, floor/Sinkhorn, frozen GNN/token path and Stage3 protocol are
retained. Mechanism contract and gradient route: `CONTRACT_MECHANISM.md`.

## Stage2

`stage2_88816eeca29442c7b3284f4d4a68f610`, 3000 epochs, four ranks, seed42:
final collision `0.04534916825964941`, wall `811.659 s`, final-epoch checkpoint.
Initial export/keypoint mismatch 238/24587 items was resolved by the existing
checkpoint-bound export audit; the manifest records the original mismatch.
§63 layer record: L3 collision `0.04547118395900274`, L1 Hard ER `8.3595751`,
L1 positive hit `0.06100385`, 57439 valid rows.

## Pre-Stage3 prediction

Frozen `SCORE_CONTRACT.md`; before Stage3/new Recall:
`R=8.386929403059185e-05`, `G_gate=0.10466435496425182`,
`G_L3=0.0005665696095017952`, **S=1.2921335084032926**, above all ten historical
scores. No clipping or refit.

## Independent full comparison

New vs #265: q Spearman `-0.03373` vs `-0.09342`; q Pearson `+0.02130` vs
`-0.03310`; q MAE `2.20395` vs `2.24595`; q prediction variance `0.88073` vs
`0.88857`; within-centre q mean/median `0.09581/0.09119` vs `0.09288/0.08695`. z
follows the same direction. The relative term improves order modestly but does
not recover full-table positive association or variance.

## Stage3

`test_recall@10 = 0.05078431031180905`, `test_recall@5 = 0.03300893121398353`,
`test_ndcg@10 = 0.027244785026023777`, `n_eval = 57439`. Parent #265:
`0.052594926791900976`. Delta `-0.0018106164800919256` (`-3.4426%`).
Run: `results/stage3_T5Train/TIGER_relative_curvature_ranking_rq/logs/Amazon_2023_Instruments/Nov-26-2026_17-58-19`.

## Eleven-arm prospective check

Recall-ascending rank of the new arm: 6/11 (above `101, 104, 100, 107, 106`;
below `105, 108, 109, 103, 102`). Correlations with S: Pearson `0.513532`,
Spearman `0.863636`, Kendall `(50, 5)` tau `0.818182`; the historical ten remain
`Pearson 0.928526`, `Kendall 45/0`. The frozen formula predicted the highest
score of all eleven and therefore the last recall rank; it was again out of range
(above) and inverted against five arms. Per the contract the prediction is
reported, not deleted or refit.

## Acceptance

C1–C9 complete: C1/C2 diagnosis, C3 mechanism/tests (139 passed, five mutations
caught), C4 pre-Stage3 score before reading Recall, C5 Stage2 collision
`0.04534916825964941 < 0.40`, C6 frozen Stage3 above, C7 independent full
comparison, C8 wall `811.659 s` vs parent `813.357 s`, C9 R2 archive
`r2:genrec/110/` and infra snapshot end diff `PASS` (0 changed, 0 missing).
Negative result retained; no retry, seed search or weight selection.
