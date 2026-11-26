# Issue #267 report

## Root cause

The #265 hard-q curvature representation is weak on the complete fixed target
population: full 28,391,948-triple q Spearman `-0.09342`, z `-0.09208`, q
prediction variance `0.88857` vs target `2.87297`, and a +1.7986 mean offset.
Within centres the median Spearman is only `0.08695` (q) / `0.08758` (z), while
the overall negative sign is dominated by centre offsets. The sample gate probe
shows learned gate improves MAE over equal weight (`2.05557` vs `2.43348`) but is
not a full-table proof. See `DIAGNOSIS.md`, `analysis/CURVATURE_FULL_109_Q.json`,
`analysis/CURVATURE_FULL_109_Z.json`, `analysis/CENTER_ORDER_109.json`,
`analysis/GATE_PROBE.json`.

## Mechanism

Parent #265 is retained: frozen GNN/token cache, Euclidean residual selection,
floor/last-level Sinkhorn, point-value local mixed curvature and its calibration.
The only new term is a same-centre softplus sign-ranking loss on the actual hard
q, with target-derived `tau=1.0516836255159168`, ties excluded at `1e-6`, and
`alpha = 0.2 * ||grad L_point|| / ||grad L_rel||` frozen on the fixed initial
sample. Full formula and gradient route: `CONTRACT_MECHANISM.md`.

Actual Stage2 calibration: alpha `0.3075788718799832`, point/relative gradient
norms `7.125525425767778 / 4.633299668611924`, 378 valid pairs, 0 ties/cross-centre
pairs. The structural budget is recomputed after adding the term; no point term
is removed.

## Stage2

To be filled after the 3000-epoch run. Run id
`stage2_88816eeca29442c7b3284f4d4a68f610`.

## Pre-Stage3 score

To be filled from `SCORE_PREDICTION.json` before Stage3.

## Stage3

To be filled from the frozen 150-epoch run.

## Independent full comparison

To be filled from `FULL_COMPARISON_109_NEW.json`.

## Conclusion

To be written in `VERDICT.md`; negative results are retained.
