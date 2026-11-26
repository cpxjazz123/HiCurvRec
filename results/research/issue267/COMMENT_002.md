## Issue #267 root diagnosis completed

- Full 28,391,948-triple fixed-distance recomputation on #265:
  - hard q: Pearson `-0.03310`, Spearman `-0.09342`, MAE `2.24595`, RMSE `2.66498`; pred/target variance `0.88857 / 2.87297`; pred/target mean `0.73624 / -1.06232`.
  - encoder z: Pearson `-0.02838`, Spearman `-0.09208`, MAE `2.24801`, RMSE `2.67032`.
  - q/z are nearly identical: the failure is not quantization-only.
- Within-centre ordering (23,819 centres): q mean/median Spearman `0.09288 / 0.08695`; z `0.09341 / 0.08758`; positive-centre fraction `0.6413`.
- 32-draw gate probe: learned gate MAE `2.05557`, Spearman `0.10452`; equal weight MAE `2.43348`, Spearman `0.01374`. Gate helps scale/order on the sample but is not a full-table proof.
- Hypotheses: H267-1 SUPPORTED, H267-2 SUPPORTED, H267-3 INCONCLUSIVE, H267-4/H267-5 UNTESTED.
- Six admission questions answered in `DIAGNOSIS.md`; mechanism contract next.
- Evidence: `results/research/issue267/{DIAGNOSIS.md,HYPOTHESES.json,analysis/CURVATURE_FULL_109_Q.json,analysis/CURVATURE_FULL_109_Z.json,analysis/CENTER_ORDER_109.json,analysis/GATE_PROBE.json}`.
- No training or parameter update has occurred.
