# Issue #269 diagnosis

## 1. Observed phenomenon (kept separate from causes)

- #268 fitted a per-triple continuous curvature on the graph-unit scale
  (`S = max||Z_ref|| = 0.5955`, coordinates `<= 0.5`).
- 99.78 % of the 28,391,948 fitted values sat on the coordinate floor `k = -0.125`;
  kappa std `0.0101`; the fit beat one global kappa on only 0.18 % of independent rows and
  on 0.0 % of the full-reference audit (`results/research/issue268/analysis/`).
- A single global Euclidean scale reached `J = 0.174575` against the per-triple family
  `0.253693`; the curvature mechanism was stopped before training.
- The #269 question is whether that failure is a *global-scale* artefact or a real absence of
  local geometric information.

## 2. Competing explanations and what measured them

| ID | explanation | evidence | status |
|---|---|---|---|
| E1 | the ruler's global scale is wrong; per-triple curvature is confounded with a single multiplicative degree of freedom | #268 `SCALE_CONTROL.json`: one scalar beats the whole per-triple family; #269 `GATE.json`: at `S*` per-triple beats one global kappa by 4.84 % held out | **SUPPORTED** |
| E2 | the curvature proxy is too weak: even a calibrated per-triple value adds no held-out information | `GATE.json`: improvement CI `[0.0419, 0.0554]`, Spearman `0.947` -> the calibrated fit is transferable | **REFUTED** |
| E3 | the frozen semantic representation `Z_ref` lacks behaviour geometry, so no valid ruler can help | `GATE.json` full-data statistics: 5.91 % mean distortion reduction, 2.97 % median, 17.7 % boundary, 2.5 % flat | **INCONCLUSIVE** (explains part: shape-free-scale `J = 0.148579` is still below the per-triple `0.173216`) |
| E4 | the #268 Stage2 NaN/floor stall is a scale-independent mechanism bug that would also block #269 | `results/research/issue269/analysis/`: `project` put `inf` in the unselected `torch.where` branch (NaN backward for `k >= 0`); the `asin(sqrt(0))` composition had an infinite intermediate derivative; `assignment_floor` then saw a NaN cost matrix and raised `InfeasibleFloor` at epoch 4 | **SUPPORTED and RESOLVED** |

Measurement detail for E4 (this is a code bug, not a research finding):

- Before the fix, the 60-epoch single-card probe at batch 4096 failed at epoch 3 with
  `NONFINITE_GRAD` on the encoder's first layer and then `InfeasibleFloor` at epoch 4.
- Term-wise autograd showed the curvature term's encoder gradient was NaN while the RQ-VAE
  term was finite.
- `project` with `k >= 0` computed `inf/norm` in the branch `torch.where` did not select;
  `0 * inf` in its backward is NaN. Replacing the `inf` radius with a finite placeholder and
  masking the scale afterwards removed it.
- `sqrt(0)` is exact in the forward but has an infinite derivative; the composition
  `asin(sqrt(t))^2` is smooth, so `_SafeSqrt` keeps the exact forward and floors only the
  backward denominator.
- After the fix the same probe passes 60 epochs at batch 4096 (57.4 s, ~0.95 s/epoch), and the
  full candidate suite passes (121 tests). The floor crash was downstream of the NaN.

## 3. Root cause adopted for this round

The #268 negative result is explained by E1: the fitted chart used the graph-unit scale, which
put almost every triple on the coordinate floor and made a free global scale look better than
any curvature table. After the weighted-least-squares calibration `S* = 0.068284`, the
per-triple curvature is identifiable on an independent reference set by the issue's own
pre-registered criteria. E3 remains partly open: a free per-row scale still reaches lower
distortion than the fitted curvature, so the mechanism is a local *shape* correction on top of a
globally calibrated ruler, not a complete geometric model.

## 4. Six pre-training questions (issue gate)

1. **What problem is solved?** The #268 scale confound that erased the per-triple signal.
2. **What evidence shows it was the cause?** `SCALE_CONTROL.json` (one scale beats the family)
   and `GATE.json` (after calibration the family beats one global kappa, CI > 0, Spearman 0.947).
3. **Why can the new mechanism act on it?** The candidate uses exactly the calibrated chart
   `w = q/(2*S*)` for every distance of a triple; `S*` is frozen and shared by fit and training,
   so the training objective is the same functional as the offline fit.
4. **What is materially different from #268?** The scale is calibrated and frozen (`S*` instead
   of `max||Z_ref||`), the legal kappa domain is `[-0.05, 1.0]` instead of `[-0.125, 1.0]`, the
   cache is refitted at `S*`, and the hard q is divided by `2*S*` before the distance. #268 never
   trained; #269 trains one candidate only.
5. **Which metric decides?** Held-out distortion for the gate (already passed) and Stage2
   collision/export diagnostics; the final judgement is Stage3 `test_R@10`.
6. **What is learned if it still fails?** That a calibrated local constant-curvature table does
   not transfer through hard quantization into SID/recall even when its offline held-out gain is
   real, which would separate "geometric fit" from "useful discrete code" for this family.

## 5. Rejected alternatives

- Training the degenerate `kappa = 0` arm: the gate passed, so the issue requires the per-triple
  arm; training both would be a second arm and is forbidden.
- Refitting `S*` after seeing Stage3, per-triple free scales, ranking losses, a gate/GNN: all
  forbidden by the issue and by the #268 evidence.
- Rebuilding the graph, targets, `Z_ref` or assets: `FROZEN`; only hash-referenced reuse.

## 6. Residual uncertainty

- The gate sample is 2,048 rows; the full fit covers 28.4 M rows, but the held-out criterion was
  measured on the sample. The full-data statistics (5.91 % mean, 17.7 % boundary) are consistent
  with it.
- `Z_ref` is the #265 encoder; a different representation could change the fitted kappa. The
  issue freezes this representation deliberately.
- The single-card regime is not comparable with four-card arms; it is reported separately.
