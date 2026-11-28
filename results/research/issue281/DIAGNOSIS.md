# Issue #281 — DIAGNOSIS (competing hypotheses, frozen before fitting)

Phenomenon: #277's linear Ridge6 leaves a systematic residual shape (low-recall arms over-predicted by
~0.0034, high-recall under-predicted by ~0.0017, #278/#279). The question is whether nonlinear or
piecewise structure *inside the six metrics* explains that shape, rather than the previously tested
external indicators (all of which failed).

| id | explanation | discriminating measurement | status |
|---|---|---|---|
| H24 | the linear monotone form is insufficient; moderate (inverted-U) ranges exist in ER / collision | inner-selected B/D vs A; implied slopes at p10/p50/p90 | UNTESTED |
| H25 | the L1 positive sharing benefit depends on the negative rate and on L2/L3 discrimination | C's interaction coefficients across folds; interaction response inside the observed range | UNTESTED |
| H26 | piecewise structure reduces the high/low calibration bias | stratified signed error of D vs A; calibration curve | UNTESTED |
| H27 | any complexity only overfits | inner selection never prefers B/C/D; OOF CI covers 0 | UNTESTED |
| null | the six features are essentially linear; the residual is noise | A wins the inner selection; the auto OOF equals A | UNTESTED |

The inner selection (CONTRACT §3) is the discriminator: if A wins in most folds and the auto OOF is not
better, H27/null hold; if a nonlinear model wins and the paired CI excludes 0, H24–H26 have support.
