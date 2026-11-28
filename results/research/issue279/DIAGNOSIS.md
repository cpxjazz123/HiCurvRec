# Issue #279 — DIAGNOSIS (competing hypotheses before any new result)

Phenomenon carried from #278: 92.4 % of validation relations are unseen in the training graph; the
frequency-weighted fidelity lowered OOF MAE by 0.000203 with a CI covering zero and one extra FP.
The question is whether *higher-order* topology (2-hop paths, common successors/predecessors, curvature)
captures associations that survive on unseen transitions, and whether the SID retains them.

| id | explanation | discriminating measurement | status |
|---|---|---|---|
| H1 | Higher-order topology provides real association strength for unseen pairs | relation-level AUC of `two_hop`/`common_*` on unseen positives vs legal negatives, anchor-clustered CI | UNTESTED |
| H2 | Any such signal is a degree/popularity proxy | degree-matched AUC within (out-decile × in-decile) cells; correlation with the #278 popularity diagnostic | UNTESTED |
| H3 | The topology is informative but the SID does not retain it | per-SID weighted/unweighted fidelity vs the relation-level strength: if higher-order relations are strong but SID sharing is flat across strata, H3 holds | UNTESTED |
| H4 | The six #277 metrics already absorb the higher-order information | cross-fitted residual analysis; candidate models' CI vs the refit Ridge6 | UNTESTED |

Curvature is diagnostic only (no obligation to show independent value). The round must be able to
distinguish H1/H2 from H4 even if the model fails: the relation-level AUC and the degree-matched AUC
answer H1/H2 independently of the Recall fit.
