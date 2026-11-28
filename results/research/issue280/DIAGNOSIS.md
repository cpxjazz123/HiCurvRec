# Issue #280 — DIAGNOSIS (competing hypotheses, frozen before decoding)

Phenomenon: #279 found real relation-level higher-order discrimination on unseen transitions
(AUC 0.62–0.63; degree-matched 0.66) but no SID-level Recall increment. The new hypothesis is that the
discrete SID can *decode* unseen directed edges directly (directional code-transition likelihood),
which would measure predictive information better than prefix sharing.

| id | explanation | discriminating measurement | status |
|---|---|---|---|
| H20 | Directional SID-code transitions carry predictive information about unseen directed edges beyond prefix sharing | decoder AUC on unseen candidates vs the prefix baseline and chance; direction asymmetry | UNTESTED |
| H21 | Any decoder signal is prefix sharing or popularity in disguise | decoder AUC vs the prefix baseline and the popularity baseline; ablations | UNTESTED |
| H22 | Higher-order structure (2-hop / common neighbours) is partly decodable from the SID | Spearman of decoder scores with graph higher-order scores; AUC stratified by higher-order strength | UNTESTED |
| H23 | The decoder metric adds a Recall increment beyond the six metrics | pre-registered Ridge6 + ≤2 decoder features vs the refit Ridge6, paired CI | UNTESTED |

The negative controls are explicit: a decoder that only memorises item pairs is impossible under the
SID-only input; a symmetric prefix score must be beaten by the directional score for H20 to hold.
