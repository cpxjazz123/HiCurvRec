# direction_decision_iter7 (Agent B — direction judge)

Inputs:
- Bottleneck (Agent G iter6): iter5/iter6 widened the behavior graph but stayed within iter4's curvature range, so the cyclic-curvature schedule kept clamping `c_layer_scale` to iter4's final values — new gradients on curvature were effectively frozen.
- Failed-mechanism ledger: empty (no TRUE_MECHANISM_FAIL entries recorded this session; iter5/iter6 are archived without formal mechanism-failure attribution).
- Mechanism pool: P1 (Riemannian Adam / wider range), P5 (warm-start embeddings only), P6+P7 (sharper behavior loss).

Scoring (a–f dimensions):

| Candidate | (a) fixes bottleneck | (b) paper support | (c) curriculum compat | (d) Stage-1 ceiling risk | (e) Gap-closing | (f) Curvature-relevant | Total |
|---|---|---|---|---|---|---|---|
| P1 (wider range + clamped layer-scale reset) | yes — unclamps c_layer_scale | yes (Guo 2022, Bécigneul 2019) | high — same cyclic schedule | medium | yes — directly explains "wider curvature reach" | yes — geometric/manifold | 6/6 |
| P5 (warm-start embeddings only) | partial — only saves embeddings | weak | high | high (Stage-1 ceiling risk flagged) | partial | low | 3/6 |
| P6+P7 (sharper behavior loss) | no — does not address clamp | medium | medium | low | partial | yes — contrastive on Poincaré ball | 4/6 |

Recommendation: **combine P1 + P5 + P6+P7 as ONE mechanism**: widen the cyclic curvature range to 0.05..1.5 with period 100k, warm-start encoder/decoder/codebooks from iter4 while resetting `c_layer_scale` to the calibrated interior, and pair with `BEHAVIOR_TEMPERATURE=0.07` + `BEHAVIOR_LOSS_WEIGHT=0.20` + `CURVATURE_REG_WEIGHT=0.005`.

Reasoning: iter4's clamp-bound c_layer_scale is the proximate cause of stage3 ceiling (no curvature gradient flow → no schedule differentiation). Widening the range without warm-start would re-randomize embeddings and waste 100k steps; warm-starting embeddings while leaving the c_layer_scale to reset at the calibrated interior lets the new wider schedule explore fresh curvature without re-learning the codebook. Lowering temperature and raising the behavior weight make the contrastive term meaningful at the wider range's high-curvature peak.

Backup: P5 alone (pure warm-start) — lower expected gain because iter4's c_layer_scale stays clamped.

Decision: iter7 is `wide_curriculum_warmstart`, single mechanism combining P1+P5+P6+P7.