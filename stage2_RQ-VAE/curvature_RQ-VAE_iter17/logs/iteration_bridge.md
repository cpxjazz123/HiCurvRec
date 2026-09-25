# iteration_bridge (Agent G — post iter16 → iter17)

## Evidence

- iter16 used fixed `c_l = [0.663, 0.435, 0.433]` for 100k steps; Agent F marked implementation, activation, geometry, and pipeline checks PASS.
- Stage3 `test_R@10=0.05688`, down `0.00289` from iter11 `0.05977` and `0.00136` from iter15 `0.05824`; hard target gap `0.00812`.
- iter16 SID remained structurally healthy (`unique=22827/24587`, `H(L1|L0)=5.4549`, `full_gini=0.0681`), so SID collapse does not explain the downstream regression.

## Root cause and gap

Dominant bottleneck: freezing curvature at the high `c0=0.663` removed cyclic low-curvature behavior and regressed Stage3 despite healthy SID utilization; this is the strongest available explanation, not a proven causal result.

Mechanism execution: implementation PASS; activation PASS; geometry aligned; downstream target FAIL.

## Forbidden next directions

- Do not repeat fixed closed-form curvature or freeze the cyclic schedule.
- Do not change Stage1 embeddings or Stage3 trainer/configuration.
- Do not stack additional loss mechanisms in this iteration.

## Next iteration objective

Keep iter8's cyclic, learnable curvature schedule and test exactly one curvature-metric optimizer mechanism whose per-layer adaptive parameter updates scale with normalized `1/(c_l + 1e-3)` and whose ON/OFF 200-step loss difference exceeds `1e-6`, while retaining finite losses and nonzero updates on every layer.

## Post iter17 → iter18

### Evidence

- The complete, successful Stage3 run finished 150 epochs on the registered iter17 SID input and reported `test_recall@10=0.059071362662999005` (`n_eval=57439`); the strict hard target `test_recall@10 > 0.065` was missed by `0.005928637337000995`.
- This observed score is below iter11 (`0.05976775361688052`) and above iter16 (`0.05687773115827226`). These are run comparisons, not causal estimates.
- The four-gate MVG and all registered direct effects passed. The three-token SID report classified geometry as descriptively `ALIGNED` (`H(L1|L0)=5.5939` bits; 6.08% three-token collision rate). The corrected successful-run pipeline checks passed; no implementation, activation, geometry, or wiring failure is evidenced.
- Stage2 geometry and direct-effect metrics are descriptive only. There is no Stage2 gate, and `hitrate@50` is not a gate.

### Dominant bottleneck

**The completed, correctly wired Stage3 run missed the downstream recall target despite passing MVG/direct-effect checks and descriptively aligned Stage2 geometry.** The exact result remained below iter11 and above iter16. Inference, not proven causality: the tested P17B normalized per-layer inverse-curvature AdamW multiplier may be insufficient by itself to close the downstream gap. This single run does not establish that it caused the score, that the target is unreachable, or any particular Stage3 cause.

### Falsifiable next objective — iter18

Carry forward exactly one distinct curvature-aware optimizer mechanism for Agent A/B/C review: **P17A, Riemannian Adam on hyperbolic factors**, a literature-backed candidate for metric-aware adaptive updates. Evaluate it instead of repeating P17B's scalar per-layer learning-rate multiplier. Preserve the existing cyclic curvature schedule; preregister and verify direct effects before Stage2 training; then complete Stage3. The objective passes only if the completed run's strict `test_recall@10 > 0.065`; otherwise it fails. Keep Stage2 metrics descriptive, do not change Stage1 or Stage3 trainer/configuration, do not stack mechanisms, and do not use `hitrate@50` as a gate.