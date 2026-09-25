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

## Post literature and direction review — iter18

- The cited Riemannian-adaptive sources support product-manifold adaptive optimization for fixed factors; they do not establish optimizer-state semantics for the live cyclic `c_l(t)` used here. The current code stores codebooks as Euclidean `nn.Embedding` parameters, and `_transport_between_t` is not point-to-point parallel transport.
- Agent B therefore deferred P17A rather than silently applying fixed-metric formulas to the cyclic schedule. P17C was also deferred because its cited preconditioner approximates Riemannian Hessian blocks; replacing that operator with a curvature scalar is unsupported.
- Selected P18B: fixed per-layer AdamW `β₂,l = 0.999 - 0.009 * clamp(2*log(c_l(0)/C_CYCLIC_MIN)/log(C_CYCLIC_MAX/C_CYCLIC_MIN), 0, 1)`. This changes second-moment memory, not the effective learning rate. `β₂,l` is frozen at initialization to preserve standard bias correction; cyclic curvature, all losses, Stage1, and Stage3 remain unchanged.
- This is a falsifiable optimizer hypothesis, not a literature-proven rule. Require the registered direct checks and one-checkpoint/one-batch gradient-path verification before Stage2 training. Stage2 quality metrics remain descriptive; only full Stage3 `test_recall@10 > 0.065` passes.

### Pre-Stage2 verification — PASS

- `mvg_check.py` loaded one iter8 checkpoint and one `(640, 768)` batch; total and component losses retained finite gradient paths. Quantizer embedding gradient norms were nonzero for all three layers.
- The fixed per-layer `β₂` values were `[0.998991, 0.990604, 0.990001]`; five-step relative updates were `[0.02443, 0.23058, 0.25270]`.
- Same-seed 200-step ON/OFF losses were `2.47846961` and `2.46642065` (`|Δ|=0.01205`); cyclic curvature was finite and returned to its initial values at step 100,000.
- Residual-scale provenance warning: iter18 has no local `layer_norms.json`, so the existing fallback uses iter1 scales `[0.001, 0.932889, 1.0]`. This matches the tested initialized curvature shown in `mvg_check_iter18.log`; retain as a provenance caveat.

## Post iter18 Stage3 result

### Evidence

- The complete Stage3 run `Sep-26-2026_04-45-57` finished all 150 epochs; final beam-20 evaluation used `n_eval=57439` and reported `test_recall@10=0.05988962203380978`.
- The strict target `test_recall@10 > 0.065` was missed by `0.005110377966190224`. Iter18 is higher than iter17 by `0.0008182593708107727`, iter11 by `0.00012186841692925915`, and iter16 by `0.0030118908755375207`. These are observed run differences, not causal estimates.
- MVG/direct effects passed; Stage2 completed 100,000 steps and exported iter18 four-token SIDs. Geometry stayed descriptive and no Stage2 gate was applied.
- Pipeline evidence confirms the Stage3 `code_path` is the iter18 SID file and `world_size=4`; the run completed and produced final metrics. The event's `variant="unknown_variant"` and NCCL cleanup warnings are recorded as metadata/runtime caveats, not proof that the result is invalid.

### Attribution and decision

**NO-GO:** P18B's completed Stage3 run did not meet the downstream target. This is a qualified target miss, not proof that P18B caused the score, has no effect, or that the target is unreachable. No implementation, activation, geometry, or pipeline failure is evidenced.

### Next iteration objective

Do not repeat fixed initial-curvature `β₂`. Before selecting iter19's single mechanism, review evidence and require a falsifiable rationale for retaining the cyclic curvature schedule. Keep Stage1 and Stage3 trainer/configuration unchanged, run the one-checkpoint/one-batch gradient/MVG verification before Stage2 training, and let only the completed Stage3 `test_recall@10 > 0.065` determine adoption. Stage2 geometry and `hitrate@50` remain descriptive only.