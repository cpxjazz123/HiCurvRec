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

## Iter19 pre-Stage2 direction

- Selected one source-level correction: in `modules/quantize.py`, the old mapping `ε(c)=sk_eps·c/c_max` paired with `_sinkhorn_algorithm`'s `exp(-distance/ε)` makes larger curvature use a larger temperature and softer assignments, opposite to the iter8 comments claiming high-curvature sharpening. Replace only the mapping with `ε(c)=sk_eps·c_min/c`, retaining the cyclic curvature schedule, Sinkhorn iterations, losses, Stage1, and Stage3.
- Cuturi's Sinkhorn formulation uses the kernel `exp(-λM)` and reports lower plan entropy as `λ` increases; the implementation's denominator temperature therefore corresponds to `ε=1/λ`. Source: [Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances* (2013)](https://arxiv.org/abs/1306.0895). This establishes the temperature direction, not that reciprocal curvature is optimal or improves recommendation recall.
- The earlier Geneva & Zabaras attribution could not be independently verified and was removed from the active implementation comment; it is not evidence for this mapping.
- Standard AdamW is restored unchanged from the baseline optimizer configuration so this trial isolates the epsilon mapping. The checker requires endpoint values, decreasing epsilon at the high-curvature cycle point, cycle return, nonzero loss-gradient paths and per-layer updates, and a finite same-seed 200-step reciprocal-vs-legacy comparison with `|Δloss|>1e-6`.
- Stage2 descriptive SID statistics are not gates; only a fully completed Stage3 run with strict `test_recall@10 > 0.065` passes. Prior iter8/9/10 results do not justify predicting a gain from this correction.

### Pre-Stage2 verification — PASS

- The Stage2 environment ran `scripts/mvg_check.py` on one iter8 checkpoint and one `(640, 768)` batch. `total_loss.requires_grad`, `grad_fn`, finite total/component losses, and nonzero reconstruction/RQ-VAE/behavior gradients passed; quantizer embedding gradient L2 norms were `[0.04520, 0.03264, 0.03019]`.
- Reciprocal-epsilon endpoints passed: at `c_min=0.05`, `ε=0.05`; at `c_max=1.5`, `ε=0.001666667`. The cycle reached `c≈[0.2743, 1.3382, 1.4997]` / `ε≈[0.009113, 0.001868, 0.001667]` at step 50,000 and returned to its step-0 values at 100,000.
- Five-step per-layer relative updates were `[0.02444, 0.23287, 0.26012]`. Same-seed 200-step reciprocal-vs-legacy losses were `2.55552244` vs `2.46642065` (`|Δ|=0.08910`); all checked losses were finite. `MVG PASS`.
- Provenance caveat: the iter19 HG-Rec config has no local `layer_norms.json`; the existing fallback used iter1 residual scales `[0.001, 0.932889, 1.0]`.

### Stage2 completion — descriptive metrics only

- The prescribed four-GPU Stage2 launcher exited `0` after `100,000` global steps (`1,362.6s`). It saved `rqvae_best.pth`, generated the raw three-token SIDs, and exported four-token `sids_for_hgrec.npy` plus `item_sids.json`; `SID_WIRING_PASS` verified the export.
- Final descriptive SID metrics: `full_gini=0.0900`, per-layer Gini `[0.1730, 0.4157, 0.4178]`, `unique=22,233/24,587`, `l01_pairs=13,342`, `H(L1|L0)=5.3839` bits, and `2,354` three-token collision rows. These are observations, not gates; no Stage2 metric controlled progression.
- Training output recorded the reciprocal epsilon tracking the live curvature schedule and standard AdamW groups. No failure/error markers were found in `train_migrated.log`.

## Post iter19 Stage3 result

- The complete four-rank, beam-20 Stage3 run `Sep-26-2026_06-26-19` finished all 150 epochs and evaluated `n_eval=57439`. Its verified `code_path` is the iter19 four-token `item_sids.json`; the first metrics event still reports `variant="unknown_variant"`.
- Final metrics: `test_recall@10=0.058618708542976024`, `test_recall@5=0.03938090844199934`, `test_ndcg@5=0.026259953201859483`, and `test_ndcg@10=0.03247000519831611`.
- The strict target `test_recall@10 > 0.065` was missed by `0.006381291457023978`. The recall is `0.001270913490833754` below iter18, `0.001149045073904495` below iter11, and `0.0004526541200229814` below iter17. These are run differences, not causal estimates.
- `HG_Rec.log` records epoch 150 and the final test event; `training_metrics.jsonl` confirms four ranks and the iter19 input path. Fatal-marker search found no traceback, runtime error, CUDA/NCCL error, `ERROR`, or `FAIL`; standard NCCL wrapper/barrier warnings remain nonfatal caveats.
- Decision: qualified downstream target miss despite MVG PASS, complete Stage2 and correct SID wiring. This does not show that the mapping caused the score or that the target is unreachable. Continue with iter20; keep Stage2 metrics descriptive and use only complete Stage3 recall as the adoption criterion.

## Iter20 pre-Stage2 direction

- Selected one Stage2-only mechanism: per-layer behavior-contrastive temperature `τ_l(t)=0.07·sqrt(C_CYCLIC_MIN/c_l(t))`, bounded by the configured curvature range. It leaves the base temperature at `0.07` at minimum curvature and sharpens high-curvature logits to approximately `0.01278` at `c=1.5`.
- The hypothesis is that stronger high-curvature next-item discrimination can improve the semantic IDs used by Stage3. This is an empirical hypothesis, not literature-established or a predicted recall gain.
- Temperature uses detached live curvature: schedule-dependent scaling is preserved without adding a separate gradient path through the temperature. Preserve iter19 reciprocal Sinkhorn epsilon and standard AdamW as the baseline; do not change losses/weights, cyclic curvature, Stage1, or Stage3.
- Stage2 geometry metrics are descriptive only. No Stage2 gate or HR@50 gate; only a complete Stage3 run with strict `test_recall@10 > 0.065` meets the target.

### Pre-Stage2 verification — PASS

- The required one-checkpoint/one-batch MVG loaded the iter8 checkpoint and a `(640, 768)` batch with 640 active source-target pairs. Total and component losses were finite and differentiable; component gradients were nonzero, including behavior-loss norm `0.1391897`; quantizer embedding gradient norms were `[0.0451957, 0.0326391, 0.0301921]`.
- Temperature endpoints passed: `τ(c_min)=0.0700000`, `τ(c_max)=0.0127802`. At steps `0/50,000/100,000`, temperatures were `[0.0699405, 0.0316665, 0.0299126]`, `[0.0298847, 0.0135307, 0.0127813]`, and `[0.0699405, 0.0316665, 0.0299126]`; the live-curvature schedule returned at the cycle boundary.
- Five-step relative parameter updates were `[0.0244374, 0.2327510, 0.2600038]`. Same-seed 200-step temperature ON/OFF losses were `2.54921436` and `2.51256108` (`Δ=0.0366533`); final behavior losses were `5.77073050` and `5.44007492` (`Δ=0.3306556`). `MVG PASS`.
- Provenance caveat: the iter20 configuration has no local `layer_norms.json`; MVG used the existing iter1 residual-scale fallback `[0.001, 0.932889, 1.0]`.

### Stage2 completion — descriptive metrics only

- The prescribed four-GPU Stage2 launcher exited `0` after `100,000` global steps (`1,324.0s`). It saved `rqvae_best.pth`, generated raw three-token SIDs, and exported the four-token `sids_for_hgrec.npy` and `item_sids.json`; `SID_WIRING_PASS` verified the export.
- Final descriptive geometry: `full_gini=0.0880`, per-layer Gini `[0.1664, 0.3789, 0.4056]`, `unique=22,301/24,587`, `l01_pairs=13,537`, `H(L1|L0)=5.3957` bits, and 2,286 three-token collision rows. Metrics did not gate progression.
- Training logs show live `behavior_temp` values tracking the curvature schedule; the fatal-marker scan found no matches.

### Stage3 result — completed

- The four-rank Stage3 run `Sep-26-2026_07-43-20` completed all 150 epochs with beam size 20 and evaluated `n_eval=57439`. `training_metrics.jsonl` confirms `world_size=4` and the iter20 four-token `item_sids.json` code path; `test_final.json` and `HG_Rec.log` contain the final event.
- Final metrics: `test_recall@10=0.05713887776597782`, `test_recall@5=0.03770957015268372`, `test_ndcg@5=0.02529898456159968`, `test_ndcg@10=0.03156023531751642`.
- The strict target `test_recall@10 > 0.065` was missed by `0.007861122234022182`. Recall was `0.0014798307769982033` below iter19, `0.0027507442678319574` below iter18, and `0.0026288758509026983` below iter11. These are observed run differences, not causal estimates.
- The event's `variant="unknown_variant"` remains a metadata caveat. Fatal-marker scans found no traceback, runtime error, CUDA/NCCL error, or worker fatal marker; the launcher emitted the known nonfatal process-group cleanup warning.
- Decision: qualified downstream target miss despite a completed, correctly wired Stage3 run and passing MVG. This does not establish that the mechanism caused the score or that the target is unreachable. Continue iter21; retain Stage2 metrics as descriptive only.

## Iter21 pre-Stage2 direction

- Selected one behavior-loss mechanism: treat same-item next-item targets in the minibatch as multiple positives, using the supervised-contrastive `L_out` objective `−mean_{p∈P(i)} log softmax(logits_i)_p`. Current code masks duplicate target candidates and keeps only the diagonal positive; iter21 instead counts all same-target candidates as positives.
- Rationale: multiple source items can share a next-item target, so those candidates are same-class examples, not negatives. Khosla et al., *Supervised Contrastive Learning* (2020), Eq. 2, defines the multi-positive mean-log-probability objective ([arXiv:2004.11362](https://arxiv.org/abs/2004.11362)). This supports the loss form, not an RQ-VAE or recommendation-recall gain.
- If a batch has no duplicate targets, the new loss reduces exactly to the existing single-positive cross-entropy. Keep behavior temperature `0.07`, behavior weight `0.20`, reciprocal Sinkhorn epsilon, standard AdamW, cyclic curvature, Stage1, and Stage3 unchanged.
- Stage2 SID statistics remain descriptive. No Stage2 gate or HR@50 gate; only a complete Stage3 run with strict `test_recall@10 > 0.065` meets the target.

### Pre-Stage2 verification — PASS

- MVG loaded one iter8 checkpoint and one `(640, 768)` batch with 640 active source-target pairs. The batch contained 79 valid rows with duplicated targets and 94 extra positive pairs. Total and component losses were finite/differentiable; behavior gradient norm was `0.1268399`, and quantizer embedding gradient norms were `[0.0451957, 0.0326391, 0.0301921]`.
- Five-step relative updates were `[0.0244391, 0.2327443, 0.2600081]`. Same-seed 200-step multi-positive/single-positive losses were `2.51468253` and `2.55552244` (`Δ=0.04084`); final behavior losses were `5.46610355` and `5.42773771` (`Δ=0.03837`). `MVG PASS`.
- Reciprocal-epsilon baseline endpoints/cycle remained finite and returned at step 100,000. Provenance caveat: iter21 has no local `layer_norms.json`; MVG used the iter1 residual-scale fallback `[0.001, 0.932889, 1.0]`.
- Stage2 completed at global step 100,000 in 1,373.1s and saved its checkpoint. SID export produced raw three-token shape `(24587, 3)` and four-token HG-Rec shape `(24587, 4)` plus 24,587 entries in `item_sids.json`; `SID_WIRING_PASS`.
- Final descriptive geometry: full Gini `0.0947`, per-layer Gini `[0.1651, 0.3895, 0.3992]`, unique codes `22,117/24,587`, L01 unique pairs `13,378`, `H(L1|L0)=5.3661` bits, and 2,470 collision rows. These are descriptive only, not gates.
### Stage3 result — completed

- Run `Sep-26-2026_09-42-56` completed all 150 epochs and the full test on four DDP ranks; `training_metrics.jsonl` confirms `world_size=4`, `n_eval=57439`, and the iter21 `item_sids.json` code path. Configuration and logs confirm beam size 20; `test_final.json` records the final metrics.
- Final metrics: `test_recall@10=0.05797454691063563`, `test_recall@5=0.03812740472501262`, `test_ndcg@5=0.025358210362482193`, and `test_ndcg@10=0.03174162401831697`.
- The strict target `test_recall@10 > 0.065` was missed by `0.007025453089364371`. Recall was `0.0019150751231741467` below the current best iter18 (`0.05988962203380978`), and `0.0008356691446578107` above iter20. These are observed run differences, not causal estimates.
- The metrics event reports `variant="unknown_variant"` despite the correct iter21 SID path. Launcher warnings included NCCL `lib wrapper not initialized`, a barrier device warning, and process-group teardown; the supervised job exited 0, and fatal-marker scans found no matches.
- Decision: qualified downstream target miss after a completed Stage3 run and passing MVG; this does not establish the mechanism's causal effect or that the target is unreachable. Continue iter22; retain Stage2 metrics as descriptive only.

## Iter22 pre-Stage2 direction

- Selected one incremental mechanism over iter21: symmetrize the existing multi-positive next-item contrastive objective. Keep its forward source→future logits and positive sets; add a reverse future→source cross-entropy using the transposed pairwise logits and transposed positive mask, then average directions and quantizer layers.
- Rationale: the same transition pair is currently trained only with source items as query anchors. Bidirectional paired retrieval is a standard symmetric cross-entropy construction (Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)); this supports the formulation, not a recommendation-recall gain.
- Preserve the multi-positive same-target masks, behavior temperature `0.07`, behavior weight `0.20`, reciprocal Sinkhorn epsilon, standard AdamW, cyclic curvature, Stage1, and Stage3. No other training or inference changes.
- Stage2 SID statistics remain descriptive only. No Stage2 gate or HR@50 gate; only a complete Stage3 run with strict `test_recall@10 > 0.065` meets the target.
- Before Stage2, the required MVG must confirm differentiable total/component losses, nonzero behavior and quantizer gradients, finite five-step layer updates, and a same-seed 200-step ON/OFF effect where ON uses bidirectional supervision and OFF uses iter21's forward-only objective. All checks must pass before GPU training.

### Pre-Stage2 verification — PASS

- MVG loaded one iter8 checkpoint and a `(640, 768)` batch with 640 active pairs, 79 rows with duplicate targets, and 94 extra positives. Total/component losses were finite and differentiable; `behavior_loss` gradient norm was `0.1541738`, and the symmetric-minus-forward behavior delta gradient norm was `0.0575924`.
- Quantizer embedding gradient norms were `[0.0451957, 0.0326391, 0.0301921]`; five-step layer-relative updates were `[0.0244260, 0.2330274, 0.2600154]`.
- Same-seed 200-step symmetric/forward-only losses were `2.50168657 / 2.51468253` (`Δ=0.01300`); behavior losses were `5.54444885 / 5.46610355` (`Δ=0.07835`). `MVG PASS`.
- Reciprocal-epsilon cycle remained finite and returned at step 100,000. Provenance caveat: the iter22 residual-scale file is absent, so MVG used the iter1 fallback `[0.001, 0.932889, 1.0]`.
- Stage2 completed in 1,401.0s at global step 100,000 on four-rank DDP and saved the checkpoint. Raw SID shape was `(24587, 3)`; the four-token NPY and 24,587-item JSON export passed `SID_WIRING_PASS`.
- Final descriptive geometry: full Gini `0.0840`, per-layer Gini `[0.1624, 0.4027, 0.4001]`, unique codes `22,399/24,587`, L01 pairs `13,477`, `H(L1|L0)=5.3947` bits, and 2,188 collision rows. These metrics are descriptive only. Fatal-marker scan found no matches.

### Stage3 result — completed

- Run `Sep-26-2026_11-08-48` completed all 150 epochs and the full test on four DDP ranks; `training_metrics.jsonl` confirms `world_size=4`, `n_eval=57439`, and the iter22 `item_sids.json` code path. Launcher configuration confirms beam size 20; `test_final.json` records the final metrics.
- Final metrics: `test_recall@10=0.05914100175838716`, `test_recall@5=0.040007660300492694`, `test_ndcg@5=0.026392291300997208`, and `test_ndcg@10=0.0325692274938095`.
- The strict target `test_recall@10 > 0.065` was missed by `0.005858998241612845`. Recall was `0.0007486202754226207` below the current best iter18 (`0.05988962203380978`) and `0.001166454847751526` above iter21. These are observed run differences, not causal estimates.
- The metrics event reports `variant="unknown_variant"` despite the correct iter22 SID path. NCCL wrapper/barrier and process-group cleanup warnings were nonfatal; the supervised job exited 0, and fatal-marker scans found no matches.
- Decision: qualified downstream target miss after a completed Stage3 run and passing MVG; this does not establish the mechanism's causal effect or that the target is unreachable. Continue iter23; retain Stage2 metrics as descriptive only.

## Iter23 pre-Stage2 direction

- Selected one incremental mechanism over iter22: compute the same symmetric, multi-positive next-item contrastive objective from the selected per-layer quantized embeddings rather than the pre-quantization residuals. The quantizer's forward representation is the selected codebook vector, with its existing straight-through gradient to the encoder.
- Testable rationale: Stage3 consumes discrete semantic IDs, while the iter22 behavior objective compared continuous residuals. Training pairwise behavior distances on the selected code vectors may reduce that representation mismatch; this is a hypothesis, not an expected recall gain.
- Preserve iter22's bidirectional and multi-positive masks, behavior temperature `0.07`, behavior weight `0.20`, reciprocal Sinkhorn epsilon, standard AdamW, cyclic curvature, Stage1, and Stage3.
- Stage2 SID statistics remain descriptive only. No Stage2 gate or HR@50 gate; only a complete Stage3 run with strict `test_recall@10 > 0.065` meets the target.
- Before Stage2, MVG must verify differentiable total/component losses, a nonzero quantized-minus-residual behavior delta gradient, nonzero quantizer gradients, finite five-step layer updates, and a same-seed 200-step quantized/residual ON/OFF effect.

### Pre-Stage2 verification — PASS

- MVG used one iter8 checkpoint and a `(640, 768)` batch with 640 active pairs, 79 rows containing duplicate targets, and 94 extra positive pairs. Total and component losses were finite and differentiable; behavior gradient norm was `0.28351067`.
- The quantized-behavior versus residual-behavior objective delta had a nonzero gradient norm of `0.17672019`; quantizer embedding gradient norms were `[0.04519572, 0.03263908, 0.03019207]`.
- Five-step per-layer relative updates were `[0.02437255, 0.23286215, 0.25940972]`. Same-seed 200-step quantized/residual ON/OFF losses were `2.62702322 / 2.50168657` (`Δ=0.12533665`) and behavior losses were `6.31473541 / 5.54444885` (`Δ=0.77028656`). `MVG PASS`.
- The iter23 residual-scale file is absent, so MVG used the iter1 fallback `[0.001, 0.932889, 1.0]`. Reciprocal-epsilon/curvature schedule checks remained finite and cyclic. Stage2 may proceed; this check proves gradient/update paths, not downstream quality.

### Stage2 result — completed

- Four-rank DDP completed all 100,000 global steps in `1,347.9s` and saved the final checkpoint. Raw SID shape was `(24587, 3)`; export produced the four-token NPY with shape `(24587, 4)` and a 24,587-item JSON. `SID_WIRING_PASS`; supervised process exited 0 and the fatal-marker scan found no matches.
- Descriptive geometry at step 100,000: full Gini `0.1415`, per-layer Gini `[0.2268, 0.4868, 0.4994]`, unique 3-token codes `20,804/24,587`, collision rows `3,783`, L01 pairs `11,299`, and `H(L1|L0)=5.0372` bits. These statistics are descriptive only, not gates.
- Next: run the full Stage3 evaluation; only strict `test_recall@10 > 0.065` meets the adoption target.

### Stage3 result — completed

- Run `Sep-26-2026_12-21-18` completed all 150 epochs and the full test on four DDP ranks. `training_metrics.jsonl` confirms `world_size=4`, `n_eval=57439`, and the iter23 `item_sids.json` path; launcher output confirms beam size 20.
- Final metrics: `test_recall@10=0.05235118995804244`, `test_recall@5=0.03508069430178102`, `test_ndcg@5=0.023477038232871904`, and `test_ndcg@10=0.029022892570999326`.
- The strict target `test_recall@10 > 0.065` was missed by `0.012648810041957559`. Recall was `0.0075384320757673345` below the current best iter18 (`0.05988962203380978`) and `0.006789811800344714` below iter22. These are observed run differences, not causal estimates.
- The metrics event reports `variant="unknown_variant"` despite the correct iter23 SID path. NCCL wrapper/barrier warnings were nonfatal; the supervised job exited 0 and the fatal-marker scan found no matches.
- Decision: qualified downstream target miss after a completed Stage3 run and passing MVG; this does not establish the mechanism's causal effect or that the target is unreachable. Continue iter24; retain Stage2 metrics as descriptive only.

## Iter24 pre-Stage2 direction

- Branch from iter22's symmetric multi-positive behavior objective on pre-quantization residuals; do not retain iter23's selected-code representation, whose completed Stage3 run scored below prior variants. That comparison is observed, not causal.
- Add one mechanism: weight each behavior-loss row by the inverse square root of its future-item frequency within the current batch (`count(ids_fut)^-1/2`), normalized by the sum of weights per direction and layer. Keep the multi-positive target sets unchanged. Hypothesis: dampening repeated head-target rows may reduce their dominance in behavior supervision.
- Preserve iter22's residual representation, bidirectional/multi-positive masks, behavior temperature `0.07`, behavior weight `0.20`, reciprocal Sinkhorn epsilon, standard AdamW, cyclic curvature, Stage1, and Stage3.
- Stage2 SID statistics remain descriptive only. No Stage2 gate or HR@50 gate; only a complete Stage3 run with strict `test_recall@10 > 0.065` meets the target.
- Before Stage2, MVG must verify differentiable total/component losses, a nonzero balanced-minus-unbalanced behavior delta gradient, nonzero quantizer gradients, finite five-step layer updates, and a same-seed 200-step balanced/unbalanced effect.

### Pre-Stage2 verification — PASS

- MVG used one iter8 checkpoint and a `(640, 768)` batch with 640 active pairs, 79 rows containing duplicate targets, and 94 extra positive pairs. Total and component losses were finite and differentiable; behavior gradient norm was `0.15442677`.
- The balanced-minus-unbalanced behavior objective delta had nonzero gradient norm `0.00834073`; quantizer embedding gradient norms were `[0.04519572, 0.03263908, 0.03019207]`.
- Five-step per-layer relative updates were `[0.02442724, 0.23307569, 0.26024604]`. Same-seed 200-step balanced/unbalanced losses were `2.48894453 / 2.51212072` (`|Δ|=0.02317619`) and behavior losses were `5.45667410 / 5.53080654` (`|Δ|=0.07413244`). `MVG PASS`.
- The iter24 residual-scale file is absent, so MVG used the iter1 fallback `[0.001, 0.932889, 1.0]`. Reciprocal-epsilon/curvature schedule checks remained finite and cyclic; Stage2 may proceed.

### Stage2 result — completed

- Four-rank DDP completed all 100,000 global steps in `1,384.4s`; the run exited 0 and the fatal-marker scan found no matches. Raw SID shape was `(24587, 3)`; export produced a four-token NPY `(24587, 4)` and `SID_WIRING_PASS`.
- The exported JSON contains 24,587 items; every row parsed as four nonnegative integer IDs. Descriptive geometry at step 100,000: full Gini `0.0850`, per-layer Gini `[0.1619, 0.4072, 0.4270]`, unique 3-token codes `22,377/24,587`, collision rows `2,210`, L01 pairs `13,480`, and `H(L1|L0)=5.3947` bits. These statistics are descriptive only, not gates.
- Next: run the full Stage3 evaluation; only strict `test_recall@10 > 0.065` meets the adoption target.

### Stage3 result — completed

- Run `Sep-26-2026_13-34-49` completed all 150 epochs and the full test on four DDP ranks. `training_metrics.jsonl` confirms `world_size=4`, `n_eval=57439`, and the iter24 `item_sids.json` path; launcher output confirms beam size 20.
- Final metrics: `test_recall@10=0.05619874997823778`, `test_recall@5=0.037343964901895926`, `test_ndcg@5=0.025041357907451947`, and `test_ndcg@10=0.031103474569520558`.
- The strict target `test_recall@10 > 0.065` was missed by `0.00880125002176222`. Recall was `0.003690872055571996` below the current best iter18 (`0.05988962203380978`) and `0.0029422517801493756` below iter22, but `0.0038475600201953383` above iter23. These are observed run differences, not causal estimates.
- The metrics event reports `variant="unknown_variant"` despite the correct iter24 SID path. NCCL wrapper/process-group warnings were nonfatal; the supervised job exited 0 and the fatal-marker scan found no matches.
- Decision: qualified downstream target miss after a completed Stage3 run and passing MVG; this does not establish the mechanism's causal effect or that the target is unreachable. Continue iter25; retain Stage2 metrics as descriptive only.
