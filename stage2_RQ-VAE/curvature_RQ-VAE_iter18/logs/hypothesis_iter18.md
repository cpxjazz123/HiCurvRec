# hypothesis_iter18 (Agent C)

## Mechanism

Change only Stage2 AdamW second-moment decay by quantizer layer. Preserve the existing cyclic, learnable `c_l(t)` schedule and all loss, assignment, Stage1, and Stage3 behavior. At optimizer initialization, set the curriculum step to 0 and read the actual initial curvature `c_l(0)` for each layer:

`u_l = clamp(2 * log(c_l(0) / C_CYCLIC_MIN) / log(C_CYCLIC_MAX / C_CYCLIC_MIN), 0, 1)`

`β₂,l = 0.999 - 0.009 * u_l`.

Use `(β₁, β₂) = (0.9, β₂,l)` in each quantizer layer's AdamW parameter group for the entire run. Keep non-quantizer parameter groups at `(0.9, 0.999)`. Keep learning rate `1e-3`, weight decay `1e-4`, and `eps=1e-8`. Do not use iter17's live `1/(c_l+1e-3)` learning-rate multiplier. This is a fixed initial-curvature-conditioned optimizer-memory hypothesis, not Riemannian Adam.

## Direct effects (must pass before Stage2 GPU training)

- **DE-1 — mapping and group wiring:** Log each layer's `c_l(0)`, `u_l`, and `β₂,l`. Assert all finite; `0 ≤ u_l ≤ 1`; `0.990 ≤ β₂,l ≤ 0.999`; and the mapping is monotone (if `c_i(0) > c_j(0)`, then `β₂,i ≤ β₂,j`). Assert quantizer groups use their layer-specific `β₂,l`, non-quantizer parameters use `0.999`, every trainable parameter occurs exactly once, and all group learning rates remain `1e-3` before stepping.
- **DE-2 — required gradient path:** On one checkpoint and one batch, require `total_loss.requires_grad` and `total_loss.grad_fn is not None`. For every applicable mechanism loss item, call `torch.autograd.grad(loss_item, model.parameters(), retain_graph=True, allow_unused=True)` and require at least one finite nonzero parameter gradient. Run `loss.backward()` and confirm the required model parameters have finite gradients.
- **DE-3 — nonzero per-layer updates:** From the same checkpoint and batch, run five ON optimizer steps; every quantizer layer must have finite, nonzero relative parameter update greater than `1e-7`. Check the encoder/other group update separately; do not let one layer's update satisfy another layer's check.
- **DE-4 — mechanism is active:** Same-seed, same-batch, 200-step ON/OFF comparison. ON uses `β₂,l`; OFF uses `0.999` for every group. Require finite losses and `abs(L_on - L_off) > 1e-6`. Record the per-layer second-moment states or update norms to make the change observable.

If any direct check fails, classify and repair the implementation/checker; do not launch Stage2 training until it passes. No Stage2 quality statistic is an early-stop or adoption gate.

## Proxy observations (descriptive only)

Record per-layer SID usage, unique pairs, collision counts, entropy, curvature, training loss, and checkpoint integrity regardless of values. Do not infer downstream success from SID metrics and do not use `hitrate@50` as a gate.

## Comparator and outcome

- iter17 used the same 100k-step cyclic-curvature curriculum with live per-layer `1/(c_l+1e-3)` AdamW LR scaling. Its complete Stage3 score was `test_recall@10=0.059071362662999005` (`n_eval=57439`), below iter11's `0.05976775361688052` and the strict target `>0.065`.
- Iter18's sole adoption criterion is a completed, correctly wired Stage3 run with final `test_recall@10 > 0.065`. Otherwise record the result and continue with a distinct mechanism; do not claim that a Stage2 proxy or this single miss proves causal failure or target impossibility.

## Failure interpretation

- DE-1 failure: `IMPLEMENTATION_FAIL`; correct parameter grouping or mapping.
- DE-2 failure: `GRADIENT_PATH_FAIL`; do not train until each required loss path has a finite nonzero gradient.
- DE-3 failure: `UPDATE_FAIL`; inspect optimizer step/group coverage and rerun the direct checks.
- DE-4 failure: `ACTIVATION_FAIL`; the mechanism has no observable effect at the registered scale, so do not spend the Stage2/Stage3 run on it.
- Direct checks pass but Stage3 misses the target: `DOWNSTREAM_TARGET_MISS`; this does not establish a causal mechanism failure.
