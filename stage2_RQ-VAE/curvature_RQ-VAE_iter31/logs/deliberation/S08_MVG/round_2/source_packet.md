# Iter31 S08 MVG — Round 2 Gradient-Repair Review

## Stage identity and Judge authorization

- Stage: `S08_MVG`, round 2; Iter31 HRA-STEP6-1, parent Iter29.
- Round-1 Judge: `logs/deliberation/S08_MVG/round_1/judge.md`, `VERDICT=REJECT_BOTH`; it authorizes only a checker repair in `scripts/mvg_check.py::_check_gradients` to add existing `output.behavior_loss` as its own component.
- No MVG runtime has been executed. This round reviews the exact authorized repair and decides whether the single MVG may run. It does not authorize a run until Judge C explicitly says so.
- S07 passed as a static preflight only; no runtime activation/domain/gradient evidence was established.

## Authorized repair now applied

Only `scripts/mvg_check.py::_check_gradients` changed for this repair:

- Added `output.behavior_loss` as a distinct component in the existing component-gradient loop.
- Require it to be scalar, require differentiability (`requires_grad` and `grad_fn`), require finite value and finite gradients, obtain component gradients by `torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True)`, and require/report at least one finite nonzero gradient among the trainable model parameters returned from that component-specific autograd path, using existing `GRAD_EPSILON`.
- Existing reconstruction/quantizer component checks, total `loss.backward()`, model/codebook checks, and result report remain.
- `grad_check.py` remains an unchanged delegating entry. No model/loss/HRA equation/weight, checkpoint/batch, optimizer operation, contracts, Stage3 wiring, or training source changed.

Round-1 Stage3 map-derived `RQVAE_VARIANT` discrepancy remains a mandatory pre-S11 correction gate, untouched by this packet.

## Required independent review

Read this packet, full S08 round-1 Judge and A/B reports, S06 plan/root §6, S07 final Judge, current Iter31 `mvg_check.py::_check_gradients`, `RqVaeComputedLosses`/forward/behavior loss, quantizer residual path, helpers, and fixed-curvature source. Verify the exact repair stayed within authorization and proves a component-specific gradient for `behavior_loss` (not just total gradients), without assuming a parameter prefix absent source evidence. Check the one-checkpoint/batch/domain/invariance/no-step/no-save obligations and existing sources unchanged. Candidate A and B independently report; neither may run commands/MVG/GPU/tests or edit.

## Frozen one-run boundary if Judge authorizes

Exactly one CUDA MVG invocation is allowed only after S08 round-2 Judge approval. Use no other checkpoint, dataset split, batch, sweep, retry, train, save, or optimizer step.

- Checkpoint: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`; pinned SHA-256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`. Main verified this hash before S08 round 1; independently reverify exact hash immediately before any Judge-authorized run and record it.
- One fixed batch: Stage1 `sentence_t5.npy`, Stage1 `item_ids.json`, Stage0 `train.parquet`; seed 42; first 640 eligible train-transition records (`active_indices[:BATCH_SIZE]`). Verify exact inputs exist before the approved invocation; do not select another batch if it fails.
- Command/cwd if authorized: `python scripts/mvg_check.py` from `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`.
- Required output: complete stdout/stderr/exit status, actual device/batch shape and input/checkpoint identity; same-quantized-tensor HRA-vs-Euclidean direct effect with no invented minimum threshold; intermediate/final finite/domain and actual projection/log-clamp/denominator incidence; exact fixed-curvature buffer/get_c invariance and optimizer exclusion with no step; component-wise reconstruction/quantizer/behavior-loss gradients including scalar/graph/finiteness/nonzero; total loss graph, actual backward and finite nonzero encoder/codebook gradients.

Round-2 Judge must either (a) reject/authorize only a further narrow verification repair and require another fresh round, or (b) authorize this exact one Main run. No S09, Stage2, Stage3, GPU training, or downstream gate is authorized by the repair itself.