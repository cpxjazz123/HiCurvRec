# implementation_plan_iter28 (CAO-1 SREMA)

## Scope

Apply the **single** new mechanism — post-`optimizer.step()` Sinkhorn-weighted Riemannian EMA codebook update on `layer.embedding.weight` only — onto the iter18 Stage2 trainer.  Every other mechanism, hyperparameter, and protocol is inherited verbatim from iter18.

The implementation is already present in the iter28 source tree:
- `modules/quantize.py::Quantize.srema_update` (~150 lines).
- `modules/quantize.py::Quantize.cache_srema_inputs` + `clear_srema_inputs`.
- `modules/quantize.py::forward` now calls `cache_srema_inputs` after computing the assignment matrix.
- `modules/rqvae.py::RqVae.srema_update` (per-layer wrapper).
- `curvature_RQ-VAE.py::build_curvature_conditioned_adamw` excludes codebook from AdamW (custom iter28 check).
- `curvature_RQ-VAE.py` post-step block runs `model.module.srema_update(beta=ITER28_SREMA_BETA)` on rank 0 after `optimizer.step()`.

## Patch summary (verified against the live source)

### `modules/quantize.py` — new methods

```python
def cache_srema_inputs(self, assignments: Tensor, residual: Tensor) -> None:
    """Cache Sinkhorn assignments [B, K] + query residual [B, D] for SREMA."""

def has_srema_inputs(self) -> bool: ...

@torch.no_grad()
def srema_update(self, beta: float = 0.99, eps: float = 1e-9) -> dict:
    """Sinkhorn-weighted Riemannian EMA codebook update (iter28 SREMA).

    ξ_j   = Σ_i A_ij · log_{p_j_old}(z_i) / (Σ_i A_ij + eps)
    p_new = exp_{p_j_old}((1 - beta) * ξ_j)   [approx via tangent interp + exp0/log0]
    w_new = log_0(p_new)
    Returns per-layer displacement / saturation summary.
    """

def clear_srema_inputs(self) -> None: ...
```

`forward()` now calls `cache_srema_inputs(assignments, residual)` after computing the Sinkhorn assignment, gated on `self.training` so eval doesn't accumulate state.

### `modules/rqvae.py` — new methods

```python
def srema_update(self, beta: float = 0.99) -> List[dict]: ...
def clear_srema_inputs(self) -> None: ...
```

### `curvature_RQ-VAE.py` — builder + post-step + log

1. `ITER28_SREMA_BETA = 0.99`, `ITER28_SREMA_LOG_EVERY = 1000` constants.
2. `build_curvature_conditioned_adamw` excludes `layers.X.embedding.weight` (the only way to prevent double-update from AdamW + SREMA).  Replaces `check_step5_optimizer` with a custom iter28 check.
3. After `optimizer.step()` (non-codebook params only):
   ```python
   srema_summaries = (
       model.module.srema_update(beta=ITER28_SREMA_BETA)
       if rank == 0 else []
   )
   ```
4. In the 5-second log block, every `ITER28_SREMA_LOG_EVERY` global steps:
   ```python
   if srema_summaries and global_step_sync % ITER28_SREMA_LOG_EVERY == 0:
       print(f"[iter28_srema_summary step=N] median_update=... max_update=...
                frac_updated=... saturation_max=...")
   ```

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tangent-space interpolation (vs full `exp_{p_old}`) drifts outside the ball | very low | final `exp_0 → log_0` round-trip keeps the result on the ball |
| SREMA breaks gradient flow | low | write under `no_grad`; next forward's autograd starts fresh |
| `Σ_i A_ij = 0` (codeword never assigned) ⇒ `NaN` | medium | explicit `denom_safe = denom.clamp_min(eps)`; verified by MVG |
| Codebook escape from ball | low | `saturation_max` printed every 1000 steps; if `≥ 1` for any layer, stop |
| DDP broadcast skew (rank 0 writes codebook, ranks 1–3 don't) | low | DDP wraps the module; the next forward's encoder/decoder consume `embedding.weight` after the standard gradient sync |
| Per-step cost | low | one extra `pairwise distance [K, B]` for `log_{p_j_old}(z_i)`; `K=256, B=640` → 163,840 entries; negligible |

## Verification

- **preflight_cao1.py**: AST-confirm `srema_update` defined on `Quantize`, `snapshot_codebooks` / `apply_hyperbolic_trust_region` removed, post-step call site present in trainer, no new `nn.Parameter`, no new loss term, contract JSON matches CAO-1 schema.  Iter28 contract schema: `srema_beta_default=0.99`, `srema_log_every_global_steps=1000`.
- **mvg_check.py**: load iter18 best checkpoint, build one batch, run optimizer.step() + SREMA for five steps.  Verify:
  - codebook excluded from AdamW groups (custom check).
  - `median_update > 0` and `saturation_max < 1` for every layer.
  - total loss finite at every step.
  - 200-step ON/OFF counterfactual: `|L_on − L_off| > 1e-6` AND codebook max-abs-diff per layer `> 1e-7`.
- **deliberation_gate.py**: S00..S09 all `MERGE_AB` → `DELIBERATION_GATE_PASS`.

## Cleanup

- No new tests added (skill discourages new tests unless they defend a contract).  Preflight + MVG are sufficient.
- `iteration_bridge.md` already records iter27→iter28 carryover.
- Throwaway MVG script: `scripts/mvg_check.py` writes to `logs/mvg_check_iter28.log` and is committed.