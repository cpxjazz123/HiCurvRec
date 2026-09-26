# implementation_plan_iter27 (CAO-1)

## Scope

Apply the **single** new mechanism — a post-`optimizer.step()` hyperbolic
codebook trust-region projection on `layer.embedding.weight` only — onto
the iter18 Stage2 trainer, keeping every other mechanism, hyperparameter,
and protocol identical.

The implementation is already present in the iter27 source tree
(`modules/quantize.py::apply_hyperbolic_trust_region`,
`modules/rqvae.py::snapshot_codebooks` / `apply_hyperbolic_trust_region`,
`curvature_RQ-VAE.py` rank-0 post-step block).  The orchestrator's
task is **not** to re-implement, but to validate that the patch
satisfies the contract, write the MVG / preflight logs, and launch
Stage2 once.

## Patch summary (independent A/B/Judge verification)

### `modules/quantize.py` — new method `Quantize.apply_hyperbolic_trust_region`

```python
def apply_hyperbolic_trust_region(
    self,
    prev_codebook: Tensor,
    trust_radius_fraction: float = 0.5,
    trust_radius_min: float = 1e-3,
    eps: float = 1e-9,
) -> dict:
    new_codebook = self.embedding.weight
    if prev_codebook.shape != new_codebook.shape: raise ValueError(...)
    if not torch.isfinite(prev_codebook).all().item() or not torch.isfinite(new_codebook).all().item():
        raise RuntimeError(...)

    curvature = self.get_c().to(dtype=new_codebook.dtype)
    prev_ball = _expmap0_t(prev_codebook.unsqueeze(0), curvature).squeeze(0)
    new_ball  = _expmap0_t(new_codebook.unsqueeze(0), curvature).squeeze(0)
    displacement = _poincare_distance_t(
        prev_ball.unsqueeze(0), new_ball.unsqueeze(0), curvature
    ).squeeze(0).squeeze(-1)

    pairwise = _poincare_distance_t(
        new_ball.unsqueeze(0), new_ball.unsqueeze(1), curvature
    ).squeeze(-1)
    pairwise.fill_diagonal_(float("inf"))
    nearest = pairwise.min(dim=1).values
    finite_nearest = nearest[torch.isfinite(nearest)]
    D_l = max(float(trust_radius_min), float(finite_nearest.median().item())) \
        if finite_nearest.numel() > 0 else float(trust_radius_min)
    tau_l = float(trust_radius_fraction) * D_l

    finite_mask = torch.isfinite(displacement)
    if not finite_mask.all().item():
        displacement = torch.where(finite_mask, displacement, torch.zeros_like(displacement))
    ratio = displacement / (tau_l + eps)
    s_k = torch.clamp(1.0 / (ratio + eps), max=1.0)

    prev_tangent = _logmap0_t(prev_ball, curvature)
    new_tangent  = _logmap0_t(new_ball, curvature)
    scaled_tangent = prev_tangent + s_k.unsqueeze(-1) * (new_tangent - prev_tangent)
    corrected_ball = _expmap0_t(scaled_tangent.unsqueeze(0), curvature).squeeze(0)
    corrected = _logmap0_t(corrected_ball, curvature)
    if not torch.isfinite(corrected).all().item():
        raise RuntimeError("iter27 trust-region: non-finite corrected tangent")
    with torch.no_grad():
        self.embedding.weight.copy_(corrected.to(dtype=new_codebook.dtype))

    clipped_mask = displacement > tau_l
    n_clipped = int(clipped_mask.sum().item())
    clip_frac = n_clipped / max(1, displacement.numel())
    delta_median = float(displacement.median().item()) if displacement.numel() > 0 else 0.0
    delta_over_tau = (displacement / (tau_l + eps)).median().item() if displacement.numel() > 0 else 0.0
    return {
        "layer_index": -1,
        "delta_median": delta_median,
        "tau_l": tau_l,
        "D_l": D_l,
        "clip_frac": float(clip_frac),
        "n_clipped": n_clipped,
        "delta_over_tau_median": float(delta_over_tau),
        "displacement_max": float(displacement.max().item()) if displacement.numel() > 0 else 0.0,
    }
```

Notes:

- The `prev_tangent + s_k * (new_tangent - prev_tangent)` interpolation
  is a **linear interpolation in tangent space** between two points
  that already live on the Poincaré ball.  After re-expmap, the result
  is on the ball.  This is geometrically equivalent to walking along
  the geodesic from `p^old` toward `p*` by fraction `s_k`.
- `_logmap0_t` / `_expmap0_t` already clip the radial coordinate so
  `‖p‖ < 1/√c`; the projection can never push a codeword outside the
  ball.
- The write happens under `torch.no_grad()`; this is correct because
  the projection is a deterministic function of the post-`step()`
  codebook and the **next** forward's autograd graph starts from the
  projected `embedding.weight`.

### `modules/rqvae.py` — `snapshot_codebooks` and `apply_hyperbolic_trust_region`

```python
def snapshot_codebooks(self) -> List[torch.Tensor]:
    return [layer.embedding.weight.detach().clone() for layer in self.layers]

def apply_hyperbolic_trust_region(
    self,
    prev_codebooks: List[torch.Tensor],
    trust_radius_fraction: float = 0.5,
    trust_radius_min: float = 1e-3,
) -> List[dict]:
    summaries = []
    for layer_index, (layer, prev_codebook) in enumerate(zip(self.layers, prev_codebooks)):
        summary = layer.apply_hyperbolic_trust_region(
            prev_codebook,
            trust_radius_fraction=trust_radius_fraction,
            trust_radius_min=trust_radius_min,
        )
        summary["layer_index"] = layer_index
        summaries.append(summary)
    return summaries
```

### `curvature_RQ-VAE.py` — post-step wiring

```python
ITER27_TRUST_RADIUS_FRAC = 0.5
ITER27_TRUST_RADIUS_MIN = 1e-3
ITER27_TRUST_LOG_EVERY = 1000

# Inside the train loop, AFTER optimizer.step() and BEFORE set_curriculum_step(...):
iter27_codebook_snapshot = (
    model.module.snapshot_codebooks() if rank == 0 else None
)
optimizer.step()
if rank == 0 and iter27_codebook_snapshot is not None:
    iter27_summaries = model.module.apply_hyperbolic_trust_region(
        iter27_codebook_snapshot,
        trust_radius_fraction=ITER27_TRUST_RADIUS_FRAC,
        trust_radius_min=ITER27_TRUST_RADIUS_MIN,
    )
else:
    iter27_summaries = []
if (
    rank == 0
    and iter27_summaries
    and global_step_sync % ITER27_TRUST_LOG_EVERY == 0
):
    clip_fracs = [summary["clip_frac"] for summary in iter27_summaries]
    deltas = [summary["delta_median"] for summary in iter27_summaries]
    taus = [summary["tau_l"] for summary in iter27_summaries]
    ratio = [summary["delta_over_tau_median"] for summary in iter27_summaries]
    print(
        f"  [iter27_trust_region_summary step={global_step_sync}] "
        f"clip_frac=[{','.join(f'{value:.3f}' for value in clip_fracs)}] "
        f"delta=[{','.join(f'{value:.4f}' for value in deltas)}] "
        f"tau=[{','.join(f'{value:.4f}' for value in taus)}] "
        f"delta_over_tau=[{','.join(f'{value:.3f}' for value in ratio)}]",
        flush=True,
    )
```

The snapshot is taken **before** `optimizer.step()` (the variable
assignment is at the top of the step block) and the projection runs
**after** `optimizer.step()`.  Rank 0 is responsible; all ranks see
the same `embedding.weight` because DDP re-broadcasts on the next
forward.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tangent-space interpolation drifts outside the ball | very low | `_expmap0_t` projects inside the ball; result re-checked |
| Projection accidentally affects the gradient path | medium | write under `no_grad`; next forward starts from the projected tensor so gradients flow normally |
| `D_l = 0` collapses the projection | medium | explicit `max(D_l, 1e-3)` floor |
| DDP broadcast skew (rank 0 wrote, ranks 1–3 did not) | low | the next forward's encoder/decoder consume `embedding.weight`; DDP wraps the module so all ranks receive the updated parameter via the standard gradient broadcast path.  We additionally rely on `dist.barrier()` already present at checkpoint boundaries. |
| Cost (pairwise distance for `[K=256, embed=32]`) | low | K=256 → 65,536 pair entries per layer; one `_poincare_distance_t` call per layer per step; 3 layers × 100k steps × 4 ranks adds negligible GPU time |

## Verification

- **preflight_cao1.py**: AST-confirm `apply_hyperbolic_trust_region`
  defined on `Quantize`, `snapshot_codebooks` defined on `RqVae`, post-
  step call site present in trainer, no new `nn.Parameter`, no new loss
  term, contract JSON matches CAO-1 schema.
- **mvg_check.py**: load iter18 best checkpoint, build one batch,
  snapshot codebooks, run `optimizer.step()` + projection for five
  steps, verify:
  - `clip_frac_l ∈ (0, 1)` for at least one layer;
  - `delta / tau` finite and `> 0` for at least one codeword;
  - total loss finite at every step;
  - `delta` is nonzero (projection actually fired);
  - encoder/decoder gradients remain finite and nonzero.

## Cleanup

- No new tests are added (skill §Verify discourages new tests unless
  they defend a contract); the preflight + MVG are sufficient.
- Throwaway MVG script: `scripts/mvg_check.py` writes to
  `logs/mvg_check_iter27.log` and is committed alongside the iter27
  source.
- `iteration_bridge.md` is updated post-hoc to record iter27's actual
  outcome (added after Stage2/Stage3 complete).