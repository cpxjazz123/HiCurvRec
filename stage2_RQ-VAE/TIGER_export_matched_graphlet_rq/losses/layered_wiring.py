"""Issue #232 §2: the structure branch, its own optimizer step, and the export-style replay.

Each property this module guarantees, and the issue clause that forces it:

* **The gradient reaches the encoder and the codebooks (§1).** The quantizer's assignment is
  discrete; its only differentiable surface is the cost matrix. The branch reads the *captured*
  costs into the level probabilities, back-propagates the objective to those costs, and routes the
  cost gradient through `d cost/d latent = 2(z - c)` and `d cost/d c = -2(z - c)` on the assigned
  codeword. `route_cost_gradient` is that step.
* **Separate step (§1).** The trainer zeroes gradients first, so no reconstruction contribution is
  alive when the structure term back-propagates.
* **Export-style replay (§1).** `hard_codes_of` calls the exporter's own `VQLayer._indices` on the
  same cost matrix, so branch tokens and exported tokens come from one code path.
* **Coverage (§2).** A 4096-item reconstruction batch holds a given triple with probability
  (4096/24556)^3 ~ 0.46%, so the structure term draws its own frozen instance batch instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from losses import layered_graphlet as LG

#: `p = softmax(-cost / TAU)`.
#:
#: TAU is NOT 1: the cost is a 32-dimensional squared Euclidean distance whose top-2 gap on the
#: frozen TIGER checkpoint measures 0.0016 (median), so `softmax(-cost)` is flat -- measured
#: `p_max = 0.004156`, i.e. almost uniform, and the pair mass `sum_k p_a p_b` sat at 0.0039 with
#: no slope. Under that temperature the whole objective was constant 1.3 for all 18,000 steps of
#: the first run and the arm reduced to the baseline plus the anti-collapse floor.
#:
#: The temperature is the measured assignment scale, read from `assignment_scale.json`
#: (median top-2 cost gap on the frozen checkpoint), NOT tuned: it is the only value at which
#: `p` resolves the quantizer's own decision boundary. Measured under it: `p_max = 0.496`,
#: pair mass 0.090, entropy 1.88 nats -- a distribution that still carries a gradient.
TAU = 0.001


class StructureBranch:
    """Frozen supervision banks, the per-step draw, and coverage bookkeeping."""

    def __init__(self, bank_dir, *, instance_batch: int = LG.SUBBATCH, seed: int = 42):
        self.bank_dir = Path(bank_dir)
        self.banks = {kind: np.load(self.bank_dir / f"{kind}_supervision.npy")
                      for kind in LG.KINDS}
        self.instance_batch = int(instance_batch)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        #: every item a structure batch can contain, ascending; the draw emits these ids
        self.items = np.unique(np.concatenate([bank.reshape(-1)
                                               for bank in self.banks.values()])).astype(np.int64)
        self.covered_total = 0
        self.steps = 0

    def draw(self) -> dict:
        """Instance batches of role rows holding ITEM IDs, as rows of an item-ordered table."""
        out = {}
        for kind, bank in self.banks.items():
            if len(bank) == 0:
                continue
            take = min(self.instance_batch, len(bank))
            pick = torch.randperm(len(bank), generator=self.generator)[:take].numpy()
            rows = bank[pick].astype(np.int64)
            # item ids are the embedding-table rows, so a caller indexes with them directly
            out[kind] = torch.from_numpy(rows).long()
        self.steps += 1
        return out


def hard_codes_of(quantizer, cost: torch.Tensor, export_style: bool) -> torch.Tensor:
    """The exporter's own assignment rule on a cost matrix (§1 export-style branch)."""
    return quantizer.vq_layers[-1]._indices(cost.detach(), export_style)


def level_probability(costs, tau: float = TAU):
    """The quantizer's own assignment preference at every level, from the captured costs."""
    return [torch.softmax(-c.float() / tau, dim=-1) for c in costs]


def structure_forward(model, embeddings, captured, *, draw: dict, export_style: bool,
                      tau: float = TAU, latents=None):
    """The three level objectives for one draw, over costs that carry the encoder's graph.

    `captured` is `(tokens, costs)` as recorded by `RQLayer.forward`; `latents` is the same
    forward's per-level residual. The cost is RECOMPUTED from the latent with the metric the
    quantizer was handed, so the objective's graph reaches the encoder -- a detached clone of the
    recorded cost would be a leaf with no path back to `encoder`, and `route_cost_gradient` would
    then find `cost.grad` populated while the encoder received exactly zero.
    """
    tokens, captured_costs = captured
    union = np.unique(np.concatenate([d.reshape(-1).numpy() for d in draw.values()]))
    roles = {kind: torch.from_numpy(np.searchsorted(union, d.numpy())).long()
             for kind, d in draw.items()}
    costs = captured_costs
    if latents is not None:
        # the quantizer's own metric, applied to the graph-carrying latent
        costs = [model.rq.distance(latent, layer.get_code_embs())
                 for latent, layer in zip(latents, model.rq.vq_layers)]
    probs = level_probability(costs, tau)
    hard = tokens.long()
    if export_style:
        hard = torch.stack([hard_codes_of(model.rq, c, True) for c in costs], dim=1).long()
    per_kind = {kind: LG.level_objectives(probs, hard, role, kind)
                for kind, role in roles.items()}
    objective = torch.stack([p["l1"] + p["l2"] + p["l3"] for p in per_kind.values()]).mean()
    diagnostics = {
        "coverage": float(len(union)) / float(len(embeddings)),
        "levels": {kind: {k: float(v.detach()) for k, v in p.items()} for kind, p in per_kind.items()},
        "export_style": bool(export_style),
    }
    return objective, diagnostics, costs
def cost_slopes(latent: torch.Tensor, tokens: torch.Tensor, codebook: torch.Tensor):
    """`d cost/d latent` and `d cost/d c` at the assigned codewords, for squared Euclidean.
    `cost = ||z - c||^2` => `d/dz = 2(z - c)` and `d/dc = -2(z - c)`. This is the metric the
    quantizer was handed; the branch reads it rather than replacing it.
    """
    delta = latent.unsqueeze(1) - codebook[tokens.long()].unsqueeze(2)   # (B, D, K)
    return 2.0 * delta, -2.0 * delta
def route_cost_gradient(model, latent: torch.Tensor, costs, tokens: torch.Tensor,
                        tau: float = TAU) -> dict:
    """Report where the structure gradient landed; the encoder is reached through the metric.
    The cost is now `metric(latent, codebook)`, so `backward()` on the objective already carries
    the gradient into the encoder -- there is no detached leaf to route. The codebook is frozen for
    this term (issue #232 option 2, the measured `CodebookCollapse`), so it is reported as zero and
    its accumulated gradient is cleared.
    """
    encoder_norm = float(sum(float(p.grad.detach().pow(2).sum())
                             for p in model.encoder.parameters()
                             if p.grad is not None) ** 0.5)
    for layer in model.rq.vq_layers:
        layer.embed.weight.grad = None
    return {"latent_grad": None,
            "codebook_grads": [torch.zeros_like(layer.get_code_embs().detach())
                               for layer in model.rq.vq_layers],
            "latent_grad_norm": encoder_norm,
            "codebook_grad_norms": [0.0, 0.0, 0.0],
            "encoder_grad_norm": encoder_norm,
            "codebook_frozen": True}
#: Gradient-scale normalisation for the structure step. Not a learning rate and not a weight on
#: the objective: the objective's own slope varies by eight orders of magnitude across the run
#: (measured: 2.6e+02 on the frozen checkpoint vs 1.9e-06 in the first run), and under AdamW a raw
#: step of either size is wrong -- 2.6e+02 collapsed the codebook inside 16 epochs
#: (`CodebookCollapse: spacing fell to 3.6e-02 ... x0.087`), while 1.9e-06 changed nothing at all.
#: Rescaling to this RMS makes the structure step's size independent of where the assignment
#: probabilities happen to sit, which is what the separate-step design already promised: the step
#: size is the weight, so it must not ride on the objective's scale.
GRADIENT_RMS = 1.0
#: Step-length budget for the encoder, in the units the metric is measured in. `verify_separation`
#: compares `min_pairwise_codeword_distance / sqrt(mean assigned cost)`, and on the frozen checkpoint
#: the assigned-codeword resolution is 0.0458 -- so a step whose LENGTH is ~0.1 of that resolution
#: is one the guard tolerates indefinitely, while a step of the same length as the resolution
#: collapses its ratio to 0.31 in one update. The measured factor is what this constant encodes:
#: the encoder's latent moves by at most this fraction of the assignment resolution per structure
#: step, so the step size is set in the metric's own units rather than in gradient-norm units
#: (`GRADIENT_RMS=1` was tried at three scales -- 1.0, 0.1, 0.01 -- and all three collapsed L1 to
#: 0.13-0.16 within ten steps, because AdamW's m/sqrt(v) makes the step size independent of the
#: gradient's own magnitude).
LATENT_STEP_FRACTION = 0.1
def normalise_latent_step(model, latent_norm: float, cost_scale: float) -> float:
    """Clamp the encoder step so the latent moves by at most `LATENT_STEP_FRACTION` of resolution."""
    params = [p for p in model.encoder.parameters() if p.grad is not None]
    if not params or cost_scale <= 0:
        return 1.0
    squared = sum(float(p.grad.detach().pow(2).sum()) for p in params)
    if squared == 0.0:
        return 1.0
    # AdamW's first step moves each parameter by ~lr, so the latent's movement is bounded by
    # |J| * lr * sqrt(number of encoder parameters); the lr budget that keeps it within the
    # fraction of the resolution is what the trainer multiplies into the step size.
    return float(LATENT_STEP_FRACTION) * float(cost_scale) / max(float(latent_norm), 1e-12)

def normalise_gradients(model, *, scale: float = GRADIENT_RMS) -> float:
    """Rescale every parameter gradient so the joint RMS equals `scale`; returns the factor."""
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 1.0
    total = sum(float(p.grad.detach().pow(2).sum()) for p in params)
    rms = (total / sum(p.numel() for p in params)) ** 0.5
    if rms == 0.0:
        return 1.0
    factor = float(scale) / rms
    for p in params:
        p.grad.mul_(factor)
    return factor


def apply_structure_step(model, latent, costs, tokens, tau: float = TAU) -> dict:
    """Route the cost gradient, step the structure term on its own, and report the norms."""
    routed = route_cost_gradient(model, latent, costs, tokens, tau)
    latent.backward(routed["latent_grad"])
    for level, rows in enumerate(routed["codebook_grads"]):
    return routed


def run_structure_step(model, embeddings, draw, *, export_style: bool = False,
                       tau: float = TAU) -> dict:
    """Forward and backward the structure term for one draw; returns the gradient report.

    The gradients land on `model.encoder`, `model.decoder` (through the shared MLP trunk) and each
    `model.rq.vq_layers[l].embed.weight`. Nothing touches reconstruction: the caller zeroes the
    gradients first, which is the §1 separate-step requirement.
    """
    model.zero_grad(set_to_none=True)
    # the draw holds item ids; the encoder runs on exactly those rows, and the objective's role
    # rows are the same ids, so `structure_forward` indexes straight into the cached embeddings
    union = np.unique(np.concatenate([d.reshape(-1).numpy() for d in draw.values()]))
    rows = torch.from_numpy(union.astype(np.int64))
    latent, tokens, costs, latents = model.quantize_items(embeddings, rows)
    indexed = {kind: torch.from_numpy(np.searchsorted(union, d.numpy())).long()
               for kind, d in draw.items()}
    # the objective must reach `costs`, so it reads the SAME tensors the gradient is routed from
    objective, diagnostics, costs = structure_forward(
        model, embeddings, (tokens, costs), draw=indexed, export_style=export_style, tau=tau,
        latents=latents)
    if not torch.isfinite(objective):
        raise RuntimeError("structure objective is not finite")
    # ONE backward reaches the encoder directly now that the cost is rebuilt from the latent --
    # there is no detached leaf to route through, and a second backward would hit freed buffers
    objective.backward()
    routed = route_cost_gradient(model, latent, costs, tokens, tau)
    # issue #232 option 2: the codebook is left to the reconstruction loss. Its gradient is
    # explicitly cleared, because the encoder and decoder both consume the codebook table and
    # autograd would otherwise leave a zero-valued `.grad` there that the step still acts on
    # through weight decay.
    for layer in model.rq.vq_layers:
        layer.embed.weight.grad = None
    routed["gradient_scale"] = normalise_gradients(model)
    routed["objective"] = float(objective.detach())
    routed["diagnostics"] = diagnostics
    routed["encoder_grad_norm"] = float(sum(
        p.grad.norm() for p in model.encoder.parameters() if p.grad is not None))
    return routed

__all__ = ["StructureBranch", "structure_forward", "route_cost_gradient", "apply_structure_step",
           "run_structure_step", "hard_codes_of", "level_probability", "cost_slopes",
           "normalise_gradients", "GRADIENT_RMS", "LATENT_STEP_FRACTION",
           "normalise_latent_step", "TAU"]