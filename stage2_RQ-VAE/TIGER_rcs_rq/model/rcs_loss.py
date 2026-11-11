"""Radial consistency between each item and the code it was assigned, issue #188.

#187 failed because a radius that depends only on the position leaves the codebook block with no structure gradient, which the frozen budget rule
correctly refuses to calibrate. This term compares two radii instead: the item's and **its own code's**, so the code vector carries gradient and
both supervised blocks are reached. Each item is compared only with the code it received -- no edges, no triples, no negatives.

    rho_k(x)  = (2/sqrt(k)) atanh(sqrt(k) |x|_eps),   |x|_eps = sqrt(|x|^2 + finfo.eps)
    L_rcs     = mean_i [rho_{k_i}(pos_i) - rho_{k_i}(c_{k(i)})]^2 / (mean_i rho_{k_i}(pos_i))^2
"""
from __future__ import annotations

import torch

#: Switch states: TERM_OFF drops the term (the frozen baseline), GEOMETRY_OFF measures with the Euclidean norm, BOTH_ON is the arm.
TERM_OFF = False
GEOMETRY_OFF = False

BOUNDARY = 1.0 - 1e-6


def radial(x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``rho_kappa(x)``, or the Euclidean norm when the geometry is off."""
    if GEOMETRY_OFF:
        return x.norm(dim=-1)
    k = kappa.reshape(-1) if kappa.dim() > 1 else kappa
    k = k.to(x.device, x.dtype).clamp_min(1e-3)
    eps = torch.finfo(x.dtype).eps
    norm = torch.sqrt(x.pow(2).sum(-1) + eps)
    arg = (torch.sqrt(k) * norm).clamp(max=BOUNDARY)
    return (2.0 / torch.sqrt(k)) * torch.atanh(arg)


def loss(item_positions: torch.Tensor, code_vectors: torch.Tensor, kappa: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """The dimensionless radial-consistency term, with the code vectors keeping their gradient."""
    if TERM_OFF:
        return item_positions.sum() * 0.0, {"scale": 0.0}
    item_radius = radial(item_positions, kappa)
    code_radius = radial(code_vectors.detach() + (code_vectors - code_vectors.detach()), kappa)
    scale = item_radius.detach().mean().clamp_min(1e-12)
    term = ((item_radius - code_radius).pow(2) / scale.pow(2)).mean()
    return term, {"scale": float(scale), "item_radius_mean": float(item_radius.detach().mean()),
                  "code_radius_mean": float(code_radius.detach().mean())}


class CalibrationDegenerate(RuntimeError):
    status = "CALIBRATION_DEGENERATE"


def calibrate_budget(blocks: dict, base_loss: torch.Tensor, struct_loss: torch.Tensor) -> tuple[float, dict]:
    """The repository's frozen rule, ``lambda* = min over blocks ||g_base|| / ||g_struct||``, over this term."""
    flat = [p for params in blocks.values() for p in params if p.requires_grad]
    if not flat:
        raise CalibrationDegenerate("no trainable parameters in the supervised blocks")

    def block_norms(term: torch.Tensor) -> dict:
        grads = torch.autograd.grad(term, flat, retain_graph=True, allow_unused=True)
        out, cursor = {}, 0
        for name, params in blocks.items():
            collected = []
            for p in params:
                if not p.requires_grad:
                    continue
                g = grads[cursor]
                cursor += 1
                collected.append(torch.zeros_like(p) if g is None else g)
            stack = torch.cat([g.reshape(-1) for g in collected]) if collected else None
            out[name] = None if stack is None else float(stack.norm())
        return out

    base = block_norms(base_loss)
    struct = block_norms(struct_loss)
    ratios = {}
    for name, value in struct.items():
        if value is None or value <= 0.0:
            raise CalibrationDegenerate(f"block {name} has no structure gradient (norm={value})")
        ratios[name] = base[name] / value if base[name] is not None else 0.0
    return min(ratios.values()), {"ratios": ratios}


def is_pairwise_free(source: str) -> bool:
    """No edge sampling, triples, margin, InfoNCE, BCE or source-target residuals in the executable code."""
    body = source.split('"""', 2)[-1]
    body = body.split("def is_pairwise_free")[0]
    forbidden = ("triple", "margin", "InfoNCE", "infonce", "BCE", "bce_loss", "edge_index", "source_target")
    return not any(token in body for token in forbidden)


def step_terms(module, embeddings: torch.Tensor, kappa: torch.Tensor):
    """``(base_loss, radial_term, record)`` for one batch along the module's own encoder-then-rq path.

    Positions come from the encoder output; the code vectors come from the level-1 codebook with their gradient intact, which is what makes the
    codebook block visible to the budget rule.
    """
    from . import poincare_contract
    z = module.encoder(embeddings)
    quantised, quant_loss, _, tokens = module.rq(z)
    recon = module.decoder(quantised)
    base = torch.nn.functional.mse_loss(recon, embeddings) + quant_loss
    e_dim = z.shape[-1] // 2
    # the item's position: the Euclidean half as it is, the hyperbolic half through the same map the stack uses
    curvature = getattr(module.rq, "curvature", 1.0)
    pos_e = z[..., :e_dim]
    pos_h = poincare_contract.expmap0(z[..., e_dim:], curvature) if curvature > 0 else z[..., e_dim:]
    item_pos = torch.cat((pos_e, pos_h), dim=-1)
    level1 = module.rq.levels[0]
    code_vectors = level1.get_code_embs()[tokens[:, 0].detach()]
    term, record = loss(item_pos, code_vectors.to(item_pos.dtype), kappa)
    return base, term, record


_TABLE = None
from pathlib import Path as _Path

TABLE_RELPATH = _Path("results") / "research" / "issue167" / "kappa.json"


def repo_root(start=None) -> _Path:
    here = (start or _Path(__file__).resolve()).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / m).is_dir() for m in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding")):
            return candidate
    raise FileNotFoundError("no repository root above this module")


def load_table() -> dict:
    global _TABLE
    if _TABLE is None:
        import json
        _TABLE = json.loads((repo_root() / TABLE_RELPATH).read_text())
    return _TABLE


def kappa_of(item_ids: torch.Tensor, table: dict | None = None) -> torch.Tensor:
    table = load_table() if table is None else table
    entries, mean_frozen = table["kappa"], float(table["mean_frozen"])
    values = [float(entries.get(str(int(i)), mean_frozen)) for i in item_ids.detach().cpu().tolist()]
    return torch.tensor(values, dtype=torch.float64).clamp_min(1e-3)
