#!/usr/bin/env python3
"""Issue #233 C16/C31/C34: the gradient audit that the training stream failed to record.

This is a REPLAY on the final checkpoint, not the training-time ledger. It measures, on the frozen
supervision draw, what the issue asks the ledger to contain:

* `|dL_graphlet/dθ|`, `|dL_collision/dθ|`, `|dL_RQVAE/dθ|` at the encoder, and the pairwise
  cosines between them (C16: "梯度范数与夹角");
* the budget the trainer actually applied, and the clipped/unclipped norm (C31);
* the latent displacement one step causes, and how many hard codes change (C31);
* C34: the local-vs-global budget denominator, i.e. how far a per-rank reference can sit from the
  four-rank one.

    python3 results/research/issue233/gradient_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "stage2_RQ-VAE" / "TIGER_export_matched_graphlet_rq"
sys.path.insert(0, str(TREE))
sys.path.insert(0, str(ROOT))

import train_rqvae as T  # noqa: E402
from losses import collision as C  # noqa: E402
from losses import layered_graphlet as LG  # noqa: E402
from losses import layered_wiring as LW  # noqa: E402
from model import RQVAE  # noqa: E402

OUT = ROOT / "results" / "research" / "issue233" / "gradient_audit.json"


def load(path: Path):
    model = RQVAE(T._tokenizer_config(), in_dim=768, distance=T.DISTANCE)
    model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    return model.eval()


def encoder_grad(model, value):
    params = [p for p in model.encoder.parameters() if p.requires_grad]
    grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
    return flat


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


def main() -> int:
    model = load(ROOT / "results" / "stage2_RQ-VAE" / "TIGER_export_matched_graphlet_rq" / "rqvae_best.pth")
    baseline = load(ROOT / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth")
    embeddings = torch.from_numpy(np.load(T.EMBEDDING_FILE)).float()
    branch = LW.StructureBranch(T.GRAPHLET_BANKS, instance_batch=T.GRAPHLET_SUBBATCH, seed=42)
    generator = torch.Generator()
    generator.manual_seed(42)

    draw = branch.draw()
    union = np.unique(np.concatenate([d.reshape(-1).numpy() for d in draw.values()]))
    rows = torch.from_numpy(union.astype(np.int64))

    # reconstruction reference on an ordinary batch
    recon_ids = torch.from_numpy(np.random.default_rng(0).choice(
        np.arange(len(embeddings)), 4096, replace=False)).long()
    reconstructed, parts, _, tokens = model(embeddings[recon_ids])
    recon_loss, _ = T.CRITERION(reconstructed, embeddings[recon_ids], parts)
    grad_recon = encoder_grad(model, recon_loss)

    # graphlet + collision on the frozen draw
    encoded, draw_tokens, costs, latents = model.quantize_items(embeddings, rows)
    indexed = {kind: torch.from_numpy(np.searchsorted(union, d.numpy())).long()
               for kind, d in draw.items()}
    objective, diag, costs = LW.structure_forward(
        model, embeddings, (draw_tokens, costs), draw=indexed, export_style=False, latents=latents)
    probs = [torch.softmax(-c / LW.TAU, dim=-1) for c in costs]
    capacity = C.capacity_term(probs[0], indexed["chain"].reshape(-1), len(union))
    with torch.no_grad():
        _, all_tokens, _, _ = model.quantize_items(embeddings, torch.arange(len(embeddings)))
    keys = (all_tokens[:, 0].long() * 1000 + all_tokens[:, 1].long()).cpu()
    sampled = C.candidate_pairs_from_prefix(keys, T.COLLISION_PAIRS, generator)
    pair_rows = torch.unique(torch.cat(sampled))
    _, pair_tokens, pair_costs, _ = model.quantize_items(embeddings, pair_rows)
    local = {int(v): i for i, v in enumerate(pair_rows.tolist())}
    left = torch.tensor([local[int(v)] for v in sampled[0].tolist()], dtype=torch.long)
    right = torch.tensor([local[int(v)] for v in sampled[1].tolist()], dtype=torch.long)
    duplicate = C.duplicate_term(torch.softmax(-pair_costs[2] / LW.TAU, dim=-1), left, right)
    col_value = capacity + duplicate

    g_g = encoder_grad(model, objective)
    g_c = encoder_grad(model, col_value)

    # C34: a per-rank reference differs from the global one by the spread of per-rank means
    per_rank = []
    for chunk in torch.chunk(recon_ids, 4):
        out, parts_i, _, _ = model(embeddings[chunk])
        loss_i, _ = T.CRITERION(out, embeddings[chunk], parts_i)
        per_rank.append(float(encoder_grad(model, loss_i).norm()))
    per_rank = torch.tensor(per_rank)

    # C31: one bounded step, then the latent displacement and how many hard codes move
    with torch.no_grad():
        before = model.get_indices(embeddings[rows], infer_use_sk=False).clone()
    scale_g = C.budget_scale(float(g_g.norm()), float(grad_recon.norm()), T.GRAPHLET_GRAD_BUDGET)
    scale_c = C.budget_scale(float(g_c.norm()), float(grad_recon.norm()), T.COLLISION_GRAD_BUDGET)
    combined = g_g * scale_g + g_c * scale_c
    unclipped = float(combined.norm())
    clipped = float(torch.nn.utils.clip_grad_norm_([combined], T.GRADIENT_CLIP_NORM))
    step_norm = min(unclipped, T.GRADIENT_CLIP_NORM)
    unit = combined / combined.norm().clamp_min(1e-12) * step_norm
    offset = 0
    for param in model.encoder.parameters():
        if param.grad is None:
            continue
        piece = unit[offset:offset + param.numel()].reshape(param.shape)
        offset += param.numel()
        with torch.no_grad():
            param.add_(piece, alpha=-T.LR)
    with torch.no_grad():
        after = model.get_indices(embeddings[rows], infer_use_sk=False)
        latent_before = model.encoder(embeddings[rows])
    moved = int((before != after).any(1).sum())

    audit = {
        "kind": "offline replay on the final checkpoint (not the training-time ledger)",
        "encoder_grad_norms": {"graphlet": float(g_g.norm()), "collision": float(g_c.norm()),
                               "reconstruction": float(grad_recon.norm())},
        "pairwise_cosine": {
            "graphlet_vs_reconstruction": cosine(g_g, grad_recon),
            "collision_vs_reconstruction": cosine(g_c, grad_recon),
            "graphlet_vs_collision": cosine(g_g, g_c),
        },
        "budget_scale_applied": {"graphlet": scale_g, "collision": scale_c},
        "budget_share_of_recon": {
            "graphlet": scale_g * float(g_g.norm()) / float(grad_recon.norm()),
            "collision": scale_c * float(g_c.norm()) / float(grad_recon.norm()),
        },
        "step": {"unclipped_norm": unclipped, "clip": T.GRADIENT_CLIP_NORM,
                 "applied_norm": step_norm,
                 "clipped": bool(unclipped > T.GRADIENT_CLIP_NORM)},
        "hard_codes_changed_by_one_step": {"items": int(len(rows)), "changed": moved,
                                           "share": moved / float(len(rows))},
        "c34_per_rank_reference": {
            "per_rank_norms": per_rank.tolist(),
            "spread_over_mean": float((per_rank.max() - per_rank.min()) / per_rank.mean()),
        },
        "diagnostics": {"capacity": float(capacity.detach()),
                        "duplicate": float(duplicate.detach()),
                        "candidates": int(left.numel())},
        "baseline_untouched_inference_note": "baseline checkpoint loaded only for reference; not trained",
    }
    OUT.write_text(json.dumps(audit, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
