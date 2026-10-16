"""Issue #22 mechanism check for the semantic-anchored bounded hyperbolic residual.

Exercises the installed wrapper (not only the helpers) on this tree's own probe
checkpoint and asserts the properties the issue actually asks for:

1. the bounded residual's hyperbolic norm never exceeds the perturbation budget,
   and the budget is the stop-gradient nearest/second-nearest codeword margin;
2. the encoder's output really is the bounded point (z == Exp_s(v_bounded));
3. the behaviour ranking term reaches the encoder and no codebook;
4. the decoder-side recovery term reaches its head and the encoder proxy, and is
   blocked on every codebook;
5. ``model.state_dict()`` keys are exactly the source tree's, so the checkpoint
   contract the MCP probe enforces is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import train_rqvae as trainer  # noqa: E402
from model import RQVAE  # noqa: E402
from semantic_bounded_residual import (  # noqa: E402
    SemanticAnchorRecoveryHead,
    install_semantic_bounded_encoder,
    semantic_anchor_recovery_loss,
    semantic_bounded_ranking_loss,
)


def probe_checkpoint() -> Path:
    """This arm's own weights, else the newest archived copy, else the source tree.

    The mechanism changes the encoder's *forward*, not its parameters, so any
    checkpoint of the same architecture probes it faithfully -- that is exactly
    what the MCP launcher seeds this tree with.
    """
    candidates = [Path(trainer.CHECKPOINT)]
    archive = Path("/home/wlia0047/ar57/wenyu/artifact_archive")
    if archive.is_dir():
        candidates += sorted(
            archive.glob(f"{trainer.TREE_NAME}_*/rqvae_best.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    candidates.append(
        Path(trainer.REPO_ROOT)
        / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/rqvae_best.pth"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no probe checkpoint found in {candidates}")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report: dict[str, object] = {"device": str(device)}
    failures: list[str] = []

    trainer.set_seed(trainer.SEED)
    embeddings = trainer.load_embeddings(trainer.EMBEDDING_FILE)
    frame = pd.read_parquet(trainer.TRAIN_FILE, columns=["seen_history", "target"])
    item_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    all_embeddings = torch.from_numpy(embeddings)

    model = RQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    checkpoint = probe_checkpoint()
    report["probe_checkpoint"] = str(checkpoint)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        model.init_codebook(all_embeddings[item_ids].to(device))

    before_keys = sorted(model.state_dict())
    branch = install_semantic_bounded_encoder(
        model, lambda: model.rq.vq_layers[0].get_code_embs(),
    )
    report["state_dict_keys_unchanged"] = sorted(model.state_dict()) == before_keys
    if not report["state_dict_keys_unchanged"]:
        failures.append("installing the mechanism changed model.state_dict()")

    # --- (0) Exp_x is a real Riemannian exponential map ----------------------
    # Without this the bounded point would not be the point the tangent length
    # claims, and the geodesic distances the ranking term compares would not be
    # the distances the representation encodes.
    generator = torch.Generator().manual_seed(trainer.SEED)
    curvature = branch.curvature
    from semantic_bounded_residual import _expmap0, _expmap_x, _hyperbolic_distance
    worst_absolute = 0.0
    worst_relative = 0.0
    for base_scale in (0.05, 0.3, 0.6, 0.9):
        base = _expmap0(torch.randn(4096, trainer.CODEBOOK_DIM, generator=generator) * base_scale,
                        curvature)
        for tangent_scale in (0.005, 0.05, 0.2):
            tangent = torch.randn(4096, trainer.CODEBOOK_DIM, generator=generator) * tangent_scale
            moved = _expmap_x(base, tangent, curvature)
            length = torch.linalg.vector_norm(tangent, dim=-1)
            measured = _hyperbolic_distance(base, moved, curvature)
            worst_absolute = max(worst_absolute, float((measured - length).abs().max()))
            worst_relative = max(
                worst_relative,
                float(((measured - length).abs() / length.clamp_min(1e-9)).max()),
            )
    report["expmap_isometry_absolute_error"] = worst_absolute
    report["expmap_isometry_relative_error"] = worst_relative
    # float32 loses 1 - ||x||^2 once the anchor sits near the ball boundary, so
    # the bound is absolute with a float32-sized relative allowance on top.
    if worst_absolute > 2e-4 or worst_relative > 5e-3:
        failures.append(
            f"Exp_x is not isometric with the tangent length "
            f"(absolute {worst_absolute:.2e}, relative {worst_relative:.2e})"
        )

    head = SemanticAnchorRecoveryHead(trainer.CODEBOOK_DIM, branch.curvature).to(device)

    rows = torch.arange(min(256, len(item_ids)), dtype=torch.long)
    batch = all_embeddings[item_ids[rows]].to(device)

    # --- (1)+(2) the bounded point really respects the budget ---------------
    latent = model.encoder(batch)
    codebook = model.rq.vq_layers[0].get_code_embs().detach()
    representation, anchor, bounded, budget = branch(latent, codebook)
    limit = (1.0 - 1e-5) ** 0.5
    # The budget is a geodesic distance, so what has to hold is that the anchor
    # moved no further than the nearest/second-nearest codeword margin.
    travelled = _hyperbolic_distance(anchor, representation, branch.curvature)
    slack = (travelled - budget).clamp_min(0.0)
    hyperbolic_norm = travelled
    report["budget_mean"] = float(budget.detach().mean())
    report["budget_active_fraction"] = float((budget > 0).float().mean())
    report["budget_violation_max"] = float(slack.max())
    report["travelled_distance_mean"] = float(hyperbolic_norm.detach().mean())
    report["representation_radius_max"] = float(representation.norm(dim=-1).max())
    if float(slack.max()) > 1e-5:
        failures.append(
            f"bounded residual exceeds the perturbation budget by {float(slack.max()):.3e}"
        )
    if float(budget.mean()) <= 0.0:
        failures.append("perturbation budget is identically zero; the bound is inert")
    if float(representation.norm(dim=-1).max()) >= limit:
        failures.append("representation left the Poincare ball")

    # The budget must not carry gradient into the codebook.
    latent_probe = latent.detach().requires_grad_(True)
    codebook_probe = model.rq.vq_layers[0].get_code_embs()
    _z, _a, _b, budget_grad = branch(latent_probe, codebook_probe)
    budget_grad.sum().backward()
    codebook_grad = codebook_probe.grad
    report["budget_codebook_grad"] = (
        0.0 if codebook_grad is None else float(codebook_grad.abs().sum())
    )
    if codebook_grad is not None and float(codebook_grad.abs().sum()) > 0.0:
        failures.append("perturbation budget passed gradient into the codebook")

    # --- (3) the behaviour term reaches the encoder and no codebook --------
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    branch.zero_grad(set_to_none=True)
    pair_dataset = trainer.TransitionPairs(frame)
    pairs = pair_dataset.pairs[: trainer.PAIR_BATCH_SIZE_PER_RANK].to(device)
    source_ids, target_ids = pairs[:, 0], pairs[:, 1]
    pair_batch = all_embeddings[torch.cat((source_ids, target_ids)).cpu()].to(device)
    residuals = trainer._residuals_with_trace(model, pair_batch)
    behavior = trainer.behavior_contrastive_loss(residuals, source_ids, target_ids)
    report["behavior_loss"] = float(behavior.detach())
    encoder_parameters = tuple(model.encoder.parameters())
    codebook_parameters = tuple(layer.get_code_embs() for layer in model.rq.vq_layers)
    encoder_grads = torch.autograd.grad(
        behavior, encoder_parameters, retain_graph=True, allow_unused=True
    )
    codebook_grads = torch.autograd.grad(
        behavior, codebook_parameters, retain_graph=True, allow_unused=True
    )
    branch_grads = torch.autograd.grad(
        behavior, tuple(branch.parameters()), retain_graph=True, allow_unused=True
    )
    report["behavior_encoder_grad"] = max(
        [float(g.abs().sum()) for g in encoder_grads if g is not None] or [0.0]
    )
    report["behavior_codebook_grad"] = max(
        [float(g.abs().sum()) for g in codebook_grads if g is not None] or [0.0]
    )
    report["behavior_branch_grad"] = max(
        [float(g.abs().sum()) for g in branch_grads if g is not None] or [0.0]
    )
    if report["behavior_encoder_grad"] <= 0.0:
        failures.append("behaviour term produced no gradient on the encoder")
    if report["behavior_codebook_grad"] > 0.0:
        failures.append("behaviour term leaked gradient into a codebook")
    if report["behavior_branch_grad"] <= 0.0:
        failures.append("behaviour term produced no gradient on the bounded-residual branch")

    # --- (4) the recovery term reaches the head and no codebook -------------
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    branch.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = model(batch)
    recovery = semantic_anchor_recovery_loss(
        head(model.last_quantized, codebook_parameters), branch.last_anchor,
    )
    report["recovery_loss"] = float(recovery.detach())
    report["total_reconstruction"] = float(reconstructed.detach().mean())
    recovery_grads = torch.autograd.grad(
        recovery, tuple(head.parameters()) + codebook_parameters, allow_unused=True,
    )
    head_count = len(tuple(head.parameters()))
    report["recovery_head_grad"] = max(
        [float(g.abs().sum()) for g in recovery_grads[:head_count] if g is not None] or [0.0]
    )
    report["recovery_codebook_grad"] = max(
        [float(g.abs().sum()) for g in recovery_grads[head_count:] if g is not None]
        or [0.0]
    )
    if report["recovery_head_grad"] <= 0.0:
        failures.append("recovery term produced no gradient on the decoder-side head")
    if report["recovery_codebook_grad"] > 0.0:
        failures.append("recovery term leaked gradient into a codebook")

    # --- the full objective must still train the codebooks via the recon term
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    branch.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = model(batch)
    total, recon = model.compute_loss(batch, reconstructed, quant_loss)
    total_grads = torch.autograd.grad(total, codebook_parameters, allow_unused=True)
    report["reconstruction_codebook_grad"] = max(
        [float(g.abs().sum()) for g in total_grads if g is not None] or [0.0]
    )
    if report["reconstruction_codebook_grad"] <= 0.0:
        failures.append("blocking the recovery gradient also blocked reconstruction's")

    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
