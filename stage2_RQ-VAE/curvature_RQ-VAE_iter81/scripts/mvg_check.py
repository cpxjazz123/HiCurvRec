"""MVG for iter81 quantized-representation transition ranking."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
import train_rqvae as training
from model import RQVAE
from model.model import behaviour_ranking_loss

PARENT_CHECKPOINT = (
    experiment.REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MVG FAIL: {message}")


def require_nonzero_finite_gradient(
    loss: torch.Tensor, parameters, label: str
) -> None:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    require(
        any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients
        ),
        f"{label} produced no finite nonzero gradient",
    )


def checkpoint_round_trip(model: RQVAE, device: torch.device) -> None:
    path = Path("/tmp/mvg_iter81_quantized_behavior.pth")
    try:
        torch.save(model.state_dict(), path)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state, strict=True)
    finally:
        path.unlink(missing_ok=True)
    require(
        all(torch.isfinite(parameter).all() for parameter in model.parameters()),
        "checkpoint reload contains non-finite weights",
    )


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(training.SEED)
    embeddings = np.asarray(
        np.load(experiment.EMBEDDING_FILE, allow_pickle=False), dtype=np.float32
    )
    checkpoint = torch.load(PARENT_CHECKPOINT, map_location=device)
    require(
        checkpoint.get("global_step") == 72_000,
        "parent checkpoint is not the completed 72k model",
    )
    require(
        tuple(float(value) for value in checkpoint.get("curvatures", ()))
        == (1.0, 1.0, 1.0)
        and tuple(float(value) for value in checkpoint.get("working_radii", ()))
        == (0.2, 0.2, 0.2),
        "parent checkpoint does not have fixed c=1 and three s=0.2 pins",
    )

    model = RQVAE(
        training._tokenizer_config(), in_dim=int(embeddings.shape[1])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    require(
        tuple(float(value) for value in model.get_curvatures().tolist())
        == (1.0, 1.0, 1.0)
        and model.rq.get_working_radii() == (0.2, 0.2, 0.2),
        "loaded parent model violates the fixed-curvature/pinned-shell contract",
    )
    require(
        training.MAX_GLOBAL_STEPS == 72_000
        and training.BATCH_SIZE_PER_RANK == 1024
        and training.MAX_GLOBAL_STEPS % 4 == 0,
        "TIGER-aligned Stage2 budget or batch contract changed",
    )
    require(
        training.BEHAVIOUR_LOSS_WEIGHT == 0.1
        and training.BEHAVIOUR_MARGIN == 0.4,
        "behavior-ranking weight or margin changed",
    )

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = training._transition_pairs(train_frame)
    require(
        len(source_ids) >= training.BATCH_SIZE_PER_RANK
        and max(int(source_ids.max()), int(successor_ids.max())) < len(embeddings),
        "training transitions do not map to the Stage1 embedding rows",
    )
    chosen = np.random.default_rng(training.SEED).choice(
        len(source_ids), size=training.BATCH_SIZE_PER_RANK, replace=False
    )
    source = torch.as_tensor(embeddings[source_ids[chosen]], device=device)
    successor = torch.as_tensor(embeddings[successor_ids[chosen]], device=device)

    pair_embeddings = torch.cat((source, successor), dim=0)
    encoded_pairs = model.encoder(pair_embeddings)
    quantized_pairs, _, _, pair_tokens = model.rq(encoded_pairs)
    quantized_source, quantized_successor = quantized_pairs.chunk(2, dim=0)
    negative_indices = torch.randperm(quantized_source.shape[0], device=device)
    quantized_negatives = quantized_source[negative_indices]
    ranking_loss = behaviour_ranking_loss(
        quantized_source,
        quantized_successor,
        quantized_negatives,
        curvature=training.LAYER_CURVATURES[0],
        margin=training.BEHAVIOUR_MARGIN,
    )
    quantization_displacement = torch.linalg.vector_norm(
        quantized_pairs.detach() - encoded_pairs.detach(), dim=-1
    ).mean()
    require(
        pair_tokens.shape == (2 * training.BATCH_SIZE_PER_RANK, 3)
        and torch.isfinite(quantized_pairs).all()
        and torch.isfinite(ranking_loss)
        and float(quantization_displacement) > 0.0,
        "the loss did not use finite, changed residual-quantized outputs",
    )

    encoder_parameters = list(model.encoder.parameters())
    codebook_parameters = [layer.embed.weight for layer in model.rq.vq_layers]
    parameters = list(model.parameters())
    require_nonzero_finite_gradient(
        ranking_loss, encoder_parameters, "quantized behavior-ranking loss"
    )
    codebook_gradients = torch.autograd.grad(
        ranking_loss, codebook_parameters, retain_graph=True, allow_unused=True
    )
    require(
        all(
            gradient is None
            or (
                torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient).item() == 0
            )
            for gradient in codebook_gradients
        ),
        "behavior loss must not directly differentiate through hard assignments/codebooks",
    )

    probe = torch.as_tensor(embeddings[:256], device=device)
    reconstructed, quant_loss, _, _ = model(probe)
    base_loss, recon_loss = model.compute_loss(probe, reconstructed, quant_loss)
    total_loss = base_loss + training.BEHAVIOUR_LOSS_WEIGHT * ranking_loss
    require(
        total_loss.requires_grad and total_loss.grad_fn is not None,
        "total Stage2 loss is detached",
    )
    require_nonzero_finite_gradient(quant_loss, parameters, "quantization loss")
    require_nonzero_finite_gradient(recon_loss, parameters, "reconstruction loss")
    require_nonzero_finite_gradient(total_loss, parameters, "total Stage2 loss")
    total_loss.backward()
    require(
        any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in parameters
        ),
        "total-loss backward produced no finite nonzero gradient",
    )

    with torch.no_grad():
        corpus = torch.as_tensor(embeddings, device=device)
        tokens, stats = model.get_indices_with_stats(corpus)
    used_codes = [
        int(torch.count_nonzero(stat["usage_counts"] > 0).item()) for stat in stats
    ]
    require(
        tokens.shape == (len(embeddings), 3)
        and all(torch.isfinite(stat["usage_counts"]).all() for stat in stats)
        and all(torch.isfinite(stat["assignment_entropy_nats"]) for stat in stats)
        and all(count > 1 for count in used_codes),
        f"parent-checkpoint Sinkhorn assignments invalid or fully collapsed: {used_codes}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.LR, weight_decay=training.WEIGHT_DECAY
    )
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    require(
        any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, parameters)
        ),
        "optimizer.step() did not update a parameter",
    )
    checkpoint_round_trip(model, device)

    print(
        "Quantized behavior ranking: "
        f"loss={float(ranking_loss.detach()):.6f} "
        f"quantization_displacement={float(quantization_displacement):.6f} "
        f"used_codes={used_codes}"
    )
    print(
        "MVG PASS: parent checkpoint, actual residual-quantized transition loss, "
        "encoder-only straight-through gradient, finite Sinkhorn assignments, "
        "non-collapsed codebooks, optimizer update, checkpoint reload, and "
        "TIGER-aligned 72k budget"
    )


if __name__ == "__main__":
    main()
