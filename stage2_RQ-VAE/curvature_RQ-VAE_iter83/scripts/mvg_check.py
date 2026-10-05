"""MVG for fixed codebook shells with free residual norms."""
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
from model.layers import _hyperbolic_residual
from model.model import behaviour_ranking_loss


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MVG FAIL: {message}")


def require_nonzero_finite_gradient(
    loss: torch.Tensor, parameters, label: str
) -> tuple[torch.Tensor | None, ...]:
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
    return gradients


def verify_projected_shells(model: RQVAE, label: str) -> None:
    for level, (layer, radius) in enumerate(
        zip(model.rq.vq_layers, experiment.CODEBOOK_SHELL_RADII)
    ):
        projected = layer.get_code_embs()
        norms = torch.linalg.vector_norm(projected, dim=-1)
        target = torch.full_like(norms, float(radius))
        require(
            torch.isfinite(projected).all()
            and torch.allclose(norms, target, rtol=1e-6, atol=1e-7),
            f"L{level + 1} projected codeword norms left radius {radius} {label}",
        )


def residual_norms_by_level(model: RQVAE, batch: torch.Tensor):
    residual = model.encoder(batch)
    previous_codes = None
    result = []
    for layer in model.rq.vq_layers:
        result.append(torch.linalg.vector_norm(residual, dim=-1))
        quantized, _, _, indices = layer(residual, bucket=previous_codes)
        residual = _hyperbolic_residual(
            residual, quantized, layer.get_curvature()
        )
        previous_codes = indices
    return result


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(training.SEED)
    np.random.seed(training.SEED)

    embedding_array = np.asarray(
        np.load(experiment.EMBEDDING_FILE), dtype=np.float32
    )
    all_embeddings = torch.from_numpy(embedding_array)
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    source_ids, successor_ids = training._transition_pairs(train_frame)
    require(
        len(train_ids) >= 1024 and len(source_ids) >= 1024,
        "Expected full one-batch Stage2 target and transition samples",
    )
    require(
        training.SEED == 42
        and training.MAX_GLOBAL_STEPS == 72_000
        and training.BEHAVIOUR_LOSS_WEIGHT == 0.1
        and training.BEHAVIOUR_MARGIN == 0.4,
        "Seed, 72k budget, or parent behavior-ranking settings changed",
    )
    batch = all_embeddings[torch.from_numpy(train_ids[:1024])].to(device)
    pair_sources = all_embeddings[torch.from_numpy(source_ids[:1024])].to(device)
    pair_successors = all_embeddings[
        torch.from_numpy(successor_ids[:1024])
    ].to(device)

    model = RQVAE(
        training._tokenizer_config(), in_dim=int(all_embeddings.shape[1])
    ).to(device)
    training.initialize_tiger_weights(model)
    checkpoint = torch.load(
        experiment.PARENT_CHECKPOINT, map_location=device, weights_only=False
    )
    state = checkpoint.get("state_dict")
    require(isinstance(state, dict), "Parent checkpoint has no state_dict")
    require(
        int(checkpoint.get("global_step", -1)) == training.MAX_GLOBAL_STEPS
        and checkpoint.get("curvatures") == [1.0, 1.0, 1.0]
        and checkpoint.get("working_radii") == [0.2, 0.2, 0.2],
        "Parent checkpoint is not the accepted full-budget pinned condition",
    )
    model.load_state_dict(state, strict=True)
    model.train()

    expected_radii = tuple(float(value) for value in experiment.CODEBOOK_SHELL_RADII)
    require(len(expected_radii) == 3, "Expected one shell radius per RQ level")
    require(
        model.rq.get_working_radii() == (0.0, 0.0, 0.0),
        "Residual hard pins are still active",
    )
    require(
        model.rq.get_codebook_shell_radii() == expected_radii,
        "Model codebook radii differ from the frozen parent P50 values",
    )
    require(
        [float(layer.curvature) for layer in model.rq.vq_layers]
        == [1.0, 1.0, 1.0],
        "Parent fixed curvature changed",
    )

    parent_state = state
    for level, radius in enumerate(expected_radii):
        key = f"rq.vq_layers.{level}.embed.weight"
        parent_norms = torch.linalg.vector_norm(
            parent_state[key].to(torch.float64), dim=-1
        )
        parent_median = torch.quantile(
            parent_norms,
            torch.tensor(
                0.5, dtype=torch.float64, device=parent_norms.device
            ),
        ).item()
        require(
            abs(parent_median - radius) <= 1e-9,
            f"L{level + 1} configured radius is not the parent P50",
        )
    verify_projected_shells(model, "after parent checkpoint load")

    with torch.no_grad():
        residual_norms = residual_norms_by_level(model, batch)
    residual_spreads = []
    for level, norms in enumerate(residual_norms):
        spread = float((norms.max() - norms.min()).item())
        std = float(norms.std(unbiased=False).item())
        require(
            torch.isfinite(norms).all() and spread > 1e-6 and std > 1e-6,
            f"L{level + 1} residual norms do not vary freely",
        )
        residual_spreads.append(spread)

    with torch.no_grad():
        tokens, stats = model.get_indices_with_stats(
            all_embeddings.to(device)
        )
    require(
        tokens.shape == (len(all_embeddings), 3),
        "Full-corpus assignment returned the wrong SID shape",
    )
    code_usage = []
    for level, stat in enumerate(stats):
        usage = stat["usage_counts"]
        entropy = stat["assignment_entropy_nats"]
        require(
            torch.isfinite(usage).all()
            and int(usage.sum().item()) == len(all_embeddings)
            and int((usage > 0).sum().item()) == 256,
            f"L{level + 1} full-corpus Sinkhorn usage is invalid or collapsed",
        )
        require(torch.isfinite(entropy), f"L{level + 1} entropy is non-finite")
        code_usage.append(int((usage > 0).sum().item()))
    verify_projected_shells(model, "after full-corpus assignment")

    model.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, batch_tokens = model(batch)
    total_reconstruction_loss, reconstruction_loss = model.compute_loss(
        batch, reconstructed, quant_loss
    )
    encoded_sources = model.encoder(pair_sources)
    encoded_successors = model.encoder(pair_successors)
    negatives = encoded_sources[
        torch.randperm(encoded_sources.shape[0], device=device)
    ]
    ranking_loss = behaviour_ranking_loss(
        encoded_sources,
        encoded_successors,
        negatives,
        curvature=1.0,
        margin=training.BEHAVIOUR_MARGIN,
    )
    total_loss = (
        total_reconstruction_loss
        + training.BEHAVIOUR_LOSS_WEIGHT * ranking_loss
    )
    require(
        batch_tokens.shape == (batch.shape[0], 3)
        and torch.isfinite(reconstructed).all()
        and torch.isfinite(quant_loss)
        and torch.isfinite(ranking_loss)
        and total_loss.requires_grad
        and total_loss.grad_fn is not None,
        "One-batch forward or total loss is invalid",
    )

    for level, layer in enumerate(model.rq.vq_layers):
        gradient = require_nonzero_finite_gradient(
            quant_loss, [layer.embed.weight], f"L{level + 1} codebook loss"
        )[0]
        require(gradient is not None, f"L{level + 1} codebook gradient is missing")
        norms = torch.linalg.vector_norm(layer.embed.weight.detach(), dim=-1)
        grad_norms = torch.linalg.vector_norm(gradient, dim=-1)
        active = grad_norms > 0
        radial_alignment = (
            (gradient[active] * layer.embed.weight.detach()[active])
            .sum(dim=-1)
            .abs()
            / (grad_norms[active] * norms[active]).clamp_min(1e-30)
        )
        require(
            active.any() and radial_alignment.max().item() < 1e-4,
            f"L{level + 1} codebook gradients are not nonzero directional gradients",
        )

    encoder_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("encoder.")
    ]
    decoder_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("decoder.")
    ]
    require_nonzero_finite_gradient(
        reconstruction_loss, encoder_parameters, "reconstruction encoder"
    )
    require_nonzero_finite_gradient(
        reconstruction_loss, decoder_parameters, "reconstruction decoder"
    )
    require_nonzero_finite_gradient(
        ranking_loss, encoder_parameters, "raw-latent behavior ranking"
    )

    total_loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("encoder.", "decoder.", "rq.vq_layers.")):
            require(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all(),
                f"{name} has no finite total-loss gradient",
            )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.LR, weight_decay=training.WEIGHT_DECAY
    )
    optimizer.step()
    verify_projected_shells(model, "after optimizer update")

    print(
        "MVG PASS: parent checkpoint + one 1024-item batch; "
        f"R={expected_radii}; residual spreads={residual_spreads}; "
        f"full-corpus code use={code_usage}; finite encoder/decoder/directional "
        "codebook gradients; fixed projected shells and normal Sinkhorn"
    )


if __name__ == "__main__":
    main()
