"""MVG for the hyperbolic RQ-VAE: geometry, gradients, and checkpoint reload."""
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


def build_model(embeddings, device):
    model = RQVAE(
        training._tokenizer_config(), in_dim=int(embeddings.shape[1])
    ).to(device)
    training.initialize_tiger_weights(model)
    return model


def checkpoint_round_trip(model, device, label: str) -> None:
    path = Path(f"/tmp/mvg_hyperbolic_{label}.pth")
    torch.save(model.state_dict(), path)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"MVG FAIL: {label} checkpoint has non-finite weights")
    path.unlink()


def main() -> None:
    training.configure_run(__file__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding_array = np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    all_embeddings = torch.from_numpy(embedding_array)
    torch.manual_seed(42)

    model = build_model(all_embeddings, device)
    layers = model.rq.vq_layers
    require(len(layers) == 3, "Expected three residual quantization levels")
    require(
        [layer.curvature for layer in layers]
        == list(training.LAYER_CURVATURES),
        "Fixed per-level curvature differs from the registered C2 values",
    )
    require(
        model.rq.get_working_radii() == (0.0, 0.0, 0.0),
        "C2 must preserve natural residual magnitudes without pinning",
    )
    require(
        [layer.sk_epsilon for layer in layers] == [0.003] * 3
        and [layer.sk_iters for layer in layers] == [50] * 3,
        "Sinkhorn settings changed",
    )
    require(
        not any(
            name.endswith("c_layer_scale") for name, _ in model.named_parameters()
        ),
        "No learnable curvature scale should remain in the fixed-curvature model",
    )
    require(
        training.BEHAVIOUR_CURVATURE == 1.0
        and training.BEHAVIOUR_LOSS_WEIGHT == 0.1
        and training.BEHAVIOUR_MARGIN == 0.4,
        "Behavior-ranking geometry or strength differs from the parent",
    )

    with torch.no_grad():
        model.init_codebook(all_embeddings[:4096].to(device))

    batch = all_embeddings[:256].to(device)
    reconstructed, quant_loss, unused_codes, tokens = model(batch)
    require(tokens.shape == (256, 3), "SID shape changed")
    require(
        reconstructed.shape == batch.shape,
        "Reconstruction shape must match the input embeddings",
    )
    require(
        torch.isfinite(reconstructed).all() and torch.isfinite(quant_loss),
        "Forward pass produced non-finite values",
    )

    rq_loss, recon_loss = model.compute_loss(batch, reconstructed, quant_loss)
    sources, successors = training._transition_pairs(
        pd.read_parquet(experiment.TRAIN_FILE)
    )
    require(len(sources) >= 256, "Training data has fewer than 256 transition pairs")
    pair_sources = all_embeddings[torch.from_numpy(sources[:256])].to(device)
    pair_successors = all_embeddings[torch.from_numpy(successors[:256])].to(device)
    encoded_source = model.encoder(pair_sources)
    encoded_successor = model.encoder(pair_successors)
    negatives = encoded_source[
        torch.randperm(encoded_source.shape[0], device=device)
    ]
    ranking_loss = behaviour_ranking_loss(
        encoded_source,
        encoded_successor,
        negatives,
        curvature=training.BEHAVIOUR_CURVATURE,
        margin=training.BEHAVIOUR_MARGIN,
    )
    total_loss = rq_loss + training.BEHAVIOUR_LOSS_WEIGHT * ranking_loss
    require(
        total_loss.requires_grad and total_loss.grad_fn is not None,
        "Combined Stage2 loss is detached",
    )
    parameters = list(model.parameters())
    require_nonzero_finite_gradient(
        quant_loss, parameters, "quantization/commitment loss"
    )
    require_nonzero_finite_gradient(recon_loss, parameters, "reconstruction loss")
    require_nonzero_finite_gradient(ranking_loss, parameters, "behavior-ranking loss")

    total_loss.backward()
    require(
        any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in parameters
        ),
        "Total-loss backward produced no finite nonzero gradient",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.LR, weight_decay=training.WEIGHT_DECAY
    )
    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer.step()
    require(
        any(
            not torch.equal(old, new)
            for old, new in zip(
                before,
                [
                    parameter.detach()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
            )
        ),
        "optimizer.step() did not update any parameter",
    )

    checkpoint_round_trip(model, device, "hyperbolic")

    tokens, stats = model.get_indices_with_stats(all_embeddings[:512].to(device))
    require(tokens.shape == (512, 3), "Final SID assignment shape changed")
    for level, stat in enumerate(stats):
        require(
            torch.isfinite(stat["usage_counts"]).all()
            and int(stat["usage_counts"].sum().item()) == 512,
            f"Level {level} Sinkhorn assignment is invalid",
        )
        require(
            torch.isfinite(stat["assignment_entropy_nats"]),
            f"Level {level} assignment entropy is non-finite",
        )
        require(
            torch.isfinite(stat["natural_residual_norm_quantiles"]).all()
            and torch.isfinite(stat["working_s_quantiles"]).all()
            and torch.isfinite(stat["ball_radius_quantiles"]).all(),
            f"Level {level} residual geometry diagnostics are non-finite",
        )
        require(
            torch.allclose(
                stat["ball_radius_quantiles"],
                torch.tanh(stat["working_s_quantiles"]),
                atol=1e-6,
                rtol=1e-6,
            ),
            f"Level {level} rho is not tanh(s)",
        )

    require(
        training.MAX_GLOBAL_STEPS == 72_000,
        "Stage2 budget must stay TIGER-aligned at 72000 global steps",
    )
    require(
        training.MAX_GLOBAL_STEPS % 4 == 0,
        "Global step budget must be divisible by the DDP world size",
    )
    require(
        training.BATCH_SIZE_PER_RANK == 1024
        and len(training.SNAPSHOT_STEPS) == 1,
        "TIGER batch size or single-snapshot contract changed",
    )

    # The inherited per-bucket Sinkhorn path must still produce a complete,
    # finite full-corpus SID matrix. Usage and prefix counts are diagnostics,
    # not C2 acceptance thresholds.
    with torch.no_grad():
        corpus = all_embeddings.to(device)
        model.init_codebook(corpus)
        corpus_tokens, _ = model.get_indices_with_stats(corpus)
    require(
        corpus_tokens.shape == (len(corpus), 3)
        and torch.isfinite(corpus_tokens.float()).all(),
        "Full-corpus RQ assignments are invalid",
    )
    full = corpus_tokens.cpu().numpy()
    prefixes = [len(np.unique(full[:, : k + 1], axis=0)) for k in range(3)]
    print(f"  full-corpus prefixes: L1={prefixes[0]} L1L2={prefixes[1]} SID={prefixes[2]}")

    print(
        "MVG PASS: fixed sequential curvature, no pin, raw RQ hyperbolic "
        "distance, behavior ranking at c=1, finite gradients, checkpoint "
        "reload, native residual/s/rho diagnostics, 72k Stage2 budget"
    )


if __name__ == "__main__":
    main()
