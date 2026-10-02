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
        [layer.curvature for layer in layers] == [1.0, 1.0, 1.0],
        "Fixed Poincare curvature must be 1.0 at every level",
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

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    radial_targets, target_stats = training.build_transition_radial_targets(
        train_frame, train_ids, len(all_embeddings)
    )
    print(
        "  radial target: "
        f"edges={int(target_stats['transition_edges'])} "
        f"items={int(target_stats['train_items'])} "
        f"radius_median={target_stats['radius_median']:.6f} "
        f"range=[{target_stats['radius_min']:.6f},"
        f"{target_stats['radius_max']:.6f}]"
    )
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    with torch.no_grad():
        model.init_codebook(train_embeddings[:4096].to(device))

    batch = train_embeddings[:256].to(device)
    batch_target_radii = torch.from_numpy(radial_targets[:256]).to(device)
    reconstructed, quant_loss, unused_codes, tokens, encoded = model(batch)
    require(tokens.shape == (256, 3), "SID shape changed")
    require(encoded.shape == (256, training.CODEBOOK_DIM), "Encoder output shape changed")
    require(
        reconstructed.shape == batch.shape,
        "Reconstruction shape must match the input embeddings",
    )
    require(
        torch.isfinite(reconstructed).all()
        and torch.isfinite(quant_loss).all()
        and torch.isfinite(encoded).all(),
        "Forward pass produced non-finite values",
    )

    base_loss, recon_loss = model.compute_loss(batch, reconstructed, quant_loss)
    radial_loss = training.radial_supervision_loss(encoded, batch_target_radii)
    total_loss = base_loss + training.RADIAL_LOSS_WEIGHT * radial_loss
    require(
        total_loss.requires_grad and total_loss.grad_fn is not None,
        "Total loss is detached",
    )
    require(
        torch.isfinite(radial_loss)
        and torch.isfinite(batch_target_radii).all()
        and bool((batch_target_radii > 0).all()),
        "Radial loss targets are invalid",
    )
    parameters = list(model.parameters())
    require_nonzero_finite_gradient(
        quant_loss, parameters, "quantization/commitment loss"
    )
    require_nonzero_finite_gradient(recon_loss, parameters, "reconstruction loss")
    require_nonzero_finite_gradient(radial_loss, parameters, "transition radial loss")

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

    # The registered mechanism is per-bucket balancing. Judge it at full corpus
    # size, not on the 512-item probe above: with 512 items and 256 first-level
    # codes each bucket holds ~2 residuals, so second-level reach is capped by
    # the sample count rather than by the mechanism.
    #
    # The mechanism's effect, measured on the parent checkpoint: global
    # Sinkhorn let each L0 bucket reach only ~37 of 256 L1 codes, capping
    # (L0, L1) at 9370 distinct prefixes. A silent fallback to global balancing
    # must fail here instead of passing MVG and reverting to parent behaviour.
    with torch.no_grad():
        corpus = all_embeddings.to(device)
        model.init_codebook(corpus)
        corpus_tokens, _ = model.get_indices_with_stats(corpus)
    full = corpus_tokens.cpu().numpy()
    prefixes = [len(np.unique(full[:, : k + 1], axis=0)) for k in range(3)]
    buckets = np.unique(full[:, 0])
    per_bucket = [np.unique(full[full[:, 0] == b, 1]).size for b in buckets]
    mean_bucket_reach = float(np.mean(per_bucket))
    print(
        f"  bucket reach: L0={prefixes[0]} L0L1={prefixes[1]} "
        f"L0L1L2={prefixes[2]} L1-per-bucket={mean_bucket_reach:.1f}"
    )
    require(
        prefixes[1] > prefixes[0] * 5,
        f"Second level adds almost no reachable capacity (L0={prefixes[0]} "
        f"L0L1={prefixes[1]}); per-bucket balancing is not in effect",
    )
    require(
        mean_bucket_reach > 40.0,
        f"Each L0 bucket reaches only {mean_bucket_reach:.1f} of 256 L1 codes; "
        "expected the per-bucket mechanism to widen this well past the "
        "global-Sinkhorn baseline of ~37",
    )

    print(
        "MVG PASS: fixed-curvature Poincare geometry, transition radial "
        "loss gradient, quantization/reconstruction gradients, optimizer "
        "update, checkpoint reload, per-bucket Sinkhorn, 72k budget"
    )


if __name__ == "__main__":
    main()
