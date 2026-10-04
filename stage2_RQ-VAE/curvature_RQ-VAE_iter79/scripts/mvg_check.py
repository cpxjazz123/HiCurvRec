"""Validate iter79 angular behavior ranking against the accepted parent."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
import train_rqvae as training
from model import RQVAE

PARENT_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
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


def build_model(embeddings: torch.Tensor, device: torch.device) -> RQVAE:
    model = RQVAE(
        training._tokenizer_config(), in_dim=int(embeddings.shape[1])
    ).to(device)
    training.initialize_tiger_weights(model)
    return model


def checkpoint_round_trip(model: RQVAE, device: torch.device) -> None:
    path = Path("/tmp/mvg_iter79_angular_ranking.pth")
    torch.save(model.state_dict(), path)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    require(
        all(torch.isfinite(parameter).all() for parameter in model.parameters()),
        "checkpoint reload produced a non-finite parameter",
    )
    path.unlink()


def main() -> None:
    training.configure_run(__file__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_embeddings = torch.from_numpy(
        training.load_embeddings(experiment.EMBEDDING_FILE)
    )
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = training._transition_pairs(train_frame)
    batch_size = training.BATCH_SIZE_PER_RANK
    selected = np.random.default_rng(training.SEED).choice(
        len(source_ids), size=batch_size, replace=False
    )
    source_batch = torch.as_tensor(
        all_embeddings[source_ids[selected]], device=device
    )
    successor_batch = torch.as_tensor(
        all_embeddings[successor_ids[selected]], device=device
    )
    training.set_seed(training.SEED)

    model = build_model(all_embeddings, device)
    checkpoint = torch.load(PARENT_CKPT, map_location=device)
    require(
        checkpoint.get("global_step") == 72_000,
        "unexpected accepted-parent checkpoint step",
    )
    require(
        checkpoint.get("curvatures") == [1.0, 1.0, 1.0]
        and checkpoint.get("working_radii") == [0.2, 0.2, 0.2],
        "accepted-parent checkpoint geometry changed",
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.train()

    layers = model.rq.vq_layers
    require(len(layers) == 3, "expected three residual quantization levels")
    require(
        tuple(layer.curvature for layer in layers) == (1.0, 1.0, 1.0)
        and tuple(training.LAYER_CURVATURES) == (1.0, 1.0, 1.0),
        "iter79 must retain parent curvature at every level",
    )
    require(
        model.rq.get_working_radii() == (0.2, 0.2, 0.2)
        and tuple(training.LAYER_WORKING_RADII) == (0.2, 0.2, 0.2),
        "iter79 must retain all parent s=0.2 pins",
    )
    require(
        [layer.sk_epsilon for layer in layers] == [0.003] * 3
        and [layer.sk_iters for layer in layers] == [50] * 3,
        "Sinkhorn settings changed",
    )
    require(
        not any(name.endswith("c_layer_scale") for name, _ in model.named_parameters()),
        "fixed-curvature model unexpectedly has a learnable curvature scale",
    )

    for level, layer in enumerate(layers):
        probe = torch.randn((64, training.CODEBOOK_DIM), device=device)
        target_norm = model.rq._radius_for_level(
            level, layer.get_curvature(), probe
        )
        pinned = model.rq._pin_to_radius(probe, target_norm)
        dimensionless_norm = (
            layer.get_curvature().sqrt()
            * torch.linalg.vector_norm(pinned, dim=-1)
        )
        require(
            torch.allclose(
                dimensionless_norm,
                torch.full_like(dimensionless_norm, 0.2),
                atol=1e-6,
                rtol=1e-6,
            ),
            f"level {level} does not preserve s=0.2",
        )

    encoded_source = model.encoder(source_batch)
    encoded_successor = model.encoder(successor_batch)
    negatives = encoded_source[torch.randperm(batch_size, device=device)]
    ranking_loss = training.behaviour_ranking_loss(
        encoded_source,
        encoded_successor,
        negatives,
        margin=training.BEHAVIOUR_MARGIN,
    )
    radial_scales = torch.linspace(
        0.25, 2.0, batch_size, device=device
    ).unsqueeze(-1)
    scaled_loss = training.behaviour_ranking_loss(
        encoded_source * radial_scales,
        encoded_successor / radial_scales,
        negatives * (3.0 - radial_scales),
        margin=training.BEHAVIOUR_MARGIN,
    )
    require(
        torch.isfinite(ranking_loss)
        and torch.allclose(ranking_loss, scaled_loss, atol=2e-6, rtol=2e-6),
        "angular ranking must be finite and invariant to positive radial rescaling",
    )
    print(
        f"angular objective radial-scale invariance PASS; "
        f"rank_loss={float(ranking_loss.detach()):.6f}"
    )

    reconstructed, quant_loss, _, tokens = model(source_batch)
    require(tokens.shape == (batch_size, 3), "SID shape changed")
    require(
        reconstructed.shape == source_batch.shape,
        "reconstruction shape must match input embeddings",
    )
    require(
        torch.isfinite(reconstructed).all() and torch.isfinite(quant_loss),
        "forward pass produced non-finite values",
    )
    base_loss, recon_loss = model.compute_loss(
        source_batch, reconstructed, quant_loss
    )
    total_loss = base_loss + training.BEHAVIOUR_LOSS_WEIGHT * ranking_loss
    require(
        total_loss.requires_grad and total_loss.grad_fn is not None,
        "total loss is detached",
    )
    require(
        torch.isfinite(total_loss) and torch.isfinite(ranking_loss),
        "training objective produced a non-finite value",
    )

    parameters = list(model.parameters())
    require_nonzero_finite_gradient(
        quant_loss, parameters, "quantization/commitment loss"
    )
    require_nonzero_finite_gradient(recon_loss, parameters, "reconstruction loss")
    require_nonzero_finite_gradient(
        ranking_loss, parameters, "angular behavior-ranking loss"
    )
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.LR, weight_decay=training.WEIGHT_DECAY
    )
    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer.step()
    after = [
        parameter.detach()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    require(
        any(not torch.equal(old, new) for old, new in zip(before, after)),
        "optimizer.step() did not update any parameter",
    )
    checkpoint_round_trip(model, device)

    model.eval()
    with torch.no_grad():
        corpus_tokens, stats = model.get_indices_with_stats(
            all_embeddings.to(device)
        )
    require(
        corpus_tokens.shape == (len(all_embeddings), 3),
        "full-corpus SID assignment shape changed",
    )
    for level, stat in enumerate(stats):
        require(
            torch.isfinite(stat["usage_counts"]).all()
            and int(stat["usage_counts"].sum().item()) == len(all_embeddings),
            f"level {level} Sinkhorn assignment is invalid",
        )
        require(
            torch.isfinite(stat["assignment_entropy_nats"]),
            f"level {level} assignment entropy is non-finite",
        )
    full = corpus_tokens.cpu().numpy()
    prefix_counts = [
        len(np.unique(full[:, :level + 1], axis=0)) for level in range(3)
    ]
    used_codes = [
        int((stat["usage_counts"] > 0).sum().item()) for stat in stats
    ]
    unique_rows = len(np.unique(full, axis=0))
    collision_rate = 1.0 - unique_rows / len(full)
    print(
        f"descriptive_only: prefix_counts={prefix_counts} "
        f"used_codes={used_codes} unique_sid_rows={unique_rows}/{len(full)} "
        f"collision_rate={collision_rate:.6f}"
    )

    require(
        training.MAX_GLOBAL_STEPS == 72_000
        and training.MAX_GLOBAL_STEPS % 4 == 0,
        "Stage2 budget must remain TIGER-aligned at 72000 global steps",
    )
    require(
        training.BATCH_SIZE_PER_RANK == 1024
        and len(training.SNAPSHOT_STEPS) == 1,
        "TIGER batch size or single-snapshot contract changed",
    )
    print(
        "MVG PASS: accepted parent checkpoint, raw-latent angular ranking, "
        "radial-scale invariance, all s=0.2 pins, finite Sinkhorn assignments, "
        "quantization/reconstruction/ranking gradients, optimizer update, "
        "checkpoint reload, and 72k Stage2 invariants"
    )


if __name__ == "__main__":
    main()
