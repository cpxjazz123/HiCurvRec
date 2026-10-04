"""MVG for the hyperbolic RQ-VAE: geometry, gradients, and checkpoint reload."""
from __future__ import annotations

import math
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
from model.layers import _poincare_distance_tangent_pairs


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
    embedding_array = training.load_embeddings(experiment.EMBEDDING_FILE)
    all_embeddings = torch.from_numpy(embedding_array)
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = training._transition_pairs(train_frame)
    batch_size = training.BATCH_SIZE_PER_RANK
    require(len(source_ids) >= batch_size, "Not enough training transitions")
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
    require(checkpoint.get("global_step") == 72_000, "Unexpected parent checkpoint step")
    require(
        tuple(float(value) for value in checkpoint.get("curvatures", ()))
        == (1.0, 1.0, 1.0)
        and tuple(float(value) for value in checkpoint.get("working_radii", ()))
        == (0.2, 0.2, 0.2),
        "Parent checkpoint geometry does not match the accepted baseline",
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.train()

    layers = model.rq.vq_layers
    require(len(layers) == 3, "Expected three residual quantization levels")
    require(
        tuple(layer.curvature for layer in layers) == (1.0, 1.5, 1.0)
        and tuple(training.LAYER_CURVATURES) == (1.0, 1.5, 1.0),
        "Only L2 curvature must change from the parent value",
    )
    require(
        model.rq.get_working_radii() == (0.2, 0.2, 0.2)
        and tuple(training.LAYER_WORKING_RADII) == (0.2, 0.2, 0.2),
        "The dimensionless working radius s must remain 0.2 at every level",
    )
    require(
        layers[0].curvature == 1.0
        and training.BEHAVIOUR_LOSS_WEIGHT == 0.1
        and training.BEHAVIOUR_MARGIN == 0.4,
        "Parent behavior-ranking geometry or settings changed",
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
            f"Level {level} does not preserve s=0.2",
        )

    angle = 0.2
    direction_a = torch.zeros((1, training.CODEBOOK_DIM), device=device)
    direction_b = torch.zeros_like(direction_a)
    direction_a[0, 0] = 1.0
    direction_b[0, 0] = math.cos(angle)
    direction_b[0, 1] = math.sin(angle)
    working_s = training.LAYER_WORKING_RADII[1]
    candidate_curvature = layers[1].curvature
    parent_a = direction_a * working_s
    parent_b = direction_b * working_s
    candidate_radius = working_s / math.sqrt(candidate_curvature)
    candidate_a = direction_a * candidate_radius
    candidate_b = direction_b * candidate_radius
    parent_distance = _poincare_distance_tangent_pairs(
        parent_a, parent_b, curvature=1.0
    )
    candidate_distance = _poincare_distance_tangent_pairs(
        candidate_a, candidate_b, curvature=candidate_curvature
    )
    measured_ratio = float((candidate_distance / parent_distance).item())
    expected_ratio = 1.0 / math.sqrt(candidate_curvature)
    require(
        math.isfinite(measured_ratio)
        and abs(measured_ratio - expected_ratio) < 1e-4,
        "L2 curvature does not produce the expected fixed-s geometry change",
    )
    print(
        f"L2 fixed-s distance ratio={measured_ratio:.6f}; "
        f"expected={expected_ratio:.6f}; behavior-ranking curvature=1.0"
    )

    reconstructed, quant_loss, _, tokens = model(source_batch)
    require(tokens.shape == (batch_size, 3), "SID shape changed")
    require(
        reconstructed.shape == source_batch.shape,
        "Reconstruction shape must match the input embeddings",
    )
    require(
        torch.isfinite(reconstructed).all() and torch.isfinite(quant_loss),
        "Forward pass produced non-finite values",
    )
    base_loss, recon_loss = model.compute_loss(
        source_batch, reconstructed, quant_loss
    )
    encoded_source = model.encoder(source_batch)
    encoded_successor = model.encoder(successor_batch)
    negatives = encoded_source[torch.randperm(batch_size, device=device)]
    ranking_loss = training.behaviour_ranking_loss(
        encoded_source,
        encoded_successor,
        negatives,
        curvature=layers[0].curvature,
        margin=training.BEHAVIOUR_MARGIN,
    )
    total_loss = base_loss + training.BEHAVIOUR_LOSS_WEIGHT * ranking_loss
    require(
        total_loss.requires_grad and total_loss.grad_fn is not None,
        "Total loss is detached",
    )
    require(
        torch.isfinite(total_loss) and torch.isfinite(ranking_loss),
        "Training objective produced a non-finite value",
    )

    parameters = list(model.parameters())
    require_nonzero_finite_gradient(
        quant_loss, parameters, "quantization/commitment loss"
    )
    require_nonzero_finite_gradient(recon_loss, parameters, "reconstruction loss")
    require_nonzero_finite_gradient(
        ranking_loss, parameters, "behavior-ranking loss"
    )
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
    checkpoint_round_trip(model, device, "iter78")

    model.eval()
    with torch.no_grad():
        corpus_tokens, stats = model.get_indices_with_stats(
            all_embeddings.to(device)
        )
    require(
        corpus_tokens.shape == (len(all_embeddings), 3),
        "Full-corpus SID assignment shape changed",
    )
    for level, stat in enumerate(stats):
        require(
            torch.isfinite(stat["usage_counts"]).all()
            and int(stat["usage_counts"].sum().item()) == len(all_embeddings),
            f"Level {level} Sinkhorn assignment is invalid",
        )
        require(
            torch.isfinite(stat["assignment_entropy_nats"]),
            f"Level {level} assignment entropy is non-finite",
        )
    full = corpus_tokens.cpu().numpy()
    prefix_counts = [
        len(np.unique(full[:, :level + 1], axis=0))
        for level in range(3)
    ]
    used_codes = [
        int((stat["usage_counts"] > 0).sum().item())
        for stat in stats
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
        "MVG PASS: accepted parent checkpoint, isolated L2 curvature=1.5 "
        "with all s=0.2 pins, unchanged L1 behavior-ranking geometry, "
        "finite Sinkhorn assignments, quantization/reconstruction/"
        "behavior-ranking gradients, optimizer update, checkpoint reload, "
        "and 72k Stage2 invariants"
    )


if __name__ == "__main__":
    main()
