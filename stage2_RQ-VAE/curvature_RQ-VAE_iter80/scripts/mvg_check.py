"""MVG for the hyperbolic RQ-VAE: geometry, gradients, and checkpoint reload."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
import train_rqvae as training
from model import RQVAE

PARENT_CHECKPOINT = Path(
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
    checkpoint = torch.load(
        PARENT_CHECKPOINT, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    require(
        checkpoint["curvatures"] == [1.0, 1.0, 1.0]
        and checkpoint["working_radii"] == [0.2, 0.2, 0.2],
        "Parent checkpoint does not preserve the fixed c=1, s=0.2 geometry",
    )
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

    model.train()

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

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    transition_sources, transition_successors = training._transition_pairs(
        train_frame
    )
    rng = np.random.default_rng(42)
    pair_rows = rng.choice(
        len(transition_sources), size=min(1024, len(transition_sources)),
        replace=False,
    )
    pair_source_ids = transition_sources[pair_rows]
    pair_successor_ids = transition_successors[pair_rows]
    pair_sources = all_embeddings[torch.from_numpy(pair_source_ids)].to(device)
    pair_successors = all_embeddings[
        torch.from_numpy(pair_successor_ids)
    ].to(device)
    encoded_source = model.encoder(pair_sources)
    encoded_successor = model.encoder(pair_successors)
    global_negatives = encoded_source[
        torch.randperm(encoded_source.shape[0], device=device)
    ]
    global_ranking_loss = training.behaviour_ranking_loss(
        encoded_source,
        encoded_successor,
        global_negatives,
        curvature=training.LAYER_CURVATURES[0],
        margin=training.BEHAVIOUR_MARGIN,
    )
    source_l2, source_prefixes = training._pinned_l2_residual(
        model, encoded_source
    )
    successor_l2, _ = training._pinned_l2_residual(
        model, encoded_successor
    )
    true_successors_by_source = [set() for _ in range(len(embedding_array))]
    for source_id, successor_id in zip(
        transition_sources, transition_successors
    ):
        true_successors_by_source[int(source_id)].add(int(successor_id))
    negative_rows, local_valid = training._prefix_conditioned_negative_rows(
        source_prefixes,
        torch.from_numpy(pair_source_ids),
        torch.from_numpy(pair_successor_ids),
        true_successors_by_source,
    )
    valid_count = int(local_valid.sum().item())
    require(valid_count > 0, "No same-L1-prefix L2 behavior negatives were valid")
    require(
        torch.equal(
            source_prefixes[local_valid],
            source_prefixes[negative_rows[local_valid]],
        ),
        "L2 negatives do not share the source L1 prefix",
    )
    valid_rows = torch.nonzero(local_valid, as_tuple=False).flatten().cpu().numpy()
    negative_source_ids = pair_source_ids[
        negative_rows[local_valid].cpu().numpy()
    ]
    require(
        all(
            int(negative_id) != int(source_id)
            and int(negative_id) != int(successor_id)
            and int(negative_id)
            not in true_successors_by_source[int(source_id)]
            for source_id, successor_id, negative_id in zip(
                pair_source_ids[valid_rows],
                pair_successor_ids[valid_rows],
                negative_source_ids,
            )
        ),
        "L2 negative sampler retained a source, positive, or observed successor",
    )
    l2_norms = torch.linalg.vector_norm(source_l2, dim=-1)
    expected_l2_norm = (
        training.LAYER_WORKING_RADII[1]
        / training.LAYER_CURVATURES[1] ** 0.5
    )
    require(
        torch.allclose(
            l2_norms,
            torch.full_like(l2_norms, expected_l2_norm),
            atol=1e-5,
            rtol=1e-5,
        ),
        "L2 residual ranking left the configured pinned shell",
    )
    local_ranking_loss = training.behaviour_ranking_loss(
        source_l2[local_valid],
        successor_l2[local_valid],
        source_l2[negative_rows[local_valid]],
        curvature=training.LAYER_CURVATURES[1],
        margin=training.BEHAVIOUR_MARGIN,
    )
    ranking_loss = (
        (1.0 - training.L2_RESIDUAL_RANKING_FRACTION)
        * global_ranking_loss
        + training.L2_RESIDUAL_RANKING_FRACTION * local_ranking_loss
    )
    total_loss, recon_loss = model.compute_loss(
        batch, reconstructed, quant_loss
    )
    total_loss = total_loss + training.BEHAVIOUR_LOSS_WEIGHT * ranking_loss
    require(
        total_loss.requires_grad and total_loss.grad_fn is not None,
        "Total loss is detached",
    )
    parameters = list(model.parameters())
    require_nonzero_finite_gradient(
        quant_loss, parameters, "quantization/commitment loss"
    )
    require_nonzero_finite_gradient(recon_loss, parameters, "reconstruction loss")
    require_nonzero_finite_gradient(
        global_ranking_loss, parameters, "global behavior ranking loss"
    )
    require_nonzero_finite_gradient(
        local_ranking_loss, parameters, "same-prefix L2 residual ranking loss"
    )
    print(
        f"  L2 residual ranking: valid={valid_count}/{len(pair_rows)} "
        f"loss={float(local_ranking_loss.detach()):.6f} "
        f"pinned_norm={float(l2_norms.detach().mean()):.6f}"
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

    # Inspect the loaded parent codebook over the full corpus; utilization
    # counts are descriptive except for a complete single-code collapse.
    with torch.no_grad():
        corpus = all_embeddings.to(device)
        corpus_tokens, _ = model.get_indices_with_stats(corpus)
    full = corpus_tokens.cpu().numpy()
    prefixes = [len(np.unique(full[:, : k + 1], axis=0)) for k in range(3)]
    used_codes = [int(np.unique(full[:, level]).size) for level in range(3)]
    buckets = np.unique(full[:, 0])
    per_bucket = [np.unique(full[full[:, 0] == b, 1]).size for b in buckets]
    mean_bucket_reach = float(np.mean(per_bucket))
    print(
        f"  parent-codebook assignments: used_codes={used_codes} "
        f"bucket reach L0={prefixes[0]} L0L1={prefixes[1]} "
        f"L0L1L2={prefixes[2]} L1-per-bucket={mean_bucket_reach:.1f}"
    )
    require(
        min(used_codes) > 1,
        f"Quantizer collapsed to <=1 code at a level: {used_codes}",
    )

    print(
        "MVG PASS: parent-checkpoint local L2 behavior gradients, pinned shell, "
        "finite Sinkhorn, non-collapsed codebooks, optimizer update, checkpoint "
        "reload, and TIGER-aligned 72k budget"
    )


if __name__ == "__main__":
    main()
