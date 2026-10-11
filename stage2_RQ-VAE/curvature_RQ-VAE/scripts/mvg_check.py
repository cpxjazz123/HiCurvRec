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
        training._tokenizer_config(),
        in_dim=int(embeddings.shape[1]),
        context_dim=int(embeddings.shape[1]),
    ).to(device)
    training.initialize_tiger_weights(model)
    return model


def encoder_inputs(embeddings, device):
    """Raw embedding plus the curvature-moved behaviour-context channel."""
    frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = training._transition_pairs(frame)
    centroid = training._behaviour_context(embeddings, source_ids, successor_ids)
    channel = training._curvature_context_channel(embeddings, centroid)
    return torch.cat([embeddings, channel], dim=1).to(device), centroid.to(device)


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
    all_inputs, all_centroid = encoder_inputs(all_embeddings, device)
    layers = model.rq.vq_layers
    require(len(layers) == 3, "Expected three residual quantization levels")
    # Curvature is per-level configuration, not a frozen constant: assert it
    # matches what the run registered and stays positive, rather than pinning
    # it to 1.0 (which is only the parent's value).
    expected_curvatures = [float(value) for value in training.LAYER_CURVATURES]
    require(
        [layer.curvature for layer in layers] == expected_curvatures,
        f"Poincare curvature {layers[0].curvature} != registered "
        f"{expected_curvatures}",
    )
    require(
        all(value > 0.0 for value in expected_curvatures),
        "Curvature must be positive at every level",
    )
    require(
        [layer.sk_epsilon for layer in layers] == [0.003] * 3
        and [layer.sk_iters for layer in layers] == [50] * 3,
        "Sinkhorn settings changed",
    )
    # The smoothness scope decides where the smoothness pairs come from, so a
    # silent fallback to the global pairing would train a different mechanism
    # than the one registered. Assert the registered value reached every layer.
    expected_scope = str(getattr(training, "QUANT_SMOOTHNESS_SCOPE", "global"))
    require(
        all(layer.smoothness_scope == expected_scope for layer in layers),
        f"smoothness scope {[layer.smoothness_scope for layer in layers]} != "
        f"registered {expected_scope}",
    )
    require(
        not any(
            name.endswith("c_layer_scale") for name, _ in model.named_parameters()
        ),
        "No learnable curvature scale should remain in the fixed-curvature model",
    )

    with torch.no_grad():
        model.init_codebook(all_inputs[:4096])

    # Assert the metric working point s = sqrt(c) * ||v|| that each level
    # actually quantizes at. This is the per-level quantity the mechanism is
    # about, so a silent fallback to the other pin reading fails here rather
    # than at Stage3.
    with torch.no_grad():
        probe = model.encoder(all_inputs[:512])
        captured = []

        def _capture(i):
            def hook(_module, args):
                captured.append(
                    (i, torch.linalg.vector_norm(args[0].detach(), dim=-1))
                )
            return hook

        handles = [
            layer.register_forward_pre_hook(_capture(i))
            for i, layer in enumerate(layers)
        ]
        model.rq(probe)
        for handle in handles:
            handle.remove()
    require(
        len(captured) == 3,
        f"Expected three VQ inputs to inspect, captured {len(captured)}",
    )
    realized = {}
    for i, measured in captured:
        curvature = float(layers[i].curvature)
        radius = float(training.LAYER_WORKING_RADII[i])
        observed_s = float((curvature ** 0.5) * float(measured.mean()))
        observed_norm = float(measured.mean())
        expected_s = (
            (curvature ** 0.5) * radius
            if not training.PIN_IN_S_COORDINATES
            else radius
        )
        require(
            abs(observed_s - expected_s) < 1e-5,
            f"L{i + 1} working point s={observed_s:.6f}, expected "
            f"{expected_s:.6f} (c={curvature}, "
            f"pin_in_s_coordinates={training.PIN_IN_S_COORDINATES})",
        )
        if not training.PIN_IN_S_COORDINATES:
            require(
                abs(observed_norm - radius) < 1e-5,
                f"L{i + 1} ||v||={observed_norm:.6f}, expected {radius} "
                "with an absolute pin",
            )
        realized[i] = (observed_norm, observed_s)
    print(
        "  working point: "
        + " ".join(
            f"L{i + 1}(c={layers[i].curvature},||v||={realized[i][0]:.4f},"
            f"s={realized[i][1]:.4f})"
            for i in sorted(realized)
        )
        + f" pin_in_s_coordinates={training.PIN_IN_S_COORDINATES}"
    )

    batch_inputs, batch_centroid = all_inputs, all_centroid
    # The context channel must be a curvature move, not a copy and not an
    # arithmetic mean: a silent fallback to either fails here.
    channel = batch_inputs[:, all_embeddings.shape[1]:]
    require(
        torch.isfinite(channel).all(),
        "Behaviour-context channel is non-finite",
    )
    require(
        float(torch.linalg.vector_norm(channel - all_embeddings.to(device), dim=-1).mean())
        > 1e-6,
        "Behaviour-context channel equals the raw embedding",
    )
    arithmetic = 0.5 * (all_embeddings.to(device) + batch_centroid)
    require(
        float((channel - arithmetic).abs().max()) > 1e-6,
        "Behaviour-context channel collapsed to the arithmetic mean",
    )
    batch = all_embeddings[:256].to(device)
    reconstructed, quant_loss, tokens = model(batch_inputs[:256])
    require(tokens.shape == (256, 3), "SID shape changed")
    require(
        reconstructed.shape == batch.shape,
        "Reconstruction shape must match the input embeddings",
    )
    require(
        torch.isfinite(reconstructed).all() and torch.isfinite(quant_loss),
        "Forward pass produced non-finite values",
    )

    total_loss, recon_loss = model.compute_loss(batch, reconstructed, quant_loss)
    require(
        total_loss.requires_grad and total_loss.grad_fn is not None,
        "Total loss is detached",
    )
    parameters = list(model.parameters())
    require_nonzero_finite_gradient(
        quant_loss, parameters, "quantization/commitment loss"
    )
    require_nonzero_finite_gradient(recon_loss, parameters, "reconstruction loss")

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

    tokens, stats = model.get_indices_with_stats(all_inputs[:512])
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
        corpus = all_inputs
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

    # --- Registered mechanism: the behaviour hinge ranks the QUANTIZED L1+L2
    # prefix, not the encoder output. Verify the space that is actually used,
    # that the prefix tensor is the quantized running sum (so the codebooks can
    # receive gradient), that gradient reaches exactly the levels that form that
    # prefix, and that the relocation has a measurable computational effect.
    require(
        training.BEHAVIOUR_RANKING_SPACE == "prefix_l12",
        f"registered ranking space {training.BEHAVIOUR_RANKING_SPACE!r} is not "
        "the one this MVG verifies",
    )
    pair_frame = pd.read_parquet(experiment.TRAIN_FILE)
    pair_source_ids, pair_successor_ids = training._transition_pairs(pair_frame)
    pair_count = 256
    pair_source = all_inputs[
        torch.from_numpy(np.asarray(pair_source_ids[:pair_count], dtype=np.int64))
    ].to(device)
    pair_successor = all_inputs[
        torch.from_numpy(np.asarray(pair_successor_ids[:pair_count], dtype=np.int64))
    ].to(device)
    # Grad enabled on purpose: the hinge has to reach the encoder as well, and a
    # detached encoder output would make that check vacuous.
    encoded_pair_source = model.encoder(pair_source)
    encoded_pair_successor = model.encoder(pair_successor)
    _, _, _, prefix_source_levels = model.rq(
        encoded_pair_source, return_prefixes=True
    )
    _, _, _, prefix_successor_levels = model.rq(
        encoded_pair_successor, return_prefixes=True
    )
    require(
        len(prefix_source_levels) == 3,
        f"expected three prefix levels, got {len(prefix_source_levels)}",
    )
    prefix_source = prefix_source_levels[1]
    prefix_successor = prefix_successor_levels[1]
    require(
        prefix_source.shape == encoded_pair_source.shape,
        f"L1+L2 prefix shape {tuple(prefix_source.shape)} != encoder shape "
        f"{tuple(encoded_pair_source.shape)}",
    )
    require(
        torch.isfinite(prefix_source).all()
        and torch.isfinite(prefix_successor).all(),
        "L1+L2 prefix is non-finite",
    )
    # The prefix must be the quantized running sum, not a passthrough of the
    # latent: a silent fallback to the encoder output fails here.
    require(
        float((prefix_source - encoded_pair_source).abs().max()) > 1e-6,
        "L1+L2 prefix equals the encoder output; the hinge would be ranking the "
        "latent, not the quantized prefix",
    )
    permutation = torch.randperm(prefix_source.shape[0], device=device)
    prefix_ranking = behaviour_ranking_loss(
        prefix_source,
        prefix_successor,
        prefix_source[permutation],
        curvature=training.LAYER_CURVATURES[0],
        margin=training.BEHAVIOUR_MARGIN,
    )
    encoder_ranking = behaviour_ranking_loss(
        encoded_pair_source,
        encoded_pair_successor,
        encoded_pair_source[permutation],
        curvature=training.LAYER_CURVATURES[0],
        margin=training.BEHAVIOUR_MARGIN,
    )
    require(
        torch.isfinite(prefix_ranking) and torch.isfinite(encoder_ranking),
        "Behaviour hinge produced a non-finite value",
    )
    require(
        abs(float(prefix_ranking) - float(encoder_ranking)) > 1e-6,
        "The prefix-space and encoder-space hinges are numerically identical; "
        "the relocation has no computational effect",
    )
    print(
        f"  behaviour hinge: prefix_l12={float(prefix_ranking):.6f} "
        f"encoder={float(encoder_ranking):.6f}"
    )
    # Gradient must reach the codebooks that form the L1+L2 prefix (levels 1 and
    # 2) and must NOT reach level 3, whose codes lie outside that prefix.
    prefix_gradients = torch.autograd.grad(
        prefix_ranking,
        [layers[i].get_code_embs() for i in range(3)],
        retain_graph=True,
        allow_unused=True,
    )
    for index in (0, 1):
        gradient = prefix_gradients[index]
        require(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient).item() > 0,
            f"L{index + 1} codebook received no finite nonzero gradient from the "
            "L1+L2 prefix hinge",
        )
    third_gradient = prefix_gradients[2]
    require(
        third_gradient is None or torch.count_nonzero(third_gradient).item() == 0,
        "L3 codebook received gradient from an L1+L2 prefix hinge",
    )
    require_nonzero_finite_gradient(
        prefix_ranking,
        list(model.encoder.parameters()),
        "L1+L2 prefix hinge into the encoder",
    )

    print(
        "MVG PASS: fixed-curvature Poincare geometry, quantization and "
        "reconstruction gradients, optimizer update, checkpoint reload, "
        "per-bucket Sinkhorn assignment, TIGER-aligned 72k budget, "
        "L1+L2 prefix behaviour hinge"
    )


if __name__ == "__main__":
    main()
