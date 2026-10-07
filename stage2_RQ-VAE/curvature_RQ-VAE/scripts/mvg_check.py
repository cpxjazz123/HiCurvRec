"""MVG for the hyperbolic RQ-VAE: geometry, gradients, and checkpoint reload."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))
from model.layers import _poincare_distance_tangent_pairs

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


def angular_margin(
    source: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    return F.cosine_similarity(source, positive, dim=-1) - F.cosine_similarity(
        source, negative, dim=-1
    )


def curvature_residual_counterfactual(
    embeddings: torch.Tensor, device: torch.device
) -> None:
    checkpoint_path = Path(experiment.RQVAE_CKPT_PATH)
    require(checkpoint_path.is_file(), f"Parent checkpoint missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["state_dict"]

    parent_config = training._tokenizer_config()
    parent_config.curvature_residual_encoder = False
    parent = RQVAE(parent_config, in_dim=int(embeddings.shape[1])).to(device)
    parent.load_state_dict(state, strict=True)

    curved_config = training._tokenizer_config()
    require(
        curved_config.curvature_residual_encoder,
        "Curvature residual encoder is not enabled",
    )
    curved = RQVAE(curved_config, in_dim=int(embeddings.shape[1])).to(device)
    missing, unexpected = curved.load_state_dict(state, strict=False)
    expected_missing = {
        name
        for name in curved.state_dict()
        if name.startswith("encoder.residual_weights.")
    }
    require(
        set(missing) == expected_missing and not unexpected,
        f"Parent checkpoint migration mismatch: missing={missing}, "
        f"unexpected={unexpected}",
    )
    parent.eval()
    curved.eval()

    frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = training._transition_pairs(frame)
    rng = np.random.default_rng(training.SEED)
    device_embeddings = embeddings.to(device)
    count = min(training.BATCH_SIZE_PER_RANK, len(source_ids))
    chosen = rng.choice(len(source_ids), size=count, replace=False)
    source = device_embeddings[torch.from_numpy(source_ids[chosen]).to(device)]
    positive = device_embeddings[torch.from_numpy(successor_ids[chosen]).to(device)]
    permutation = torch.from_numpy(rng.permutation(count)).to(device)

    with torch.no_grad():
        probe_inputs = torch.cat((source, positive), dim=0)
        parent_initial = parent.encoder(probe_inputs)
        curved_initial = curved.encoder(probe_inputs)
        identity_error = float((parent_initial - curved_initial).abs().max().item())
    require(
        identity_error < 2e-5,
        f"Zero residual branch changed parent encoder outputs: max_abs={identity_error}",
    )

    parent_curvatures = tuple(float(layer.curvature) for layer in parent.rq.vq_layers)
    curved_curvatures = tuple(float(layer.curvature) for layer in curved.rq.vq_layers)
    require(
        curved_curvatures == parent_curvatures
        and curved_curvatures
        == tuple(float(value) for value in training.LAYER_CURVATURES),
        "The residual encoder changed fixed per-layer curvature",
    )
    adapters = list(curved.encoder.residual_weights)
    linear_layers = sum(
        isinstance(module, torch.nn.Linear) for module in curved.encoder.mlp
    )
    require(len(adapters) == linear_layers, "Residual block count mismatch")
    require(
        all(torch.count_nonzero(parameter).item() == 0 for parameter in adapters),
        "Residual adapter weights must start at zero",
    )

    with torch.no_grad():
        parent_source = parent.encoder(source)
        parent_positive = parent.encoder(positive)
        parent_negative = parent_source[permutation]
        positive_distance = _poincare_distance_tangent_pairs(
            parent_source, parent_positive, parent_curvatures[0]
        )
        negative_distance = _poincare_distance_tangent_pairs(
            parent_source, parent_negative, parent_curvatures[0]
        )
        active = (
            positive_distance + training.BEHAVIOUR_MARGIN > negative_distance
        )
    require(active.any().item(), "Frozen probe sampled no active behavior pairs")

    def take_training_step(model: RQVAE, *, inspect_adapters: bool):
        optimizer = torch.optim.AdamW(
            model.encoder.parameters(),
            lr=training.LR,
            betas=(training.ADAMW_BETA1, training.ADAMW_BASE_BETA2),
            eps=training.ADAMW_EPS,
            weight_decay=training.WEIGHT_DECAY,
        )
        optimizer.zero_grad(set_to_none=True)
        encoded_source = model.encoder(source)
        encoded_positive = model.encoder(positive)
        encoded_negative = encoded_source[permutation]
        ranking_loss = training.behaviour_ranking_loss(
            encoded_source,
            encoded_positive,
            encoded_negative,
            curvature=training.LAYER_CURVATURES[0],
            margin=training.BEHAVIOUR_MARGIN,
        )
        loss = training.BEHAVIOUR_LOSS_WEIGHT * ranking_loss
        adapter_grad_norm = 0.0
        if inspect_adapters:
            gradients = torch.autograd.grad(
                loss, adapters, retain_graph=True, allow_unused=True
            )
            require(
                all(
                    gradient is not None
                    and torch.isfinite(gradient).all()
                    and torch.count_nonzero(gradient).item() > 0
                    for gradient in gradients
                ),
                "A curvature residual block has no finite nonzero behavior gradient",
            )
            adapter_grad_norm = float(
                torch.sqrt(
                    sum(torch.sum(gradient.detach().square()) for gradient in gradients)
                ).item()
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.encoder.parameters(), training.GRADIENT_CLIP_NORM
        )
        optimizer.step()
        with torch.no_grad():
            after_source = model.encoder(source)
            after_positive = model.encoder(positive)
            after_negative = after_source[permutation]
            margins = angular_margin(after_source, after_positive, after_negative)
        require(torch.isfinite(margins).all(), "Counterfactual margins are non-finite")
        return margins.detach(), adapter_grad_norm

    parent_before = angular_margin(parent_source, parent_positive, parent_negative)
    curved_before = angular_margin(
        curved_initial[:count], curved_initial[count:], curved_initial[:count][permutation]
    )
    require(
        torch.allclose(parent_before, curved_before, atol=2e-5, rtol=2e-5),
        "Parent and curvature-residual margins differ before the step",
    )
    parent_after, _ = take_training_step(parent, inspect_adapters=False)
    curved_after, adapter_grad_norm = take_training_step(curved, inspect_adapters=True)
    parent_delta = (parent_after - parent_before)[active].cpu().numpy()
    curved_delta = (curved_after - curved_before)[active].cpu().numpy()
    paired_gain = curved_delta - parent_delta
    bootstrap_rng = np.random.default_rng(training.SEED)
    bootstrap_indices = bootstrap_rng.integers(
        0, len(paired_gain), size=(2000, len(paired_gain))
    )
    lower, upper = np.quantile(paired_gain[bootstrap_indices].mean(axis=1), [0.025, 0.975])
    supported = float(paired_gain.mean()) > 1e-5 and float(lower) > 0.0
    print(
        "  curvature residual frozen step: "
        f"active={int(active.sum())}/{count} identity_max_abs={identity_error:.3e} "
        f"adapter_grad_norm={adapter_grad_norm:.3e}"
    )
    print(
        "  angular-margin delta: "
        f"parent={float(parent_delta.mean()):+.6e} "
        f"curved={float(curved_delta.mean()):+.6e} "
        f"paired_gain={float(paired_gain.mean()):+.6e} "
        f"bootstrap95=[{float(lower):+.6e},{float(upper):+.6e}] "
        f"COUNTERFACTUAL_SUPPORT={supported}"
    )
    require(
        tuple(float(layer.curvature) for layer in curved.rq.vq_layers)
        == curved_curvatures,
        "Fixed curvature changed during the counterfactual",
    )
    del parent, curved, checkpoint, state
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    training.configure_run(__file__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding_array = np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    all_embeddings = torch.from_numpy(embedding_array)
    torch.manual_seed(42)
    curvature_residual_counterfactual(all_embeddings, device)

    model = build_model(all_embeddings, device)
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
    require(
        not any(
            name.endswith("c_layer_scale") for name, _ in model.named_parameters()
        ),
        "No learnable curvature scale should remain in the fixed-curvature model",
    )

    with torch.no_grad():
        model.init_codebook(all_embeddings[:4096].to(device))

    # Assert the metric working point s = sqrt(c) * ||v|| that each level
    # actually quantizes at. This is the per-level quantity the mechanism is
    # about, so a silent fallback to the other pin reading fails here rather
    # than at Stage3.
    with torch.no_grad():
        probe = model.encoder(all_embeddings[:512].to(device))
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
    adapters = list(model.encoder.residual_weights)
    require_nonzero_finite_gradient(
        quant_loss, adapters, "curvature residual quantization path"
    )
    require_nonzero_finite_gradient(
        recon_loss, adapters, "curvature residual reconstruction path"
    )
    require_nonzero_finite_gradient(
        total_loss, adapters, "curvature residual total Stage2 path"
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
        "MVG PASS: fixed-curvature Möbius encoder residuals, finite/nonzero "
        "behavior/reconstruction/quantization gradients, optimizer update, "
        "checkpoint reload, per-bucket Sinkhorn assignment, "
        "TIGER-aligned 72k budget"
    )


if __name__ == "__main__":
    main()
