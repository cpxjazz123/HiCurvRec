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


def check_adaptive_controller(model, batch, device) -> None:
    """The controller must hold ``sqrt(c_l) * EMA[P50(||r_l||)]`` at target.

    The statistic the controller reads is injected directly rather than
    manufactured through the network: the hyperbolic residual is bounded by
    the codebook, so no amount of latent rescaling reproduces the drift the
    run will see. The contract under test is the update law itself, on both a
    shrinking and a growing residual scale, plus its boundedness and its
    stability when the same statistic keeps arriving.
    """
    rq = model.rq
    target = training.ADAPTIVE_TARGET_S
    levels = training.CODEBOOK_NUM
    for label, medians in (
        ("shrink", torch.tensor([1.20, 0.32, 0.21], device=device)),
        ("grow", torch.tensor([2.90, 0.034, 0.023], device=device)),
    ):
        # Let the EMA forget the seeded value before feeding the new regime.
        for _ in range(200):
            rq.observe_residual_norms(medians)
        before_c = rq.get_curvature_values()
        curvatures = rq.retune_curvatures()
        working_s = rq.effective_working_s()
        require(
            before_c != curvatures,
            f"Controller ignored a {label}ing residual scale "
            f"({before_c} -> {curvatures})",
        )
        require(
            torch.allclose(
                working_s,
                torch.full_like(working_s, target),
                atol=0.01 * target,
            ),
            f"Controller missed its target on {label}: s={working_s.tolist()} "
            f"target={target}",
        )
        require(
            all(
                training.ADAPTIVE_CURVATURE_MIN
                <= value
                <= training.ADAPTIVE_CURVATURE_MAX
                for value in curvatures
            ),
            f"Curvature escaped its configured bounds: {curvatures}",
        )
        expected = [(target / float(median)) ** 2 for median in medians]
        require(
            np.allclose(curvatures, expected, rtol=1e-4),
            f"Controller does not implement c=(s*/ema)^2 on {label}: "
            f"{curvatures} vs {expected}",
        )
        # A repeated identical statistic must not keep moving the ball.
        for _ in range(50):
            rq.observe_residual_norms(medians)
        rq.retune_curvatures()
        settled = rq.get_curvature_values()
        require(
            all(
                abs(settled[index] - curvatures[index]) <= 1e-4 * curvatures[index]
                for index in range(levels)
            ),
            f"Controller oscillates on a steady {label}ing residual: "
            f"{curvatures} -> {settled}",
        )
        print(
            f"  controller[{label}]: c={[round(v, 5) for v in curvatures]} "
            f"s={[round(v, 5) for v in rq.effective_working_s().tolist()]} "
            f"(target {target})"
        )
    # Out-of-range statistics must clamp instead of producing a degenerate ball.
    rq.retune_curvatures()
    for _ in range(300):
        rq.observe_residual_norms(torch.full((levels,), 1e-9, device=device))
    clamped = rq.retune_curvatures()
    require(
        all(
            abs(value - training.ADAPTIVE_CURVATURE_MAX)
            <= 1e-6 * training.ADAPTIVE_CURVATURE_MAX
            for value in clamped
        ),
        f"Controller failed to clamp a vanishing residual scale: {clamped}",
    )
    print(
        f"  controller[clamp]: c={[round(v, 5) for v in clamped]} "
        f"(ceiling {training.ADAPTIVE_CURVATURE_MAX})"
    )
    # The clamp probe is deliberately destructive; restore the registered seed so
    # the gradient, assignment and geometry checks below run at the real start.
    rq.set_curvatures(training.LAYER_CURVATURES)
    rq.residual_norm_ema.copy_(
        torch.tensor(
            [
                training.ADAPTIVE_TARGET_S / math.sqrt(value)
                for value in training.LAYER_CURVATURES
            ],
            device=device,
            dtype=rq.residual_norm_ema.dtype,
        )
    )
    with torch.no_grad():
        model.init_codebook(batch[:4096].to(device))


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
        np.allclose(
            [layer.curvature.item() for layer in layers],
            list(training.LAYER_CURVATURES),
            rtol=1e-6,
            atol=0.0,
        ),
        "Initial per-level curvature differs from the registered C3 seed",
    )
    require(
        model.rq.get_working_radii() == (0.0, 0.0, 0.0),
        "C3 must preserve natural residual magnitudes without pinning",
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
        "No learnable curvature scale should remain in the adaptive model",
    )
    require(
        not any(
            name.endswith("curvature") for name, _ in model.named_parameters()
        ),
        "Adaptive curvature must live in a buffer, not in the optimizer's "
        "parameters",
    )
    require(
        model.rq.target_s == training.ADAPTIVE_TARGET_S
        and model.rq.adaptive_ema_decay == training.ADAPTIVE_EMA_DECAY,
        "Controller constants differ from the registered C3 values",
    )
    require(
        training.ADAPTIVE_RETUNE_STEPS == 500
        and training.ADAPTIVE_RETUNE_STEPS % 4 == 0,
        "Retune window must be 500 synchronized global steps",
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
    check_adaptive_controller(model, batch, device)
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

    # Drive the controller the way the trainer does: several retune windows over
    # real forward passes on real data, converging on the target working point.
    with torch.no_grad():
        corpus = all_embeddings.to(device)
        model.init_codebook(corpus)
        probe = corpus[:2048]
        for _ in range(200):
            model.rq(model.encoder(probe))
            model.update_adaptive_curvature()
        settled_c = model.get_curvature_values()
        settled_s = model.rq.effective_working_s()
    require(
        torch.allclose(
            settled_s,
            torch.full_like(settled_s, training.ADAPTIVE_TARGET_S),
            atol=0.05 * training.ADAPTIVE_TARGET_S,
        ),
        f"Controller did not converge on real data: s={settled_s.tolist()} "
        f"target={training.ADAPTIVE_TARGET_S}",
    )
    print(
        f"  controller[corpus]: c={[round(v, 6) for v in settled_c]} "
        f"s={[round(v, 5) for v in settled_s.tolist()]}"
    )

    # The inherited per-bucket Sinkhorn path must still produce a complete,
    # finite full-corpus SID matrix. Usage and prefix counts are diagnostics,
    # not C3 acceptance thresholds.
    with torch.no_grad():
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
        "MVG PASS: adaptive per-level curvature converging to the target "
        "working point on real residuals, no pin, no radial loss, raw RQ "
        "hyperbolic distance, behavior ranking at c=1, finite gradients, "
        "checkpoint reload, native residual/s/rho diagnostics, 72k budget"
    )


if __name__ == "__main__":
    main()
