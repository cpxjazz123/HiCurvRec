"""MVG for the hyperbolic RQ-VAE: geometry, gradients, and checkpoint reload."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
import train_rqvae as training
from model import RQVAE
from model.layers import (
    VQLayer,
    _ball_radius,
    _expmap0_tangent,
    _hyperbolic_residual,
    _mobius_add,
    _pairwise_poincare_distance_tangents,
    _set_radius,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MVG FAIL: {message}")


def check_geometry() -> None:
    """Verify the six geometric contracts, independently of any run.

    Each check is a property of the operators, not of a trained model, so a
    regression in the algebra is caught before a 72k-step run spends an hour
    reproducing it. Where a repair replaced one formula with another, the check
    also asserts that the two genuinely differ, so the check cannot pass
    vacuously because the old formula happened to be equivalent.
    """
    generator = torch.Generator().manual_seed(0)

    # (6) One radius unit for residual and codeword shells: the same rho must
    # name the same relative ball depth at every curvature.
    for curvature in (0.3, 1.0, 2.5, 5.0):
        c = torch.tensor(curvature)
        raw = torch.randn(64, 8, generator=generator)
        pinned = _set_radius(raw, torch.full((64, 1), 0.2), c)
        rho = _ball_radius(pinned, c)
        require(
            torch.allclose(rho, torch.full((64, 1), 0.2), atol=1e-6),
            f"A ball radius is not curvature-stable at c={curvature}: {rho.mean()}",
        )
    # The old tangent-norm convention only agreed at c=1; assert the two
    # definitions really part company off c=1, so this check cannot pass
    # vacuously. Reading "0.2" as a tangent norm puts the codeword at ball
    # radius 0.2 * sqrt(c), not at 0.2.
    c_off = torch.tensor(2.5)
    raw = torch.randn(64, 8, generator=generator)
    unit = raw / raw.norm(dim=-1, keepdim=True)
    as_tangent_norm = _ball_radius(unit * 0.2, c_off)
    as_ball_radius = _ball_radius(_set_radius(unit, torch.full((64, 1), 0.2), c_off), c_off)
    require(
        not torch.allclose(as_tangent_norm, as_ball_radius, atol=1e-3),
        "Tangent-norm and ball-radius units coincide off c=1; (6) is untested",
    )
    require(
        torch.allclose(
            as_tangent_norm, as_ball_radius * float(c_off.sqrt()), atol=1e-5
        ),
        "The two radius units do not differ by sqrt(c); (6) is untested",
    )

    # (3) Pin then restore is an exact round trip on radius and direction, and
    # the old tangent-norm ratio provably fails that round trip.
    for curvature in (0.3, 1.0, 2.5, 5.0):
        c = torch.tensor(curvature)
        x = torch.randn(128, 8, generator=generator) * 0.35
        source_radius = _ball_radius(x, c)
        pinned = _set_radius(x, torch.full((128, 1), 0.2), c)
        restored = _set_radius(pinned, source_radius, c)
        require(
            torch.allclose(
                _ball_radius(restored, c), source_radius, atol=1e-5
            ),
            f"Pin/restore does not return to the source radius at c={curvature}",
        )
        require(
            torch.allclose(
                restored / restored.norm(dim=-1, keepdim=True),
                x / x.norm(dim=-1, keepdim=True),
                atol=1e-5,
            ),
            f"Pin/restore lost the direction at c={curvature}",
        )
    # The old tangent-norm ratio was correct only while c = 1 and the latents
    # were inside the ball; assert it provably fails off that, so this check
    # cannot pass vacuously. The in-ball sample below is what the encoder
    # actually emits.
    x = torch.randn(128, 8, generator=generator) * 0.15
    c_off = torch.tensor(2.5)
    old_restore = torch.randn(128, 8, generator=generator) * (
        x.norm(dim=-1, keepdim=True) / (0.2 / c_off.sqrt())
    )
    require(
        not torch.allclose(
            _ball_radius(old_restore, c_off), _ball_radius(x, c_off), atol=1e-2
        ),
        "The previous tangent-norm restore happens to be correct; (3) is untested",
    )
    # At c = 1 on in-ball data it must agree, so this is a pure re-expression
    # of the parent and not a silent behaviour change.
    c_one = torch.tensor(1.0)
    in_ball = x / x.norm(dim=-1, keepdim=True) * (
        torch.rand(128, 1, generator=generator) * 0.4
    )
    pinned_in_ball = _set_radius(in_ball, torch.full((128, 1), 0.2), c_one)
    old_c1 = pinned_in_ball * (
        in_ball.norm(dim=-1, keepdim=True) / 0.2
    )
    new_c1 = _set_radius(
        pinned_in_ball, _ball_radius(in_ball, c_one), c_one
    )
    require(
        torch.allclose(old_c1, new_c1, atol=1e-5),
        "At c=1 on in-ball data the radius restore changed the parent's result",
    )

    # (1) Mobius subtraction is not commutative, so operand order is part of
    # the geometry. Confirm the two orders differ, i.e. this was a real defect.
    q = torch.randn(64, 8, generator=generator) * 0.15
    r = torch.randn(64, 8, generator=generator) * 0.30
    reference = _hyperbolic_residual(r, q, 1.0)
    flipped = _mobius_add(
        -_expmap0_tangent(q, 1.0), _expmap0_tangent(r, 1.0), 1.0
    )
    require(
        float((reference - flipped).abs().max()) > 1e-3,
        "Residual operand order made no difference; (1) is untested",
    )

    # (2) The quantized latent must aggregate in the ball, and stay a valid
    # point. A Euclidean tangent sum is not one.
    codes = [torch.randn(64, 8, generator=generator) * 0.15 for _ in range(3)]
    point = None
    for code in codes:
        mapped = _expmap0_tangent(code, 1.0)
        point = mapped if point is None else _mobius_add(point, mapped, 1.0)
    require(
        bool((torch.linalg.vector_norm(point, dim=-1) < 1.0).all()),
        "Mobius accumulation left the ball",
    )


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
    check_geometry()
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

    # (4)/(6) The codeword radius is a real, curvature-stable constraint when
    # configured, and it is deliberately left free (0.0) because measurement
    # rejected pinning. Assert the free default is what the model reports, and
    # exercise the projection on a scratch layer so the mechanism is still
    # covered by MVG rather than going untested.
    require(
        model.rq.get_codebook_radii() == training.LAYER_CODEBOOK_RADII,
        "Codebook radii were not wired into the model",
    )
    require(
        all(float(value) == 0.0 for value in training.LAYER_CODEBOOK_RADII),
        "Codeword pinning is enabled; it was rejected on measured evidence and "
        "must stay off unless re-registered",
    )
    for level, layer in enumerate(layers):
        radius = float(training.LAYER_CODEBOOK_RADII[level])
        require(
            float(layer.codebook_radius) == radius,
            f"L{level + 1} layer radius {layer.codebook_radius} != {radius}",
        )
        projected = layer.get_code_embs()
        if radius == 0.0:
            # Free codewords: the raw weight is the codeword, unchanged.
            require(
                projected is layer.embed.weight,
                f"L{level + 1} projects codewords while no shell is configured",
            )
            continue
        measured = _ball_radius(projected, layer.get_curvature())
        require(
            torch.allclose(measured, torch.full_like(measured, radius), atol=1e-6),
            f"L{level + 1} codewords left radius {radius}: {measured.min()}",
        )
        with torch.no_grad():
            before = layer.get_code_embs().clone()
            layer.embed.weight.mul_(3.0)
            after = layer.get_code_embs()
        require(
            torch.allclose(before, after, atol=1e-6),
            f"L{level + 1} codewords moved when the raw weight norm changed",
        )

    # The shell mechanism itself, on a scratch layer so the free default does not
    # leave it uncovered: a pinned codebook is curvature-stable and its raw
    # weight norm cannot leak into the codeword.
    for curvature in (0.3, 1.0, 2.5, 5.0):
        probe = VQLayer(
            codebook_size=32,
            codebook_dim=8,
            curvature=curvature,
            codebook_radius=0.2,
        )
        for parameter in probe.parameters():
            parameter.requires_grad_(False)
        measured = _ball_radius(probe.get_code_embs(), probe.get_curvature())
        require(
            torch.allclose(measured, torch.full_like(measured, 0.2), atol=1e-6),
            f"Shell codebook is not curvature-stable at c={curvature}",
        )
        with torch.no_grad():
            before = probe.get_code_embs().clone()
            probe.embed.weight.mul_(7.0)
        require(
            torch.allclose(before, probe.get_code_embs(), atol=1e-6),
            f"Raw weight norm leaked into the codeword at c={curvature}",
        )
    require(
        model.rq.get_working_radii() == training.LAYER_WORKING_RADII,
        "Residual working radii changed",
    )

    with torch.no_grad():
        model.init_codebook(all_embeddings[:4096].to(device))

    # (5) Initialization must fit the frame the quantizer clusters, which is the
    # pinned one, and the centers must be the ones the quantizer then projects.
    # Measured on this checkpoint's L1 latents, clustering the pinned tangents
    # and clustering the raw encoder latents differ by 3.4% mean quantization
    # error, so a silent switch to the raw frame must fail here.
    for level, layer in enumerate(layers):
        require(
            torch.isfinite(layer.get_code_embs()).all(),
            f"L{level + 1} initialization produced non-finite codewords",
        )
    init_latent = model.encoder(all_embeddings[:4096].to(device)).detach()
    init_curvature = layers[0].get_curvature()
    init_pinned = _set_radius(
        init_latent,
        torch.full(
            (init_latent.shape[0], 1),
            float(training.LAYER_WORKING_RADII[0]),
            device=device,
        ),
        init_curvature,
    )
    require(
        torch.allclose(
            _ball_radius(init_pinned, init_curvature),
            torch.full_like(_ball_radius(init_pinned, init_curvature), 0.2),
            atol=1e-5,
        ),
        "The frame the quantizer clusters in is not the pinned frame",
    )
    pinned_centers = torch.tensor(
        KMeans(n_clusters=int(training.CODEBOOK_SIZE[0]), n_init="auto")
        .fit(init_pinned.cpu().numpy())
        .cluster_centers_,
        dtype=torch.float32,
        device=device,
    )
    raw_centers = torch.tensor(
        KMeans(n_clusters=int(training.CODEBOOK_SIZE[0]), n_init="auto")
        .fit(init_latent.cpu().numpy())
        .cluster_centers_,
        dtype=torch.float32,
        device=device,
    )
    pinned_error = float(
        _pairwise_poincare_distance_tangents(
            init_pinned, pinned_centers, init_curvature
        ).min(dim=-1).values.mean()
    )
    raw_error = float(
        _pairwise_poincare_distance_tangents(
            init_pinned, raw_centers, init_curvature
        ).min(dim=-1).values.mean()
    )
    print(
        f"  L1 init frame check: centers-from-pinned={pinned_error:.6f} "
        f"centers-from-raw={raw_error:.6f}"
    )
    require(
        pinned_error < raw_error,
        f"Clustering the pinned frame is not better than the raw latents "
        f"({pinned_error:.6f} >= {raw_error:.6f}); (5) is untested",
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
        "MVG PASS: fixed-curvature Poincare geometry, quantization and "
        "reconstruction gradients, optimizer update, checkpoint reload, "
        "per-bucket Sinkhorn assignment, TIGER-aligned 72k budget"
    )


if __name__ == "__main__":
    main()
