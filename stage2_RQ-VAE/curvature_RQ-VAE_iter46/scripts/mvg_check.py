"""Exercise Iter46's c=1 Sinkhorn quantizer, residual, loss, and gradients."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

ITER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ITER_DIR))

import curvature_config as experiment
import train_rqvae as training
from model.layers import (
    CURVATURE,
    _hyperbolic_residual,
    _pairwise_poincare_distance_tangents,
)
from model.utils import expmap0, logmap0, mobius_add, poincare_distance


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def nonzero_finite_gradients(loss, parameters, label):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    nonzero = [
        gradient
        for gradient in gradients
        if gradient is not None and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient).item() > 0
    ]
    require(nonzero, f"{label} has no finite nonzero parameter gradient")


def main():
    require(CURVATURE == 1.0, "Strict GFC curvature must remain c=1")
    training.set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embeddings = np.load(experiment.EMBEDDING_FILE, mmap_mode="r")
    init_batch = torch.from_numpy(
        np.array(embeddings[:512], dtype=np.float32, copy=True)
    ).to(device)
    batch = init_batch[:16]

    model = training.RQVAE(
        training._tokenizer_config(), in_dim=int(embeddings.shape[1])
    ).to(device)
    training.initialize_tiger_weights(model)
    model.init_codebook(init_batch)
    require(len(model.rq.vq_layers) == 3, "Expected TIGER's three RQ levels")
    require(
        [layer.use_sk for layer in model.rq.vq_layers] == [True, True, True],
        "Geodesic Sinkhorn must be enabled on all three levels",
    )
    require(
        [layer.sk_epsilon for layer in model.rq.vq_layers] == [0.003] * 3
        and [layer.sk_iters for layer in model.rq.vq_layers] == [50] * 3,
        "Sinkhorn configuration must remain TIGER epsilon/iterations",
    )
    require(model.rq.vq_layers[-1].sk_epsilon == 0.003, "Sinkhorn epsilon changed")

    tokens, assignment_stats = model.get_indices_with_stats(init_batch)
    require(tokens.shape == (len(init_batch), 3), "Sinkhorn SID export shape changed")
    for level, stats in enumerate(assignment_stats):
        usage = stats["usage_counts"]
        entropy = stats["assignment_entropy_nats"]
        require(
            int(usage.sum().item()) == len(init_batch),
            f"Level {level} usage counts do not cover the input batch",
        )
        require(
            torch.isfinite(entropy) and 0.0 <= float(entropy) <= np.log(256) + 1e-6,
            f"Level {level} assignment entropy is invalid",
        )

    with tempfile.TemporaryDirectory(prefix="iter46_gfc_mvg_") as temp_dir:
        checkpoint = Path(temp_dir) / "initialized_model.pth"
        torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    tangent_x = torch.tensor([[0.2, -0.1], [0.35, 0.05]], device=device)
    tangent_q = torch.tensor([[-0.1, 0.15], [0.1, 0.2], [0.3, -0.25]], device=device)
    pairwise = _pairwise_poincare_distance_tangents(tangent_x, tangent_q)
    reference = torch.empty_like(pairwise)
    for i in range(len(tangent_x)):
        for j in range(len(tangent_q)):
            reference[i, j] = poincare_distance(
                expmap0(tangent_x[i], 1.0),
                expmap0(tangent_q[j], 1.0),
                1.0,
            ).squeeze()
    require(torch.allclose(pairwise, reference, atol=1e-6, rtol=1e-6),
            "Pairwise assignment distance disagrees with TIGER Poincare formula")

    residual = torch.tensor([[0.3, -0.2], [-0.15, 0.25]], device=device)
    code = torch.tensor([[0.1, 0.05], [0.2, -0.1]], device=device)
    residual_actual = _hyperbolic_residual(residual, code)
    residual_reference = logmap0(
        mobius_add(
            -expmap0(code, 1.0), expmap0(residual, 1.0), 1.0
        ),
        1.0,
    )
    require(
        torch.allclose(residual_actual, residual_reference, atol=1e-6, rtol=1e-6),
        "Möbius residual disagrees with the specified log0((-exp0(q))⊕exp0(r))",
    )

    reconstructed, quant_loss, _, tokens = model(batch)
    total_loss, recon_loss = model.compute_loss(batch, reconstructed, quant_loss)
    require(tokens.shape == (len(batch), 3), "Expected three TIGER SID levels")
    require(torch.isfinite(total_loss), "Total loss is non-finite")
    require(total_loss.requires_grad and total_loss.grad_fn is not None,
            "Total loss is detached")
    require(
        torch.allclose(recon_loss, F.mse_loss(reconstructed, batch)),
        "TIGER Euclidean reconstruction MSE changed",
    )
    parameters = list(model.parameters())
    nonzero_finite_gradients(quant_loss, parameters, "Poincare quantization loss")
    nonzero_finite_gradients(recon_loss, parameters, "TIGER MSE reconstruction loss")
    total_loss.backward()
    require(
        any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in parameters
        ),
        "Total loss backward produced no finite nonzero gradient",
    )
    require(torch.isfinite(reconstructed).all(), "Reconstruction contains non-finite values")
    require(torch.isfinite(quant_loss), "Quantization loss contains non-finite values")
    print(
        "MVG PASS: c=1 Poincare Sinkhorn on all three levels, residual, "
        "Euclidean reconstruction MSE, checkpoint reload, and nonzero gradients"
    )


if __name__ == "__main__":
    main()
