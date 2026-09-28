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
from model.layers import _hyperbolic_residual, _pairwise_poincare_distance_tangents
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
    layers = model.rq.vq_layers
    require(len(layers) == 3, "Expected TIGER's three RQ levels")
    require(
        [layer.use_sk for layer in layers] == [True, True, True],
        "Geodesic Sinkhorn must be enabled on all three levels",
    )
    require(
        [layer.sk_epsilon for layer in layers] == [0.003] * 3
        and [layer.sk_iters for layer in layers] == [50] * 3,
        "Sinkhorn configuration must match Iter46",
    )
    require(
        [layer.c_layer_norm for layer in layers]
        == list(training.LAYER_CURVATURE_NORMS),
        "Iter18 residual-calibrated layer scales changed",
    )

    model.set_curriculum_step(0)
    c_start = model.get_curvatures().detach()
    model.set_curriculum_step(training.C_CYCLIC_PERIOD // 2)
    c_peak = model.get_curvatures().detach()
    model.set_curriculum_step(training.C_CYCLIC_PERIOD)
    c_end = model.get_curvatures().detach()
    require(torch.all(c_peak > c_start), "Cyclic curvature did not rise at half-period")
    require(
        torch.allclose(c_start, c_end, atol=1e-6, rtol=1e-6),
        "Cyclic curvature did not return at the configured period",
    )
    require(
        torch.isfinite(c_peak).all()
        and torch.all(c_peak > 0)
        and torch.all(c_peak <= training.C_CYCLIC_MAX + 1e-6),
        "Cyclic layer curvatures are invalid",
    )

    optimizer, initial_curvatures, curvature_u, layer_beta2 = (
        training.build_curvature_conditioned_adamw(model)
    )
    groups = {
        group["curvature_layer_index"]: group["betas"][1]
        for group in optimizer.param_groups
        if group["curvature_layer_index"] is not None
    }
    require(
        sorted(groups) == [0, 1, 2]
        and np.allclose(
            [groups[index] for index in range(3)], layer_beta2, atol=1e-12
        ),
        "AdamW beta2 groups do not match each layer's initial curvature",
    )
    require(
        len(initial_curvatures) == len(curvature_u) == len(layer_beta2) == 3,
        "Curvature-conditioned optimizer metadata has the wrong layer count",
    )

    model.set_curriculum_step(10_000)
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
            torch.isfinite(entropy)
            and 0.0 <= float(entropy) <= np.log(256) + 1e-6,
            f"Level {level} assignment entropy is invalid",
        )

    with tempfile.TemporaryDirectory(prefix="iter47_curvature_mvg_") as temp_dir:
        checkpoint = Path(temp_dir) / "initialized_model.pth"
        torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    tangent_x = torch.tensor([[0.2, -0.1], [0.35, 0.05]], device=device)
    tangent_q = torch.tensor(
        [[-0.1, 0.15], [0.1, 0.2], [0.3, -0.25]], device=device
    )
    for curvature in (0.05, 0.3, 1.0, 1.5):
        pairwise = _pairwise_poincare_distance_tangents(
            tangent_x, tangent_q, curvature
        )
        reference = torch.empty_like(pairwise)
        for i in range(len(tangent_x)):
            for j in range(len(tangent_q)):
                reference[i, j] = poincare_distance(
                    expmap0(tangent_x[i], curvature),
                    expmap0(tangent_q[j], curvature),
                    curvature,
                ).squeeze()
        require(
            torch.allclose(pairwise, reference, atol=1e-6, rtol=1e-6),
            f"Pairwise Poincare distance failed at c={curvature}",
        )

        residual = torch.tensor([[0.3, -0.2], [-0.15, 0.25]], device=device)
        code = torch.tensor([[0.1, 0.05], [0.2, -0.1]], device=device)
        residual_actual = _hyperbolic_residual(residual, code, curvature)
        residual_reference = logmap0(
            mobius_add(
                -expmap0(code, curvature),
                expmap0(residual, curvature),
                curvature,
            ),
            curvature,
        )
        require(
            torch.allclose(
                residual_actual, residual_reference, atol=1e-6, rtol=1e-6
            ),
            f"Möbius residual failed at c={curvature}",
        )

    model.set_curriculum_step(10_000)
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
    curvature_gradients = torch.autograd.grad(
        quant_loss,
        [layer.c_layer_scale for layer in layers],
        retain_graph=True,
        allow_unused=True,
    )
    require(
        all(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient).item() > 0
            for gradient in curvature_gradients
        ),
        "Every layer curvature must receive a finite nonzero quantization gradient",
    )
    curvature_reg_gradients = torch.autograd.grad(
        model.rq.curvature_regularization(),
        [layer.c_layer_scale for layer in layers],
        retain_graph=True,
        allow_unused=True,
    )
    require(
        any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient).item() > 0
            for gradient in curvature_reg_gradients
        ),
        "Curvature-scale regularization has no finite nonzero gradient",
    )
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
    require(torch.isfinite(reconstructed).all(), "Reconstruction is non-finite")
    require(torch.isfinite(quant_loss), "Quantization loss is non-finite")
    print(
        "MVG PASS: cyclic learnable curvature, per-layer AdamW beta2, "
        "curved Sinkhorn/residual geometry, curvature gradients, and checkpoint reload"
    )


if __name__ == "__main__":
    main()
