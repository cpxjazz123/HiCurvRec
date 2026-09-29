"""Check Iter48 cold-start, mechanism schedules, and gradient paths."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
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


def require_nonzero_finite_gradient(loss, parameters, label):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    nonzero = any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
    )
    require(nonzero, f"{label} has no finite nonzero parameter gradient")


def initialize_model(mode, embeddings, device):
    training.configure_run(mode, __file__)
    model = training.RQVAE(
        training._tokenizer_config(), in_dim=int(embeddings.shape[1])
    ).to(device)
    training.initialize_tiger_weights(model)
    model.init_codebook(embeddings[: min(4096, len(embeddings))].to(device))
    return model

def checkpoint_round_trip(model, device, name):
    with tempfile.TemporaryDirectory(prefix=f"iter48_{name}_mvg_") as temp_dir:
        checkpoint = Path(temp_dir) / "initialized_model.pth"
        torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device))


def main():
    training.set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding_array = np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    all_embeddings = torch.from_numpy(embedding_array)
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    transition_dataset = training.TransitionDataset(all_embeddings, train_frame)
    pair_count = min(128, len(transition_dataset))
    pair_indices = np.linspace(
        0, len(transition_dataset) - 1, num=pair_count, dtype=np.int64
    )
    rows = [transition_dataset[int(index)] for index in pair_indices]
    source_ids = torch.stack([row[0] for row in rows]).to(device)
    target_ids = torch.stack([row[1] for row in rows]).to(device)
    source = torch.stack([row[2] for row in rows]).to(device)
    target = torch.stack([row[3] for row in rows]).to(device)
    paired_batch = torch.cat((source, target), dim=0)

    baseline = initialize_model("A", all_embeddings, device)
    checkpoint_round_trip(baseline, device, "A")
    baseline.set_curriculum_step(0)
    require(
        torch.allclose(baseline.get_curvatures(), torch.ones(3, device=device)),
        "Iter48-A must use fixed c=1 at all levels",
    )
    require(
        baseline.rq.get_effective_epsilons() == [0.003] * 3,
        "Iter48-A Sinkhorn epsilon changed",
    )
    baseline_batch = all_embeddings[:32].to(device)
    reconstructed, quant_loss, _, baseline_tokens, behavior_loss = baseline(
        baseline_batch
    )
    baseline_total, baseline_recon = baseline.compute_loss(
        baseline_batch, reconstructed, quant_loss, behavior_loss
    )
    require(baseline_tokens.shape == (len(baseline_batch), 3), "A SID shape changed")
    require(
        baseline_total.requires_grad and baseline_total.grad_fn is not None,
        "A total loss is detached",
    )
    require(torch.isfinite(baseline_total), "A total loss is non-finite")
    require(
        torch.allclose(baseline_recon, F.mse_loss(reconstructed, baseline_batch)),
        "TIGER reconstruction loss changed",
    )
    baseline_parameters = [
        parameter for parameter in baseline.parameters() if parameter.requires_grad
    ]
    require_nonzero_finite_gradient(
        quant_loss, baseline_parameters, "A quantization loss"
    )
    baseline_total.backward()
    require(
        any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in baseline_parameters
        ),
        "A total-loss backward produced no finite nonzero gradient",
    )

    training.set_seed(42)
    curriculum = initialize_model("CURRICULUM", all_embeddings, device)
    checkpoint_round_trip(curriculum, device, "curriculum")
    layers = curriculum.rq.vq_layers
    require(len(layers) == 3, "Expected three residual quantization levels")
    require(
        [layer.sk_epsilon for layer in layers] == [0.003] * 3
        and [layer.sk_iters for layer in layers] == [50] * 3,
        "Curriculum Sinkhorn baseline changed",
    )
    require(
        [layer.c_layer_norm for layer in layers]
        == list(training.LAYER_CURVATURE_NORMS),
        "Residual-calibrated layer scales changed",
    )

    curriculum.set_curriculum_step(10_000)
    bootstrap_curvature = curriculum.get_curvatures().detach()
    require(
        torch.allclose(
            bootstrap_curvature,
            torch.full_like(bootstrap_curvature, training.C_CYCLIC_MIN),
            atol=1e-7,
        ),
        "B must hold all curvature levels at c_min",
    )
    require(curriculum.rq.get_behavior_weight() == 0.0, "B behavior loss must be off")
    require(
        curriculum.rq.get_effective_epsilons() == [0.003] * 3,
        "B must keep fixed epsilon",
    )

    curriculum.set_curriculum_step(20_000)
    require(curriculum.rq.get_curriculum_alpha() == 0.0, "C ramp start changed")
    curriculum.set_curriculum_step(30_000)
    require(
        abs(curriculum.rq.get_curriculum_alpha() - 0.5) < 1e-12
        and abs(curriculum.rq.get_behavior_weight() - 0.1) < 1e-12,
        "C curvature/behavior ramp is not halfway at 30k",
    )
    c_mid = curriculum.get_curvatures().detach()
    require(
        torch.all(c_mid > training.C_CYCLIC_MIN)
        and torch.all(c_mid <= training.C_CYCLIC_MAX),
        "C curvature interpolation is out of range",
    )
    require(
        curriculum.rq.get_effective_epsilons() == [0.003] * 3,
        "C must keep fixed epsilon",
    )

    optimizer, initial_curvatures, _, layer_beta2 = (
        training.build_curvature_conditioned_adamw(curriculum)
    )
    grouped_beta2 = {
        group["curvature_layer_index"]: group["betas"][1]
        for group in optimizer.param_groups
        if group["curvature_layer_index"] is not None
        and group.get("parameter_role") != "curvature_scale"
    }
    require(
        sorted(grouped_beta2) == [0, 1, 2]
        and np.allclose(
            [grouped_beta2[index] for index in range(3)], layer_beta2, atol=1e-12
        ),
        "Curvature-conditioned AdamW groups do not match initial c",
    )
    require(
        np.allclose(initial_curvatures, [training.C_CYCLIC_MIN] * 3, atol=1e-8),
        "Curriculum optimizer did not initialize at c_min",
    )
    curriculum.set_curriculum_step(30_000)

    reconstructed, quant_loss, _, tokens, behavior_loss = curriculum(
        paired_batch, behavior_ids=(source_ids, target_ids)
    )
    total_loss, recon_loss = curriculum.compute_loss(
        paired_batch, reconstructed, quant_loss, behavior_loss
    )
    require(tokens.shape == (len(paired_batch), 3), "Curriculum SID shape changed")
    require(
        total_loss.requires_grad and total_loss.grad_fn is not None,
        "Curriculum total loss is detached",
    )
    require(
        torch.isfinite(total_loss) and torch.isfinite(behavior_loss),
        "Curriculum loss is non-finite",
    )
    parameters = list(curriculum.parameters())
    scale_parameters = [layer.c_layer_scale for layer in layers]
    require_nonzero_finite_gradient(quant_loss, parameters, "quantization/commitment loss")
    require_nonzero_finite_gradient(behavior_loss, parameters, "behavior contrastive loss")
    require_nonzero_finite_gradient(
        curriculum.rq.curvature_regularization(), scale_parameters,
        "curvature-scale regularization",
    )
    scale_gradients = torch.autograd.grad(
        behavior_loss, scale_parameters, retain_graph=True, allow_unused=True
    )
    require(
        any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient).item() > 0
            for gradient in scale_gradients
        ),
        "Behavior loss does not differentiate through curvature",
    )
    total_loss.backward()
    require(
        any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in parameters
        ),
        "Curriculum total-loss backward produced no finite nonzero gradient",
    )

    curriculum.set_curriculum_step(40_000)
    require(curriculum.rq.get_behavior_weight() == 0.2, "C behavior weight endpoint changed")
    require(
        curriculum.rq.get_effective_epsilons() == [0.003] * 3,
        "D epsilon ramp must start at the fixed base value",
    )
    curriculum.set_curriculum_step(50_000)
    epsilon_mid = curriculum.rq.get_effective_epsilons()
    for layer, epsilon in zip(layers, epsilon_mid):
        conditioned = max(
            0.003 * float(layer.get_curvature().detach().item())
            / training.C_CYCLIC_MAX,
            1e-4,
        )
        require(
            abs(epsilon - (0.003 + conditioned) / 2.0) < 1e-9,
            "D epsilon interpolation changed",
        )
    curriculum.set_curriculum_step(60_000)
    epsilon_full = curriculum.rq.get_effective_epsilons()
    for layer, epsilon in zip(layers, epsilon_full):
        require(
            abs(
                epsilon
                - max(
                    0.003 * float(layer.get_curvature().detach().item())
                    / training.C_CYCLIC_MAX,
                    1e-4,
                )
            ) < 1e-9,
            "Epsilon endpoint is not curvature-conditioned",
        )
    curriculum.set_curriculum_step(100_000)
    tokens, assignment_stats = curriculum.get_indices_with_stats(
        all_embeddings[:512].to(device)
    )
    require(tokens.shape == (512, 3), "Final curriculum SID assignment shape changed")
    for level, stats in enumerate(assignment_stats):
        require(
            torch.isfinite(stats["usage_counts"]).all()
            and int(stats["usage_counts"].sum().item()) == 512,
            f"Level {level} stabilized Sinkhorn assignment is invalid",
        )
        require(
            torch.isfinite(stats["assignment_entropy_nats"]),
            f"Level {level} assignment entropy is non-finite",
        )
        require(
            float(stats["effective_epsilon"]) >= 1e-4,
            f"Level {level} epsilon violated its floor",
        )

    side_arms = (
        ("BASELINE", False, False, 1.0),
        ("CURVATURE_ONLY", True, False, training.C_CYCLIC_MIN),
        ("BEHAVIOR_ONLY", False, True, 1.0),
    )
    for mode, curvature_enabled, behavior_enabled, fixed_curvature in side_arms:
        training.set_seed(42)
        arm = initialize_model(mode, all_embeddings, device)
        checkpoint_round_trip(arm, device, mode.lower())
        layers = arm.rq.vq_layers
        require(training.MAX_GLOBAL_STEPS == 40_000, f"{mode} step budget changed")
        require(
            all(layer.curriculum_enabled is curvature_enabled for layer in layers)
            and all(
                layer.behavior_curriculum_enabled is behavior_enabled
                for layer in layers
            ),
            f"{mode} mechanism toggles do not match the registered arm",
        )
        require(
            [layer.sk_epsilon for layer in layers] == [0.003] * 3
            and [layer.sk_iters for layer in layers] == [50] * 3,
            f"{mode} Sinkhorn settings changed",
        )
        arm.set_curriculum_step(10_000)
        require(
            torch.allclose(
                arm.get_curvatures(),
                torch.full((3,), fixed_curvature, device=device),
                atol=1e-7,
            ),
            f"{mode} initial curvature changed",
        )
        require(
            arm.rq.get_effective_epsilons() == [0.003] * 3,
            f"{mode} must use fixed Sinkhorn epsilon",
        )
        arm.set_curriculum_step(30_000)
        require(
            abs(
                arm.rq.get_curriculum_alpha()
                - (0.5 if mode == "CURVATURE_ONLY" else 0.0)
            ) < 1e-12,
            f"{mode} curvature schedule is not isolated",
        )
        expected_behavior_weight = 0.1 if mode == "BEHAVIOR_ONLY" else 0.0
        require(
            abs(arm.rq.get_behavior_weight() - expected_behavior_weight) < 1e-12,
            f"{mode} behavior schedule is not isolated",
        )
        current_curvatures = arm.get_curvatures().detach()
        if mode == "CURVATURE_ONLY":
            require(
                torch.all(current_curvatures > training.C_CYCLIC_MIN),
                "Curvature-only arm failed to activate the C curvature ramp",
            )
            require_nonzero_finite_gradient(
                arm.rq.curvature_regularization(),
                [layer.c_layer_scale for layer in layers],
                "curvature-only regularization",
            )
        else:
            require(
                torch.allclose(
                    current_curvatures,
                    torch.ones_like(current_curvatures),
                    atol=1e-7,
                ),
                f"{mode} must keep baseline curvature fixed",
            )

        reconstructed, quant_loss, _, _, behavior_loss = arm(
            paired_batch, behavior_ids=(source_ids, target_ids)
        )
        total_loss, _ = arm.compute_loss(
            paired_batch, reconstructed, quant_loss, behavior_loss
        )
        require(
            total_loss.requires_grad and total_loss.grad_fn is not None,
            f"{mode} total loss is detached",
        )
        require(
            torch.isfinite(total_loss) and torch.isfinite(behavior_loss),
            f"{mode} loss is non-finite",
        )
        trainable_parameters = [
            parameter for parameter in arm.parameters() if parameter.requires_grad
        ]
        require_nonzero_finite_gradient(
            quant_loss, trainable_parameters, f"{mode} quantization loss"
        )
        if mode == "BEHAVIOR_ONLY":
            require(
                behavior_loss.item() > 0.0,
                "Behavior-only arm did not compute its active behavior loss",
            )
            require_nonzero_finite_gradient(
                behavior_loss, trainable_parameters, "behavior-only contrastive loss"
            )
            require(
                all(not layer.c_layer_scale.requires_grad for layer in layers),
                "Behavior-only arm unexpectedly trains curvature scales",
            )
        else:
            require(
                behavior_loss.item() == 0.0,
                f"{mode} behavior loss must remain zero",
            )
        total_loss.backward()
        require(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in arm.parameters()
            ),
            f"{mode} total-loss backward produced no finite nonzero gradient",
        )



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
                -expmap0(code, curvature), expmap0(residual, curvature), curvature
            ),
            curvature,
        )
        require(
            torch.allclose(residual_actual, residual_reference, atol=1e-6, rtol=1e-6),
            f"Möbius residual failed at c={curvature}",
        )

    print(
        "MVG PASS: matched baseline, curvature-only, behavior-only, and Iter48-C "
        "schedule/gradient checks; stable Sinkhorn; checkpoint reload"
    )


if __name__ == "__main__":
    main()
