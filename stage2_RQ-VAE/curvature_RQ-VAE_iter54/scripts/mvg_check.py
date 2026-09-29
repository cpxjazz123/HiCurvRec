"""Build Iter54 graph signals and verify controller learnability and gradient paths."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(SOURCE_DIR / "scripts"))

import curvature_config as experiment
from behavior_ricci import (
    _distance_bitsets,
    _edge_negative_curvature_contribution,
    _make_outgoing_tables,
    _transition_records,
    _undirected_graph_tables,
    build_item_signals,
)
from train_rqvae import (
    EMBEDDING_FILE,
    TRAIN_FILE,
    _model_config,
    build_curvature_features,
    load_shared_initialization,
    initialize_tiger_weights,
)
from tiger_learnable_curvature_model import (
    CurvatureController,
    TIGERLearnableCurvatureRQVAE,
    _poincare_pairwise_distances,
)

ITER53_DIR = experiment.REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter53"


def _check_exact_transport_example() -> None:
    frame = pd.DataFrame({"history": [[0], [0], [1], [2]], "target": [1, 2, 3, 4]})
    sources, targets, counts = _transition_records(frame, 5)
    outgoing, probabilities = _make_outgoing_tables(sources, targets, counts, 5)
    undirected = _undirected_graph_tables(sources, targets, 5)
    adjacency, two_hop = _distance_bitsets(undirected, 5)
    import behavior_ricci as ricci

    ricci._WORK_OUT_NEIGHBORS = outgoing
    ricci._WORK_OUT_PROBABILITIES = probabilities
    ricci._WORK_ADJACENCY_BITS = adjacency
    ricci._WORK_TWO_HOP_BITS = two_hop
    contributions = [
        _edge_negative_curvature_contribution((int(source), int(target)))
        for source, target in zip(sources, targets)
    ]
    item_zero_need = sum(value for source, value in contributions if source == 0)
    if not np.isclose(item_zero_need, 1.0, atol=1e-8):
        raise RuntimeError(f"Exact toy-graph ORC need mismatch: {item_zero_need}")
    if any(value != 0.0 for source, value in contributions if source != 0):
        raise RuntimeError("Toy graph sink transitions must have zero negative-curvature need")
    print("[LearnableCurvature] exact toy-graph transport: PASS", flush=True)


def _check_init_reproduces_iter53(features: np.ndarray) -> None:
    """The controller must start exactly at the validated Iter53 linear map."""
    reference = np.asarray(np.load(ITER53_DIR / "item_curvatures.npy"), dtype=np.float64)
    model = TIGERLearnableCurvatureRQVAE(
        _model_config(), in_dim=8, curvature_features=torch.from_numpy(features)
    )
    with torch.no_grad():
        learned = model.controller(model.curvature_features).numpy()
    gap = float(np.abs(learned - reference).max())
    if gap > 1e-4:
        raise RuntimeError(f"Controller init deviates from Iter53 curvature by {gap}")
    with torch.no_grad():
        controller = model.controller
        primary_weights = controller.effective_primary_weight()
        pairwise_gaps = torch.pdist(primary_weights)
        if float(pairwise_gaps.min()) <= 1e-6:
            raise RuntimeError("controller hidden units are symmetric at initialization")
        initial_hidden = (
            torch.from_numpy(features[:, :3]) @ primary_weights.t()
            + controller.primary_bias
            + controller.interaction_bias
        )
        if float(initial_hidden.min()) <= 0.0:
            raise RuntimeError("diverse initialization leaves the active ReLU region")
        if not torch.allclose(
            primary_weights.mean(dim=0),
            torch.full((3,), 1.0 / 3.0),
            atol=1e-7,
            rtol=0.0,
        ):
            raise RuntimeError("diverse initialization does not preserve the Iter53 linear map")
    print(
        f"[LearnableCurvature] init reproduces Iter53 linear c_i "
        f"(max abs diff={gap:.2e}, mean={learned.mean():.6f}): PASS",
        flush=True,
    )


def _check_quantization_independent_of_controller(embeddings: np.ndarray) -> None:
    """Perturbing every controller weight must not change any quantizer output."""
    target_ids = np.unique(pd.read_parquet(TRAIN_FILE)["target"].to_numpy(dtype=np.int64))
    model = TIGERLearnableCurvatureRQVAE(
        _model_config(),
        in_dim=embeddings.shape[1],
        curvature_features=torch.from_numpy(
            build_curvature_features(
                np.load(ITER53_DIR / "item_behavior_ricci_signals.npy").astype(np.float32)
            )
        ),
    )
    initialize_tiger_weights(model)
    load_shared_initialization(model, embeddings, target_ids)
    item_ids = torch.arange(8, dtype=torch.long)
    batch = torch.from_numpy(embeddings[item_ids.numpy()].copy())
    model.eval()
    with torch.no_grad():
        before = model(batch, item_ids=item_ids)
        base_tokens, base_quant, base_recon, base_unused = (
            before[3],
            before[1],
            before[0],
            before[2],
        )
        base_curvature = before[5].clone()
    with torch.no_grad():
        for parameter in model.controller.parameters():
            parameter.add_(0.37)
    with torch.no_grad():
        after = model(batch, item_ids=item_ids)
    if not torch.equal(base_tokens, after[3]):
        raise RuntimeError("Quantizer tokens changed when the controller moved")
    if not torch.equal(base_quant, after[1]):
        raise RuntimeError("Quantization loss changed when the controller moved")
    if not torch.equal(base_recon, after[0]):
        raise RuntimeError("Reconstruction changed when the controller moved")
    if int(base_unused) != int(after[2]):
        raise RuntimeError("Unused-code count changed when the controller moved")
    if torch.allclose(base_curvature, after[5]):
        raise RuntimeError("Perturbing the controller did not change c_i")
    print(
        "[LearnableCurvature] Euclidean TIGER quantization independent of controller: PASS",
        flush=True,
    )



def _check_primary_monotonicity(controller, features: np.ndarray) -> None:
    with torch.no_grad():
        if not bool((controller.effective_primary_weight() > 0).all()):
            raise RuntimeError("Effective primary-feature weights must stay positive")
        if not bool((torch.nn.functional.softplus(controller.output_weight) > 0).all()):
            raise RuntimeError("Output weights must stay positive")
    sample = features[::97][:256].copy()
    baseline = torch.from_numpy(sample)
    with torch.no_grad():
        base_curvature = controller(baseline)
    epsilon = 1e-3
    bounds = controller.primary_abs_max.cpu().numpy()
    for index in range(3):
        changed = sample.copy()
        changed[:, index] += epsilon
        if index == 0:
            changed[:, 3] += epsilon * changed[:, 1]
            changed[:, 4] += epsilon * changed[:, 2]
        elif index == 1:
            changed[:, 3] += epsilon * sample[:, 0]
            changed[:, 5] += epsilon * changed[:, 2]
        else:
            changed[:, 4] += epsilon * sample[:, 0]
            changed[:, 5] += epsilon * sample[:, 1]
        valid = np.abs(changed[:, index]) <= bounds[index]
        with torch.no_grad():
            shifted = controller(torch.from_numpy(changed[valid]))
            if bool((shifted + 1e-7 < base_curvature[valid]).any()):
                raise RuntimeError(f"Increasing primary feature {index} reduced curvature")
    print(
        "[LearnableCurvature] R/G/H monotone over observed z-score bounds: PASS",
        flush=True,
    )
def _gradient_gate(embeddings: np.ndarray, features: np.ndarray) -> None:
    model = TIGERLearnableCurvatureRQVAE(
        _model_config(), in_dim=embeddings.shape[1], curvature_features=torch.from_numpy(features)
    )
    initialize_tiger_weights(model)
    target_ids = np.unique(pd.read_parquet(TRAIN_FILE)["target"].to_numpy(dtype=np.int64))
    load_shared_initialization(model, embeddings, target_ids)
    monotone_controller = CurvatureController(
        _model_config(), torch.from_numpy(features)
    )
    with torch.no_grad():
        monotone_controller.interaction_weight.fill_(0.01)
    _check_primary_monotonicity(monotone_controller, features)
    controller_parameters = tuple(model.controller.parameters())
    if not controller_parameters:
        raise RuntimeError("Controller has no learnable parameters")
    if model.curvature_features.requires_grad:
        raise RuntimeError("Structure feature buffer must stay fixed")

    model.set_global_step(30_000)
    model.train()
    source_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target_ids_batch = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    item_ids = torch.cat((source_ids, target_ids_batch))
    batch = torch.from_numpy(embeddings[item_ids.numpy()].copy())
    reconstructed, quant_loss, _, _, behavior_loss, curvature, curvature_reg = model(
        batch, item_ids=item_ids, behavior_ids=(source_ids, target_ids_batch)
    )
    for name, loss_item in (("quantization", quant_loss), ("behavior", behavior_loss)):
        if not torch.isfinite(loss_item) or not loss_item.requires_grad:
            raise RuntimeError(f"{name} loss lacks a finite autograd path")
        grads = torch.autograd.grad(
            loss_item, tuple(model.parameters()), retain_graph=True, allow_unused=True
        )
        if not any(
            g is not None and torch.isfinite(g).all() and bool((g != 0).any()) for g in grads
        ):
            raise RuntimeError(f"{name} loss has no nonzero model-parameter gradient")

    # The behavior loss must reach the controller; that is the learning signal.
    controller_grads = torch.autograd.grad(
        model.get_behavior_weight() * behavior_loss,
        controller_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    if not any(
        g is not None and torch.isfinite(g).all() and bool((g != 0).any())
        for g in controller_grads
    ):
        raise RuntimeError("Behavior loss does not reach the controller")
    if any(
        g is not None and bool((g != 0).any())
        for g in torch.autograd.grad(quant_loss, controller_parameters, allow_unused=True)
    ):
        raise RuntimeError("Quantization loss must not reach the controller")

    reg_grads = torch.autograd.grad(
        curvature_reg, controller_parameters, retain_graph=True, allow_unused=True
    )
    if not any(g is not None and bool((g != 0).any()) for g in reg_grads):
        raise RuntimeError("Curvature regularizer does not reach the controller")

    # Primary-feature positivity must survive the softplus parameterization.
    with torch.no_grad():
        primary = torch.nn.functional.softplus(model.controller.primary_weight)
    if not bool((primary > 0).all()):
        raise RuntimeError("Primary structure weights must stay positive")

    if not torch.isfinite(curvature).all() or not torch.isfinite(curvature_reg):
        raise RuntimeError("Curvature or regularizer became non-finite")
    if bool((curvature <= 0).any()):
        raise RuntimeError("Curvature must stay positive")

    residuals = model.rq(model.encoder(batch), return_residuals=True)[4]
    level = 0
    source = residuals[level, :4]
    candidates = residuals[level, 4:]
    pair = torch.sqrt(curvature[:4][:, None] * curvature[4:][None, :])
    ricci = _poincare_pairwise_distances(source, candidates, pair)
    euclidean = torch.cdist(source, candidates, p=2)
    if not torch.isfinite(ricci).all() or bool((ricci <= 0).any()):
        raise RuntimeError("Poincare behavior distances are invalid")
    if torch.allclose(ricci, euclidean):
        raise RuntimeError("Pair curvature does not change the behavior-loss distance")

    total = (
        torch.nn.functional.mse_loss(reconstructed, batch)
        + quant_loss
        + model.get_behavior_weight() * behavior_loss
    )
    if not total.requires_grad or total.grad_fn is None or not torch.isfinite(total):
        raise RuntimeError("Total Stage2 loss has no finite autograd path")
    (total + curvature_reg).backward()
    for name, parameter in model.controller.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Controller parameter {name} received no gradient")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Controller parameter {name} has a non-finite gradient")
    print(
        "[LearnableCurvature] controller learnable, positive primaries, quantization-free, "
        "behavior/regularizer gradients, total backward: PASS",
        flush=True,
    )


def main() -> None:
    _check_exact_transport_example()
    embeddings = np.asarray(np.load(EMBEDDING_FILE), dtype=np.float32)
    frame = pd.read_parquet(TRAIN_FILE)
    signals = build_item_signals(frame, len(embeddings))
    experiment.STAGE2_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(experiment.ITEM_SIGNALS_PATH, signals.astype(np.float32))
    parent_signals = np.asarray(
        np.load(ITER53_DIR / "item_behavior_ricci_signals.npy"), dtype=np.float32
    )
    if not np.array_equal(signals.astype(np.float32), parent_signals):
        raise RuntimeError("Iter54 graph signals differ from the frozen Iter53 signals")
    print(
        "[LearnableCurvature] graph signals bit-identical to Iter53: PASS", flush=True
    )
    features = build_curvature_features(signals.astype(np.float32))
    _check_init_reproduces_iter53(features)
    _check_quantization_independent_of_controller(embeddings)
    _gradient_gate(embeddings, features)


if __name__ == "__main__":
    main()
