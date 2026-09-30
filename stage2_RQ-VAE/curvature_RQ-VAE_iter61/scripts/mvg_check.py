"""Mechanism verification for Iter61 topology-gated mixed geometry."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
from train_rqvae import (
    EMBEDDING_FILE,
    TRAIN_FILE,
    _model_config,
    build_model,
    load_shared_initialization,
    load_embeddings,
    topology_scores_from_signals,
    initialize_tiger_weights,
)
from tiger_ricci_behavior_model import _poincare_pairwise_distances


def _load_fixed_parent_inputs():
    curvatures = np.asarray(np.load(experiment.PARENT_ITEM_CURVATURES_PATH), dtype=np.float32)
    signals = np.asarray(np.load(experiment.PARENT_ITEM_SIGNALS_PATH), dtype=np.float32)
    if curvatures.ndim != 1 or signals.shape != (len(curvatures), 3):
        raise RuntimeError("Iter53 fixed geometry inputs have incompatible shapes")
    if not np.isfinite(curvatures).all() or not np.isfinite(signals).all():
        raise RuntimeError("Iter53 fixed geometry inputs contain non-finite values")
    return curvatures, signals


def _check_structure_score_and_curvature(curvatures, signals):
    means = signals.mean(axis=0, dtype=np.float64)
    scales = signals.std(axis=0, dtype=np.float64)
    scales[scales < 1e-8] = 1.0
    normalized = (signals.astype(np.float64) - means) / scales
    weights = np.asarray(
        [experiment.RICCI_WEIGHT_R, experiment.RICCI_WEIGHT_G, experiment.RICCI_WEIGHT_H],
        dtype=np.float64,
    )
    q = normalized @ weights
    expected_curvatures = (
        experiment.CURVATURE_MIN
        + (experiment.CURVATURE_MAX - experiment.CURVATURE_MIN) * expit(q)
    ).astype(np.float32)
    if not np.array_equal(curvatures, expected_curvatures):
        raise RuntimeError("Iter61 fixed c_i no longer matches Iter53's unchanged q_i -> c_i")
    scores = topology_scores_from_signals(signals)
    if not np.allclose(scores, q, rtol=1e-7, atol=1e-7):
        raise RuntimeError("Gate q_i differs from Iter53's weighted normalized structure score")
    print("[Iter61] Iter53 fixed c_i and structure score q_i identity: PASS", flush=True)
    return scores


def _check_quantization_independence(embeddings, curvatures, scores, target_ids):
    ids = torch.arange(8, dtype=torch.long)
    batch = torch.from_numpy(embeddings[ids.numpy()].copy())
    low = build_model(
        embeddings.shape[1], np.full_like(curvatures, 0.05), np.full_like(scores, -3.0)
    )
    high = build_model(
        embeddings.shape[1], np.full_like(curvatures, 1.5), np.full_like(scores, 3.0)
    )
    initialize_tiger_weights(low)
    initialize_tiger_weights(high)
    load_shared_initialization(low, embeddings, target_ids)
    load_shared_initialization(high, embeddings, target_ids)
    for name, value in low.state_dict().items():
        if name in {"item_curvatures", "item_topology_scores"}:
            continue
        if not torch.equal(value, high.state_dict()[name]):
            raise RuntimeError(f"Matched quantizers differ in state entry {name}")
    low.eval()
    high.eval()
    with torch.no_grad():
        low_recon, low_quant, low_unused, low_tokens, _ = low(batch, item_ids=ids)
        high_recon, high_quant, high_unused, high_tokens, _ = high(batch, item_ids=ids)
    if not torch.equal(low_tokens, high_tokens):
        raise RuntimeError("Euclidean TIGER assignments depend on curvature/topology inputs")
    if not torch.equal(low_quant, high_quant) or not torch.equal(low_recon, high_recon):
        raise RuntimeError("Quantization or reconstruction depends on behavior-only inputs")
    if int(low_unused) != int(high_unused):
        raise RuntimeError("Unused-code count depends on behavior-only inputs")
    print("[Iter61] Euclidean TIGER quantization independent of c_i and q_i: PASS", flush=True)


def _check_mixed_metric_and_gradients(embeddings, curvatures, scores, target_ids):
    model = build_model(embeddings.shape[1], curvatures, scores)
    initialize_tiger_weights(model)
    load_shared_initialization(model, embeddings, target_ids)
    if model.item_curvatures.requires_grad or model.item_topology_scores.requires_grad:
        raise RuntimeError("Iter53 curvature and topology score tables must remain fixed")
    if any(name in {"item_curvatures", "item_topology_scores"} for name, _ in model.named_parameters()):
        raise RuntimeError("Fixed curvature/topology tables were registered as parameters")
    trainable = {name for name, _ in model.named_parameters() if "geometry_" in name}
    if trainable != {"geometry_alpha_raw", "geometry_tau"}:
        raise RuntimeError(f"Expected exactly two trainable gate parameters; found {trainable}")

    score_order = np.argsort(scores)
    source_ids = torch.as_tensor(
        score_order[np.rint(np.asarray([0.08, 0.34, 0.66, 0.92]) * (len(scores) - 1)).astype(np.int64)],
        dtype=torch.long,
    )
    target_ids_batch = torch.as_tensor(
        score_order[np.rint(np.asarray([0.20, 0.46, 0.74, 0.99]) * (len(scores) - 1)).astype(np.int64)],
        dtype=torch.long,
    )
    if len(torch.unique(torch.cat((source_ids, target_ids_batch)))) != 8:
        raise RuntimeError("Gradient check did not select eight distinct item IDs")
    item_ids = torch.cat((source_ids, target_ids_batch))
    batch = torch.from_numpy(embeddings[item_ids.numpy()].copy())
    model.set_global_step(30_000)
    model.eval()
    reconstructed, quant_loss, _, _, behavior_loss = model(
        batch, item_ids=item_ids, behavior_ids=(source_ids, target_ids_batch)
    )
    if not torch.isfinite(behavior_loss) or not behavior_loss.requires_grad:
        raise RuntimeError("Mixed-geometry behavior loss lacks a finite autograd path")

    residuals = model.rq(model.encoder(batch), return_residuals=True)[4]
    pair_curvature = torch.sqrt(
        model.item_curvatures[source_ids][:, None]
        * model.item_curvatures[target_ids_batch][None, :]
    )
    item_gates = model._geometry_gate(model.item_topology_scores)
    pair_gate = torch.sqrt(
        item_gates[source_ids][:, None] * item_gates[target_ids_batch][None, :]
    )
    duplicate_targets = target_ids_batch[:, None].eq(target_ids_batch[None, :])
    duplicate_targets.fill_diagonal_(False)
    valid = source_ids.ne(target_ids_batch)
    labels = torch.arange(4, dtype=torch.long)
    expected_losses = []
    first_level_mixed = None
    first_level_euclidean = None
    first_level_hyperbolic = None
    for level in range(model.rq.codebook_num):
        source = residuals[level, :4]
        candidates = residuals[level, 4:]
        euclidean = torch.cdist(source, candidates, p=2)
        hyperbolic = _poincare_pairwise_distances(source, candidates, pair_curvature)
        mixed = (1.0 - pair_gate) * euclidean + pair_gate * hyperbolic
        if not torch.isfinite(mixed).all():
            raise RuntimeError("Specified topology-gated pair distances are not finite")
        if level == 0:
            first_level_mixed = mixed
            first_level_euclidean = euclidean
            first_level_hyperbolic = hyperbolic
        logits = -mixed / model.behavior_temperature
        logits = logits.masked_fill(duplicate_targets, -torch.inf)
        expected_losses.append(torch.nn.functional.cross_entropy(logits[valid], labels[valid]))
    expected_behavior_loss = torch.stack(expected_losses).mean()
    if not torch.allclose(behavior_loss, expected_behavior_loss, rtol=1e-5, atol=1e-5):
        raise RuntimeError("Model behavior loss does not implement the topology-gated distance equation")
    if torch.allclose(first_level_mixed, first_level_euclidean) or torch.allclose(
        first_level_mixed, first_level_hyperbolic
    ):
        raise RuntimeError("Mixed distance collapsed to a single geometry at initialization")
    if not bool(((item_gates > 0.0) & (item_gates < 1.0)).all()):
        raise RuntimeError("Gate values must be strictly between zero and one")

    gate_parameters = (model.geometry_alpha_raw, model.geometry_tau)
    gate_gradients = torch.autograd.grad(
        behavior_loss, gate_parameters, retain_graph=True, allow_unused=True
    )
    if not all(
        grad is not None and torch.isfinite(grad).all() and bool((grad != 0).any())
        for grad in gate_gradients
    ):
        raise RuntimeError("Both alpha and tau must receive finite nonzero behavior gradients")
    for name, loss_item in (("quantization", quant_loss), ("behavior", behavior_loss)):
        if not torch.isfinite(loss_item) or not loss_item.requires_grad or loss_item.grad_fn is None:
            raise RuntimeError(f"{name} loss lacks a finite autograd path")
        grads = torch.autograd.grad(
            loss_item, tuple(model.parameters()), retain_graph=True, allow_unused=True
        )
        if not any(
            grad is not None and torch.isfinite(grad).all() and bool((grad != 0).any())
            for grad in grads
        ):
            raise RuntimeError(f"{name} loss has no nonzero model-parameter gradient")
    total_loss = torch.nn.functional.mse_loss(reconstructed, batch) + quant_loss + (
        model.get_behavior_weight() * behavior_loss
    )
    if not total_loss.requires_grad or total_loss.grad_fn is None or not torch.isfinite(total_loss):
        raise RuntimeError("Total Stage2 loss has no finite autograd path")
    total_loss.backward()
    if any(parameter.grad is None for parameter in gate_parameters):
        raise RuntimeError("Gate parameters are unused by early-ramp DDP total loss")
    diagnostics = model.gate_diagnostics()
    if diagnostics["gate_max"] - diagnostics["gate_min"] < 1e-3:
        raise RuntimeError("Initial item gates have no measurable topology-dependent variation")
    print(
        "[Iter61] mixed metric, alpha/tau gradients, early-ramp DDP graph, and gate spread: PASS "
        f"(alpha={diagnostics['geometry_alpha']:.6f}, tau={diagnostics['geometry_tau']:.6f}, "
        f"gate range={diagnostics['gate_min']:.6f}..{diagnostics['gate_max']:.6f})",
        flush=True,
    )


def main() -> None:
    curvatures, signals = _load_fixed_parent_inputs()
    scores = _check_structure_score_and_curvature(curvatures, signals)
    embeddings = load_embeddings(EMBEDDING_FILE)
    if len(embeddings) != len(curvatures):
        raise RuntimeError("Iter53 fixed inputs do not align with Stage1 embeddings")
    frame = pd.read_parquet(TRAIN_FILE)
    target_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    _check_quantization_independence(embeddings, curvatures, scores, target_ids)
    _check_mixed_metric_and_gradients(embeddings, curvatures, scores, target_ids)


if __name__ == "__main__":
    main()
