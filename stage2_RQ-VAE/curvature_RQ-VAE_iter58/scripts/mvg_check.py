"""Build fixed graph curvature and verify Iter58 hierarchy-joint gradients."""
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
    build_item_curvatures,
)
from train_rqvae import (
    EMBEDDING_FILE,
    TRAIN_FILE,
    TransitionDataset,
    _model_config,
    load_shared_initialization,
    initialize_tiger_weights,
)
from tiger_ricci_behavior_model import (
    TIGERRicciBehaviorRQVAE,
    _poincare_pairwise_distances,
)


def _check_exact_transport_example() -> None:
    frame = pd.DataFrame(
        {"history": [[0], [0], [1], [2]], "target": [1, 2, 3, 4]}
    )
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
    print("[RicciBehavior] exact toy-graph transport: PASS", flush=True)


def _check_training_pairs(frame: pd.DataFrame) -> None:
    expected = np.asarray(
        [
            (int(history[-1]), int(target))
            for history, target in frame[["history", "target"]].itertuples(
                index=False, name=None
            )
            if isinstance(history, (list, tuple, np.ndarray)) and len(history)
        ],
        dtype=np.int64,
    )
    actual = TransitionDataset(frame).pairs.numpy()
    if len(actual) != 339_519 or not np.array_equal(actual, expected):
        raise RuntimeError(
            f"Iter58 must preserve all immediate-history pairs: {actual.shape}"
        )
    print(
        f"[Iter58] immediate history[-1] pairs={len(actual)}; data identity: PASS",
        flush=True,
    )


def _check_graph_inputs_match_iter53(curvatures: np.ndarray, signals: np.ndarray) -> None:
    """Keep fixed per-item geometry unchanged from the chosen parent."""
    parent_curvatures = np.asarray(
        np.load(
            SOURCE_DIR.parents[1]
            / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter53/item_curvatures.npy"
        ),
        dtype=np.float32,
    )
    parent_signals = np.asarray(
        np.load(
            SOURCE_DIR.parents[1]
            / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter53/item_behavior_ricci_signals.npy"
        ),
        dtype=np.float32,
    )
    if parent_curvatures.shape != curvatures.shape or parent_signals.shape != signals.shape:
        raise RuntimeError("Iter58 graph-input shapes differ from Iter53")
    if not np.array_equal(parent_curvatures, curvatures):
        raise RuntimeError("Iter58 curvatures differ from Iter53")
    if not np.array_equal(parent_signals, signals):
        raise RuntimeError("Iter58 graph signals differ from Iter53")
    print(
        f"[Iter58] fixed c_i and graph signals bit-identical to Iter53; "
        f"range=({curvatures.min():.6f},{curvatures.max():.6f}): PASS",
        flush=True,
    )


def _check_quantization_is_euclidean(embeddings: np.ndarray) -> None:
    """Quantization output must not depend on the curvature table."""
    config = _model_config()
    target_ids = np.unique(pd.read_parquet(TRAIN_FILE)["target"].to_numpy(dtype=np.int64))
    item_ids = torch.arange(8, dtype=torch.long)
    batch = torch.from_numpy(embeddings[item_ids.numpy()].copy())
    low = TIGERRicciBehaviorRQVAE(
        config, in_dim=embeddings.shape[1], item_curvatures=torch.full((len(embeddings),), 0.05)
    )
    high = TIGERRicciBehaviorRQVAE(
        config, in_dim=embeddings.shape[1], item_curvatures=torch.full((len(embeddings),), 1.5)
    )
    initialize_tiger_weights(low)
    initialize_tiger_weights(high)
    load_shared_initialization(low, embeddings, target_ids)
    load_shared_initialization(high, embeddings, target_ids)
    for name, value in low.state_dict().items():
        if name == "item_curvatures":
            continue
        if not torch.equal(value, high.state_dict()[name]):
            raise RuntimeError(f"Matched arms must share every parameter: {name}")
    low.eval()
    high.eval()
    with torch.no_grad():
        low_recon, low_quant, low_unused, low_tokens, _ = low(batch, item_ids=item_ids)
        high_recon, high_quant, high_unused, high_tokens, _ = high(batch, item_ids=item_ids)
    if not torch.equal(low_tokens, high_tokens):
        raise RuntimeError("Quantizer assignment changed with the curvature table")
    if not torch.equal(low_quant, high_quant):
        raise RuntimeError("Quantization loss changed with the curvature table")
    if not torch.equal(low_recon, high_recon):
        raise RuntimeError("Reconstruction changed with the curvature table")
    if int(low_unused) != int(high_unused):
        raise RuntimeError("Unused-code count changed with the curvature table")
    print(
        "[RicciBehavior] Euclidean TIGER quantization independent of c_i: PASS",
        flush=True,
    )


def _gradient_gate(embeddings: np.ndarray, curvatures: np.ndarray) -> None:
    if len(embeddings) < 16:
        raise ValueError("Gradient gate needs at least sixteen item embeddings")
    pair_dataset = TransitionDataset(pd.read_parquet(TRAIN_FILE))
    selected_pairs = []
    seen_targets = set()
    for source, target in pair_dataset.pairs.tolist():
        if source != target and target not in seen_targets:
            selected_pairs.append((source, target))
            seen_targets.add(target)
            if len(selected_pairs) == 16:
                break
    if len(selected_pairs) != 16:
        raise RuntimeError("Could not select sixteen distinct immediate-history pairs")
    batch_pairs = torch.tensor(selected_pairs, dtype=torch.long)
    source_ids, target_ids_batch = batch_pairs[:, 0], batch_pairs[:, 1]
    item_ids = torch.cat((source_ids, target_ids_batch))
    batch = torch.from_numpy(embeddings[item_ids.numpy()].copy())

    model = TIGERRicciBehaviorRQVAE(
        _model_config(),
        in_dim=embeddings.shape[1],
        item_curvatures=torch.from_numpy(curvatures),
    )
    initialize_tiger_weights(model)
    target_ids = np.unique(pd.read_parquet(TRAIN_FILE)["target"].to_numpy(dtype=np.int64))
    load_shared_initialization(model, embeddings, target_ids)
    if model.item_curvatures.requires_grad or "item_curvatures" in dict(model.named_parameters()):
        raise RuntimeError("Curvature table must remain a fixed, nontrainable buffer")

    model.set_global_step(30_000)
    model.train()
    reconstructed, quant_loss, _, _, behavior_loss = model(
        batch, item_ids=item_ids, behavior_ids=(source_ids, target_ids_batch)
    )
    if not torch.isfinite(behavior_loss) or not behavior_loss.requires_grad:
        raise RuntimeError("Hierarchy-joint behavior loss lacks a finite autograd path")

    residuals = model.rq(model.encoder(batch), return_residuals=True)[4]
    batch_size = len(source_ids)
    curvature_table = torch.from_numpy(np.asarray(curvatures, dtype=np.float32))
    pair_curvature = torch.sqrt(
        curvature_table[source_ids][:, None] * curvature_table[target_ids_batch][None, :]
    )
    level_distances = [
        _poincare_pairwise_distances(
            residuals[level, :batch_size],
            residuals[level, batch_size:],
            pair_curvature,
        )
        for level in range(model.rq.codebook_num)
    ]
    if not all(torch.isfinite(distances).all() for distances in level_distances):
        raise RuntimeError("Hierarchy-joint Poincare distances are not finite")
    if any(bool((distances <= 0).any()) for distances in level_distances):
        raise RuntimeError("Hierarchy-joint Poincare distances must be positive")
    duplicate_targets = target_ids_batch[:, None].eq(target_ids_batch[None, :])
    duplicate_targets.fill_diagonal_(False)
    valid = source_ids.ne(target_ids_batch)
    labels = torch.arange(batch_size, dtype=torch.long)

    joint_distances = torch.stack(level_distances, dim=0).mean(dim=0)
    joint_logits = (-joint_distances / model.behavior_temperature).masked_fill(
        duplicate_targets, -torch.inf
    )
    joint_loss = torch.nn.functional.cross_entropy(joint_logits[valid], labels[valid])
    actual_joint_loss = model._behavior_contrastive_loss(
        residuals, source_ids, target_ids_batch, model._curvature_for(item_ids)
    )
    if not torch.allclose(actual_joint_loss, joint_loss, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Behavior loss is not CE over the mean residual-level distance")

    independent_level_losses = []
    for distances in level_distances:
        logits = (-distances / model.behavior_temperature).masked_fill(
            duplicate_targets, -torch.inf
        )
        independent_level_losses.append(
            torch.nn.functional.cross_entropy(logits[valid], labels[valid])
        )
    independent_mean = torch.stack(independent_level_losses).mean()
    if torch.allclose(joint_loss, independent_mean, rtol=1e-6, atol=1e-6):
        raise RuntimeError("MVG batch does not distinguish joint from independent ranking")
    residual_level_grads = torch.autograd.grad(
        joint_loss, residuals, retain_graph=True, allow_unused=True
    )[0]
    if residual_level_grads is None or not all(
        torch.isfinite(residual_level_grads[level]).all()
        and bool((residual_level_grads[level] != 0).any())
        for level in range(model.rq.codebook_num)
    ):
        raise RuntimeError("Joint behavior loss must reach every residual level")

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
    if not any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool((parameter.grad != 0).any())
        for parameter in model.parameters()
    ):
        raise RuntimeError("Total Stage2 loss backward produced no nonzero gradients")
    print(
        "[Iter58] joint CE semantics, three-level residual gradients, "
        "quantization/behavior/total gradients: PASS",
        flush=True,
    )


def main() -> None:
    _check_exact_transport_example()
    embeddings = np.asarray(np.load(EMBEDDING_FILE), dtype=np.float32)
    frame = pd.read_parquet(TRAIN_FILE)
    _check_training_pairs(frame)
    curvatures, signals = build_item_curvatures(frame, len(embeddings))
    if signals.shape != (len(embeddings), 3):
        raise RuntimeError(f"Unexpected item graph-signal shape: {signals.shape}")
    experiment.STAGE2_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(experiment.ITEM_CURVATURES_PATH, curvatures)
    np.save(experiment.ITEM_SIGNALS_PATH, signals)
    print(
        f"[Iter58] saved fixed c_i to {experiment.ITEM_CURVATURES_PATH}",
        flush=True,
    )
    _check_graph_inputs_match_iter53(curvatures, signals)
    _check_quantization_is_euclidean(embeddings)
    _gradient_gate(embeddings, curvatures)


if __name__ == "__main__":
    main()
