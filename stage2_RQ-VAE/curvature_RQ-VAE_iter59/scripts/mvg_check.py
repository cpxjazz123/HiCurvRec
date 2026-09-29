"""Build fixed graph curvature and verify Iter59 quantized-codeword gradients."""
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
    TransitionDataset,
    TRAIN_FILE,
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


def _check_graph_inputs_match_iter58(
    curvatures: np.ndarray, signals: np.ndarray
) -> None:
    """Iter58 independently verified these fixed values against Iter53."""
    parent_dir = SOURCE_DIR.parents[1] / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter58"
    parent_curvatures = np.asarray(
        np.load(parent_dir / "item_curvatures.npy"), dtype=np.float32
    )
    parent_signals = np.asarray(
        np.load(parent_dir / "item_behavior_ricci_signals.npy"), dtype=np.float32
    )
    if parent_curvatures.shape != curvatures.shape or parent_signals.shape != signals.shape:
        raise RuntimeError("Iter59 graph-signal shapes differ from the fixed parent values")
    if not np.array_equal(parent_curvatures, curvatures):
        raise RuntimeError("Iter59 curvatures differ from the fixed parent values")
    if not np.array_equal(parent_signals, signals):
        raise RuntimeError("Iter59 graph signals differ from the fixed parent values")
    print(
        f"[Iter59] fixed c_i and graph signals match Iter58/Iter53; "
        f"range=({curvatures.min():.6f},{curvatures.max():.6f}): PASS",
        flush=True,
    )

def _check_training_pairs(frame: pd.DataFrame) -> None:
    expected = np.asarray(
        [
            (int(history[-1]), int(target))
            for history, target in frame[["history", "target"]].itertuples(
                index=False, name=None
            )
            if history is not None
            and isinstance(history, (list, tuple, np.ndarray))
            and len(history)
        ],
        dtype=np.int64,
    )
    actual = TransitionDataset(frame).pairs.numpy()
    if len(actual) != 339_519 or not np.array_equal(actual, expected):
        raise RuntimeError("Iter59 must preserve all immediate-history training pairs")
    print(f"[Iter59] immediate history[-1] pairs={len(actual)}; data identity: PASS")

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
    source_ids, target_ids = batch_pairs[:, 0], batch_pairs[:, 1]
    item_ids = torch.cat((source_ids, target_ids))
    batch = torch.from_numpy(embeddings[item_ids.numpy()].copy())

    model = TIGERRicciBehaviorRQVAE(
        _model_config(),
        in_dim=embeddings.shape[1],
        item_curvatures=torch.from_numpy(curvatures),
    )
    initialize_tiger_weights(model)
    target_ids_all = np.unique(
        pd.read_parquet(TRAIN_FILE)["target"].to_numpy(dtype=np.int64)
    )
    load_shared_initialization(model, embeddings, target_ids_all)
    if model.item_curvatures.requires_grad or "item_curvatures" in dict(model.named_parameters()):
        raise RuntimeError("Curvature table must remain a fixed, nontrainable buffer")

    captured = {}
    encoder_hook = model.encoder.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("encoded", output)
    )
    quantizer_hook = model.rq.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("levels", output[4])
    )
    model.set_global_step(30_000)
    model.train()
    reconstructed, quant_loss, _, _, behavior_loss = model(
        batch, item_ids=item_ids, behavior_ids=(source_ids, target_ids)
    )
    encoder_hook.remove()
    quantizer_hook.remove()
    levels = captured["levels"]
    encoded = captured["encoded"]
    if levels.shape[:2] != (model.rq.codebook_num, len(item_ids)):
        raise RuntimeError("Expected one hard-quantized vector for each SID level")
    if not torch.isfinite(levels).all() or not behavior_loss.requires_grad:
        raise RuntimeError("Quantized-codeword behavior loss is not finite/differentiable")

    source_curvature = model._curvature_for(source_ids)
    target_curvature = model._curvature_for(target_ids)
    pair_curvature = torch.sqrt(source_curvature[:, None] * target_curvature[None, :])
    duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
    duplicate_targets.fill_diagonal_(False)
    valid = source_ids.ne(target_ids)
    labels = torch.arange(len(source_ids), dtype=torch.long)
    per_level_losses = []
    per_level_distances = []
    for level in range(model.rq.codebook_num):
        source = levels[level, : len(source_ids)]
        candidates = levels[level, len(source_ids) :]
        distances = _poincare_pairwise_distances(source, candidates, pair_curvature)
        if not torch.isfinite(distances).all() or bool((distances < 0).any()):
            raise RuntimeError("Quantized-level Poincare distances must be finite and nonnegative")
        logits = (-distances / model.behavior_temperature).masked_fill(
            duplicate_targets, -torch.inf
        )
        per_level_losses.append(torch.nn.functional.cross_entropy(logits[valid], labels[valid]))
        per_level_distances.append(distances)
    expected_loss = torch.stack(per_level_losses).mean()
    if not torch.allclose(behavior_loss, expected_loss, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Behavior loss is not the mean per-level quantized-codeword CE")

    residuals = torch.stack(
        [encoded - levels[:level].sum(dim=0) for level in range(model.rq.codebook_num)]
    )
    residual_losses = []
    for level in range(model.rq.codebook_num):
        distances = _poincare_pairwise_distances(
            residuals[level, : len(source_ids)],
            residuals[level, len(source_ids) :],
            pair_curvature,
        )
        logits = (-distances / model.behavior_temperature).masked_fill(
            duplicate_targets, -torch.inf
        )
        residual_losses.append(
            torch.nn.functional.cross_entropy(logits[valid], labels[valid])
        )
    if torch.allclose(
        behavior_loss, torch.stack(residual_losses).mean(), rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError("MVG batch does not distinguish quantized codes from residuals")

    level_grads = torch.autograd.grad(
        behavior_loss, levels, retain_graph=True, allow_unused=True
    )[0]
    if level_grads is None or not all(
        torch.isfinite(level_grads[level]).all()
        and bool((level_grads[level] != 0).any())
        for level in range(model.rq.codebook_num)
    ):
        raise RuntimeError("Behavior loss must reach every quantized SID level")
    euclidean = torch.cdist(
        levels[0, : len(source_ids)], levels[0, len(source_ids) :], p=2
    )
    if torch.allclose(per_level_distances[0], euclidean):
        raise RuntimeError("Pair curvature does not change quantized-code distances")

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
        "[Iter59] per-level hard-quantized codeword CE, three-level gradients, "
        "residual distinction, curvature, and total backward: PASS",
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
        f"[Iter59] saved fixed c_i to {experiment.ITEM_CURVATURES_PATH}",
        flush=True,
    )
    _check_graph_inputs_match_iter58(curvatures, signals)
    _check_quantization_is_euclidean(embeddings)
    _gradient_gate(embeddings, curvatures)


if __name__ == "__main__":
    main()
