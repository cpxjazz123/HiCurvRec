"""Build fixed train-graph curvatures and verify Iter55 pair semantics and gradients."""
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
    print("[Iter55] exact toy-graph transport: PASS", flush=True)


def _check_pair_semantics(frame: pd.DataFrame) -> None:
    toy = pd.DataFrame(
        {
            "history": [[0, 1], [2], [3, 4, 5], [], None],
            "target": [6, 7, 8, 9, 10],
        }
    )
    expected = torch.tensor([[0, 6], [2, 7], [4, 8]], dtype=torch.long)
    actual = TransitionDataset(toy).pairs
    if not torch.equal(actual, expected):
        raise RuntimeError(f"Two-step/fallback pair semantics mismatch: {actual.tolist()}")

    dataset = TransitionDataset(frame)
    valid_count = 0
    two_step_count = 0
    fallback_count = 0
    for history in frame["history"]:
        if history is None or not isinstance(history, (list, tuple, np.ndarray)):
            continue
        if len(history) == 0:
            continue
        valid_count += 1
        if len(history) >= 2:
            two_step_count += 1
        else:
            fallback_count += 1
    if len(dataset) != valid_count:
        raise RuntimeError(
            f"Pair count changed: dataset={len(dataset)}, nonempty histories={valid_count}"
        )
    if two_step_count + fallback_count != len(dataset):
        raise RuntimeError("Two-step and single-step fallback counts do not cover the dataset")
    print(
        f"[Iter55] two-step pairs={two_step_count}, single-step fallback={fallback_count}, "
        f"total={len(dataset)}; pair semantics and coverage: PASS",
        flush=True,
    )


def _check_signals_match_iter53(curvatures: np.ndarray, signals: np.ndarray) -> None:
    """The selected fixed FCCR-1 curvature table must match Iter53 exactly."""
    parent_dir = SOURCE_DIR.parents[1] / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter53"
    parent_curvatures = np.asarray(
        np.load(parent_dir / "item_curvatures.npy"), dtype=np.float32
    )
    parent_signals = np.asarray(
        np.load(parent_dir / "item_behavior_ricci_signals.npy"), dtype=np.float32
    )
    if parent_curvatures.shape != curvatures.shape or parent_signals.shape != signals.shape:
        raise RuntimeError("Iter55 graph-signal shapes differ from the fixed Iter53 parent")
    if not np.array_equal(parent_curvatures, curvatures):
        raise RuntimeError("Iter55 curvatures differ from the fixed Iter53 parent")
    if not np.array_equal(parent_signals, signals):
        raise RuntimeError("Iter55 graph signals differ from the fixed Iter53 parent")
    print(
        f"[Iter55] fixed c_i and train-graph signals bit-identical to Iter53; "
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
    print("[Iter55] Euclidean TIGER quantization independent of c_i: PASS", flush=True)


def _gradient_gate(embeddings: np.ndarray, curvatures: np.ndarray) -> None:
    if len(embeddings) < 8:
        raise ValueError("Gradient gate needs at least eight item embeddings")
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
    source_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target_ids_batch = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    item_ids = torch.cat((source_ids, target_ids_batch))
    batch = torch.from_numpy(embeddings[item_ids.numpy()].copy())
    reconstructed, quant_loss, _, _, behavior_loss = model(
        batch, item_ids=item_ids, behavior_ids=(source_ids, target_ids_batch)
    )
    if not torch.isfinite(behavior_loss) or not behavior_loss.requires_grad:
        raise RuntimeError("Ricci-conditioned behavior loss lacks a finite autograd path")

    level = 0
    residuals = model.rq(model.encoder(batch), return_residuals=True)[4]
    source = residuals[level, :4]
    candidates = residuals[level, 4:]
    curvature_table = torch.from_numpy(np.asarray(curvatures, dtype=np.float32))
    pair_curvature = torch.sqrt(
        curvature_table[source_ids][:, None] * curvature_table[target_ids_batch][None, :]
    )
    ricci_distances = _poincare_pairwise_distances(source, candidates, pair_curvature)
    euclidean_distances = torch.cdist(source, candidates, p=2)
    if not torch.isfinite(ricci_distances).all():
        raise RuntimeError("Ricci-conditioned behavior distances are not finite")
    if torch.allclose(ricci_distances, euclidean_distances):
        raise RuntimeError("Pair curvature does not change the behavior-loss distance")
    if bool((ricci_distances <= 0).any()):
        raise RuntimeError("Poincare behavior distances must be positive")

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
        "[Iter55] shared initialization, fixed FCCR-1 curvature, Euclidean quantization, "
        "Ricci pair distance, quantization/behavior/total gradients: PASS",
        flush=True,
    )


def main() -> None:
    _check_exact_transport_example()
    embeddings = np.asarray(np.load(EMBEDDING_FILE), dtype=np.float32)
    frame = pd.read_parquet(TRAIN_FILE)
    _check_pair_semantics(frame)
    curvatures, signals = build_item_curvatures(frame, len(embeddings))
    if signals.shape != (len(embeddings), 3):
        raise RuntimeError(f"Unexpected item graph-signal shape: {signals.shape}")
    experiment.STAGE2_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(experiment.ITEM_CURVATURES_PATH, curvatures)
    np.save(experiment.ITEM_SIGNALS_PATH, signals)
    print(
        f"[Iter55] saved fixed c_i to {experiment.ITEM_CURVATURES_PATH}",
        flush=True,
    )
    _check_signals_match_iter53(curvatures, signals)
    _check_quantization_is_euclidean(embeddings)
    _gradient_gate(embeddings, curvatures)


if __name__ == "__main__":
    main()
