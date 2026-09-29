"""Build fixed train-graph curvatures and verify the Stage2 gradient path."""
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
    _model_config,
    load_shared_initialization,
    initialize_tiger_weights,
)
from tiger_ricci_model import (
    TIGERRicciBehaviorRQVAE,
    _poincare_item_code_distances,
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
    print("[BehaviorRicci] exact toy-graph transport: PASS", flush=True)


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

    first_layer = model.rq.vq_layers[0]
    probe = torch.from_numpy(embeddings[:2].copy())
    encoded = model.encoder(probe)
    c_pair = torch.tensor([experiment.CURVATURE_MIN, experiment.CURVATURE_MAX])
    geometric_distances = _poincare_item_code_distances(
        encoded, first_layer.get_code_embs(), c_pair
    )
    if torch.allclose(geometric_distances[0], geometric_distances[1]):
        raise RuntimeError("Per-item curvature does not change Poincare code distances")

    model.set_global_step(30_000)
    model.train()
    source_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target_ids_batch = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    item_ids = torch.cat((source_ids, target_ids_batch))
    batch = torch.from_numpy(embeddings[item_ids.numpy()].copy())
    reconstructed, quant_loss, _, _, behavior_loss = model(
        batch, item_ids=item_ids, behavior_ids=(source_ids, target_ids_batch)
    )
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
        "[BehaviorRicci] shared initialization, fixed curvature, Poincare geometry, "
        "quantization/behavior/total gradients: PASS",
        flush=True,
    )


def main() -> None:
    _check_exact_transport_example()
    embeddings = np.asarray(np.load(EMBEDDING_FILE), dtype=np.float32)
    frame = pd.read_parquet(TRAIN_FILE)
    curvatures, signals = build_item_curvatures(frame, len(embeddings))
    if signals.shape != (len(embeddings), 3):
        raise RuntimeError(f"Unexpected item graph-signal shape: {signals.shape}")
    experiment.STAGE2_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(experiment.ITEM_CURVATURES_PATH, curvatures)
    np.save(experiment.ITEM_SIGNALS_PATH, signals)
    print(
        f"[BehaviorRicci] saved fixed c_i to {experiment.ITEM_CURVATURES_PATH}; "
        f"range=({curvatures.min():.6f},{curvatures.max():.6f})",
        flush=True,
    )
    _gradient_gate(embeddings, curvatures)


if __name__ == "__main__":
    main()
