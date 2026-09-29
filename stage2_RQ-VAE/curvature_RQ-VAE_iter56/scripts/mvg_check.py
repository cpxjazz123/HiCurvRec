"""Build fixed train-graph curvatures and verify the Iter56 gradient path."""
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
    print("[Iter56] exact toy-graph transport: PASS", flush=True)

def _check_immediate_transition_pairs(frame: pd.DataFrame) -> None:
    dataset = TransitionDataset(frame)
    expected = []
    for history, target in frame[["history", "target"]].itertuples(
        index=False, name=None
    ):
        if history is None or not isinstance(history, (list, tuple, np.ndarray)):
            continue
        if len(history):
            expected.append((int(history[-1]), int(target)))
    expected_pairs = torch.tensor(expected, dtype=torch.long)
    if not torch.equal(dataset.pairs, expected_pairs):
        raise RuntimeError("Iter56 pair data differ from Iter53 immediate transitions")
    if len(dataset) != 339_519:
        raise RuntimeError(f"Unexpected immediate-transition pair count: {len(dataset)}")
    print(
        f"[Iter56] immediate history[-1] pairs={len(dataset)}; data identity: PASS",
        flush=True,
    )


def _check_signals_match_iter53(curvatures: np.ndarray, signals: np.ndarray) -> None:
    """The graph signals remain bit-identical to the Iter53 parent."""
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
        raise RuntimeError("Iter56 graph signal shapes differ from the Iter53 parent")
    if not np.array_equal(parent_curvatures, curvatures):
        raise RuntimeError("Iter56 curvatures differ from the Iter53 parent")
    if not np.array_equal(parent_signals, signals):
        raise RuntimeError("Iter56 graph signals differ from the Iter53 parent")
    print(
        f"[Iter56] fixed c_i and graph signals bit-identical to Iter53; "
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
        "[Iter56] Euclidean TIGER quantization independent of c_i: PASS",
        flush=True,
    )


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
    source_curvature = curvature_table[source_ids]
    ricci_distances = _poincare_pairwise_distances(source, candidates, source_curvature)
    reference_distances = torch.cat(
        [
            _poincare_pairwise_distances(
                source[row : row + 1],
                candidates,
                source_curvature[row : row + 1],
            )
            for row in range(len(source))
        ],
        dim=0,
    )
    if not torch.allclose(ricci_distances, reference_distances, rtol=1e-6, atol=1e-7):
        raise RuntimeError("Batched distances differ from per-source curvature geometry")
    euclidean_distances = torch.cdist(source, candidates, p=2)
    if not torch.isfinite(ricci_distances).all():
        raise RuntimeError("Source-anchored behavior distances are not finite")
    if torch.allclose(ricci_distances, euclidean_distances):
        raise RuntimeError("Source curvature does not change the behavior distance")
    if bool((ricci_distances <= 0).any()):
        raise RuntimeError("Source-anchored Poincare distances must be positive")
    behavior_curvatures = torch.cat(
        (source_curvature, curvature_table[target_ids_batch])
    )
    expected_behavior_loss = model._behavior_contrastive_loss(
        residuals, source_ids, target_ids_batch, behavior_curvatures
    )
    if not torch.allclose(expected_behavior_loss, behavior_loss):
        raise RuntimeError("Forward behavior loss does not use source-anchored curvature")
    candidate_curvatures_changed = behavior_curvatures.clone()
    candidate_curvatures_changed[4:] = candidate_curvatures_changed[4:] * 1.7 + 0.03
    candidate_changed_loss = model._behavior_contrastive_loss(
        residuals, source_ids, target_ids_batch, candidate_curvatures_changed
    )
    if not torch.equal(expected_behavior_loss, candidate_changed_loss):
        raise RuntimeError("Candidate curvature changes source-anchored behavior loss")

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
        "[Iter56] shared initialization, fixed curvature, Euclidean quantization, "
        "source-anchored behavior distance, quantization/behavior/total gradients: PASS",
        flush=True,
    )


def main() -> None:
    _check_exact_transport_example()
    embeddings = np.asarray(np.load(EMBEDDING_FILE), dtype=np.float32)
    frame = pd.read_parquet(TRAIN_FILE)
    _check_immediate_transition_pairs(frame)
    curvatures, signals = build_item_curvatures(frame, len(embeddings))
    if signals.shape != (len(embeddings), 3):
        raise RuntimeError(f"Unexpected item graph-signal shape: {signals.shape}")
    experiment.STAGE2_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(experiment.ITEM_CURVATURES_PATH, curvatures)
    np.save(experiment.ITEM_SIGNALS_PATH, signals)
    print(
        f"[Iter56] saved fixed c_i to {experiment.ITEM_CURVATURES_PATH}",
        flush=True,
    )
    _check_signals_match_iter53(curvatures, signals)
    _check_quantization_is_euclidean(embeddings)
    _gradient_gate(embeddings, curvatures)


if __name__ == "__main__":
    main()
