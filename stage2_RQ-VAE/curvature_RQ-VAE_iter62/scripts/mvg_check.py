"""Derive fixed train-graph curvature and verify Iter62 geometry/gradients."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(SOURCE_DIR / "scripts"))

import curvature_config as experiment
import topology_curvature as topology
from train_rqvae import (
    EMBEDDING_FILE,
    TRAIN_FILE,
    _model_config,
    load_shared_initialization,
    initialize_tiger_weights,
)
from tiger_ricci_model import (
    TIGERTopologyHyperbolicRQVAE,
    _expmap0_tangent,
    _hyperbolic_residual,
    _logmap0_point,
    _mobius_add,
    _normalized_poincare_item_code_distances,
    _poincare_item_code_distances,
)


def _check_exact_transport_example() -> None:
    frame = pd.DataFrame(
        {"history": [[0], [0], [1], [2]], "target": [1, 2, 3, 4]}
    )
    sources, targets, counts = topology._transition_records(frame, 5)
    outgoing, probabilities = topology._make_outgoing_tables(
        sources, targets, counts, 5
    )
    undirected = topology._undirected_graph_tables(sources, targets, 5)
    adjacency, two_hop = topology._distance_bitsets(undirected, 5)
    topology._WORK_OUT_NEIGHBORS = outgoing
    topology._WORK_OUT_PROBABILITIES = probabilities
    topology._WORK_ADJACENCY_BITS = adjacency
    topology._WORK_TWO_HOP_BITS = two_hop
    contributions = [
        topology._edge_negative_curvature_contribution((int(source), int(target)))
        for source, target in zip(sources, targets)
    ]
    item_zero_need = sum(value for source, value in contributions if source == 0)
    if not np.isclose(item_zero_need, 1.0, atol=1e-8):
        raise RuntimeError(f"Exact toy-graph ORC need mismatch: {item_zero_need}")
    if any(value != 0.0 for source, value in contributions if source != 0):
        raise RuntimeError("Toy graph sink transitions must have zero negative-curvature need")
    print("[Iter62] exact train-graph transport example: PASS", flush=True)


def _check_geometry() -> None:
    torch.manual_seed(7)
    dtype = torch.float64
    curvature = torch.tensor([0.05, 0.4, 1.5], dtype=dtype)
    tangent = torch.tensor(
        [[0.2, -0.1], [0.05, 0.15], [-0.12, 0.08]], dtype=dtype
    )
    mapped = _expmap0_tangent(tangent, curvature)
    round_trip = _logmap0_point(mapped, curvature)
    if not torch.allclose(round_trip, tangent, atol=1e-10, rtol=1e-9):
        raise RuntimeError("Poincare exp/log map round trip failed")
    if bool((torch.linalg.vector_norm(mapped, dim=-1) >= 1.0 / curvature.sqrt()).any()):
        raise RuntimeError("Exp map returned a point outside its item-specific ball")

    codes = torch.tensor(
        [[0.1, 0.0], [0.0, 0.12], [-0.08, 0.06], [0.03, -0.11]],
        dtype=dtype,
    )
    physical = _poincare_item_code_distances(tangent, codes, curvature)
    normalized = _normalized_poincare_item_code_distances(tangent, codes, curvature)
    expected = curvature.sqrt().unsqueeze(1) * physical
    if not torch.allclose(normalized, expected, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Normalized assignment distance is not sqrt(c_i) * d_c")
    if not torch.equal(normalized.argmin(dim=1), physical.argmin(dim=1)):
        raise RuntimeError("Positive per-item normalization changed nearest-code ordering")

    residual = torch.tensor([[0.2, -0.1], [0.05, 0.15], [-0.12, 0.08]], dtype=dtype)
    code = torch.tensor([[0.1, 0.02], [-0.03, 0.09], [0.04, -0.02]], dtype=dtype)
    actual = _hyperbolic_residual(residual, code, curvature)
    expected_point = _mobius_add(
        -_expmap0_tangent(code, curvature),
        _expmap0_tangent(residual, curvature),
        curvature,
    )
    expected_residual = _logmap0_point(expected_point, curvature)
    if not torch.allclose(actual, expected_residual, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Residual update does not match Mobius subtraction + log map")
    if torch.allclose(actual, residual - code, atol=1e-6, rtol=1e-6):
        raise RuntimeError("Hyperbolic residual update collapsed to Euclidean subtraction")
    print("[Iter62] Poincare maps, normalized assignment, hyperbolic residual: PASS", flush=True)


def _require_gradient(loss: torch.Tensor, parameters, label: str) -> None:
    if not torch.isfinite(loss) or not loss.requires_grad or loss.grad_fn is None:
        raise RuntimeError(f"{label} has no finite autograd path")
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    if not any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and bool((gradient != 0).any())
        for gradient in gradients
    ):
        raise RuntimeError(f"{label} has no nonzero model-parameter gradient")


def _gradient_gate(embeddings: np.ndarray, curvatures: np.ndarray) -> None:
    if len(embeddings) < 8:
        raise ValueError("Gradient gate needs at least eight item embeddings")
    model = TIGERTopologyHyperbolicRQVAE(
        _model_config(),
        in_dim=embeddings.shape[1],
        item_curvatures=torch.from_numpy(curvatures),
    )
    initialize_tiger_weights(model)
    target_ids = np.unique(pd.read_parquet(TRAIN_FILE)["target"].to_numpy(dtype=np.int64))
    load_shared_initialization(model, embeddings, target_ids)
    if model.item_curvatures.requires_grad or "item_curvatures" in dict(model.named_parameters()):
        raise RuntimeError("Curvature table must remain a fixed, nontrainable buffer")

    baseline_state = torch.load(
        Path(
            "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter48/versions/TIGERBaseline/out/rqvae/instruments/rqvae_init.pth"
        ),
        map_location="cpu",
    )["state_dict"]
    mismatches = [
        key for key, value in baseline_state.items()
        if not torch.equal(value, model.state_dict()[key])
    ]
    if mismatches:
        raise RuntimeError(f"Shared Euclidean TIGER initialization changed: {mismatches[:4]}")

    probe = torch.from_numpy(embeddings[:8].copy())
    item_ids = torch.arange(8, dtype=torch.long)
    model.train()
    reconstructed, quant_loss, _, tokens = model(probe, item_ids=item_ids)
    if tokens.shape != (8, 3):
        raise RuntimeError(f"Unexpected SID shape: {tokens.shape}")
    recon_loss = F.mse_loss(reconstructed, probe)
    total_loss = recon_loss + quant_loss
    if not total_loss.requires_grad or total_loss.grad_fn is None:
        raise RuntimeError("Pure reconstruction + quantization loss has no autograd path")
    parameters = tuple(model.parameters())
    _require_gradient(recon_loss, parameters, "reconstruction loss")
    _require_gradient(quant_loss, parameters, "hyperbolic quantization loss")
    _require_gradient(total_loss, parameters, "total Stage2 loss")
    total_loss.backward()
    if not any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and bool((parameter.grad != 0).any())
        for parameter in parameters
    ):
        raise RuntimeError("Total loss backward produced no finite nonzero gradients")
    print(
        "[Iter62] shared baseline initialization, fixed c_i, recon/quant gradients: PASS",
        flush=True,
    )


def main() -> None:
    _check_exact_transport_example()
    _check_geometry()
    embeddings = np.asarray(np.load(EMBEDDING_FILE), dtype=np.float32)
    frame = pd.read_parquet(TRAIN_FILE)
    curvatures, signals = topology.build_item_curvatures(frame, len(embeddings))
    if signals.shape != (len(embeddings), 3):
        raise RuntimeError(f"Unexpected item topology-signal shape: {signals.shape}")
    if not np.isfinite(curvatures).all() or not np.isfinite(signals).all():
        raise RuntimeError("Train-only topology calculation returned non-finite values")
    experiment.STAGE2_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(experiment.ITEM_CURVATURES_PATH, curvatures)
    np.save(experiment.ITEM_SIGNALS_PATH, signals)
    print(
        f"[Iter62] fixed train-graph c_i saved: {experiment.ITEM_CURVATURES_PATH}; "
        f"range=({curvatures.min():.6f},{curvatures.max():.6f})",
        flush=True,
    )
    _gradient_gate(embeddings, curvatures)


if __name__ == "__main__":
    main()
