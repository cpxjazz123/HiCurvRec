"""Gradient audit with the *synchronised* q memory used by the fixed trainer.

This is still read-only with respect to Stage2: it performs at most ten tiny KL-only
optimizer steps on a copy of the checkpoint and never starts a training run.

For every step it recomputes the full export-path q from the current parameters, so the
candidate side is not frozen at the initial checkpoint.  It reports soft KL, hard q KL,
KL gradient norm, and the parameter-state change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TREE = REPO / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ"
sys.path.insert(0, str(TREE))
CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ" / "rqvae_best.pth"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = HERE / "gradient_memory_audit.json"


def load_model():
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "audit_rqvae2", TREE / "model" / "__init__.py",
        submodule_search_locations=[str(TREE / "model")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["audit_rqvae2"] = package
    spec.loader.exec_module(package)
    config = SimpleNamespace(
        hidden_sizes=(512, 256, 128), codebook_num=3, codebook_size=(256, 256, 256),
        codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=50,
    )
    model = package.RQVAE(config, in_dim=768)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"])
    for layer in model.rq.vq_layers:
        layer.use_sk = False
    model.eval()
    return model


def main() -> int:
    from losses.knn import (
        build_knn_targets,
        hard_quantized_from_embeddings,
        knn_neighbourhood_kl,
        soft_quantized_readout,
    )

    model = load_model()
    x = torch.from_numpy(np.load(EMBEDDING).astype(np.float32))
    candidates, log_target, _ = build_knn_targets(x.numpy())
    ids = torch.arange(1024)
    candidate_ids = candidates[ids]
    target = torch.from_numpy(log_target[ids])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    def q_all() -> torch.Tensor:
        with torch.no_grad():
            q, _ = hard_quantized_from_embeddings(model, x)
        return q

    def hard_and_soft_kl(q_memory: torch.Tensor) -> tuple[float, float]:
        q_proxy, q_hard = soft_quantized_readout(model, x[ids], assignment="export")
        cand = q_memory[candidate_ids]
        soft = float(knn_neighbourhood_kl(q_proxy, cand, target))
        hard = float(knn_neighbourhood_kl(q_hard, cand, target))
        return soft, hard

    initial_memory = q_all()
    soft_before, hard_before = hard_and_soft_kl(initial_memory)

    # Initial gradient norms, before any update, on the same batch and the same state.
    def grad_norm() -> float:
        total = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        return total ** 0.5

    model.zero_grad(set_to_none=True)
    q_proxy, _ = soft_quantized_readout(model, x[ids], assignment="export")
    initial_kl = knn_neighbourhood_kl(q_proxy, initial_memory[candidate_ids], target)
    initial_kl.backward()
    initial_kl_grad = grad_norm()
    model.zero_grad(set_to_none=True)
    reconstructed, parts, _, _ = model(x[ids])
    base_loss, _ = model.compute_loss(x[ids], reconstructed, parts)
    base_loss.backward()
    initial_base_grad = grad_norm()
    model.zero_grad(set_to_none=True)

    trace = []
    for step in range(10):
        q_memory = q_all()
        q_proxy, _ = soft_quantized_readout(model, x[ids], assignment="export")
        cand = q_memory[candidate_ids]
        soft = float(knn_neighbourhood_kl(q_proxy, cand, target))
        # One differentiable soft-proxy step; then refresh all q from the new parameters.
        model.zero_grad(set_to_none=True)
        loss = knn_neighbourhood_kl(q_proxy, cand, target)
        loss.backward()
        optimizer.step()
        updated_memory = q_all()
        updated_soft, updated_hard = hard_and_soft_kl(updated_memory)
        trace.append({
            "step": step + 1,
            "soft_before": soft,
            "soft_after": updated_soft,
            "hard_after": updated_hard,
        })

    result = {
        "scope": "read-only after loading a checkpoint; 10 one-step KL diagnostics, no Stage2 run",
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "batch_items": int(len(ids)),
        "lr": 1e-6,
        "initial": {
            "soft_kl": soft_before,
            "hard_kl": hard_before,
            "kl_gradient_norm": initial_kl_grad,
            "base_gradient_norm": initial_base_grad,
            "kl_to_base_ratio": initial_kl_grad / max(initial_base_grad, 1e-12),
        },
        "trace": trace,
        "final": trace[-1] if trace else None,
        "note": "the candidate q is recomputed from the current model at every diagnostic step",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
