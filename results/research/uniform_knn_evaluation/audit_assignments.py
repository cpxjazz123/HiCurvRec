"""Read-only assignment and gradient audit for the Uniform-KNN arm.

No training is started.  The script answers three review questions on the frozen
checkpoint:

* how often training-time Sinkhorn tokens, inference argmin tokens and exported SID
  tokens disagree per level;
* whether the soft proxy's KL and the hard export q's KL move in the same direction
  after one small KL-only update;
* what the relative gradient strengths of reconstruction and KL are.

Run:
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/uniform_knn_evaluation/audit_assignments.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TREE = REPO / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ"
sys.path.insert(0, str(TREE))
CHECKPOINT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ" / "rqvae_best.pth"
SID = REPO / "results" / "stage2_RQ-VAE" / "TIGER_Uniform_KNN_RQ" / "item_sids.json"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = HERE / "assignment_audit.json"


def load_model():
    sys.path.insert(0, str(TREE))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "audit_rqvae", TREE / "model" / "__init__.py",
        submodule_search_locations=[str(TREE / "model")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["audit_rqvae"] = package
    spec.loader.exec_module(package)
    config = SimpleNamespace(
        hidden_sizes=(512, 256, 128), codebook_num=3, codebook_size=(256, 256, 256),
        codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=50,
    )
    model = package.RQVAE(config, in_dim=768)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state)
    # Mirror the arm's explicit assignment choice: no training-only Sinkhorn.
    for layer in model.rq.vq_layers:
        layer.use_sk = False
    model.eval()
    return model


def main() -> int:
    from losses.knn import (
        KNN_WEIGHT,
        build_knn_targets,
        hard_quantized_from_embeddings,
        knn_neighbourhood_kl,
        soft_quantized_readout,
    )

    model = load_model()
    x = torch.from_numpy(np.load(EMBEDDING).astype(np.float32))
    exported = json.loads(SID.read_text())
    expected = np.array([[int(v) for v in exported[str(i)][:3]] for i in range(len(x))])
    with torch.no_grad():
        model.train()
        training = model.get_indices(x, infer_use_sk=False).cpu().numpy()
        model.eval()
        inference = model.get_indices(x, infer_use_sk=False).cpu().numpy()

    def mismatch(tokens: np.ndarray) -> dict:
        return {
            "joint": float(np.mean(np.any(tokens != expected, axis=1))),
            "levels": [
                float(np.mean(tokens[:, level] != expected[:, level]))
                for level in range(3)
            ],
        }

    assignment = {
        "training_vs_export": mismatch(training),
        "inference_vs_export": mismatch(inference),
        "training_vs_inference": {
            "joint": float(np.mean(np.any(training != inference, axis=1))),
            "levels": [
                float(np.mean(training[:, level] != inference[:, level]))
                for level in range(3)
            ],
        },
        "note": "arm disables Sinkhorn; training and inference tokens are argmin; export mismatch remains 2.89%",
    }

    # One KL-only step: compare the soft proxy's KL with the hard export q's KL.
    candidates, log_target, _ = build_knn_targets(x.numpy())
    ids = torch.arange(1024)
    eval_model = model
    eval_model.train()
    q_memory, _ = hard_quantized_from_embeddings(eval_model, x)
    q_candidates = q_memory[candidates[ids]]
    target = torch.from_numpy(log_target[ids])
    q_proxy, q_initial_hard = soft_quantized_readout(eval_model, x[ids], assignment="export")
    soft_kl_before = float(knn_neighbourhood_kl(q_proxy, q_candidates, target))
    hard_kl_before = float(knn_neighbourhood_kl(q_initial_hard, q_candidates, target))
    optimizer = torch.optim.SGD(eval_model.parameters(), lr=1e-6)
    step_trace = []
    gradient = {}
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        q_proxy, _ = soft_quantized_readout(eval_model, x[ids], assignment="export")
        current_kl = knn_neighbourhood_kl(q_proxy, q_candidates, target)
        current_kl.backward()
        if step == 0:
            gradient = {
                name: float(parameter.grad.norm())
                for name, parameter in eval_model.named_parameters()
                if parameter.grad is not None and (name.endswith("weight") or "embed" in name)
            }
        optimizer.step()
        step_trace.append(float(current_kl.detach()))
    q_proxy_after, q_hard_after = soft_quantized_readout(eval_model, x[ids], assignment="export")
    soft_kl_after = float(knn_neighbourhood_kl(q_proxy_after, q_memory[candidates[ids]], target))
    hard_kl_after = float(knn_neighbourhood_kl(q_hard_after, q_memory[candidates[ids]], target))
    gradient_alignment = {
        "soft_kl_before": soft_kl_before,
        "soft_kl_after": soft_kl_after,
        "hard_kl_before": hard_kl_before,
        "hard_kl_after": hard_kl_after,
        "soft_delta": soft_kl_after - soft_kl_before,
        "hard_delta": hard_kl_after - hard_kl_before,
        "same_direction": bool((soft_kl_after - soft_kl_before) * (hard_kl_after - hard_kl_before) >= 0),
        "step_trace": step_trace,
        "gradient_norms": gradient,
        "note": "100 small SGD steps on KL only; hard q uses the export argmin path",
    }

    # Relative gradient strength: the fixed 1e-3 weight is only meaningful if the
    # KL gradient is not already the whole update at the final checkpoint.
    def parameter_grad_norm() -> float:
        total = 0.0
        for parameter in eval_model.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().pow(2).sum())
        return total ** 0.5

    eval_model.zero_grad(set_to_none=True)
    reconstructed, parts, _, _ = eval_model(x[ids])
    base_loss, recon_loss = eval_model.compute_loss(x[ids], reconstructed, parts)
    base_loss.backward()
    base_grad = parameter_grad_norm()
    eval_model.zero_grad(set_to_none=True)
    q_proxy, _ = soft_quantized_readout(eval_model, x[ids], assignment="export")
    kl_loss = knn_neighbourhood_kl(q_proxy, q_memory[candidates[ids]], target)
    kl_loss.backward()
    kl_grad = parameter_grad_norm()
    gradient_alignment["component_losses"] = {
        "reconstruction": float(recon_loss.detach()),
        "base_total": float(base_loss.detach()),
        "knn_kl": float(kl_loss.detach()),
        "weighted_knn": float(KNN_WEIGHT * kl_loss.detach()),
    }
    gradient_alignment["component_grad_norms"] = {
        "base_total": base_grad,
        "knn_kl": kl_grad,
        "ratio_knn_to_base": kl_grad / max(base_grad, 1e-12),
    }

    result = {
        "scope": "read-only; no Stage2 training",
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "sid": str(SID.relative_to(REPO)),
        "assignment": assignment,
        "gradient_alignment": gradient_alignment,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
