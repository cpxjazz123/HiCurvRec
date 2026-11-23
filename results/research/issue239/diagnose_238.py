"""C7 of issue #239: what exactly degenerates at step 2 in the #238 candidate?

Issue #239 forbids papering over the failure with a constant-matrix fallback, so this records, for
each of the first three optimisation steps, the quantities whose failure would each produce the
observed `Cannot center a constant distance matrix`:

  encoder output   -> is the latent finite and does it still have spread?
  L3 residual      -> the tensor the mixed distance is evaluated on
  codebook         -> did the 256 codewords collapse onto each other?
  mixed distance   -> the assignment matrix itself: range, spread, finiteness
  gradients        -> finite? how large?
  parameter update -> how far did one step move the weights?

The run is the #238 configuration exactly: mixed distance wired into the balanced level.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

REPO = "/fs04/ar57/wenyu/GeneRec"
sys.path.insert(0, REPO + "/stage2_RQ-VAE/TIGER_curvfid_rq")

import train_rqvae as entry  # noqa: E402
import curv_supervision as cs  # noqa: E402
from distances import mixed  # noqa: E402
from losses.curvature_fidelity import curvature_fidelity_loss  # noqa: E402
from model.model import RQVAE  # noqa: E402


def stats(name, tensor, extra=""):
    t = tensor.detach()
    finite = torch.isfinite(t).all().item()
    return {
        "quantity": name,
        "finite": bool(finite),
        "min": float(t.min()) if finite else float("nan"),
        "max": float(t.max()) if finite else float("nan"),
        "mean": float(t.mean()) if finite else float("nan"),
        "std": float(t.std()) if finite else float("nan"),
        "extra": extra,
    }


def residual_stack(model, latent):
    residual = latent
    for layer in model.rq.vq_layers:
        codes = layer.get_code_embs()
        residual = residual - codes[mixed.distances(residual, codes).argmin(dim=1)]
    return residual


def main() -> int:
    calibration = json.load(open(
        REPO + "/results/stage2_RQ-VAE/TIGER_curvfid_rq/calibration.json", encoding="utf-8"))
    mixed.set_scale(float(calibration["distance"]["mapping_scale"]))
    scale_s = float(calibration["target"]["global_scale_s"])
    delta = float(calibration["target"]["huber_delta"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = entry._tokenizer_config()
    torch.manual_seed(42)
    model = RQVAE(config, in_dim=768, distance=mixed.distances).to(device)
    entry.initialize_tiger_weights(model)
    embeddings = entry.load_embeddings(entry.EMBEDDING_FILE)
    vectors = torch.from_numpy(entry.maybe_apply_pca(embeddings, entry.PCA_DIM, entry.SEED)).to(device)
    targets = cs.prepare(calibration, entry.REPO_ROOT / "results" / "research" / "behavior_graph",
                         vectors.shape[0], seed=entry.SEED)
    with torch.no_grad():
        model.rq.init_codebook(model.encoder(vectors), device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    weight = None
    rows = []
    failure = None
    for step in range(3):
        try:
            batch = vectors[step * 4096:(step + 1) * 4096]
            optimizer.zero_grad(set_to_none=True)
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}

            reconstructed, parts, _, _, _ = model(batch, return_quantized=True)
            loss, components = entry.CRITERION(reconstructed, batch, parts)
            recon_loss = components["reconstruction"]

            index = cs.sample_indices(targets, step, seed=entry.SEED).to(device)
            lookup = cs.q_lookup_from_model(model, vectors, device)
            k_q = cs.curvature_of_nodes(lookup, targets, index, device)
            l_curv = curvature_fidelity_loss(k_q, targets["k_g"][index.cpu()].to(device),
                                             scale=scale_s, delta=delta)

            with torch.no_grad():
                latent = model.encoder(batch)
                residual = residual_stack(model, latent)
                codes = model.rq.vq_layers[-1].get_code_embs()
                dist = mixed.distances(residual, codes)
                codes3 = codes

            rows.append({"step": step, "stage": "forward"})
            rows.append(stats("encoder_output", latent))
            rows.append(stats("l3_residual", residual))
            rows.append(stats("l3_codebook", codes3,
                              f"unique(6dp)={int(torch.unique(codes3.round(decimals=6), dim=0).shape[0])}"))
            rows.append(stats("mixed_cost_matrix", dist,
                              f"per_row_span_mean={float((dist.max(1).values - dist.min(1).values).mean()):.6g}"))
            rows.append(stats("k_q", k_q))
            rows.append(stats("l_curv", l_curv.reshape(1)))

            if weight is None:
                params = [p for p in model.parameters() if p.requires_grad]

                def grad_norm(target):
                    grads = torch.autograd.grad(target, params, retain_graph=True, allow_unused=True)
                    total = torch.zeros((), device=device)
                    for g in grads:
                        if g is not None:
                            total = total + g.detach().pow(2).sum()
                    return float(total.sqrt())

                rn, cn = grad_norm(recon_loss), grad_norm(l_curv)
                weight = 0.05 * rn / max(cn, 1e-12)
                rows.append({"step": step, "stage": "weight",
                             "quantity": "lambda_curv", "finite": True,
                             "min": weight, "max": weight, "mean": weight, "std": 0.0,
                             "extra": f"recon_grad={rn:.6g} curv_grad={cn:.6g}"})

            (loss + weight * l_curv).backward()
            grad = torch.zeros((), device=device)
            nonfinite = 0
            bad = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad = grad + p.grad.detach().pow(2).sum()
                    if not torch.isfinite(p.grad).all():
                        nonfinite += 1
                        bad.append(name)
            rows.append({"step": step, "stage": "backward", "quantity": "grad_total_norm",
                         "finite": nonfinite == 0 and torch.isfinite(grad).item(),
                         "min": float(grad.sqrt()), "max": float(grad.sqrt()),
                         "mean": float(grad.sqrt()), "std": 0.0,
                         "extra": f"nonfinite_tensors={nonfinite} names={bad}"})

            optimizer.step()
            moved = 0.0
            for key, value in model.state_dict().items():
                if key in before and value.dtype.is_floating_point:
                    moved += float((value.detach() - before[key]).pow(2).sum())
            rows.append({"step": step, "stage": "update", "quantity": "param_update_norm",
                         "finite": True, "min": moved ** 0.5, "max": moved ** 0.5,
                         "mean": moved ** 0.5, "std": 0.0, "extra": ""})
        except Exception as exc:  # noqa: BLE001
            failure = {"step": step, "error": f"{type(exc).__name__}: {exc}"}
            rows.append({"step": step, "stage": "exception", "quantity": "raised",
                         "finite": False, "min": None, "max": None, "mean": None, "std": None,
                         "extra": failure["error"]})
            break

    out = {"configuration": "issue238 (mixed distance as the Sinkhorn cost)",
           "failure": failure, "rows": rows}
    path = REPO + "/results/research/issue239/diagnose_238.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
