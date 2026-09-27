"""One-checkpoint, one-real-batch MVG for Iter33 tangent residual subtraction."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"
REFERENCE_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
)
EXPECTED_FIXED_C = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
GRAD_EPSILON = 1e-12


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter33", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_batch(rqtrain, device):
    dataset = rqtrain.TransitionDataset(
        rqtrain.EMB_NPY, rqtrain.ITEM_IDS_JSON, rqtrain.TRAIN_PARQUET
    )
    active_indices = [
        index for index, targets in enumerate(dataset.next_items) if targets
    ]
    if len(active_indices) < rqtrain.BATCH_SIZE:
        raise RuntimeError(
            f"Only {len(active_indices)} transition sources; "
            f"need {rqtrain.BATCH_SIZE}"
        )
    raw_batch = rqtrain.collate_items(
        [dataset[index] for index in active_indices[: rqtrain.BATCH_SIZE]]
    )
    return rqtrain._build_seq_batch(*(value.to(device) for value in raw_batch))


def _build_model(rqtrain, device, fixed_c, state):
    model = rqtrain.RqVae(
        input_dim=rqtrain.INPUT_DIM,
        embed_dim=rqtrain.EMBED_DIM,
        hidden_dims=rqtrain.HIDDEN_DIMS,
        codebook_size=rqtrain.CODEBOOK_SIZE,
        codebook_kmeans_init=False,
        n_layers=rqtrain.N_LAYERS,
        commitment_weight=rqtrain.COMMITMENT_WEIGHT,
        sk_eps=0.05,
        sk_iters=3,
        c_cyclic_min=rqtrain.C_CYCLIC_MIN,
        c_cyclic_max=rqtrain.C_CYCLIC_MAX,
        c_cyclic_period=rqtrain.C_CYCLIC_PERIOD,
        midpoint_layer_mask=rqtrain.MIDPOINT_LAYER_MASK,
        residual_layer_norms=[1.0] * rqtrain.N_LAYERS,
        fixed_layer_curvatures=fixed_c,
    ).to(device)
    missing, unexpected = model.load_state_dict(
        {
            key: value
            for key, value in state["model"].items()
            if not key.endswith((".c_layer_scale", "._fixed_c"))
        },
        strict=False,
    )
    expected_missing = {f"layers.{i}._fixed_c" for i in range(rqtrain.N_LAYERS)}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"warm-start mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return model


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MVG requires CUDA")
    rqtrain = _load_training_module()
    from modules.hyperbolic import _expmap0_t, _logmap0_t, _mobius_add_t

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    fixed_c = rqtrain._load_closed_form_curvatures()
    if not np.allclose(fixed_c, EXPECTED_FIXED_C, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"fixed curvature changed: {fixed_c}")
    if not REFERENCE_CKPT.is_file():
        raise FileNotFoundError(REFERENCE_CKPT)
    checkpoint = torch.load(REFERENCE_CKPT, map_location=device, weights_only=False)
    if not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("warm-start checkpoint has no model state")
    model = _build_model(rqtrain, device, fixed_c, checkpoint)
    batch = _load_batch(rqtrain, device)

    observed = []
    original_step4 = model._step4_tangent_residual
    original_step5 = model._step5_transport

    def observe_step4(residual, embedding, layer_index):
        result = original_step4(residual, embedding, layer_index)
        expected = residual - embedding
        if not torch.equal(result, expected):
            raise RuntimeError(f"Step4 layer {layer_index} does not implement r - q")
        if result.shape != residual.shape or not torch.isfinite(result).all().item():
            raise RuntimeError(f"Step4 layer {layer_index} produced invalid residual")
        curvature = model.layers[layer_index].get_c().view(1, 1)
        parent_residual = _logmap0_t(
            _mobius_add_t(
                -_expmap0_t(embedding, curvature),
                _expmap0_t(residual, curvature),
                curvature,
            ),
            curvature,
        )
        max_parent_delta = float((result - parent_residual).abs().max().item())
        observed.append(
            {
                "layer": layer_index,
                "residual": residual,
                "embedding": embedding,
                "result": result,
                "max_parent_delta": max_parent_delta,
            }
        )
        return result

    def observe_step5(residual, layer_index):
        result = original_step5(residual, layer_index)
        if result is not residual or not torch.equal(result, residual):
            raise RuntimeError(f"inherited RIRT changed coordinates at layer {layer_index}")
        return result

    model._step4_tangent_residual = observe_step4
    model._step5_transport = observe_step5
    model.train()
    before_curvature = [float(layer.get_c().item()) for layer in model.layers]
    output = model(batch)
    if [entry["layer"] for entry in observed] != [0, 1, 2]:
        raise RuntimeError(f"Step4 activation evidence incomplete: {observed}")
    if not np.isfinite([entry["max_parent_delta"] for entry in observed]).all():
        raise RuntimeError("parent/current Step4 comparison is non-finite")
    if max(entry["max_parent_delta"] for entry in observed) <= 1e-8:
        raise RuntimeError("Iter33 Step4 has no measurable difference from Iter32 M2")

    first = observed[0]["residual"]
    reconstructed = sum(entry["embedding"] for entry in observed) + observed[-1]["result"]
    telescoping_error = float((reconstructed - first).abs().max().item())
    if not np.isfinite(telescoping_error) or telescoping_error > 2e-6:
        raise RuntimeError(f"tangent residual telescoping failed: {telescoping_error}")
    step4_residual_gradient = torch.autograd.grad(
        observed[0]["result"].sum(),
        observed[0]["residual"],
        retain_graph=True,
    )[0]
    if not torch.equal(
        step4_residual_gradient, torch.ones_like(step4_residual_gradient)
    ):
        raise RuntimeError("Step4 residual gradient is not +1")
    step4_code_gradient = torch.autograd.grad(
        observed[0]["result"].sum(),
        observed[0]["embedding"],
        retain_graph=True,
        allow_unused=True,
    )[0]
    if step4_code_gradient is not None and torch.count_nonzero(step4_code_gradient).item():
        raise RuntimeError("Step4 detached hard code still receives residual-path gradients")
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("total loss has no autograd path")
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("total loss is non-finite")
    output.loss.backward()
    nonzero_gradients = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"non-finite gradient: {name}")
        if parameter.grad.abs().sum().item() > GRAD_EPSILON:
            nonzero_gradients.append(name)
    if not nonzero_gradients:
        raise RuntimeError("no trainable parameter received a non-zero gradient")
    after_curvature = [float(layer.get_c().item()) for layer in model.layers]
    if before_curvature != after_curvature or not np.allclose(
        after_curvature, fixed_c, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError(
            f"fixed curvature changed: before={before_curvature}, after={after_curvature}"
        )

    print("MVG PASS")
    print(f"checkpoint={REFERENCE_CKPT}")
    print(f"closed_form_c_l={fixed_c}; fixed_after_forward={after_curvature}")
    print(
        "step4_layers="
        f"{[(entry['layer'], entry['max_parent_delta']) for entry in observed]}"
    )
    print(f"tangent_telescoping_max_error={telescoping_error:.8g}")
    print(f"loss={float(output.loss.detach().item()):.8g}")
    print(f"nonzero_gradient_parameters={len(nonzero_gradients)}")


if __name__ == "__main__":
    main()
