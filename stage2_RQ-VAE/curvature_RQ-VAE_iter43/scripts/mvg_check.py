"""One-checkpoint, one-real-batch MVG for Iter43 volume-growth curvature."""
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
EXPECTED_SHELL_TOTALS = [323633, 17943179, 230372961]
EXPECTED_GROWTH_BY_HOP = [
    13.22354335212879,
    55.44298325572485,
    12.839027075414005,
]
EXPECTED_FIXED_C = [0.627109001269444, 1.477631716406078, 0.6402592823244784]

GRAD_EPSILON = 1e-12
RADIAL_RELATIVE_TOL = 1e-3


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter43", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_batch(rqtrain, device):
    dataset = rqtrain.TransitionDataset(
        rqtrain.EMB_NPY, rqtrain.ITEM_IDS_JSON, rqtrain.TRAIN_PARQUET
    )
    raw_embeddings = np.load(rqtrain.EMB_NPY).astype(np.float32)
    if not np.array_equal(dataset.embeddings.numpy(), raw_embeddings):
        raise RuntimeError("Iter43 must leave the Stage1 semantic embeddings unchanged")
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
    batch = rqtrain._build_seq_batch(*(value.to(device) for value in raw_batch))
    return dataset, batch


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
    from modules.hyperbolic import (
        _expmap0_t,
        _poincare_distance_t,
        _transport_between_t,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)

    dataset, batch = _load_batch(rqtrain, device)
    graph_curvature = rqtrain.compute_behavior_graph_curvature(dataset.next_items)
    if graph_curvature["mapping"] != rqtrain.MAPPING_ID:
        raise RuntimeError(f"unexpected graph-curvature mapping: {graph_curvature}")
    if graph_curvature["shell_totals_by_hop"] != EXPECTED_SHELL_TOTALS:
        raise RuntimeError(
            f"train graph shell counts changed: {graph_curvature['shell_totals_by_hop']}"
        )
    if not np.allclose(
        graph_curvature["growth_by_hop"],
        EXPECTED_GROWTH_BY_HOP,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"train graph growth rates changed: {graph_curvature['growth_by_hop']}"
        )
    fixed_c = graph_curvature["closed_form_c_l"]
    if not np.allclose(fixed_c, EXPECTED_FIXED_C, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"fixed curvature changed: {fixed_c}")
    if not np.isclose(np.mean(fixed_c), 0.915, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"mean curvature budget changed: {fixed_c}")
    if not REFERENCE_CKPT.is_file():
        raise FileNotFoundError(REFERENCE_CKPT)
    checkpoint = torch.load(REFERENCE_CKPT, map_location=device, weights_only=False)
    if not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("warm-start checkpoint has no model state")
    model = _build_model(rqtrain, device, fixed_c, checkpoint)

    observed = []
    original_step5 = model._step5_transport

    def observe_step5(residual, layer_index):
        transported = original_step5(residual, layer_index)
        if transported is not residual or not torch.equal(transported, residual):
            raise RuntimeError(
                f"Step5 layer {layer_index} did not preserve tangent coordinates"
            )
        if not torch.isfinite(transported).all().item():
            raise RuntimeError(f"Step5 layer {layer_index} produced non-finite values")
        if layer_index < model.n_layers - 1:
            c_from = model.layers[layer_index].get_c().view(1, 1)
            c_next = model.layers[layer_index + 1].get_c().view(1, 1)
            old_output = _transport_between_t(residual, c_from, c_next)
            max_delta = float((old_output - transported).abs().max().item())
            if not np.isfinite(max_delta):
                raise RuntimeError("legacy/new Step5 comparison is non-finite")
            zero = torch.zeros_like(residual)
            current_ball = _expmap0_t(residual, c_from)
            next_ball = _expmap0_t(transported, c_next)
            current_distance = _poincare_distance_t(
                zero, current_ball, c_from
            ).squeeze(-1)
            next_distance = _poincare_distance_t(
                zero, next_ball, c_next
            ).squeeze(-1)
            target_distance = 2.0 * residual.norm(dim=-1)
            relative_error = torch.maximum(
                (current_distance - target_distance).abs(),
                (next_distance - target_distance).abs(),
            ) / target_distance.clamp_min(1e-6)
            max_radial_error = float(relative_error.max().item())
            if (
                not np.isfinite(max_radial_error)
                or max_radial_error > RADIAL_RELATIVE_TOL
            ):
                raise RuntimeError(
                    f"radial-distance preservation failed at layer {layer_index}: "
                    f"relative error={max_radial_error:.6g}"
                )
            observed.append(
                {
                    "layer": layer_index,
                    "max_legacy_delta": max_delta,
                    "max_radial_relative_error": max_radial_error,
                }
            )
        return transported

    model._step5_transport = observe_step5
    model.train()
    before_curvature = [float(layer.get_c().item()) for layer in model.layers]
    output = model(batch)
    if [entry["layer"] for entry in observed] != [0, 1]:
        raise RuntimeError(f"Step5 activation evidence incomplete: {observed}")
    if max(entry["max_legacy_delta"] for entry in observed) <= 1e-8:
        raise RuntimeError("new Step5 has no measurable difference from Iter29 transport")
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
        f"mapping={graph_curvature['mapping']} hops={graph_curvature['layer_hops']} "
        f"shell_totals={graph_curvature['shell_totals_by_hop']} "
        f"q_by_hop={graph_curvature['growth_by_hop']} "
        f"q_by_layer={graph_curvature['growth_by_layer']} "
        f"sectional_curvature_l={graph_curvature['sectional_curvature_l']} "
        "raw_semantic_embeddings_unchanged=PASS"
    )
    print(f"step5_layers={observed}")
    print(f"loss={float(output.loss.detach().item()):.8g}")
    print(f"nonzero_gradient_parameters={len(nonzero_gradients)}")


if __name__ == "__main__":
    main()
