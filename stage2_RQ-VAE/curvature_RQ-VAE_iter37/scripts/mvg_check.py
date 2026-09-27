"""One-checkpoint, one-real-batch MVG for Iter37 layer concatenation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"
REFERENCE_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
)
EXPECTED_FIXED_C = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
GRAD_EPSILON = 1e-12


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter37", TRAIN_SCRIPT)
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
    transferred = {
        key: value
        for key, value in state["model"].items()
        if not key.endswith((".c_layer_scale", "._fixed_c"))
    }
    transferred = model._expand_decoder_input_state(transferred)
    missing, unexpected = model.load_state_dict(transferred, strict=False)
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
    from modules.encoder import MLP

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
    model.train()
    before_curvature = [float(layer.get_c().item()) for layer in model.layers]

    legacy_decoder = MLP(
        input_dim=rqtrain.EMBED_DIM,
        hidden_dims=rqtrain.HIDDEN_DIMS[-1::-1],
        out_dim=rqtrain.INPUT_DIM,
        normalize=False,
    ).to(device)
    legacy_decoder.load_state_dict(
        {
            key.removeprefix("decoder."): value
            for key, value in checkpoint["model"].items()
            if key.startswith("decoder.")
        },
        strict=True,
    )
    legacy_decoder.eval()

    paired_x = torch.cat((batch.x, batch.x_fut), dim=0)
    quantized = model.get_semantic_ids(paired_x)
    concatenated = model._step6_concat_embeddings(quantized)
    expected_concat = quantized.embeddings.permute(2, 0, 1).reshape(
        paired_x.shape[0], rqtrain.N_LAYERS * rqtrain.EMBED_DIM
    )
    if concatenated.shape != expected_concat.shape or not torch.equal(
        concatenated, expected_concat
    ):
        raise RuntimeError("Step6 did not preserve layer-major decoder blocks")
    legacy_sum = quantized.embeddings.sum(dim=0).transpose(0, 1)
    legacy_reconstruction = legacy_decoder(legacy_sum)
    tiled_reconstruction = model.decoder(concatenated)
    initial_function_delta = float(
        (legacy_reconstruction - tiled_reconstruction).abs().max().item()
    )
    expected_first_weight = legacy_decoder.mlp[0].weight.repeat(
        1, rqtrain.N_LAYERS
    )
    first_weight_delta = float(
        (model.decoder.mlp[0].weight - expected_first_weight).abs().max().item()
    )
    old_preactivation = legacy_decoder.mlp[0](legacy_sum)
    new_preactivation = model.decoder.mlp[0](concatenated)
    preactivation_delta = float(
        (old_preactivation - new_preactivation).abs().max().item()
    )
    legacy_normalized = F.normalize(legacy_reconstruction, p=2, dim=-1)
    tiled_normalized = F.normalize(tiled_reconstruction, p=2, dim=-1)
    normalized_delta = float(
        (legacy_normalized - tiled_normalized).abs().max().item()
    )
    normalized_cosine = float(
        F.cosine_similarity(legacy_reconstruction, tiled_reconstruction, dim=-1)
        .min()
        .item()
    )
    if (
        not torch.allclose(
            legacy_normalized, tiled_normalized, rtol=1e-4, atol=5e-4
        )
        or normalized_cosine < 0.99999
    ):
        raise RuntimeError(
            "tiled decoder warm-start changed normalized function: "
            f"normalized_delta={normalized_delta:.8g}, "
            f"min_cosine={normalized_cosine:.8g}, "
            f"output_delta={initial_function_delta:.8g}, "
            f"weight_delta={first_weight_delta:.8g}, "
            f"preactivation_delta={preactivation_delta:.8g}"
        )

    output = model(batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("total loss has no autograd path")
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("total loss is non-finite")
    output.loss.backward()
    first_weight_gradient = model.decoder.mlp[0].weight.grad
    if first_weight_gradient is None or not torch.isfinite(first_weight_gradient).all().item():
        raise RuntimeError("concatenated decoder first layer has no finite gradient")
    gradient_blocks = first_weight_gradient.reshape(
        first_weight_gradient.shape[0], rqtrain.N_LAYERS, rqtrain.EMBED_DIM
    )
    block_norms = [
        float(gradient_blocks[:, index].abs().sum().item())
        for index in range(rqtrain.N_LAYERS)
    ]
    if any(norm <= GRAD_EPSILON for norm in block_norms):
        raise RuntimeError(f"one or more decoder layer blocks have zero gradient: {block_norms}")

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
    print(f"decoder_input_shape={tuple(concatenated.shape)}; layer_block_gradient_norms={block_norms}")
    print("tiled_warm_start_preserves_legacy_sum_function=True")
    print(f"loss={float(output.loss.detach().item()):.8g}")
    print(f"nonzero_gradient_parameters={len(nonzero_gradients)}")


if __name__ == "__main__":
    main()
