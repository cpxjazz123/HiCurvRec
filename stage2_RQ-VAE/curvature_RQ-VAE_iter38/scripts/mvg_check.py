"""One-checkpoint, one-real-batch MVG for Iter38 prefix reconstruction."""
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
PREFIX_GRAD_EPSILON = 1e-12


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter38", TRAIN_SCRIPT)
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
    paired_x = torch.cat((batch.x, batch.x_fut), dim=0)
    quantized = model.get_semantic_ids(paired_x)
    prefix_embeddings = quantized.embeddings.cumsum(dim=0)
    expected_shape = (
        rqtrain.N_LAYERS,
        rqtrain.EMBED_DIM,
        paired_x.shape[0],
    )
    if tuple(prefix_embeddings.shape) != expected_shape:
        raise RuntimeError(
            f"prefix embedding shape {tuple(prefix_embeddings.shape)} "
            f"does not match {expected_shape}"
        )
    final_prefix = prefix_embeddings[-1].transpose(0, 1)
    summed_embeddings = model._step6_sum_embeddings(quantized)
    if not torch.allclose(
        final_prefix, summed_embeddings, rtol=1e-5, atol=2e-6
    ):
        raise RuntimeError("final cumulative prefix differs from full residual sum")

    from modules.normalize import l2norm

    curvature = model.layers[0].get_c().view(1, 1)
    encoder_parameters = tuple(model.encoder.parameters())
    decoder_parameters = tuple(model.decoder.parameters())
    prefix_losses = []
    for prefix_index, prefix_embedding in enumerate(
        prefix_embeddings.unbind(dim=0)
    ):
        prediction = l2norm(model.decode(prefix_embedding.transpose(0, 1)))
        prefix_loss = model.reconstruction_loss(
            prediction, paired_x, c=curvature
        ).mean()
        if not torch.isfinite(prefix_loss).item() or prefix_loss.item() <= 0:
            raise RuntimeError(
                f"prefix {prefix_index + 1} reconstruction loss is invalid"
            )
        gradients = torch.autograd.grad(
            prefix_loss,
            encoder_parameters + decoder_parameters,
            retain_graph=prefix_index < rqtrain.N_LAYERS - 1,
            allow_unused=True,
        )
        encoder_gradients = gradients[: len(encoder_parameters)]
        decoder_gradients = gradients[len(encoder_parameters) :]
        if not any(
            gradient is not None
            and torch.isfinite(gradient).all().item()
            and gradient.abs().sum().item() > PREFIX_GRAD_EPSILON
            for gradient in encoder_gradients
        ):
            raise RuntimeError(
                f"prefix {prefix_index + 1} has no finite encoder gradient"
            )
        if not any(
            gradient is not None
            and torch.isfinite(gradient).all().item()
            and gradient.abs().sum().item() > PREFIX_GRAD_EPSILON
            for gradient in decoder_gradients
        ):
            raise RuntimeError(
                f"prefix {prefix_index + 1} has no finite decoder gradient"
            )
        prefix_losses.append(float(prefix_loss.detach().item()))

    del quantized, prefix_embeddings, final_prefix, summed_embeddings
    model.zero_grad(set_to_none=True)
    output = model(batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("total loss has no autograd path")
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("total loss is non-finite")
    expected_reconstruction_loss = float(np.mean(prefix_losses))
    observed_reconstruction_loss = float(
        output.reconstruction_loss.detach().item()
    )
    if not np.isclose(
        observed_reconstruction_loss,
        expected_reconstruction_loss,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError(
            "forward reconstruction loss does not equal the mean of prefix losses: "
            f"observed={observed_reconstruction_loss:.8g}, "
            f"expected={expected_reconstruction_loss:.8g}"
        )
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
    print(f"cumulative_prefix_losses={prefix_losses}")
    print(f"loss={float(output.loss.detach().item()):.8g}")
    print(f"nonzero_gradient_parameters={len(nonzero_gradients)}")


if __name__ == "__main__":
    main()
