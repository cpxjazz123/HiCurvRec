"""MVG preflight for iter16 closed-form fixed layer curvature."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"
REFERENCE_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
)
GRAD_EPSILON = 1e-12


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter16", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MVG requires CUDA")
    rqtrain = _load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)

    fixed_c = rqtrain._load_closed_form_curvatures()
    dataset = rqtrain.TransitionDataset(
        rqtrain.EMB_NPY, rqtrain.ITEM_IDS_JSON, rqtrain.TRAIN_PARQUET
    )
    active_indices = [i for i, targets in enumerate(dataset.next_items) if targets]
    raw_batch = rqtrain.collate_items(
        [dataset[index] for index in active_indices[: rqtrain.BATCH_SIZE]]
    )
    batch = rqtrain._build_seq_batch(*(value.to(device) for value in raw_batch))

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

    state = torch.load(REFERENCE_CKPT, map_location=device, weights_only=False)
    skip = {
        f"layers.{i}.c_layer_scale" for i in range(rqtrain.N_LAYERS)
    } | {f"layers.{i}._fixed_c" for i in range(rqtrain.N_LAYERS)}
    transferable = {
        name: value for name, value in state["model"].items() if name not in skip
    }
    model.load_state_dict(transferable, strict=False)

    model.set_curriculum_step(0)
    c0 = [float(layer.get_c().item()) for layer in model.layers]
    model.set_curriculum_step(50_000)
    c1 = [float(layer.get_c().item()) for layer in model.layers]
    if max(abs(a - b) for a, b in zip(c0, c1)) > 1e-6:
        raise RuntimeError(f"MVG FAIL: curvature drifted with step: {c0} vs {c1}")
    if not (fixed_c[0] > fixed_c[1] > fixed_c[2]):
        raise RuntimeError(f"MVG FAIL: expected c0>c1>c2, got {fixed_c}")

    model.train()
    output = model(batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("MVG FAIL: total loss detached")
    if output.behavior_loss.item() <= 0:
        raise RuntimeError("MVG FAIL: behavior loss inactive")
    if output.curvature_regularization.item() != 0.0:
        raise RuntimeError("MVG FAIL: curvature_reg must be 0")

    output.loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    before = [p.detach().clone() for p in trainable]
    optimizer.step()
    if not any(
        float((p - old).abs().sum()) > GRAD_EPSILON for old, p in zip(before, trainable)
    ):
        raise RuntimeError("MVG FAIL: no parameter update")

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        path = Path(handle.name)
    try:
        torch.save({"model": model.state_dict()}, path)
        model.load_state_dict(torch.load(path, weights_only=False)["model"], strict=True)
    finally:
        path.unlink(missing_ok=True)

    print("MVG PASS")
    print(f"closed_form_c_l={fixed_c}")
    print(f"c_after_steps={c0}")
    print(f"behavior_loss={float(output.behavior_loss):.6f}")


if __name__ == "__main__":
    main()
