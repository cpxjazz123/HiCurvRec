"""iter27 grad-check: layer-1 + layer-2 compat entrypoint."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter27", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("grad_check requires CUDA")
    rqtrain = _load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)

    dataset = rqtrain.TransitionDataset(
        rqtrain.EMB_NPY, rqtrain.ITEM_IDS_JSON, rqtrain.TRAIN_PARQUET
    )
    active = [i for i, t in enumerate(dataset.next_items) if t]
    raw_batch = rqtrain.collate_items(
        [dataset[i] for i in active[: rqtrain.BATCH_SIZE]]
    )
    seq_batch = rqtrain._build_seq_batch(*(value.to(device) for value in raw_batch))

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
    ).to(device)
    model.train()

    output = model(seq_batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("GRAD_CHECK FAIL: total loss detached")
    output.loss.backward()
    for layer_index in range(rqtrain.N_LAYERS):
        param = model.layers[layer_index].embedding.weight
        if param.grad is None or float(param.grad.norm()) < 1e-12:
            raise RuntimeError(
                f"GRAD_CHECK FAIL: layer {layer_index} codebook has no gradient"
            )

    print("GRAD_CHECK PASS")
    print(f"closed_form_c_l=N/A (iter27 inherits cyclic c from iter18)")
    print(f"trust_region_constants=({rqtrain.ITER27_TRUST_RADIUS_FRAC}, {rqtrain.ITER27_TRUST_RADIUS_MIN})")


if __name__ == "__main__":
    main()
