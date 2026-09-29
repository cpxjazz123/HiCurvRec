"""Verify cold-start parity and nonzero gradients before Euclidean Stage2 runs."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))
import train_euclidean_tiger as trainer


def initialized_model(arm, embeddings, target_ids, device):
    trainer.set_seed(trainer.SEED)
    model = trainer.build_model(arm, embeddings.shape[1]).to(device)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(embeddings[torch.as_tensor(target_ids, dtype=torch.long)].to(device))
    return model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Gradient check requires the project CUDA environment")
    device = torch.device("cuda:0")
    embeddings_np = trainer.load_embeddings(trainer.EMBEDDING_FILE)
    embeddings = torch.from_numpy(embeddings_np)
    frame = pd.read_parquet(trainer.TRAIN_FILE)
    dataset = trainer.TransitionDataset(frame)
    target_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))

    baseline = initialized_model("BASELINE", embeddings, target_ids, device)
    trainer.set_seed(trainer.SEED)
    behavior = trainer.build_model("BEHAVIOR", embeddings.shape[1]).to(device)
    trainer.initialize_tiger_weights(behavior)
    baseline_state = baseline.state_dict()
    behavior_state = behavior.state_dict()
    if baseline_state.keys() != behavior_state.keys():
        raise AssertionError("Euclidean arms must have identical trainable state keys")
    unequal = [
        name
        for name in baseline_state
        if not name.startswith("rq.vq_layers.")
        and not torch.equal(baseline_state[name], behavior_state[name])
    ]
    if unequal:
        raise AssertionError(f"Xavier model weights differ: {unequal[:5]}")

    with tempfile.NamedTemporaryFile(prefix="iter48_tiger_mvg_", suffix=".pth", delete=False) as file:
        checkpoint = Path(file.name)
    try:
        torch.save({name: value.detach().cpu() for name, value in baseline_state.items()}, checkpoint)
        loaded = torch.load(checkpoint, map_location=device)
        behavior.load_state_dict(loaded, strict=True)
        if any(
            not torch.equal(baseline_state[name], behavior.state_dict()[name])
            for name in baseline_state
        ):
            raise AssertionError("Behavior arm did not load the shared TIGER initialization")
    finally:
        checkpoint.unlink(missing_ok=True)

    pairs = dataset.pairs[:128].to(device)
    source_ids, target_ids_batch = pairs[:, 0], pairs[:, 1]
    batch_ids = torch.cat((source_ids, target_ids_batch), dim=0)
    batch = embeddings[batch_ids.cpu()].to(device)

    baseline.train()
    base_recon, base_quant, _, _ = baseline(batch)
    baseline_loss = F.mse_loss(base_recon, batch) + base_quant
    if not baseline_loss.requires_grad or baseline_loss.grad_fn is None:
        raise AssertionError("Euclidean baseline total loss has no autograd path")
    baseline_loss.backward()
    if not any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in baseline.parameters()):
        raise AssertionError("Euclidean baseline total loss has no nonzero parameter gradient")

    behavior.train()
    behavior.set_global_step(30_000)
    behavior_weight = behavior.get_behavior_weight()
    if not np.isclose(behavior_weight, 0.1, atol=1e-12):
        raise AssertionError(f"Unexpected 30k behavior weight: {behavior_weight}")
    rec, quant, _, _, contrastive = behavior(
        batch, behavior_ids=(source_ids, target_ids_batch)
    )
    total = F.mse_loss(rec, batch) + quant + behavior_weight * contrastive
    if not total.requires_grad or total.grad_fn is None or not torch.isfinite(total):
        raise AssertionError("Euclidean behavior total loss has no finite autograd path")
    if not torch.isfinite(contrastive) or contrastive <= 0:
        raise AssertionError(f"Behavior contrastive loss is invalid: {contrastive.item()}")
    behavior_grads = torch.autograd.grad(
        contrastive,
        tuple(behavior.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    nonzero_behavior_grads = [
        gradient
        for gradient in behavior_grads
        if gradient is not None
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient).item() > 0
    ]
    if not nonzero_behavior_grads:
        raise AssertionError("Euclidean behavior loss has no finite nonzero parameter gradient")
    total.backward()
    if not any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in behavior.parameters()):
        raise AssertionError("Euclidean behavior total loss has no nonzero parameter gradient")

    behavior.set_global_step(20_000)
    at_20k = behavior.get_behavior_weight()
    behavior.set_global_step(40_000)
    at_40k = behavior.get_behavior_weight()
    if not np.isclose(at_20k, 0.0) or not np.isclose(at_40k, 0.2):
        raise AssertionError(f"Incorrect ramp endpoints: {at_20k}, {at_40k}")
    print(
        "MVG_OK "
        f"state_equal=True baseline_loss={baseline_loss.item():.8f} "
        f"behavior_loss={contrastive.item():.8f} weight_30k={behavior_weight:.3f} "
        f"behavior_grad_tensors={len(nonzero_behavior_grads)} weight_20k={at_20k:.3f} "
        f"weight_40k={at_40k:.3f} batch_pairs={len(pairs)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
