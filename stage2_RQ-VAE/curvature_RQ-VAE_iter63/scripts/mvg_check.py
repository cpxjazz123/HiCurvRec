from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
import train_rqvae as train
from model import RQVAE
from model.layers import (
    _expmap0_tangent,
    _hyperbolic_residual,
    _mobius_add,
    _poincare_item_code_distances,
    _poincare_row_distances,
)


def main():
    if train.BEHAVIOR_WEIGHT != 0.0:
        raise RuntimeError("Behavior loss weight is not permanently zero")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train.set_seed(train.SEED)
    embeddings = torch.from_numpy(train.load_embeddings(train.EMBEDDING_FILE))
    frame = __import__("pandas").read_parquet(train.TRAIN_FILE)
    raw_features, features_np, _, _ = train.build_transition_features(frame, len(embeddings))
    reference = np.load(train.FREE_CURVATURE_FILE).astype(np.float32, copy=False)
    curvatures_np, _ = train.build_level_curvatures(features_np, reference)
    sorted_reference = np.sort(reference, kind="stable")
    item_ids = np.arange(len(features_np), dtype=np.int64)
    expected_shape = (len(embeddings), len(train.STRUCTURE_LEVEL_COLUMNS))
    if curvatures_np.shape != expected_shape:
        raise RuntimeError(f"Per-level curvature table shape {curvatures_np.shape} != {expected_shape}")
    for level, column in enumerate(train.STRUCTURE_LEVEL_COLUMNS):
        order = np.lexsort((item_ids, features_np[:, column]))
        if not np.array_equal(curvatures_np[order, level], sorted_reference):
            raise RuntimeError(f"Residual level {level} changed the Iter51 Free-distribution mapping")
        if not np.isclose(curvatures_np[:, level].mean(), reference.mean(), rtol=0.0, atol=1e-8):
            raise RuntimeError(f"Residual level {level} curvature mean differs from Iter51")
        if not np.isclose(curvatures_np[:, level].std(), reference.std(), rtol=0.0, atol=1e-8):
            raise RuntimeError(f"Residual level {level} curvature std differs from Iter51")
        correlation = train._spearman(curvatures_np[:, level], raw_features[:, column])
        if correlation <= 0.0:
            raise RuntimeError(f"Residual level {level} lost its positive feature-curvature relation")
        print(f"level_{level}_{train.FEATURE_NAMES[column]}_spearman={correlation:.6f}")
    if float(reference.min()) < 0.05 or float(reference.max()) > 0.40:
        raise RuntimeError("Iter51 weak Free-curvature distribution is outside [0.05, 0.40]")

    target_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    if len(target_ids) < 1024:
        raise RuntimeError("Need at least 1024 train targets for Iter51 KMeans initialization")
    model = RQVAE(
        train.tokenizer_config(),
        in_dim=embeddings.shape[1],
        item_curvatures=torch.from_numpy(curvatures_np).to(device),
    ).to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    codebook_ids = torch.as_tensor(target_ids[:1024], dtype=torch.long, device=device)
    model.init_codebook(embeddings[codebook_ids.cpu()].to(device), codebook_ids)

    with tempfile.TemporaryDirectory(prefix="iter63_mvg_") as temporary_dir:
        checkpoint_path = Path(temporary_dir) / "cold_start_state.pth"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    if hasattr(model, "_behavior_contrastive_loss") or hasattr(model.rq, "get_behavior_weight"):
        raise RuntimeError("Behavior loss path remains in the Iter63 model")
    source_ids = torch.as_tensor(target_ids[:32], dtype=torch.long, device=device)
    paired_target_ids = torch.as_tensor(target_ids[32:64], dtype=torch.long, device=device)
    batch_ids = torch.cat((source_ids, paired_target_ids))
    batch = embeddings[batch_ids.cpu()].to(device)
    model.train()
    reconstructed, quant_loss, _, _, used_curvatures = model(batch, batch_ids)
    total_loss, recon_loss = model.compute_loss(batch, reconstructed, quant_loss)
    expected_total = F.mse_loss(reconstructed, batch) + quant_loss
    if not torch.equal(total_loss, expected_total):
        raise RuntimeError("The Iter63 objective is not exactly reconstruction + quantization")
    if not total_loss.requires_grad or total_loss.grad_fn is None:
        raise RuntimeError("total_loss lacks an autograd graph")

    parameters = tuple(model.parameters())
    for name, loss_value in (("reconstruction", recon_loss), ("quantization", quant_loss)):
        gradients = torch.autograd.grad(loss_value, parameters, retain_graph=True, allow_unused=True)
        nonzero = [
            float(gradient.detach().abs().sum())
            for gradient in gradients
            if gradient is not None and torch.isfinite(gradient).all()
        ]
        if not nonzero or max(nonzero) <= 0.0:
            raise RuntimeError(f"{name} loss has no nonzero finite gradient to model parameters")
        print(f"{name}_gradient_l1={sum(nonzero):.9g}")
    if model.item_curvatures.requires_grad or any(
        name == "item_curvatures" for name, _ in model.named_parameters()
    ):
        raise RuntimeError("Fixed per-level item curvatures unexpectedly became trainable")
    fixed_table = model.item_curvatures.detach().clone()
    total_loss.backward()
    encoder_grad = model.encoder.mlp[1].weight.grad
    if encoder_grad is None or not torch.isfinite(encoder_grad).all() or encoder_grad.abs().sum() == 0:
        raise RuntimeError("Encoder backward gradient is absent, zero, or non-finite")
    if not torch.equal(model.item_curvatures, fixed_table):
        raise RuntimeError("Fixed per-level curvature table changed during backward")
    expected_curvatures = model.item_curvatures.index_select(0, batch_ids)
    if not torch.equal(used_curvatures, expected_curvatures):
        raise RuntimeError("The forward pass did not use the fixed item/level curvature table")
    if not torch.isfinite(reconstructed).all() or not torch.isfinite(quant_loss):
        raise RuntimeError("Geometry or objective produced non-finite values")

    sample_c = model.get_item_curvatures(batch_ids[:8])[:, 0]
    latent = model.encoder(batch[:8])
    distances = _poincare_item_code_distances(
        latent, model.rq.vq_layers[0].get_code_embs()[:8], sample_c
    )
    point_query = _expmap0_tangent(latent, sample_c[:, None])
    point_codes = _expmap0_tangent(
        model.rq.vq_layers[0].get_code_embs()[:8].unsqueeze(0).expand(8, -1, -1),
        sample_c.reshape(8, 1, 1),
    )
    difference = _mobius_add(-point_query.unsqueeze(1), point_codes, sample_c.reshape(8, 1, 1))
    distance_reference = (2.0 / sample_c.sqrt().unsqueeze(1)) * torch.atanh(
        (sample_c.sqrt().unsqueeze(1) * torch.linalg.vector_norm(difference, dim=-1)).clamp(max=1.0 - 1e-6)
    )
    if not torch.allclose(distances, distance_reference, rtol=2e-4, atol=2e-4):
        raise RuntimeError("Poincare item-code distance differs from the direct Mobius distance")
    residual = _hyperbolic_residual(latent, model.rq.vq_layers[0].get_code_embs()[:8], sample_c)
    if not torch.isfinite(residual).all():
        raise RuntimeError("Hyperbolic residual update produced non-finite values")
    row_distances = _poincare_row_distances(latent, latent, sample_c)
    if not torch.isfinite(row_distances).all() or not torch.allclose(
        row_distances, row_distances.T, rtol=2e-4, atol=2e-4
    ):
        raise RuntimeError("Poincare row distances are non-finite or asymmetric")

    print(
        f"MVG_OK device={device} total_requires_grad={total_loss.requires_grad} "
        f"grad_fn={type(total_loss.grad_fn).__name__} curvature_shape={tuple(curvatures_np.shape)} "
        f"curvature_min={float(reference.min()):.9g} curvature_max={float(reference.max()):.9g} "
        f"curvature_mean={float(reference.mean()):.9g} curvature_std={float(reference.std()):.9g} "
        f"behavior_weight={train.BEHAVIOR_WEIGHT:.1f} objective=reconstruction+quantization"
    )


if __name__ == "__main__":
    main()
