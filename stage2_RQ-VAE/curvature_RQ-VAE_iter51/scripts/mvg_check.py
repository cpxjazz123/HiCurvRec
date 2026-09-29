from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
import train_rqvae as train
from model import RQVAE
from model.layers import (
    _expmap0_tangent,
    _mobius_add,
    _poincare_item_code_distances,
    _poincare_pairwise_distances,
    _poincare_row_distances,
)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embeddings = torch.from_numpy(train.load_embeddings(train.EMBEDDING_FILE))
    frame = __import__("pandas").read_parquet(train.TRAIN_FILE)
    raw_features, features_np, _, _ = train.build_transition_features(frame, len(embeddings))
    reference = np.load(train.FREE_CURVATURE_FILE).astype(np.float32, copy=False)
    curvatures_np, _ = train.build_level_curvatures(features_np, reference)
    sorted_reference = np.sort(reference, kind="stable")
    item_ids = np.arange(len(features_np), dtype=np.int64)
    if curvatures_np.shape != (len(embeddings), len(train.STRUCTURE_LEVEL_COLUMNS)):
        raise RuntimeError("Per-level curvature table has the wrong shape")
    for level, column in enumerate(train.STRUCTURE_LEVEL_COLUMNS):
        order = np.lexsort((item_ids, features_np[:, column]))
        if not np.array_equal(curvatures_np[order, level], sorted_reference):
            raise RuntimeError(f"Residual level {level} does not preserve the Free curvature distribution")
        if np.any(np.diff(curvatures_np[order, level]) < 0.0):
            raise RuntimeError(f"Residual level {level} curvature is not monotonic in its assigned feature")
        if not np.isclose(curvatures_np[:, level].mean(), reference.mean(), rtol=0.0, atol=1e-8):
            raise RuntimeError(f"Residual level {level} curvature mean differs from Free")
        if not np.isclose(curvatures_np[:, level].std(), reference.std(), rtol=0.0, atol=1e-8):
            raise RuntimeError(f"Residual level {level} curvature standard deviation differs from Free")
        correlation = train._spearman(curvatures_np[:, level], raw_features[:, column])
        if correlation <= 0.0:
            raise RuntimeError(f"Residual level {level} lost its positive feature-curvature relation")
        print(f"level_{level}_{train.FEATURE_NAMES[column]}_spearman={correlation:.6f}")

    target_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    if len(target_ids) < 64:
        raise RuntimeError("Need at least 64 training target items for the gradient preflight")
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
    model.rq.behavior_loss_weight = train.BEHAVIOR_WEIGHT
    model.rq.behavior_temperature = train.BEHAVIOR_TEMPERATURE
    model.rq.set_curriculum_step(30_000)
    codebook_ids = torch.as_tensor(target_ids[:1024], dtype=torch.long, device=device)
    model.init_codebook(embeddings[codebook_ids.cpu()].to(device), codebook_ids)
    checkpoint = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(checkpoint)

    source_ids = torch.as_tensor(target_ids[:32], dtype=torch.long, device=device)
    paired_target_ids = torch.as_tensor(target_ids[32:64], dtype=torch.long, device=device)
    batch_ids = torch.cat((source_ids, paired_target_ids))
    batch = embeddings[batch_ids.cpu()].to(device)
    model.train()
    reconstructed, quant_loss, _, _, behavior_loss, used_curvatures = model(
        batch, batch_ids, behavior_ids=(source_ids, paired_target_ids)
    )
    total_loss, _ = model.compute_loss(batch, reconstructed, quant_loss, behavior_loss)
    if not total_loss.requires_grad or total_loss.grad_fn is None:
        raise RuntimeError("total_loss lacks an autograd graph")
    parameters = tuple(model.parameters())
    for name, loss_value in (("quantization", quant_loss), ("behavior", behavior_loss)):
        gradients = torch.autograd.grad(loss_value, parameters, retain_graph=True, allow_unused=True)
        nonzero = [
            float(gradient.detach().abs().sum())
            for gradient in gradients
            if gradient is not None and torch.isfinite(gradient).all()
        ]
        if not nonzero or max(nonzero) <= 0.0:
            raise RuntimeError(f"{name} loss has no nonzero gradient to trainable model parameters")
        print(f"{name}_gradient_l1={sum(nonzero):.9g}")
    if model.item_curvatures.requires_grad or any(
        name == "item_curvatures" for name, _ in model.named_parameters()
    ):
        raise RuntimeError("Fixed per-level item curvatures unexpectedly became trainable")
    fixed_table = model.item_curvatures.detach().clone()
    total_loss.backward()
    encoder_grad = model.encoder.mlp[1].weight.grad
    if encoder_grad is None or not torch.isfinite(encoder_grad).all():
        raise RuntimeError("Encoder backward gradient is absent or non-finite")
    if not torch.equal(model.item_curvatures, fixed_table):
        raise RuntimeError("Fixed per-level curvature table changed during backward")
    expected_curvatures = model.item_curvatures.index_select(0, batch_ids)
    if not torch.isfinite(used_curvatures).all() or not torch.equal(used_curvatures, expected_curvatures):
        raise RuntimeError("The forward pass did not use the fixed per-item, per-level curvatures")
    if not torch.isfinite(reconstructed).all() or not torch.isfinite(quant_loss) or not torch.isfinite(behavior_loss):
        raise RuntimeError("Geometry or loss produced non-finite values")

    sample_c = model.get_item_curvatures(batch_ids[:8])[:, 0]
    pair_c = torch.sqrt(sample_c[:, None] * sample_c[None, :])
    latent = model.encoder(batch[:8])
    distances = _poincare_pairwise_distances(latent, latent, pair_c)
    c_pair = pair_c.unsqueeze(-1)
    point_x = _expmap0_tangent(latent[:, None, :], c_pair)
    point_y = _expmap0_tangent(latent[None, :, :], c_pair)
    difference = _mobius_add(-point_x, point_y, c_pair)
    norm = torch.linalg.vector_norm(difference, dim=-1)
    sqrt_pair_c = pair_c.sqrt()
    reference_distances = (2.0 / sqrt_pair_c) * torch.atanh(
        (sqrt_pair_c * norm).clamp(max=1.0 - 1e-6)
    )
    asymmetry = float((distances - distances.t()).abs().max().detach())
    if not torch.isfinite(distances).all() or asymmetry > 1e-5 or not torch.allclose(
        distances, reference_distances, rtol=2e-4, atol=2e-4
    ):
        raise RuntimeError("Pairwise Poincare kernel is non-finite, asymmetric, or inaccurate")

    codebook = model.rq.vq_layers[0].get_code_embs()[:8]
    item_distances = _poincare_item_code_distances(latent, codebook, sample_c)
    c_rows = sample_c.reshape(8, 1, 1)
    point_query = _expmap0_tangent(latent, sample_c[:, None])
    point_codes = _expmap0_tangent(codebook.unsqueeze(0).expand(8, -1, -1), c_rows)
    item_difference = _mobius_add(-point_query.unsqueeze(1), point_codes, c_rows)
    item_norm = torch.linalg.vector_norm(item_difference, dim=-1)
    sqrt_item_c = sample_c.sqrt().unsqueeze(1)
    item_reference = (2.0 / sqrt_item_c) * torch.atanh(
        (sqrt_item_c * item_norm).clamp(max=1.0 - 1e-6)
    )
    if not torch.allclose(item_distances, item_reference, rtol=2e-4, atol=2e-4):
        raise RuntimeError("Item-code Poincare kernel disagrees with direct Mobius distance")
    print(
        f"MVG_OK total_requires_grad={total_loss.requires_grad} grad_fn={type(total_loss.grad_fn).__name__} "
        f"curvature_shape={tuple(curvatures_np.shape)} "
        f"curvature_mean={float(curvatures_np.mean()):.9g} "
        f"curvature_std={float(curvatures_np.std()):.9g} "
        f"pairwise_asymmetry={asymmetry:.3g} behavior_loss={float(behavior_loss.detach()):.6f}"
    )


if __name__ == "__main__":
    main()
