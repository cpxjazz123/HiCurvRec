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
    _, features_np, _, _ = train.build_transition_features(frame, len(embeddings))
    features = torch.from_numpy(features_np).to(device)
    target_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))[:1024]
    model = RQVAE(train.tokenizer_config(), in_dim=embeddings.shape[1], structural_dim=features.shape[1]).to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    model.reset_curvature_head()
    model.rq.set_curriculum_step(30_000)
    model.init_codebook(embeddings[torch.as_tensor(target_ids)].to(device), features[torch.as_tensor(target_ids, device=device)])
    checkpoint = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(checkpoint)
    pair_count = 32
    source_ids = torch.as_tensor(target_ids[:pair_count], dtype=torch.long, device=device)
    target_ids_t = torch.as_tensor(target_ids[pair_count:2 * pair_count], dtype=torch.long, device=device)
    batch_ids = torch.cat((source_ids, target_ids_t))
    batch = embeddings[batch_ids.cpu()].to(device)
    structural = features[batch_ids]
    model.train()
    reconstructed, quant_loss, _, _, behavior_loss, curvatures = model(
        batch, structural, behavior_ids=(source_ids, target_ids_t)
    )
    total_loss, _ = model.compute_loss(batch, reconstructed, quant_loss, behavior_loss)
    if not total_loss.requires_grad or total_loss.grad_fn is None:
        raise RuntimeError("total_loss lacks an autograd graph")
    head_parameters = tuple(model.curvature_head.parameters())
    for name, loss_value in (("quantization", quant_loss), ("behavior", behavior_loss)):
        gradients = torch.autograd.grad(loss_value, head_parameters, retain_graph=True, allow_unused=True)
        nonzero = [float(gradient.detach().abs().sum()) for gradient in gradients if gradient is not None]
        if not nonzero or max(nonzero) <= 0.0:
            raise RuntimeError(f"{name} loss has no nonzero gradient to item-curvature head")
        print(f"{name}_curvature_gradient_l1={sum(nonzero):.9g}")
    total_loss.backward()
    head_grad = model.curvature_head.weight.grad
    if head_grad is None or not torch.isfinite(head_grad).all() or float(head_grad.detach().abs().sum()) <= 0.0:
        raise RuntimeError("curvature head backward gradient is absent, zero, or non-finite")
    if not torch.isfinite(curvatures).all() or not torch.allclose(curvatures, torch.ones_like(curvatures), atol=1e-6):
        raise RuntimeError("Item curvatures do not initialize uniformly to c=1")
    if not torch.isfinite(reconstructed).all() or not torch.isfinite(quant_loss) or not torch.isfinite(behavior_loss):
        raise RuntimeError("Geometry or loss produced non-finite values")
    torch.optim.SGD(head_parameters, lr=1.0).step()
    with torch.no_grad():
        varied_curvatures = model.get_item_curvatures(features[torch.as_tensor(target_ids, device=device)])
        if float(varied_curvatures.std()) <= 0.0:
            raise RuntimeError("A curvature-head update did not produce item-conditional c values")
        if float(varied_curvatures.min()) < model.curvature_min or float(varied_curvatures.max()) > model.curvature_max:
            raise RuntimeError("Learned item curvature escaped its positive bounded range")
        sample_c = varied_curvatures[:8]
        pair_c = torch.sqrt(sample_c[:, None] * sample_c[None, :])
        latent = model.encoder(batch[:8])
        distances = _poincare_pairwise_distances(latent, latent, pair_c)
        c_pair = pair_c.unsqueeze(-1)
        point_x = _expmap0_tangent(latent[:, None, :], c_pair)
        point_y = _expmap0_tangent(latent[None, :, :], c_pair)
        difference = _mobius_add(-point_x, point_y, c_pair)
        norm = torch.linalg.vector_norm(difference, dim=-1)
        sqrt_pair_c = pair_c.sqrt()
        reference = (2.0 / sqrt_pair_c) * torch.atanh(
            (sqrt_pair_c * norm).clamp(max=1.0 - 1e-6)
        )
        asymmetry = float((distances - distances.t()).abs().max())
        if not torch.isfinite(distances).all() or asymmetry > 1e-5 or not torch.allclose(
            distances, reference, rtol=2e-4, atol=2e-4
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
            raise RuntimeError("Item-code Poincare kernel disagrees with direct Möbius distance")
        if not torch.isfinite(distances).all():
            raise RuntimeError("Pairwise Poincare distance is non-finite")
    coincident_x = torch.zeros(4, 32, device=device, requires_grad=True)
    coincident_y = torch.zeros(4, 32, device=device, requires_grad=True)
    coincident_c = torch.ones(4, device=device)
    coincident_row_distance = _poincare_row_distances(
        coincident_x, coincident_y, coincident_c
    ).sum()
    coincident_row_grad = torch.autograd.grad(
        coincident_row_distance, (coincident_x, coincident_y), retain_graph=False
    )
    if not all(torch.isfinite(gradient).all() for gradient in coincident_row_grad):
        raise RuntimeError("Coincident row-distance gradients are non-finite")
    coincident_pair_x = torch.zeros(4, 32, device=device, requires_grad=True)
    coincident_pair_distance = _poincare_pairwise_distances(
        coincident_pair_x, coincident_pair_x, torch.ones(4, 4, device=device)
    ).sum()
    coincident_pair_grad = torch.autograd.grad(
        coincident_pair_distance, coincident_pair_x, retain_graph=False
    )[0]
    if not torch.isfinite(coincident_pair_grad).all():
        raise RuntimeError("Coincident pairwise-distance gradients are non-finite")
    print(
        f"MVG_OK total_requires_grad={total_loss.requires_grad} grad_fn={type(total_loss.grad_fn).__name__} "
        f"initial_c_min={float(curvatures.min().detach()):.6f} initial_c_max={float(curvatures.max().detach()):.6f} "
        f"updated_c_std={float(varied_curvatures.std()):.9g} pairwise_asymmetry={asymmetry:.3g} "
        f"behavior_loss={float(behavior_loss.detach()):.6f}"
    )


if __name__ == "__main__":
    main()
