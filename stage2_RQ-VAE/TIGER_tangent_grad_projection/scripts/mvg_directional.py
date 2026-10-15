from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_rqvae as tr


def main():
    torch.set_num_threads(1)
    tr.set_seed(tr.SEED)
    embeddings = tr.maybe_apply_pca(tr.load_embeddings(tr.EMBEDDING_FILE), tr.PCA_DIM, tr.SEED)
    frame = pd.read_parquet(tr.TRAIN_FILE)
    pairs = tr.TransitionPairs(frame)
    batch_size = min(tr.PAIR_BATCH_SIZE_PER_RANK, len(pairs))
    if batch_size < 2:
        raise RuntimeError("Directional MVG requires at least two real transition pairs")
    pair_ids = pairs.pairs[:batch_size]
    source_ids, target_ids = pair_ids[:, 0], pair_ids[:, 1]
    all_embeddings = torch.from_numpy(embeddings)
    pair_batch = all_embeddings[torch.cat((source_ids, target_ids))]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = tr.RQVAE(tr._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    if tr.XAVIER_INIT:
        tr.initialize_tiger_weights(model)
    heads = tr.DirectionalHeads().to(device)
    model.directional_heads = heads
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    if train_ids.size < max(tr.CODEBOOK_SIZE):
        raise RuntimeError(f"Codebook init needs at least {max(tr.CODEBOOK_SIZE)} distinct train targets; got {train_ids.size}")
    model.init_codebook(all_embeddings[torch.from_numpy(train_ids)].to(device))

    pair_batch = pair_batch.to(device)
    source_ids, target_ids = source_ids.to(device), target_ids.to(device)
    residuals = tr._residuals_with_trace(model, pair_batch)
    loss = tr.directional_residual_loss(residuals, source_ids, target_ids, heads, symmetric=False)
    per_level = [tr.directional_residual_loss(residuals[level:level + 1], source_ids, target_ids, heads, symmetric=False) for level in range(residuals.shape[0])]
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite directional loss: {loss.item()}")
    loss.backward()
    encoder_grads = [p.grad for p in model.encoder.parameters()]
    head_grads = {name: [p.grad for p in module.parameters()] for name, module in (("f_source", heads.f_source), ("f_target", heads.f_target))}
    encoder_norm = math.sqrt(sum(float(g.detach().pow(2).sum()) for g in encoder_grads if g is not None))
    head_norms = {name: math.sqrt(sum(float(g.detach().pow(2).sum()) for g in grads if g is not None)) for name, grads in head_grads.items()}
    residual_norms = [float(torch.linalg.vector_norm(x.detach())) for x in residuals]
    values = [loss.detach(), *[x.detach() for x in per_level], residuals.detach(), torch.tensor(encoder_norm), *[torch.tensor(v) for v in head_norms.values()]]
    if not all(torch.isfinite(x).all() for x in values) or float(loss.detach()) <= 0 or any(float(x.detach()) <= 0 for x in per_level) or any(v <= 0 for v in residual_norms) or encoder_norm <= 0 or any(v <= 0 for v in head_norms.values()):
        raise RuntimeError(f"non-finite or zero MVG value: loss={loss.item()}, level_losses={[float(x.detach()) for x in per_level]}, residual_norms={residual_norms}, encoder={encoder_norm}, heads={head_norms}")
    params = list(model.parameters())
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=tr.LR, weight_decay=tr.WEIGHT_DECAY)
    optimizer.step()
    moved = any(not torch.equal(old, new.detach()) for old, new in zip(before, params))
    if not moved:
        raise RuntimeError("AdamW step did not move any parameter")
    print(f"loss={loss.item():.9g}")
    print("per_level_loss=" + repr([float(x.detach()) for x in per_level]))
    print(f"residual_shapes={tuple(residuals.shape)}")
    print("per_level_residual_norm=" + repr(residual_norms))
    print(f"encoder_gradient_norm={encoder_norm:.9g}")
    print("head_gradient_norms=" + repr({name: float(value) for name, value in head_norms.items()}))
    print(f"adamw_parameters_moved={moved}")


if __name__ == "__main__":
    main()
