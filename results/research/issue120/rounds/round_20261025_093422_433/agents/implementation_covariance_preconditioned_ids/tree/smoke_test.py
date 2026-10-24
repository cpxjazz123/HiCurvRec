"""Minimal synthetic proof of whitening, RQ-VAE backward, and unique SID export."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model import RQVAE
from train_rqvae import (
    _extend_collisions,
    apply_train_whitener,
    fit_train_whitener,
)


def main() -> None:
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    # Deliberately anisotropic data; fit only first 48 rows and transform all.
    x = (rng.normal(size=(64, 8)) * np.array([8, 4, 2, 1, .5, .3, .2, .1])).astype("float32")
    mean, matrix = fit_train_whitener(x, np.arange(48))
    y = apply_train_whitener(x, mean, matrix)
    assert np.isfinite(y).all() and y.shape == x.shape
    cfg = SimpleNamespace(
        hidden_sizes=(16, 8), codebook_num=3, codebook_size=(8, 8, 8),
        codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq",
        ema_decay=0.99, fix_code_embs=False, sk_epsilon=0.003, sk_iters=5,
    )
    model = RQVAE(cfg, in_dim=y.shape[1])
    model.init_codebook(torch.from_numpy(y[:48]))
    batch = torch.from_numpy(y[:16])
    reconstructed, quant_loss, _, _ = model(batch)
    loss, _ = model.compute_loss(batch, reconstructed, quant_loss)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g).item() for g in grads)
    with torch.no_grad():
        tokens = model.get_indices(torch.from_numpy(y), infer_use_sk=False).numpy()
    sid = _extend_collisions(tokens, [8, 8, 8])
    assert sid.shape == (len(x), 4) and len(np.unique(sid, axis=0)) == len(x)
    assert np.isfinite(loss.item())
    print(f"PASS loss={loss.item():.6g} tokens={tokens.shape} unique_sid={len(sid)}")


if __name__ == "__main__":
    main()
