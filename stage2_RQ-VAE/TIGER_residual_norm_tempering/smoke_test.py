"""Minimal CPU check for residual scaling, gradients, and unique SID export."""
from types import SimpleNamespace

import numpy as np
import torch

from model import RQVAE
from train_rqvae import _extend_collisions


def main():
    torch.manual_seed(7)
    config = SimpleNamespace(
        hidden_sizes=(16, 8), dropout=0.0, codebook_num=3,
        codebook_size=(8, 8, 8), codebook_dim=4, vq_type="vq",
        beta=0.25, sk_epsilon=-1.0, sk_iters=-1,
        ema_decay=0.99, fix_code_embs=False,
    )
    model = RQVAE(config, in_dim=6)
    x = torch.randn(24, 6)
    model.init_codebook(x)
    with torch.no_grad():
        scales = model.rq.calibrate_scales(model.encoder(x))
    model.train()
    reconstruction, quant_loss, _, tokens = model(x)
    loss, _ = model.compute_loss(x, reconstruction, quant_loss)
    assert torch.isfinite(loss) and torch.isfinite(scales).all()
    loss.backward()
    changed_params = [p for p in model.parameters() if p.requires_grad]
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in changed_params)
    assert model.rq.stage_scales.requires_grad is False and bool(model.rq.scales_frozen)
    sid = _extend_collisions(tokens.detach().numpy().astype(np.int64), list(config.codebook_size))
    assert sid.shape[0] == x.shape[0] and len(np.unique(sid, axis=0)) == len(sid)
    print(f"PASS loss={loss.item():.6f}; scales={scales.tolist()}; tokens={tuple(tokens.shape)}; unique SID rows={len(sid)}")


if __name__ == "__main__":
    main()
