"""Fast forward/backward and SID-export check for the promoted candidate."""
from types import SimpleNamespace
import numpy as np
import torch

from model import RQVAE
from train_rqvae import _extend_collisions, _repository_root, ARM_NAME, RESULT_DIR


def main():
    torch.manual_seed(7)
    cfg = SimpleNamespace(hidden_sizes=(16, 8), codebook_num=3,
                          codebook_size=(8, 8, 8), codebook_dim=4,
                          dropout=0.0, beta=0.25, vq_type="vq",
                          ema_decay=0.99, fix_code_embs=False,
                          sk_epsilon=-1.0, sk_iters=-1)
    model = RQVAE(cfg, in_dim=6)
    model.rq.hyperbolic_on = True
    x = torch.randn(5, 6)
    successors = torch.randn(5, 6)
    pred, qloss, _, tokens = model(x, successors)
    loss = model.compute_loss(x, pred, qloss)[0]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.mlp[1].weight.grad is not None
    assert torch.isfinite(model.encoder.mlp[1].weight.grad).all()
    with torch.no_grad():
        tokens = model.get_indices(x, successor_means=successors).numpy()
    exported = _extend_collisions(tokens, [8, 8, 8])
    assert exported.shape == (5, 4) and len(np.unique(exported, axis=0)) == 5
    assert _repository_root().is_dir() and ARM_NAME == "TIGER_tangent_successor_residual"
    assert RESULT_DIR.name == ARM_NAME
    print("PASS forward_backward_finite_gradients_sid_export promoted_root", _repository_root())


if __name__ == "__main__":
    main()
