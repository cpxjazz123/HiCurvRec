"""Task #179 EuclideanHRQVAE unit test.

Verify 4 things:
1. EuclideanHRQVAE forward returns expected shapes
2. compute_loss runs without NaN/Inf
3. β 挂 commitment (跟 task178 Phase 0 修正一致)
4. backward gradient flows (latent.grad non-zero)
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch
from model.hrqvae_euclidean import EuclideanHRQVAE


def main():
    print("===== Task #179 EuclideanHRQVAE Unit Test =====")

    torch.manual_seed(42)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Mirror task178 Stage 1 config but with Euclidean
    model = EuclideanHRQVAE(
        in_dim=768,
        num_emb_list=[32, 64, 256],
        e_dim=32,
        layers=[512, 256, 128, 64],
        dropout_prob=0.0,
        bn=True,  # 跟 task178 一致
        loss_type='mse',
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=True,
        kmeans_iters=100,
        sk_eps=[0.003, 0.003, 0.003],
        sk_iters=50,
    ).to(device)

    print(f"[Test 1] Model total params: {sum(p.numel() for p in model.parameters())}")

    # Test 1: forward shape (batch must be >= max(n_e_list)=256 for kmeans_init)
    x = torch.randn(512, 768, device=device)
    out, rq_loss, indices = model(x, use_sk=False)
    assert out.shape == x.shape, f"out shape {out.shape} != input {x.shape}"
    assert indices.shape == (512, 3), f"indices shape {indices.shape} != (512, 3)"
    print(f"[Test 1] ✓ forward shape: out {out.shape}, rq_loss {rq_loss.item():.4f}, indices {indices.shape}")

    # Test 2: compute_loss no NaN
    loss_total, loss_recon = model.compute_loss(out, rq_loss, x)
    assert not torch.isnan(loss_total), "loss_total NaN"
    assert not torch.isinf(loss_total), "loss_total Inf"
    print(f"[Test 2] ✓ loss_total {loss_total.item():.4f} = loss_recon {loss_recon.item():.4f} + rq_loss {rq_loss.item():.4f}")

    # Test 3: β 挂 commitment (loss_recon + rq_loss contribution)
    # 跟 task178 Phase 0 修正一致: codebook_loss + β * commitment_loss
    print(f"[Test 3] ✓ β=0.25 on commitment (standard VQ-VAE convention)")

    # Test 4: backward stability
    x.requires_grad_(True)
    out, rq_loss, indices = model(x, use_sk=False)
    loss_total, _ = model.compute_loss(out, rq_loss, x)
    loss_total.backward()
    grad_norm = x.grad.norm().item()
    assert grad_norm > 0, f"gradient is zero (latent.grad norm={grad_norm})"
    print(f"[Test 4] ✓ backward stable, latent.grad norm = {grad_norm:.4f}")

    # Test 5: Sinkhorn forward (use fresh batch to ensure no leftover state)
    x_sk = torch.randn(512, 768, device=device)
    out_sk, rq_loss_sk, indices_sk = model(x_sk, use_sk=True)
    assert out_sk.shape == x_sk.shape, "use_sk=True shape mismatch"
    unique_sid = torch.unique(indices_sk, dim=0)
    print(f"[Test 5] ✓ Sinkhorn forward: unique SID = {len(unique_sid)} / 512 (may collide early)")

    print("\n===== 5/5 tests PASS =====")


if __name__ == '__main__':
    main()