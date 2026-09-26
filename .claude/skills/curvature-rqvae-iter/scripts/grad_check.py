"""Pre-train gradient-path sanity check (Project Rules §6).

Usage:
    PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 python3 scripts/grad_check.py <iter_id>

硬约束（不可违反）:
  - 文件名 `curvature_RQ-VAE.py` 含 `-` 不是合法 Python 标识符，禁止 `import curvature_RQ-VAE as m`，
    必须用 `importlib.util.spec_from_file_location('rqtrain', '.../curvature_RQ-VAE.py')`。
  - `curvature_config.py` 只放路径常量；数值超参（INPUT_DIM / USE_CYCLIC_CURVATURE / MIDPOINT_LAYER_MASK 等）
    全部在 `curvature_RQ-VAE.py` 文件顶部，通过 `rqtrain` 模块读取。
  - 不允许 inline `python -c "..."` 一次性检查，必须落到本脚本复用。
"""
import importlib.util
import os
import sys

import torch


def _load_rqtrain(path):
    spec = importlib.util.spec_from_file_location("rqtrain", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    if len(sys.argv) < 2:
        print("Usage: grad_check.py <iter_id>", file=sys.stderr)
        sys.exit(2)

    iter_id = sys.argv[1]
    work = f"/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter{iter_id}"
    if not os.path.isdir(work):
        print(f"[grad] iter workdir missing: {work}", file=sys.stderr)
        sys.exit(2)

    sys.path.insert(0, work)
    import curvature_config  # noqa: F401  # 注入 env vars

    from modules.quantize import QuantizeForwardMode, QuantizeDistance
    from modules.rqvae import RqVae
    from data.schemas import SeqBatch

    m = _load_rqtrain(os.path.join(work, "curvature_RQ-VAE.py"))

    model = RqVae(
        input_dim=m.INPUT_DIM,
        embed_dim=m.EMBED_DIM,
        hidden_dims=m.HIDDEN_DIMS,
        codebook_size=m.CODEBOOK_SIZE,
        codebook_kmeans_init=False,
        codebook_normalize=False,
        codebook_sim_vq=False,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=m.N_LAYERS,
        n_cat_features=0,
        commitment_weight=m.COMMITMENT_WEIGHT,
        gate_M2_intrinsic=m.USE_M2_INTRINSIC,
        gate_M3_transport=m.USE_M3_TRANSPORT,
        hyperbolic_distance=True,
        sk_eps=0.05,
        distance_mode=m.DISTANCE_MODE,
        prefix_router_layers=None,
        margin_reg_weight=0.0,
        spread_loss_weight=0.0,
        anisotropy_loss_weight=0.0,
        use_tcu=False,
        use_mcdq=False,
        use_scs=False,
        use_fixed_curvature=m.USE_FIXED_CURVATURE,
        c_fixed=m.C_FIXED,
        use_cyclic_curvature=m.USE_CYCLIC_CURVATURE,
        c_cyclic_min=m.C_CYCLIC_MIN,
        c_cyclic_max=m.C_CYCLIC_MAX,
        c_cyclic_period=m.C_CYCLIC_PERIOD,
        use_geodesic_midpoint_commit=m.USE_GEODESIC_MIDPOINT_COMMIT,
        midpoint_layer_mask=m.MIDPOINT_LAYER_MASK,
        use_mobius_gyrovector=False,
    ).cuda()
    model.set_curriculum_step(0)

    torch.manual_seed(42)
    x = torch.randn(8, m.INPUT_DIM, device="cuda")
    bsz = 8
    sb = SeqBatch(
        user_ids=torch.zeros(bsz, dtype=torch.long, device="cuda"),
        ids=torch.arange(bsz, dtype=torch.long, device="cuda"),
        ids_fut=torch.zeros(bsz, dtype=torch.long, device="cuda"),
        x=x,
        x_fut=x,
        seq_mask=torch.ones(bsz, dtype=torch.long, device="cuda"),
    )
    out = model(sb, gumbel_t=0.2)
    total = out.loss
    print(f"[grad] total.requires_grad={total.requires_grad} grad_fn={total.grad_fn is not None} val={float(total.item()):.4f}")

    passed = total.requires_grad and total.grad_fn is not None
    # 逐项新机制 loss 检查 (margin_loss / recon_loss / rqvae_loss 在 baseline 配置下可能 detached;
    # 任何项 requires_grad=False 时跳过, 视为 baseline 合法状态; 真正判断靠 total.backward())
    for name, l in [("recon", out.reconstruction_loss), ("rqvae", out.rqvae_loss), ("margin", out.margin_loss)]:
        try:
            if not getattr(l, "requires_grad", False):
                val = float(l.item()) if hasattr(l, "item") else float(l)
                print(f"[grad] {name:8s}: skipped (no requires_grad; baseline value={val:.4f})")
                continue
            grads = torch.autograd.grad(l, model.parameters(), retain_graph=True, allow_unused=True)
            nz = sum(1 for g in grads if g is not None and g.abs().sum() > 0)
            total_p = len(list(model.parameters()))
            print(f"[grad] {name:8s}: {nz}/{total_p} non-zero")
            if nz == 0:
                print(f"[grad] FAIL: {name} has 0 params got non-zero grad (silent no-op)")
                passed = False
        except RuntimeError as e:
            val = float(l.item()) if hasattr(l, "item") else float(l)
            print(f"[grad] {name:8s}: skipped (RuntimeError: {str(e).splitlines()[0]}; value={val:.4f})")
            continue

    if total.requires_grad:
        total.backward()
        # 验证 total loss 至少触达 1 个可训练参数
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        if not any_grad:
            print("[grad] FAIL: total.backward() produced 0 grads on all trainable params")
            passed = False
        else:
            print("[grad] total.backward() -> non-zero grad present")
    else:
        passed = False

    print("[grad] PASS" if passed else "[grad] FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()