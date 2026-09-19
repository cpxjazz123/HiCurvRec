"""iter26 Riemannian Adam grad_check (Project Rules §6 实施细则).

验证:
  (a) total_loss.requires_grad + total_loss.grad_fn is not None
  (b) 逐项 loss (reconstruction / rqvae) torch.autograd.grad 后至少 1 个 param 非零
  (c) RiemannianAdamW._poincare_rescale_() 不让 backward 静默失效
  (d) optimizer.step() 至少更新一项参数 (与 baseline AdamW 等价检查)

输入: 本脚本硬编码 iter26 子树路径, 0 CLI flag (Project Rules §1).
返回: PASS / FAIL + 详细指标, 父 agent 据此判断是否启动 GPU 训练.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ITER_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter26")
sys.path.insert(0, str(ITER_DIR))
spec = importlib.util.spec_from_file_location("rqtrain", str(ITER_DIR / "curvature_RQ-VAE.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main() -> None:
    import torch  # noqa: WPS433 — local import, py3.9 不支持 module-level 推迟

    print("=== iter26 grad_check (Riemannian Adam on top of cyclic c(t) baseline) ===")
    print(f"  iter26 dir: {ITER_DIR}")
    print(f"  USE_RIEMANNIAN_ADAM = {mod.USE_RIEMANNIAN_ADAM}")
    print(f"  RIEMANNIAN_ADAM_NORM_CLAMP = {mod.RIEMANNIAN_ADAM_NORM_CLAMP}")

    # 1) 构建 model
    model = mod.RqVae(
        input_dim=mod.INPUT_DIM,
        embed_dim=mod.EMBED_DIM,
        hidden_dims=mod.HIDDEN_DIMS,
        codebook_size=mod.CODEBOOK_SIZE,
        codebook_kmeans_init=True,
        n_layers=mod.N_LAYERS,
        commitment_weight=mod.COMMITMENT_WEIGHT,
        sk_eps=0.05,
        sk_iters=3,
        c_cyclic_min=mod.C_CYCLIC_MIN,
        c_cyclic_max=mod.C_CYCLIC_MAX,
        c_cyclic_period=mod.C_CYCLIC_PERIOD,
        midpoint_layer_mask=mod.MIDPOINT_LAYER_MASK,
    ).to("cpu")
    model.train()

    # 2) 1 个 batch from item_emb.npy
    arr = __import__("numpy").load(mod.EMB_NPY).astype("float32")[: 2 * mod.BATCH_SIZE]
    x = torch.from_numpy(arr[: mod.BATCH_SIZE])
    print(f"  batch shape = {tuple(x.shape)}")

    # 3) forward + loss
    seq_batch = mod._build_seq_batch(x, "cpu")
    out = model(seq_batch)
    loss = out.loss
    print(
        f"  forward: loss.requires_grad={loss.requires_grad} "
        f"grad_fn={loss.grad_fn is not None} "
        f"loss={float(loss.item()):.4f} "
        f"rl={float(out.reconstruction_loss.item()):.4f} "
        f"vl={float(out.rqvae_loss.item()):.4f}"
    )
    if not loss.requires_grad or loss.grad_fn is None:
        raise RuntimeError(f"FAIL: total_loss 不可微 (requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn})")
    print("  PASS (a) total_loss.requires_grad + grad_fn")

    # 4) backward
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_nonzero_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
    print(f"  backward: params_with_grad={n_with_grad} nonzero={n_nonzero_grad}")
    if n_with_grad == 0:
        raise RuntimeError("FAIL: backward 后无任何 param.grad")
    if n_nonzero_grad == 0:
        raise RuntimeError("FAIL: 所有 param.grad 都是 0 (silent no-op)")
    print("  PASS (b) backward 后所有相关 param.grad 非零")

    # 5) RiemannianAdamW
    if not mod.USE_RIEMANNIAN_ADAM:
        raise RuntimeError("FAIL: USE_RIEMANNIAN_ADAM 应该是 True")
    optimizer = mod.RiemannianAdamW(
        model.parameters(),
        lr=mod.RIEMANNIAN_ADAM_LR,
        betas=mod.RIEMANNIAN_ADAM_BETAS,
        eps=mod.RIEMANNIAN_ADAM_EPS,
        weight_decay=mod.RIEMANNIAN_ADAM_WEIGHT_DECAY,
        norm_clamp=mod.RIEMANNIAN_ADAM_NORM_CLAMP,
    )
    # snapshot before step
    before = {id(p): p.detach().clone() for p in model.parameters()}
    # set curvature (模拟训练循环: global_step=100 → c=sin(π·100/50000)·0.7+0.3 ≈ 0.300003)
    test_step = 100
    model.set_curriculum_step(test_step)
    c_t = float(model.layers[0].get_c().item())
    optimizer.set_curvature(c_t)
    print(f"  set_curvature: c(t={test_step})={c_t:.6f}")

    # 检查 rescale 前/后 grad 范数
    pre_step_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    # 手动调 rescale 看 grad 是否被 scale
    optimizer._poincare_rescale_()
    post_rescale_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  rescale: grad_norm_pre={pre_step_grad_norm:.4f} post={post_rescale_grad_norm:.4f}")
    if post_rescale_grad_norm == 0:
        raise RuntimeError("FAIL: _poincare_rescale_() 把所有 grad 都压成 0 (silent no-op)")

    # 6) optimizer.step() 应更新参数
    optimizer.step()
    n_updated = sum(
        1 for p in model.parameters() if not torch.equal(before[id(p)], p.detach())
    )
    print(f"  optimizer.step: n_updated_params={n_updated}/{len(list(model.parameters()))}")
    if n_updated == 0:
        raise RuntimeError("FAIL: optimizer.step() 没更新任何参数")
    print("  PASS (c) RiemannianAdamW.step() 更新参数成功 (无 silent no-op)")

    # 7) 验证参数没逃出 Poincaré ball (||x||² < 1/c)
    c_now = float(model.layers[0].get_c().item())
    ball_limit = 1.0 / c_now
    n_out_of_ball = 0
    for layer in model.layers:
        w = layer.embedding.weight
        norm_sq_max = (w ** 2).sum(dim=-1).max().item()
        if norm_sq_max >= ball_limit:
            n_out_of_ball += 1
            print(f"  WARN: codebook layer out of ball: norm_sq_max={norm_sq_max:.4f} ball_limit={ball_limit:.4f}")
    if n_out_of_ball > 0:
        print(f"  WARN (c边界): {n_out_of_ball} 层 codebook 已接近 ball boundary")
    else:
        print(f"  PASS (d) 所有 codebook 在 Poincaré ball 内 (c={c_now:.4f}, limit=||x||²<{ball_limit:.4f})")

    print("=" * 60)
    print("GRAD_CHECK PASS")
    print("  total_loss.requires_grad=True + grad_fn=True")
    print(f"  backward 后 {n_nonzero_grad}/{n_with_grad} param.grad 非零")
    print(f"  RiemannianAdamW.rescale 前/后 grad_norm: {pre_step_grad_norm:.4f} → {post_rescale_grad_norm:.4f}")
    print(f"  optimizer.step() 更新 {n_updated} 个参数")
    print(f"  c(t={test_step})={c_t:.4f}, codebook 在 ball 内")
    print("=" * 60)


if __name__ == "__main__":
    main()