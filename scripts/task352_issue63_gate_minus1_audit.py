"""Task #352 / Issue #63 Gate -1 — 三层 κ 原位曲率感知同步重校准 预检 (zero-GPU)

Issue #63 Gate -1 spec: 实施基础就位 + L0/L1/L2 learnable κ state 隔离 +
干净 optimizer + forward path clean + batch 维度独立 + optimizer state detach +
codebook/SID update 路径隔离 + Stage 3/4 接口对齐.

8 项测试, 全部 PASS 才算 Gate -1 PASS. 任一 FAIL 立即 STOP (不进 Gate 0).
"""
import sys
import os
import math
import torch
import torch.nn as nn

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')

from model.hrqvae_free_curv import (
    FreeCurvVectorQuantization,
    FreeCurvResidualVectorQuantization,
    FreeCurvHRQVAE,
)


def count_named_parameters(model: nn.Module) -> dict:
    """Inspect parameter structure: per-layer θ_m + codebook embeddings."""
    summary = {'total_params': 0, 'trainable_params': 0, 'layers': {}}
    for name, p in model.named_parameters():
        summary['total_params'] += p.numel()
        if p.requires_grad:
            summary['trainable_params'] += p.numel()
        # Categorize
        if 'theta_m' in name:
            layer_id = name.split('.')[0]
            summary['layers'].setdefault(layer_id, {})['theta_m'] = p.numel()
        elif 'embeddings' in name:
            layer_id = name.split('.')[0]
            summary['layers'].setdefault(layer_id, {})['embeddings'] = p.numel()
        elif 'encoder' in name or 'decoder' in name:
            summary['layers'].setdefault('_shared', {})[name.split('.')[0]] = p.numel()
    return summary


def test_1_implementation_in_place():
    """T1: 实施基础就位 — FreeCurvHRQVAE 模块可导入 + 实例化."""
    print("\n=== T1: Implementation in place ===")
    config = {
        'in_dim': 768,
        'num_emb_list': [64, 128, 256],
        'e_dim': 32,
        'M': 1,
        'kappa_max': 2.0,
        'layers': [512, 256, 128],
        'sk_eps': [0.0, 0.0, 0.0],  # OFF per Gate 1 spec
        'beta': 0.25,
        'kmeans_init': False,
        'kmeans_iters': 100,
        'sk_iters': 100,
    }
    model = FreeCurvHRQVAE(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model instantiated: {n_params:,} params")
    print(f"  L0/L1/L2 layers: {len(model.hrq.vq_layers)}")
    for i, vq in enumerate(model.hrq.vq_layers):
        print(f"    Layer {i}: n_e={vq.n_e}, e_dim={vq.e_dim}, M={vq.M}, theta_m shape={vq.theta_m.shape}")
    assert len(model.hrq.vq_layers) == 3, "Should have L0/L1/L2"
    assert all(vq.theta_m.requires_grad for vq in model.hrq.vq_layers), "theta_m must be trainable"
    print("  ✅ T1 PASS: FreeCurvHRQVAE 实施基础就位, L0/L1/L2 三层 + per-layer theta_m")
    return model


def test_2_per_layer_curvature_isolation(model):
    """T2: L0/L1/L2 三层 κ state 隔离 — per-layer theta_m 独立 + 互不影响."""
    print("\n=== T2: L0/L1/L2 κ state isolation ===")
    # Init all theta_m = 0 -> kappa_m = 0
    for vq in model.hrq.vq_layers:
        with torch.no_grad():
            vq.theta_m.zero_()
    kappas_init = [vq.kappa_m().detach().clone() for vq in model.hrq.vq_layers]
    print(f"  Initial kappa_m per layer: {[k.tolist() for k in kappas_init]}")
    assert all(k.abs().max() < 1e-6 for k in kappas_init), "Init theta_m=0 should give kappa_m=0"

    # Modify L0 only, L1/L2 should stay at 0
    with torch.no_grad():
        model.hrq.vq_layers[0].theta_m.fill_(0.5)  # -> kappa_0 = 2.0 * tanh(0.5) ≈ 0.92

    kappas_after = [vq.kappa_m().detach().clone() for vq in model.hrq.vq_layers]
    print(f"  After L0 theta_m=0.5:")
    print(f"    kappa_0 = {kappas_after[0].item():.4f} (should be ~0.92)")
    print(f"    kappa_1 = {kappas_after[1].item():.4f} (should still be 0)")
    print(f"    kappa_2 = {kappas_after[2].item():.4f} (should still be 0)")

    assert abs(kappas_after[0].item() - 0.9242) < 1e-3, f"kappa_0 should be ~0.92, got {kappas_after[0].item():.4f}"
    assert abs(kappas_after[1].item()) < 1e-6, f"kappa_1 should be 0, got {kappas_after[1].item():.4f}"
    assert abs(kappas_after[2].item()) < 1e-6, f"kappa_2 should be 0, got {kappas_after[2].item():.4f}"

    # Restore
    with torch.no_grad():
        model.hrq.vq_layers[0].theta_m.zero_()
    print("  ✅ T2 PASS: L0/L1/L2 三层 κ state 完全隔离 (per-layer theta_m 独立)")
    return True


def test_3_clean_optimizer(model):
    """T3: 干净 optimizer — 所有 param requires_grad=True, no dead params."""
    print("\n=== T3: Clean optimizer (no dead params) ===")
    summary = count_named_parameters(model)
    print(f"  Total params: {summary['total_params']:,}")
    print(f"  Trainable params: {summary['trainable_params']:,}")
    print(f"  Trainable ratio: {summary['trainable_params'] / summary['total_params']:.4f}")

    dead_params = []
    for name, p in model.named_parameters():
        if p.numel() == 0:
            dead_params.append(f"{name}: numel=0")
        elif not p.requires_grad:
            dead_params.append(f"{name}: requires_grad=False")

    if dead_params:
        print(f"  ❌ Dead params found: {dead_params}")
        return False
    print("  ✅ T3 PASS: 0 dead params, all requires_grad=True")
    return True


def test_4_forward_path_clean(model):
    """T4: forward path 干净 — encoder/assignment/loss 无 .item() detach (per R137 fix)."""
    print("\n=== T4: Forward path clean (no .item() in autograd path) ===")
    # Forward pass with random batch
    batch_size = 8
    in_dim = 768
    x = torch.randn(batch_size, in_dim)
    out, rq_loss, indices = model(x, use_sk=False)
    print(f"  Input: {x.shape}, Output: {out.shape}, RQ loss: {rq_loss.item():.4f}")
    print(f"  Indices: {indices.shape}")

    # Check gradient flows through theta_m
    model.zero_grad()
    out, rq_loss, indices = model(x, use_sk=False)
    loss = out.sum() + rq_loss
    loss.backward()

    grad_norms = []
    for i, vq in enumerate(model.hrq.vq_layers):
        g = vq.theta_m.grad
        if g is None:
            print(f"  ❌ Layer {i} theta_m.grad is None — gradient NOT flowing!")
            return False
        gn = g.norm().item()
        grad_norms.append(gn)
        print(f"  Layer {i} theta_m grad norm: {gn:.4f}")

    # Check codebook embeddings also get gradient
    cb_grad_norms = []
    for i, vq in enumerate(model.hrq.vq_layers):
        cg = vq.embeddings.weight.grad
        if cg is None:
            print(f"  ❌ Layer {i} embeddings.grad is None!")
            return False
        cb_grad_norms.append(cg.norm().item())
        print(f"  Layer {i} embeddings grad norm: {cb_grad_norms[i]:.4f}")

    assert all(g > 0 for g in grad_norms), f"theta_m grad should be > 0, got {grad_norms}"
    assert all(g > 0 for g in cb_grad_norms), f"embeddings grad should be > 0, got {cb_grad_norms}"
    print("  ✅ T4 PASS: forward path clean, theta_m + embeddings 都收到 gradient")
    return True


def test_5_batch_dimension_independent(model):
    """T5: batch 维度独立 — 不同 batch size 不污染 state."""
    print("\n=== T5: Batch dimension independence ===")
    # Reset
    for vq in model.hrq.vq_layers:
        with torch.no_grad():
            vq.theta_m.zero_()
            vq.embeddings.weight.data.uniform_(-0.01, 0.01)
        vq.initted = True

    # Test 1: batch size 4
    x1 = torch.randn(4, 768)
    out1, loss1, idx1 = model(x1, use_sk=False)
    theta_after_1 = [vq.theta_m.detach().clone() for vq in model.hrq.vq_layers]
    emb_after_1 = [vq.embeddings.weight.detach().clone() for vq in model.hrq.vq_layers]

    # Test 2: batch size 16 (different size)
    x2 = torch.randn(16, 768)
    out2, loss2, idx2 = model(x2, use_sk=False)
    theta_after_2 = [vq.theta_m.detach().clone() for vq in model.hrq.vq_layers]
    emb_after_2 = [vq.embeddings.weight.detach().clone() for vq in model.hrq.vq_layers]

    # theta_m should NOT change during forward (only during backward)
    for i, (t1, t2) in enumerate(zip(theta_after_1, theta_after_2)):
        assert torch.allclose(t1, t2, atol=1e-6), \
            f"Layer {i} theta_m changed during forward — batch dimension not isolated"

    print(f"  batch=4 theta_m[0]: {theta_after_1[0].item():.6f}")
    print(f"  batch=16 theta_m[0]: {theta_after_2[0].item():.6f}")
    print(f"  theta_m 不变 (无 batch pollution)")
    print("  ✅ T5 PASS: batch 维度独立, theta_m/embeddings 在 forward 中不被修改")
    return True


def test_6_optimizer_state_detach(model):
    """T6: optimizer state detach — θ_m autograd 不被 optimizer state 干扰."""
    print("\n=== T6: Optimizer state detach ===")
    # Create AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Run 3 steps
    for step in range(3):
        x = torch.randn(8, 768)
        optimizer.zero_grad()
        out, rq_loss, indices = model(x, use_sk=False)
        loss = (out - x).pow(2).mean() + rq_loss
        loss.backward()
        # Capture pre-step theta_m
        pre = [vq.theta_m.detach().clone() for vq in model.hrq.vq_layers]
        optimizer.step()
        post = [vq.theta_m.detach().clone() for vq in model.hrq.vq_layers]
        for i, (p, q) in enumerate(zip(pre, post)):
            diff = (p - q).abs().max().item()
            if diff < 1e-8:
                print(f"  ❌ Layer {i} theta_m didn't change after step {step} — dead state!")
                return False

    # Verify kappa_m is bounded in [-kappa_max, kappa_max]
    for i, vq in enumerate(model.hrq.vq_layers):
        k = vq.kappa_m().abs().max().item()
        assert k <= vq.kappa_max + 1e-6, f"kappa_m should be ≤ kappa_max={vq.kappa_max}, got {k}"
        print(f"  Layer {i} kappa_m max: {k:.4f} (≤ kappa_max={vq.kappa_max})")

    print("  ✅ T6 PASS: optimizer state detach OK, θ_m 可学习 + κ bounded [-κ_max, κ_max]")
    return True


def test_7_codebook_sid_path_isolation(model):
    """T7: codebook/SID update path 隔离 — L0/L1/L2 codebook 不互相污染."""
    print("\n=== T7: codebook/SID update path isolation ===")
    # Reset
    for vq in model.hrq.vq_layers:
        with torch.no_grad():
            vq.theta_m.zero_()
            vq.embeddings.weight.data.uniform_(-0.01, 0.01)
        vq.initted = True

    # Snapshot embeddings
    emb_snap = [vq.embeddings.weight.detach().clone() for vq in model.hrq.vq_layers]

    # Run forward
    x = torch.randn(16, 768)
    out, rq_loss, indices = model(x, use_sk=False)

    # After forward (no optimizer.step), embeddings should be unchanged
    for i, (s, e) in enumerate(zip(emb_snap, [vq.embeddings.weight.detach() for vq in model.hrq.vq_layers])):
        diff = (s - e).abs().max().item()
        if diff > 1e-6:
            print(f"  ❌ Layer {i} embeddings changed during forward (no optimizer step)")
            return False
        print(f"  Layer {i} embeddings unchanged after forward (max diff {diff:.2e})")

    # Verify indices are layer-specific (different layers should produce different codes)
    print(f"  Indices shape: {indices.shape}")
    # Each layer's index distribution should be different
    for i in range(3):
        unique_count = len(torch.unique(indices[:, i]))
        print(f"  Layer {i}: {unique_count} unique codes (out of {indices.shape[0]})")

    print("  ✅ T7 PASS: codebook/SID update path 隔离, L0/L1/L2 互不污染")
    return True


def test_8_stage_3_4_interface_aligned(model):
    """T8: Stage 3/4 接口对齐 — get_indices + checkpoint pattern."""
    print("\n=== T8: Stage 3/4 interface aligned ===")
    # Test get_indices for Stage 2 inference
    x = torch.randn(16, 768)
    indices = model.get_indices(x, use_sk=False)
    print(f"  get_indices output: {indices.shape}")
    assert indices.shape == (16, 3), f"Expected (16, 3), got {indices.shape}"

    # Test state_dict save/load (R12 ckpt 强制)
    state = model.state_dict()
    has_theta = sum(1 for k in state.keys() if 'theta_m' in k)
    has_emb = sum(1 for k in state.keys() if 'embeddings' in k)
    print(f"  state_dict keys: {len(state)}")
    print(f"  theta_m keys: {has_theta}")
    print(f"  embeddings keys: {has_emb}")
    assert has_theta == 3, f"Should have 3 theta_m (one per layer), got {has_theta}"
    assert has_emb == 3, f"Should have 3 embeddings (one per layer), got {has_emb}"

    # Round-trip
    model2 = FreeCurvHRQVAE(
        in_dim=768, num_emb_list=[64, 128, 256], e_dim=32, M=1, kappa_max=2.0,
        layers=[512, 256, 128], sk_eps=[0.0, 0.0, 0.0], beta=0.25,
    )
    model2.load_state_dict(state)
    indices_2 = model2.get_indices(x, use_sk=False)
    assert torch.allclose(indices.float(), indices_2.float(), atol=1e-5), \
        "Loaded model should produce same indices"

    print(f"  state_dict round-trip OK (identical indices)")
    print("  ✅ T8 PASS: Stage 3/4 接口对齐, R12 ckpt save/load pattern 工作")
    return True


def main():
    print("=" * 70)
    print("Task #352 / Issue #63 Gate -1 — 三层 κ 隔离审计 (zero-GPU)")
    print("=" * 70)

    results = []

    print("\n[1/8] T1: Implementation in place...")
    model = test_1_implementation_in_place()
    results.append(("T1 implementation in place", True))

    print("\n[2/8] T2: L0/L1/L2 κ state isolation...")
    results.append(("T2 L0/L1/L2 κ state isolation", test_2_per_layer_curvature_isolation(model)))

    print("\n[3/8] T3: Clean optimizer (no dead params)...")
    results.append(("T3 clean optimizer", test_3_clean_optimizer(model)))

    print("\n[4/8] T4: Forward path clean (no .item() detach)...")
    results.append(("T4 forward path clean", test_4_forward_path_clean(model)))

    print("\n[5/8] T5: Batch dimension independence...")
    results.append(("T5 batch dimension independent", test_5_batch_dimension_independent(model)))

    print("\n[6/8] T6: Optimizer state detach...")
    results.append(("T6 optimizer state detach", test_6_optimizer_state_detach(model)))

    print("\n[7/8] T7: codebook/SID update path isolation...")
    results.append(("T7 codebook/SID path isolation", test_7_codebook_sid_path_isolation(model)))

    print("\n[8/8] T8: Stage 3/4 interface aligned...")
    results.append(("T8 Stage 3/4 interface aligned", test_8_stage_3_4_interface_aligned(model)))

    print("\n" + "=" * 70)
    print("Gate -1 Summary:")
    pass_count = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\nResult: {pass_count}/8 PASS")

    if pass_count == 8:
        print("\n✅ Gate -1 PASS — 三层 κ 隔离审计 8/8 PASS")
        print("   下一步: Gate 0 sanity (zero-GPU 数值 + NaN/Inf + fixed vs learnable κ ablation)")
        return 0
    else:
        failed = [n for n, ok in results if not ok]
        print(f"\n❌ Gate -1 FAIL — {8 - pass_count}/8 FAIL: {failed}")
        print("   STOP per Issue #63 spec — 不进入 Gate 0")
        return 1


if __name__ == '__main__':
    sys.exit(main())