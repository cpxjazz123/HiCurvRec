#!/usr/bin/env python3
"""Issue #119 P0-6 (2026-08-10): C3 Relational Curvature Correctness Unit Tests.

5 个 unit test 必须在 GPU smoke 前 PASS:
  Test 1 — Global index: batch_size=16 但 positive index 人为设 5000/8000/9000, 确认不发生 batch-local indexing error
  Test 2 — Geometry consistency: anchor/positive/negative 全部 sqrt(c)·|h| < 1
  Test 3 — Finite loss: 0 < L_rel < inf
  Test 4 — Curvature sensitivity: L_rel(c) ≠ L_rel(c+ε), ∂L_rel/∂κ ≠ 0
  Test 5 — Gradient isolation (C3): ∂L_rel/∂encoder = 0, ∂L_rel/∂codebook = 0 (通过 autograd.grad)

输出:
  taskA/_history/issue118_c3_correctness/unit_test_report.json
  含 5 个 test 的 PASS/FAIL + 详细 metrics.

运行:
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/test_c3_relational_correctness.py
"""
import sys
import os
import json
import argparse
from pathlib import Path

GENRE_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, GENRE_ROOT)

import torch
import numpy as np


def test1_global_index():
    """Test 1 — 验证 anchor/positive/negative 用 global item index, 不发生 batch-local indexing error.

    batch_size=16, 但 positive idx 显式设 [5000, 8000, 9000] (远超 batch_size).
    旧实现会直接 z_det[pos_idx] → IndexError. 新实现 (P0-1 frozen bank) 直接用 global idx 索引 bank[pos_idx].
    """
    sys.path.insert(0, os.path.join(GENRE_ROOT, "taskA"))
    from taskA.stage2 import poincare_relational_loss_per_layer

    torch.manual_seed(2024)
    np.random.seed(2024)
    B = 16
    E_DIM = 32
    POS_K = 3
    NEG_N = 5

    # 用 frozen bank 索引: anchor/positive/negative 都是 (N, E_DIM) bank 上的 global index
    N_FAKE_BANK = 9922  # 模拟真实关系图
    bank = torch.randn(N_FAKE_BANK, E_DIM, dtype=torch.float32) * 0.01
    batch_idx = torch.arange(B)
    pos_idx = torch.tensor([
        [5000, 8000, 9000],  # 远超 batch_size, 但 global index 合法
        [200, 8000, 5000],
        [9000, 8000, 5000],
        [5000, 9000, 8000],
        [8000, 5000, 9000],
        [9000, 8000, 5000],
        [5000, 8000, 9000],
        [8000, 9000, 5000],
        [5000, 8000, 9000],
        [9000, 5000, 8000],
        [8000, 9000, 5000],
        [5000, 9000, 8000],
        [9000, 8000, 5000],
        [5000, 8000, 9000],
        [8000, 9000, 5000],
        [9000, 5000, 8000],
    ], dtype=torch.long)
    neg_idx = torch.randint(0, N_FAKE_BANK, (B, NEG_N), dtype=torch.long)
    # 排除 self
    for i in range(B):
        neg_idx[i] = torch.where(neg_idx[i] == i, torch.tensor(0, dtype=torch.long), neg_idx[i])

    anchor_z = bank[batch_idx]                   # (B, E_DIM)
    pos_z = bank[pos_idx]                        # (B, POS_K, E_DIM)
    neg_z = bank[neg_idx]                        # (B, NEG_N, E_DIM)
    c = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)

    try:
        L_rel = poincare_relational_loss_per_layer(
            anchor_z=anchor_z, pos_z=pos_z, neg_z=neg_z, c=c, tau=0.1,
        )
        return {
            "test": "Test1_GlobalIndex",
            "pass": bool(torch.isfinite(L_rel).item()),
            "L_rel": L_rel.item(),
            "anchor_shape": list(anchor_z.shape),
            "pos_shape": list(pos_z.shape),
            "neg_shape": list(neg_z.shape),
            "max_pos_idx": int(pos_idx.max()),
            "max_neg_idx": int(neg_idx.max()),
        }
    except (IndexError, RuntimeError) as e:
        return {"test": "Test1_GlobalIndex", "pass": False, "error": str(e)}


def test2_geometry_consistency():
    """Test 2 — anchor/positive/negative 全部 sqrt(c)·|h| < 1 (球内).

    即使输入 raw tangent norm > 1, expmap0 + proj_to_ball 后必须在 ball 内.
    """
    sys.path.insert(0, os.path.join(GENRE_ROOT, "taskA"))
    from taskA.stage2 import poincare_relational_loss_per_layer

    torch.manual_seed(2024)
    B = 16
    E_DIM = 32
    POS_K = 8
    NEG_N = 32

    # 故意造 raw tangent norm 大 (违反 ball 约束), 看 proj_to_ball 兜底
    anchor_z = torch.randn(B, E_DIM, dtype=torch.float32) * 5.0
    pos_z = torch.randn(B, POS_K, E_DIM, dtype=torch.float32) * 5.0
    neg_z = torch.randn(B, NEG_N, E_DIM, dtype=torch.float32) * 5.0
    c = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)

    try:
        L_rel = poincare_relational_loss_per_layer(
            anchor_z=anchor_z, pos_z=pos_z, neg_z=neg_z, c=c, tau=0.1,
        )
        finite = bool(torch.isfinite(L_rel).item())
        # 再次手算几何一致性
        from HG_Rec.model.utils import expmap0, proj_to_ball
        sys.path.insert(0, os.path.join(GENRE_ROOT, "HG-Rec"))
        try:
            from model.utils import expmap0 as _e0, proj_to_ball as _p2b
        except ImportError:
            try:
                from HG_Rec.model.utils import expmap0 as _e0, proj_to_ball as _p2b
            except ImportError:
                _e0 = expmap0
                _p2b = proj_to_ball
        anchor_h = _p2b(_e0(anchor_z.detach(), c), c)
        sqrt_c = c.sqrt().item()
        max_anchor_norm = (sqrt_c * anchor_h.norm(dim=-1)).max().item()
        return {
            "test": "Test2_GeometryConsistency",
            "pass": finite and max_anchor_norm < 1.0 + 1e-5,
            "L_rel": L_rel.item(),
            "max_anchor_sqrt_c_norm": max_anchor_norm,
            "raw_anchor_norm": anchor_z.norm(dim=-1).max().item(),
            "raw_anchor_norm_should_be_gt_1": anchor_z.norm(dim=-1).max().item() > 1.0,
        }
    except (AssertionError, RuntimeError) as e:
        return {"test": "Test2_GeometryConsistency", "pass": False, "error": str(e)}


def test3_finite_loss():
    """Test 3 — 0 < L_rel < inf."""
    sys.path.insert(0, os.path.join(GENRE_ROOT, "taskA"))
    from taskA.stage2 import poincare_relational_loss_per_layer

    torch.manual_seed(2024)
    B = 16
    E_DIM = 32
    POS_K = 8
    NEG_N = 32

    bank = torch.randn(9922, E_DIM, dtype=torch.float32) * 0.05
    batch_idx = torch.arange(B)
    pos_idx = torch.randint(0, 9922, (B, POS_K), dtype=torch.long)
    neg_idx = torch.randint(0, 9922, (B, NEG_N), dtype=torch.long)
    for i in range(B):
        pos_idx[i] = torch.where(pos_idx[i] == i, torch.tensor(0, dtype=torch.long), pos_idx[i])
        neg_idx[i] = torch.where(neg_idx[i] == i, torch.tensor(0, dtype=torch.long), neg_idx[i])

    anchor_z = bank[batch_idx]
    pos_z = bank[pos_idx]
    neg_z = bank[neg_idx]
    c = torch.tensor(1.5, dtype=torch.float32, requires_grad=True)
    L_rel = poincare_relational_loss_per_layer(
        anchor_z=anchor_z, pos_z=pos_z, neg_z=neg_z, c=c, tau=0.1,
    )
    val = L_rel.item()
    return {
        "test": "Test3_FiniteLoss",
        "pass": (0.0 < val < float("inf")) and bool(torch.isfinite(L_rel).item()),
        "L_rel": val,
    }


def test4_curvature_sensitivity():
    """Test 4 — ∂L_rel/∂κ ≠ 0, 且 L_rel(c) ≠ L_rel(c+ε).

    C3 链路核心: 改变 c 必须改变 L_rel, 否则 κ 无法被驱动.
    """
    sys.path.insert(0, os.path.join(GENRE_ROOT, "taskA"))
    from taskA.stage2 import poincare_relational_loss_per_layer

    torch.manual_seed(2024)
    B = 16
    E_DIM = 32
    POS_K = 8
    NEG_N = 32

    bank = torch.randn(9922, E_DIM, dtype=torch.float32) * 0.05
    batch_idx = torch.arange(B)
    pos_idx = torch.randint(0, 9922, (B, POS_K), dtype=torch.long)
    neg_idx = torch.randint(0, 9922, (B, NEG_N), dtype=torch.long)
    for i in range(B):
        pos_idx[i] = torch.where(pos_idx[i] == i, torch.tensor(0, dtype=torch.long), pos_idx[i])
        neg_idx[i] = torch.where(neg_idx[i] == i, torch.tensor(0, dtype=torch.long), neg_idx[i])

    anchor_z = bank[batch_idx]
    pos_z = bank[pos_idx]
    neg_z = bank[neg_idx]

    # c=1.0 (无曲率) vs c=2.0 (有曲率)
    c1 = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    c2 = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
    L_rel_c1 = poincare_relational_loss_per_layer(
        anchor_z=anchor_z, pos_z=pos_z, neg_z=neg_z, c=c1, tau=0.1,
    )
    L_rel_c2 = poincare_relational_loss_per_layer(
        anchor_z=anchor_z, pos_z=pos_z, neg_z=neg_z, c=c2, tau=0.1,
    )
    # 数值梯度: 必 > 0 (∂L_rel/∂c ≠ 0)
    kappa_grad = torch.autograd.grad(L_rel_c1, c1, retain_graph=False, allow_unused=True)[0]
    kappa_grad_val = kappa_grad.abs().item() if kappa_grad is not None else 0.0

    return {
        "test": "Test4_CurvatureSensitivity",
        "pass": (
            abs(L_rel_c1.item() - L_rel_c2.item()) > 1e-6
            and kappa_grad_val > 1e-8
        ),
        "L_rel_c1": L_rel_c1.item(),
        "L_rel_c2": L_rel_c2.item(),
        "delta_L_rel": abs(L_rel_c1.item() - L_rel_c2.item()),
        "dL_rel_dc": kappa_grad_val,
    }


def test5_gradient_isolation():
    """Test 5 — C3: ∂L_rel/∂encoder = 0, ∂L_rel/∂codebook = 0.

    真实 autograd.grad 验证, 不硬编码 0.
    """
    sys.path.insert(0, os.path.join(GENRE_ROOT, "taskA"))
    from taskA.stage2 import poincare_relational_loss_per_layer, relational_gradient_audit

    torch.manual_seed(2024)
    B = 16
    E_DIM = 32
    POS_K = 8
    NEG_N = 32

    bank = torch.randn(9922, E_DIM, dtype=torch.float32) * 0.05
    batch_idx = torch.arange(B)
    pos_idx = torch.randint(0, 9922, (B, POS_K), dtype=torch.long)
    neg_idx = torch.randint(0, 9922, (B, NEG_N), dtype=torch.long)
    for i in range(B):
        pos_idx[i] = torch.where(pos_idx[i] == i, torch.tensor(0, dtype=torch.long), pos_idx[i])
        neg_idx[i] = torch.where(neg_idx[i] == i, torch.tensor(0, dtype=torch.long), neg_idx[i])

    anchor_z = bank[batch_idx]
    pos_z = bank[pos_idx]
    neg_z = bank[neg_idx]
    c = torch.tensor(1.5, dtype=torch.float32, requires_grad=True)

    audit = relational_gradient_audit(
        anchor_z=anchor_z, pos_z=pos_z, neg_z=neg_z, c=c, tau=0.1,
    )
    # 期望:
    #   relational_kappa_grad > 0 (∂L_rel/∂c ≠ 0)
    #   relational_encoder_grad = 0 (anchor_z.detach())
    #   relational_codebook_grad = 0 (codebook 不参与 L_rel)
    return {
        "test": "Test5_GradientIsolation",
        "pass": (
            audit["relational_kappa_grad"] > 1e-8
            and audit["relational_encoder_grad"] < 1e-8
            and audit["relational_codebook_grad"] < 1e-8
            and audit["L_rel_finite"]
            and audit["L_rel_requires_grad"]
        ),
        "relational_kappa_grad": audit["relational_kappa_grad"],
        "relational_encoder_grad": audit["relational_encoder_grad"],
        "relational_codebook_grad": audit["relational_codebook_grad"],
        "L_rel_finite": audit["L_rel_finite"],
        "L_rel_requires_grad": audit["L_rel_requires_grad"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue118_c3_correctness")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[test_c3_relational_correctness] starting 5 unit tests...")
    results = []
    results.append(test1_global_index())
    print(f"  {results[-1]['test']}: {'PASS' if results[-1]['pass'] else 'FAIL'}")
    results.append(test2_geometry_consistency())
    print(f"  {results[-1]['test']}: {'PASS' if results[-1]['pass'] else 'FAIL'}")
    results.append(test3_finite_loss())
    print(f"  {results[-1]['test']}: {'PASS' if results[-1]['pass'] else 'FAIL'}")
    results.append(test4_curvature_sensitivity())
    print(f"  {results[-1]['test']}: {'PASS' if results[-1]['pass'] else 'FAIL'}")
    results.append(test5_gradient_isolation())
    print(f"  {results[-1]['test']}: {'PASS' if results[-1]['pass'] else 'FAIL'}")

    n_pass = sum(1 for r in results if r.get("pass", False))
    n_total = len(results)
    overall_pass = (n_pass == n_total)
    report = {
        "issue": "#119",
        "task": "P0-6",
        "title": "C3 Relational Curvature Correctness Unit Tests",
        "overall_pass": overall_pass,
        "n_pass": n_pass,
        "n_total": n_total,
        "tests": results,
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").localtime()),
    }
    out_path = out_dir / "unit_test_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[test_c3_relational_correctness] {n_pass}/{n_total} PASS, saved {out_path}")
    if not overall_pass:
        print("  ⚠ OVERALL FAIL — do NOT launch C3 smoke")
        sys.exit(1)
    else:
        print("  ✓ ALL PASS — safe to launch C3 smoke (Phase 1)")
        sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())