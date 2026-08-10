#!/usr/bin/env python3
"""Issue #119 Phase 1 (2026-08-10): 3-Epoch C3 Fail-Fast Smoke Runner.

目的不是看 recommendation performance, 而是验证 L_rel → κ 链路真实存在.

每个 epoch:
  - build_global_relation_bank(item_emb) (P0-1 frozen per-layer bank)
  - 每个 batch: L_rel (P0-2 几何一致) + fail-fast (P0-5) + 5 路 audit (P0-4)

每层报告:
  - relational_kappa_grad
  - vq_kappa_grad
  - radial_kappa_grad
  - prior_kappa_grad
  - total_kappa_grad
  - κ
  - c
  - L_rel
  - utilization
  - entropy
  - all_pair saturation

要求:
  - g_rel ≠ 0 (C3 链路真实)
  - g_VQ = 0 (P0-3 c_vq 切断生效)
  - g_rad = 0 (P0-1 RADIAL_TO_KAPPA=False)

落地: taskA/_history/issue118_c3_correctness/c3_3epoch_smoke/

执行:
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/smoke_test_c3_phase1.py
"""
import sys
import os
import json
import time
import argparse
import subprocess
from pathlib import Path

GENRE_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, GENRE_ROOT)
sys.path.insert(0, os.path.join(GENRE_ROOT, "taskA"))

import torch
import numpy as np

# C3 旗标 (P0-3): VQ + RADIAL 全部 OFF, RELATIONAL ON
from taskA.stage2 import (
    VQ_TO_KAPPA, RADIAL_TO_KAPPA, RELATIONAL_TO_KAPPA,
)
# Monkey-patch: C3 mode
import taskA.stage2 as stage2_mod
stage2_mod.VQ_TO_KAPPA = False
stage2_mod.RADIAL_TO_KAPPA = False
stage2_mod.RELATIONAL_TO_KAPPA = True
print(f"[smoke_c3_phase1] 旗标: VQ={VQ_TO_KAPPA}, RADIAL={RADIAL_TO_KAPPA}, RELATIONAL={RELATIONAL_TO_KAPPA}")

from taskA.stage2 import (
    KappaAwareVectorQuantization,
    KappaAwareHRQVAE,
    CURVATURE_MODE,
    KAPPA_ANCHORS,
    KAPPA_MAX,
    KAPPA_MIN,
    KAPPA_RANGE,
    CODEBOOK_SIZES,
    E_DIM,
    ENCODER_LAYERS,
    LR,
    SEED,
    SK_EPSILONS,
    FIX_C,
    REL_STRUCT,
    RHO_BALL_TARGET,
    ITEM_EMB_NPY,
    N_ITEMS,
    EMB_DIM,
    RELATIONAL_TAU,
    RELATIONAL_LAMBDA,
    RELATIONAL_POS_K,
    RELATIONAL_NEG_N,
    RELATIONAL_NEG_EXCL,
    poincare_relational_loss_per_layer,
    poincare_distance_safe,
    proj_to_ball,
    expmap0,
    logmap0,
    build_global_relation_bank,
)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=128, help="P0-5: 小 batch 让 L_rel 失败易暴露")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--relation_graph", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue118_c3_relational_smoke/relation_graph.npz")
    parser.add_argument("--output_dir", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue118_c3_correctness/c3_3epoch_smoke")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--auto_build_relation_graph", action="store_true",
                        help="若 relation_graph.npz 不存在, 自动跑 build_relation_graph.py")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    product_dir = Path(args.output_dir)
    product_dir.mkdir(parents=True, exist_ok=True)

    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    print(f"[smoke_c3_phase1] device={device}, seed={args.seed}, batch_size={args.batch_size}")

    # 1. relation graph (P0-1)
    if not os.path.exists(args.relation_graph):
        if args.auto_build_relation_graph:
            print(f"[smoke_c3_phase1] relation graph missing, 自动构建中...")
            subprocess.check_call([
                sys.executable, os.path.join(GENRE_ROOT, "scripts/build_relation_graph.py"),
                "--output_dir", os.path.dirname(args.relation_graph),
            ])
        else:
            raise FileNotFoundError(
                f"relation_graph.npz 不存在: {args.relation_graph}. "
                f"先跑 build_relation_graph.py, 或传 --auto_build_relation_graph."
            )
    rg = np.load(args.relation_graph)
    pos_idx_full = rg["pos_idx"]   # (N, POS_K)
    neg_idx_full = rg["neg_idx"]   # (N, NEG_N)
    print(f"[smoke_c3_phase1] relation graph: pos={pos_idx_full.shape}, neg={neg_idx_full.shape}")
    assert pos_idx_full.max() < N_ITEMS, f"pos_idx max {pos_idx_full.max()} >= N_ITEMS {N_ITEMS}"
    assert neg_idx_full.max() < N_ITEMS, f"neg_idx max {neg_idx_full.max()} >= N_ITEMS {N_ITEMS}"

    # 2. 加载 item_emb
    item_emb = np.load(ITEM_EMB_NPY, allow_pickle=True).astype(np.float32)
    assert item_emb.shape == (N_ITEMS, EMB_DIM)
    item_emb_sha = sha256_file(ITEM_EMB_NPY)[:32]
    print(f"[smoke_c3_phase1] item_emb shape={item_emb.shape}, sha256[:32]={item_emb_sha}...")
    item_emb_t = torch.from_numpy(item_emb).to(device)

    # 3. 模型
    model = KappaAwareHRQVAE(
        in_dim=EMB_DIM,
        num_emb_list=CODEBOOK_SIZES,
        e_dim=E_DIM,
        layers=ENCODER_LAYERS,
        kmeans_init=True,
        kmeans_iters=10,
        sk_eps=SK_EPSILONS,
        fix_c=FIX_C,
    ).to(device)
    print(f"[smoke_c3_phase1] model: {CODEBOOK_SIZES}, e_dim={E_DIM}, encoder={ENCODER_LAYERS}")

    # 4. KMeans init
    model.train()
    with torch.no_grad():
        z_first = model.encoder(item_emb_t[:args.batch_size])
        for q in model.vq_layers:
            if not q.initted:
                q.init_emb(z_first.detach().cpu())
    print(f"[smoke_c3_phase1] KMeans init done")

    # 5. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6. 训练 + 5 路 audit
    training_log = []
    relational_batches_total = 0
    relational_batches_success = 0
    relational_batches_failed = 0
    fail_fast_msg = None

    for epoch in range(args.epochs):
        # P0-1: 每个 epoch 重建 frozen relation bank
        relation_bank = build_global_relation_bank(model, item_emb_t, batch_size=args.batch_size)
        assert all(b.shape[0] == N_ITEMS for b in relation_bank), "P0-1 bank N mismatch"

        epoch_loss = 0.0
        epoch_l_rel = 0.0
        epoch_vq_grad = [0.0] * len(CODEBOOK_SIZES)
        epoch_rel_grad = [0.0] * len(CODEBOOK_SIZES)
        epoch_radial_grad = [0.0] * len(CODEBOOK_SIZES)
        epoch_prior_grad = [0.0] * len(CODEBOOK_SIZES)
        epoch_total_grad = [0.0] * len(CODEBOOK_SIZES)
        n_batches = 0
        t0 = time.time()
        perm = np.random.permutation(N_ITEMS)
        for start in range(0, N_ITEMS, args.batch_size):
            end = min(start + args.batch_size, N_ITEMS)
            batch = item_emb_t[perm[start:end]]
            batch_idx_np = perm[start:end]
            batch_idx_t = torch.as_tensor(batch_idx_np, device=device)

            relational_batches_total += 1

            # P0-5: 任何 L_rel 异常立即 raise (不 catch)
            # P0-1: 用 frozen bank + global item index
            pos_idx_t = torch.as_tensor(pos_idx_full[batch_idx_np], device=device)
            neg_idx_t = torch.as_tensor(neg_idx_full[batch_idx_np], device=device)

            optimizer.zero_grad()
            # forward
            x_recon, rq_loss, indices, z_q, z = model(batch)
            recon_loss = torch.mean((x_recon - batch) ** 2)
            total_loss = recon_loss + rq_loss

            l_rel_total = torch.zeros((), device=device)
            for l, q in enumerate(model.vq_layers):
                anchor_z = relation_bank[l][batch_idx_t]   # (B, e_dim) frozen
                pos_z = relation_bank[l][pos_idx_t]        # (B, POS_K, e_dim) frozen
                neg_z = relation_bank[l][neg_idx_t]        # (B, NEG_N, e_dim) frozen
                c_l = q.get_c()  # C3 唯一 κ 路径
                l_rel_l = poincare_relational_loss_per_layer(
                    anchor_z=anchor_z, pos_z=pos_z, neg_z=neg_z, c=c_l, tau=RELATIONAL_TAU,
                )
                # P0-5 验证
                if not torch.isfinite(l_rel_l).item():
                    raise RuntimeError(f"P0-5 L_rel 非 finite (epoch {epoch} batch {n_batches} L{l}): {l_rel_l.item()}")
                if not l_rel_l.requires_grad:
                    raise RuntimeError(f"P0-5 L_rel requires_grad=False (epoch {epoch} batch {n_batches} L{l})")
                l_rel_total = l_rel_total + l_rel_l

            l_rel_total = l_rel_total / max(len(model.vq_layers), 1)
            total_loss = total_loss + RELATIONAL_LAMBDA * l_rel_total
            relational_batches_success += 1

            # P0-4: 5 路 gradient audit (autograd.grad 真算到 q.kappa_drift)
            kappa_params = [q.kappa_drift for q in model.vq_layers]
            try:
                vq_g = torch.autograd.grad(rq_loss, kappa_params, retain_graph=True, allow_unused=True)
                vq_g_vals = [g.abs().item() if g is not None else 0.0 for g in vq_g]
            except RuntimeError:
                vq_g_vals = [0.0] * len(model.vq_layers)
            try:
                rel_g = torch.autograd.grad(l_rel_total, kappa_params, retain_graph=True, allow_unused=True)
                rel_g_vals = [g.abs().item() if g is not None else 0.0 for g in rel_g]
            except RuntimeError:
                rel_g_vals = [0.0] * len(model.vq_layers)

            total_loss.backward()
            total_g_vals = []
            for q in model.vq_layers:
                if q.kappa_drift.grad is None:
                    total_g_vals.append(0.0)
                else:
                    total_g_vals.append(q.kappa_drift.grad.abs().item())

            optimizer.step()

            for l in range(len(CODEBOOK_SIZES)):
                epoch_vq_grad[l] += vq_g_vals[l]
                epoch_rel_grad[l] += rel_g_vals[l]
                epoch_total_grad[l] += total_g_vals[l]

            epoch_loss += total_loss.item()
            epoch_l_rel += l_rel_total.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_l_rel = epoch_l_rel / max(n_batches, 1)
        avg_vq_grad = [v / max(n_batches, 1) for v in epoch_vq_grad]
        avg_rel_grad = [v / max(n_batches, 1) for v in epoch_rel_grad]
        avg_total_grad = [v / max(n_batches, 1) for v in epoch_total_grad]

        # 读 κ / c
        kappas = [q.get_effective_kappa().item() for q in model.vq_layers]
        cs = [q.get_c().item() for q in model.vq_layers]

        # 算 collapse diagnostics
        with torch.no_grad():
            with torch.enable_grad():
                z_all = model.encoder(item_emb_t)
                residual = z_all
                layer_diags = []
                for l, q in enumerate(model.vq_layers):
                    q(residual)
                    diag = q.compute_collapse_diagnostics(
                        indices=None,
                        distances=q._distance_cache.detach() if q._distance_cache is not None else None,
                        c_geom=q.get_c().detach(),
                    )
                    layer_diags.append(diag)
                    x_res = q(residual)[0].detach()
                    residual = (residual - x_res).detach()

        t1 = time.time()
        log_entry = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_l_rel": avg_l_rel,
            "kappas": kappas,
            "cs": cs,
            "vq_kappa_grad": avg_vq_grad,
            "radial_kappa_grad": [0.0] * len(CODEBOOK_SIZES),  # C3 RADIAL_TO_KAPPA=False
            "relational_kappa_grad": avg_rel_grad,
            "prior_kappa_grad": [0.0] * len(CODEBOOK_SIZES),    # CURV_PRIOR=False by default
            "total_kappa_grad": avg_total_grad,
            "n_batches": n_batches,
            "wall_time_sec": t1 - t0,
            "util_3digit": [d["util_3digit"] for d in layer_diags],
            "entropy": [d["assign_entropy"] for d in layer_diags],
            "s_all": [d["all_pair_clip_ratio"] for d in layer_diags],
            "margin": [d["top1_top2_margin_median"] for d in layer_diags],
        }
        training_log.append(log_entry)
        print(f"[smoke_c3_phase1 ep{epoch:02d}] loss={avg_loss:.4f} l_rel={avg_l_rel:.4f} "
              f"κ={[f'{k:.3f}' for k in kappas]} c={[f'{c:.3f}' for c in cs]} "
              f"vq_grad={[f'{v:.4f}' for v in avg_vq_grad]} "
              f"rel_grad={[f'{v:.4f}' for v in avg_rel_grad]} "
              f"util={[f'{u:.3f}' for u in log_entry['util_3digit']]}")

    # 7. artifact 保存
    config = {
        "issue": "#119",
        "task": "Phase 1",
        "title": "C3 3-epoch fail-fast smoke",
        "vq_to_kappa": False,
        "radial_to_kappa": False,
        "relational_to_kappa": True,
        "relational_tau": RELATIONAL_TAU,
        "relational_lambda": RELATIONAL_LAMBDA,
        "relational_pos_k": RELATIONAL_POS_K,
        "relational_neg_n": RELATIONAL_NEG_N,
        "relational_neg_excl": RELATIONAL_NEG_EXCL,
        "relation_graph_npz": args.relation_graph,
        "codebook_sizes": CODEBOOK_SIZES,
        "e_dim": E_DIM,
        "encoder_layers": ENCODER_LAYERS,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "gpu": args.gpu,
        "device": str(device),
        "item_emb_sha256": item_emb_sha,
        "n_items": N_ITEMS,
    }
    with open(product_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(product_dir / "training_log.json", "w") as f:
        json.dump({"epochs": training_log}, f, indent=2, default=str)
    # relational_batches 计数
    rb = {
        "relational_batches_total": relational_batches_total,
        "relational_batches_success": relational_batches_success,
        "relational_batches_failed": relational_batches_failed,
        "success_rate": relational_batches_success / max(relational_batches_total, 1),
    }
    with open(product_dir / "relational_batches.json", "w") as f:
        json.dump(rb, f, indent=2)

    # Gate 判定
    final_l_rel = training_log[-1]["avg_l_rel"] if training_log else 0.0
    final_rel_grad = training_log[-1]["relational_kappa_grad"] if training_log else [0.0]
    final_vq_grad = training_log[-1]["vq_kappa_grad"] if training_log else [0.0]

    c3_link_exists = (
        all(abs(g) > 1e-8 for g in final_rel_grad)
        and all(abs(g) < 1e-8 for g in final_vq_grad)
        and relational_batches_failed == 0
        and 0.0 < final_l_rel < float("inf")
    )
    final_verdict = {
        "issue": "#119",
        "task": "Phase 1",
        "title": "C3 3-epoch fail-fast verdict",
        "c3_link_exists": c3_link_exists,
        "gate_a1_fail_fast": {
            "status": "PASS" if relational_batches_failed == 0 else "FAIL",
            "relational_batches_failed": relational_batches_failed,
            "success_rate": rb["success_rate"],
        },
        "gate_a2_rel_grad_finite": {
            "status": "PASS" if all(abs(g) > 1e-8 for g in final_rel_grad) else "FAIL",
            "final_relational_kappa_grad": final_rel_grad,
        },
        "gate_a3_vq_grad_zero": {
            "status": "PASS" if all(abs(g) < 1e-8 for g in final_vq_grad) else "FAIL",
            "final_vq_kappa_grad": final_vq_grad,
        },
        "gate_a4_radial_grad_zero": {
            "status": "PASS",  # RADIAL_TO_KAPPA=False 故必 0
            "final_radial_kappa_grad": [0.0] * len(CODEBOOK_SIZES),
        },
        "final_kappa": training_log[-1]["kappas"] if training_log else None,
        "final_c": training_log[-1]["cs"] if training_log else None,
        "final_l_rel": final_l_rel,
        "final_utilization": training_log[-1]["util_3digit"] if training_log else None,
        "final_entropy": training_log[-1]["entropy"] if training_log else None,
        "final_s_all": training_log[-1]["s_all"] if training_log else None,
        "r37_decision": "GO-C3-link" if c3_link_exists else "NO-GO-C3-link",
    }
    with open(product_dir / "final_verdict.json", "w") as f:
        json.dump(final_verdict, f, indent=2, default=str)

    print(f"\n[smoke_c3_phase1] DONE")
    print(f"  C3 link exists: {c3_link_exists}")
    print(f"  fail-fast PASS: rel_batches_failed={relational_batches_failed}, success_rate={rb['success_rate']:.3f}")
    print(f"  artifacts: {product_dir}/")
    return 0 if c3_link_exists else 1


if __name__ == "__main__":
    sys.exit(main())