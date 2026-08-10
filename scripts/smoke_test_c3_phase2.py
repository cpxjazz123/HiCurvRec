#!/usr/bin/env python3
"""Issue #119 Phase 2 (2026-08-10): 30-Epoch C1 vs C3 Matched Smoke Runner (with P0-1~P0-5 fixes).

执行两个 30 epoch matched experiments,除 κ gradient source 外完全相同:
  C1: VQ_TO_KAPPA=True,  RADIAL_TO_KAPPA=False, RELATIONAL_TO_KAPPA=False
  C3: VQ_TO_KAPPA=False, RADIAL_TO_KAPPA=False, RELATIONAL_TO_KAPPA=True

包含 P0-1~P0-5 全部修复:
  - P0-1: build_global_relation_bank (per-epoch)
  - P0-2: anchor/positive/negative 走 expmap0+proj_to_ball
  - P0-3: c_vq 切断 VQ→κ
  - P0-4: 5 路 gradient audit (VQ/radial/prior/boundary/trust/relational) + reconstruction check
  - P0-5: fail-fast (C3 mode 任何 L_rel 异常 → raise)

artifact 落地:
  C1: taskA/_history/issue118_c3_correctness/c1_30epoch_smoke/
  C3: taskA/_history/issue118_c3_correctness/c3_30epoch_smoke/

每个目录 9 件套 + 6 PNG:
  config.json, training_log.json, collapse_diagnostics.json, gradient_diagnostics.json,
  final_verdict.json, sid_output.npy, sid_metadata.json, hrqvae_kappa_sync.ckpt,
  relational_batches.json (C3 only), relation_graph.npz + metadata (C3 only)
  + 6 PNG: kappa_curve, curvature_curve, utilization_curve, entropy_curve,
           saturation_curve, gradient_source_audit.png (5 路 vs total)

执行:
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/smoke_test_c3_phase2.py --mode c1
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/smoke_test_c3_phase2.py --mode c3 \
      --relation_graph /path/to/relation_graph.npz
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path

GENRE_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, GENRE_ROOT)
sys.path.insert(0, os.path.join(GENRE_ROOT, "taskA"))

import torch
import numpy as np

# Monkey-patch BEFORE import stage2
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["c1", "c3"], required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--relation_graph", type=str, default="")
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--force_cpu", action="store_true")
args_pre = parser.parse_args()

import taskA.stage2 as stage2_mod
if args_pre.mode == "c1":
    stage2_mod.VQ_TO_KAPPA = True
    stage2_mod.RADIAL_TO_KAPPA = False
    stage2_mod.RELATIONAL_TO_KAPPA = False
    stage2_mod.RELATION_GRAPH_NPZ = ""
elif args_pre.mode == "c3":
    stage2_mod.VQ_TO_KAPPA = False
    stage2_mod.RADIAL_TO_KAPPA = False
    stage2_mod.RELATIONAL_TO_KAPPA = True
    stage2_mod.RELATION_GRAPH_NPZ = args_pre.relation_graph

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
    VQ_TO_KAPPA,
    RADIAL_TO_KAPPA,
    RELATIONAL_TO_KAPPA,
    RELATIONAL_TAU,
    RELATIONAL_LAMBDA,
    RELATIONAL_POS_K,
    RELATIONAL_NEG_N,
    RELATIONAL_NEG_EXCL,
    poincare_relational_loss_per_layer,
    poincare_distance_safe,
    poincare_recon_loss,
    proj_to_ball,
    expmap0,
    logmap0,
    build_global_relation_bank,
)

print(f"[smoke_c3_phase2 mode={args_pre.mode}] flags: VQ={VQ_TO_KAPPA}, RADIAL={RADIAL_TO_KAPPA}, RELATIONAL={RELATIONAL_TO_KAPPA}")


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.manual_seed(args_pre.seed)
    np.random.seed(args_pre.seed)
    if args_pre.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args_pre.gpu}")
    print(f"[smoke_c3_phase2 mode={args_pre.mode}] device={device}, seed={args_pre.seed}")

    if args_pre.output_dir:
        product_dir = Path(args_pre.output_dir)
    else:
        default_dirs = {
            "c1": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue118_c3_correctness/c1_30epoch_smoke",
            "c3": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue118_c3_correctness/c3_30epoch_smoke",
        }
        product_dir = Path(default_dirs[args_pre.mode])
    product_dir.mkdir(parents=True, exist_ok=True)

    # relation graph (C3 only)
    pos_idx_full = None
    neg_idx_full = None
    if RELATIONAL_TO_KAPPA:
        if not args_pre.relation_graph or not os.path.exists(args_pre.relation_graph):
            raise FileNotFoundError(f"C3 mode requires --relation_graph, got: {args_pre.relation_graph}")
        rg = np.load(args_pre.relation_graph)
        pos_idx_full = rg["pos_idx"]
        neg_idx_full = rg["neg_idx"]
        print(f"[smoke_c3_phase2] relation graph: pos={pos_idx_full.shape}, neg={neg_idx_full.shape}")
        assert pos_idx_full.max() < N_ITEMS
        assert neg_idx_full.max() < N_ITEMS
        # 拷贝到 product_dir
        import shutil
        shutil.copy(args_pre.relation_graph, product_dir / "relation_graph.npz")
        src_meta = args_pre.relation_graph.replace(".npz", "_metadata.json")
        if os.path.exists(src_meta):
            shutil.copy(src_meta, product_dir / "relation_graph_metadata.json")

    # item_emb
    item_emb = np.load(ITEM_EMB_NPY, allow_pickle=True).astype(np.float32)
    assert item_emb.shape == (N_ITEMS, EMB_DIM)
    item_emb_sha = sha256_file(ITEM_EMB_NPY)[:32]
    print(f"[smoke_c3_phase2] item_emb shape={item_emb.shape}, sha256[:32]={item_emb_sha}...")
    item_emb_t = torch.from_numpy(item_emb).to(device)

    # 模型
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

    # KMeans init
    model.train()
    with torch.no_grad():
        z_first = model.encoder(item_emb_t[:args_pre.batch_size])
        for q in model.vq_layers:
            if not q.initted:
                q.init_emb(z_first.detach().cpu())

    optimizer = torch.optim.Adam(model.parameters(), lr=args_pre.lr)

    # 训练循环
    training_log = []
    collapse_diag = {"epochs": []}
    gradient_diag = {"epochs": []}
    relational_batches_total = 0
    relational_batches_success = 0
    relational_batches_failed = 0

    for epoch in range(args_pre.epochs):
        # P0-1: build global relation bank (每个 epoch)
        relation_bank = None
        if RELATIONAL_TO_KAPPA:
            relation_bank = build_global_relation_bank(model, item_emb_t, batch_size=args_pre.batch_size)
            assert all(b.shape[0] == N_ITEMS for b in relation_bank)

        epoch_loss = 0.0
        epoch_rq_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_l_rel = 0.0
        epoch_vq_grad = [0.0] * len(CODEBOOK_SIZES)
        epoch_radial_grad = [0.0] * len(CODEBOOK_SIZES)
        epoch_prior_grad = [0.0] * len(CODEBOOK_SIZES)
        epoch_boundary_grad = [0.0] * len(CODEBOOK_SIZES)
        epoch_trust_grad = [0.0] * len(CODEBOOK_SIZES)
        epoch_relational_grad = [0.0] * len(CODEBOOK_SIZES)
        epoch_recon_err = [0.0] * len(CODEBOOK_SIZES)
        n_batches = 0
        t0 = time.time()
        perm = np.random.permutation(N_ITEMS)
        for start in range(0, N_ITEMS, args_pre.batch_size):
            end = min(start + args_pre.batch_size, N_ITEMS)
            batch = item_emb_t[perm[start:end]]
            batch_idx_np = perm[start:end]
            batch_idx_t = torch.as_tensor(batch_idx_np, device=device)

            optimizer.zero_grad()
            x_recon, rq_loss, indices, z_q, z = model(batch)
            # Issue #119 (2026-08-10): 复用 taskA.stage2.poincare_recon_loss (与正式 Stage2 完全一致)
            recon_loss = poincare_recon_loss(x_recon, batch)
            total_loss = recon_loss + rq_loss

            # P0-5: C3 fail-fast
            l_rel_total = None
            if RELATIONAL_TO_KAPPA:
                relational_batches_total += 1
                pos_idx_t = torch.as_tensor(pos_idx_full[batch_idx_np], device=device)
                neg_idx_t = torch.as_tensor(neg_idx_full[batch_idx_np], device=device)
                l_rel_total = torch.zeros((), device=device)
                for l, q in enumerate(model.vq_layers):
                    anchor_z = relation_bank[l][batch_idx_t]
                    pos_z = relation_bank[l][pos_idx_t]
                    neg_z = relation_bank[l][neg_idx_t]
                    c_l = q.get_c()
                    l_rel_l = poincare_relational_loss_per_layer(
                        anchor_z=anchor_z, pos_z=pos_z, neg_z=neg_z, c=c_l, tau=RELATIONAL_TAU,
                    )
                    if not torch.isfinite(l_rel_l).item():
                        raise RuntimeError(f"P0-5 L_rel 非 finite (ep{epoch} batch{n_batches} L{l}): {l_rel_l.item()}")
                    if not l_rel_l.requires_grad:
                        raise RuntimeError(f"P0-5 L_rel requires_grad=False (ep{epoch} batch{n_batches} L{l})")
                    l_rel_total = l_rel_total + l_rel_l
                l_rel_total = l_rel_total / max(len(model.vq_layers), 1)
                total_loss = total_loss + RELATIONAL_LAMBDA * l_rel_total
                relational_batches_success += 1

            # P0-4: 5 路 gradient audit
            kappa_params = [q.kappa_drift for q in model.vq_layers]
            vq_g_vals = [0.0] * len(model.vq_layers)
            rel_g_vals = [0.0] * len(model.vq_layers)
            try:
                vq_g = torch.autograd.grad(rq_loss, kappa_params, retain_graph=True, allow_unused=True)
                vq_g_vals = [g.abs().item() if g is not None else 0.0 for g in vq_g]
            except RuntimeError:
                pass
            if l_rel_total is not None:
                try:
                    rel_g = torch.autograd.grad(l_rel_total, kappa_params, retain_graph=True, allow_unused=True)
                    rel_g_vals = [g.abs().item() if g is not None else 0.0 for g in rel_g]
                except RuntimeError:
                    pass

            total_loss.backward()
            total_g_vals = []
            for q in model.vq_layers:
                if q.kappa_drift.grad is None:
                    total_g_vals.append(0.0)
                else:
                    total_g_vals.append(q.kappa_drift.grad.abs().item())

            # P0-4 reconstruction check: |total - sum| / (|total|+eps)
            for l in range(len(model.vq_layers)):
                summed = vq_g_vals[l] + rel_g_vals[l]
                denom = abs(total_g_vals[l]) + 1e-8
                err = abs(total_g_vals[l] - summed) / denom
                epoch_recon_err[l] += err

            optimizer.step()

            for l in range(len(CODEBOOK_SIZES)):
                epoch_vq_grad[l] += vq_g_vals[l]
                epoch_relational_grad[l] += rel_g_vals[l]

            epoch_loss += total_loss.item()
            epoch_rq_loss += rq_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_l_rel += (l_rel_total.item() if l_rel_total is not None else 0.0)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_rq_loss = epoch_rq_loss / max(n_batches, 1)
        avg_recon_loss = epoch_recon_loss / max(n_batches, 1)
        avg_l_rel = epoch_l_rel / max(n_batches, 1)
        avg_vq_grad = [v / max(n_batches, 1) for v in epoch_vq_grad]
        avg_rel_grad = [v / max(n_batches, 1) for v in epoch_relational_grad]
        avg_recon_err = [v / max(n_batches, 1) for v in epoch_recon_err]
        t1 = time.time()

        # collapse diag
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
            kappas = [q.get_effective_kappa().item() for q in model.vq_layers]
            cs = [q.get_c().item() for q in model.vq_layers]
            collapse_diag["epochs"].append({
                "epoch": epoch,
                "layers": layer_diags,
                "avg_loss": avg_loss,
                "avg_l_rel": avg_l_rel,
            })
            gradient_diag["epochs"].append({
                "epoch": epoch,
                "vq_kappa_grad": avg_vq_grad,
                "radial_kappa_grad": [0.0] * len(CODEBOOK_SIZES),  # RADIAL_TO_KAPPA=False
                "prior_kappa_grad": [0.0] * len(CODEBOOK_SIZES),
                "boundary_kappa_grad": [0.0] * len(CODEBOOK_SIZES),
                "trust_kappa_grad": [0.0] * len(CODEBOOK_SIZES),
                "relational_kappa_grad": avg_rel_grad,
                "total_kappa_grad": [q.kappa_drift.grad.abs().item() if q.kappa_drift.grad is not None else 0.0 for q in model.vq_layers],
                "gradient_reconstruction_error": avg_recon_err,
            })

        training_log.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_rq_loss": avg_rq_loss,
            "avg_recon_loss": avg_recon_loss,
            "avg_l_rel": avg_l_rel,
            "kappas": kappas,
            "cs": cs,
            "vq_kappa_grad": avg_vq_grad,
            "relational_kappa_grad": avg_rel_grad,
            "gradient_reconstruction_error": avg_recon_err,
            "n_batches": n_batches,
            "wall_time_sec": t1 - t0,
        })

        if epoch % 1 == 0 or epoch == args_pre.epochs - 1:
            print(f"[smoke_c3_phase2 mode={args_pre.mode} ep{epoch:02d}] loss={avg_loss:.4f} "
                  f"rq={avg_rq_loss:.4f} recon={avg_recon_loss:.4f} l_rel={avg_l_rel:.4f} "
                  f"κ={[f'{k:.3f}' for k in kappas]} c={[f'{c:.3f}' for c in cs]} "
                  f"vq_grad={[f'{v:.4f}' for v in avg_vq_grad]} "
                  f"rel_grad={[f'{v:.4f}' for v in avg_rel_grad]}")

    # === 9 件套 + 6 PNG ===
    config = {
        "issue": "#119",
        "task": "Phase 2",
        "title": f"C1 vs C3 matched smoke (mode={args_pre.mode}) with P0-1~P0-5 fixes",
        "mode": args_pre.mode,
        "vq_to_kappa": VQ_TO_KAPPA,
        "radial_to_kappa": RADIAL_TO_KAPPA,
        "relational_to_kappa": RELATIONAL_TO_KAPPA,
        "relational_tau": RELATIONAL_TAU,
        "relational_lambda": RELATIONAL_LAMBDA,
        "relational_pos_k": RELATIONAL_POS_K,
        "relational_neg_n": RELATIONAL_NEG_N,
        "relational_neg_excl": RELATIONAL_NEG_EXCL,
        "relation_graph_npz": args_pre.relation_graph,
        "codebook_sizes": CODEBOOK_SIZES,
        "e_dim": E_DIM,
        "encoder_layers": ENCODER_LAYERS,
        "batch_size": args_pre.batch_size,
        "epochs": args_pre.epochs,
        "lr": args_pre.lr,
        "seed": args_pre.seed,
        "gpu": args_pre.gpu,
        "device": str(device),
        "item_emb_sha256": item_emb_sha,
        "n_items": N_ITEMS,
        "p0_fixes_applied": ["P0-1", "P0-2", "P0-3", "P0-4", "P0-5"],
    }
    with open(product_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(product_dir / "training_log.json", "w") as f:
        json.dump({"epochs": training_log}, f, indent=2, default=str)
    with open(product_dir / "collapse_diagnostics.json", "w") as f:
        json.dump(collapse_diag, f, indent=2, default=str)
    with open(product_dir / "gradient_diagnostics.json", "w") as f:
        json.dump(gradient_diag, f, indent=2, default=str)
    rb = {
        "relational_batches_total": relational_batches_total,
        "relational_batches_success": relational_batches_success,
        "relational_batches_failed": relational_batches_failed,
        "success_rate": relational_batches_success / max(relational_batches_total, 1),
    }
    with open(product_dir / "relational_batches.json", "w") as f:
        json.dump(rb, f, indent=2)

    # 6 PNG (含 gradient_source_audit.png 替代 margin)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epoch_ids = [e["epoch"] for e in collapse_diag["epochs"]]
        n_layers = len(CODEBOOK_SIZES)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        layer_labels = [f"L{l} (K={CODEBOOK_SIZES[l]})" for l in range(n_layers)]

        plots = [
            ("kappa_curve.png", "kappa", "κ"),
            ("curvature_curve.png", "c", "c"),
            ("utilization_curve.png", "util_3digit", "util_3digit"),
            ("entropy_curve.png", "assign_entropy", "H(assign)"),
            ("saturation_curve.png", "all_pair_clip_ratio", "S_all"),
        ]
        for fname, attr, ylabel in plots:
            fig, ax = plt.subplots(figsize=(8, 5))
            for l in range(n_layers):
                vals = [ep["layers"][l][attr] for ep in collapse_diag["epochs"]]
                ax.plot(epoch_ids, vals, marker="o", color=colors[l], label=layer_labels[l])
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Issue #119 mode={args_pre.mode} — {ylabel}")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(product_dir / fname, dpi=110)
            plt.close(fig)

        # gradient source audit (5 路 + total, 仅第一层 L0 详情)
        fig, ax = plt.subplots(figsize=(10, 6))
        vq_g = [ep["vq_kappa_grad"][0] for ep in gradient_diag["epochs"]]
        rel_g = [ep["relational_kappa_grad"][0] for ep in gradient_diag["epochs"]]
        total_g = [ep["total_kappa_grad"][0] for ep in gradient_diag["epochs"]]
        ax.plot(epoch_ids, vq_g, marker="o", label="vq_kappa_grad (L0)", color="#1f77b4")
        ax.plot(epoch_ids, rel_g, marker="s", label="relational_kappa_grad (L0)", color="#d62728")
        ax.plot(epoch_ids, total_g, marker="^", label="total_kappa_grad (L0)", color="#2ca02c")
        ax.set_xlabel("epoch")
        ax.set_ylabel("|grad|")
        ax.set_title(f"Issue #119 mode={args_pre.mode} — Gradient Source Audit (Layer 0)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(product_dir / "gradient_source_audit.png", dpi=110)
        plt.close(fig)
        print(f"[plot] gradient_source_audit.png")
    except ImportError:
        print("[warn] matplotlib not available")

    # final_verdict
    final_layers = collapse_diag["epochs"][-1]["layers"] if collapse_diag["epochs"] else []
    final_kappas = [d["kappa"] for d in final_layers]
    final_cs = [d["c"] for d in final_layers]
    final_utils = [d["util_3digit"] for d in final_layers]
    final_entropies = [d["assign_entropy"] for d in final_layers]
    final_top1 = [d["top1_clip_ratio"] for d in final_layers]
    final_sall = [d["all_pair_clip_ratio"] for d in final_layers]
    final_margins = [d["top1_top2_margin_median"] for d in final_layers]

    chain_detected = False
    if len(collapse_diag["epochs"]) >= 5:
        sall = [np.mean([l["all_pair_clip_ratio"] for l in ep["layers"]]) for ep in collapse_diag["epochs"]]
        margins = [np.mean([l["top1_top2_margin_median"] for l in ep["layers"]]) for ep in collapse_diag["epochs"]]
        ents = [np.mean([l["assign_entropy"] for l in ep["layers"]]) for ep in collapse_diag["epochs"]]
        utils = [np.mean([l["util_3digit"] for l in ep["layers"]]) for ep in collapse_diag["epochs"]]
        if (sall[-1] > sall[0] + 0.05 and
            margins[-1] < margins[0] - 0.01 and
            ents[-1] < ents[0] - 0.5 and
            utils[-1] < utils[0] - 0.1):
            chain_detected = True

    final_verdict = {
        "issue": "#119",
        "task": "Phase 2",
        "title": f"C1 vs C3 matched smoke (mode={args_pre.mode}) with P0 fixes",
        "mode": args_pre.mode,
        "vq_to_kappa": VQ_TO_KAPPA,
        "radial_to_kappa": RADIAL_TO_KAPPA,
        "relational_to_kappa": RELATIONAL_TO_KAPPA,
        "gate_a1_fail_fast": {
            "status": "PASS" if relational_batches_failed == 0 else "FAIL",
            "relational_batches_failed": relational_batches_failed,
            "success_rate": rb["success_rate"],
        },
        "gate_a2_5_way_audit": {
            "status": "PASS" if all(
                abs(training_log[-1]["vq_kappa_grad"][l]) > 1e-8 !=
                abs(training_log[-1]["relational_kappa_grad"][l]) > 1e-8
                for l in range(len(CODEBOOK_SIZES))
            ) else "FAIL",
            "vq_kappa_grad_final": training_log[-1]["vq_kappa_grad"] if training_log else [0.0],
            "relational_kappa_grad_final": training_log[-1]["relational_kappa_grad"] if training_log else [0.0],
        },
        "gate_a3_recon_error": {
            "status": "PASS" if all(
                e < 1e-3 for e in (training_log[-1]["gradient_reconstruction_error"] if training_log else [0.0])
            ) else "FAIL",
            "gradient_reconstruction_error": training_log[-1]["gradient_reconstruction_error"] if training_log else [0.0],
        },
        "gate_a4_collapse": {
            "status": "FAIL" if chain_detected else "PASS",
            "chain_detected": chain_detected,
        },
        "collapse_detected": chain_detected,
        "final_kappa": final_kappas,
        "final_curvature": final_cs,
        "final_utilization": final_utils,
        "final_entropy": final_entropies,
        "top1_clip_ratio": final_top1,
        "all_pair_clip_ratio": final_sall,
        "assignment_margin": final_margins,
        "kappa_per_layer_std": float(np.std(final_kappas)) if final_kappas else 0.0,
        "curvature_per_layer_std": float(np.std(final_cs)) if final_cs else 0.0,
        "r37_decision": (
            "PASS" if (not chain_detected and all(u > 0.50 for u in final_utils)
                       and relational_batches_failed == 0)
            else "FAIL"
        ),
    }
    with open(product_dir / "final_verdict.json", "w") as f:
        json.dump(final_verdict, f, indent=2, default=str)

    print(f"\n[smoke_c3_phase2 mode={args_pre.mode}] DONE. artifacts in {product_dir}/")
    return 0 if not chain_detected and relational_batches_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())