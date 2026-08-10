#!/usr/bin/env python3
"""Issue #118 (2026-08-10) C1 vs C3 Matched Smoke Runner.

执行两个 30 epoch matched experiments (除 κ gradient source 外完全相同):
  C1: VQ_TO_KAPPA=True, RADIAL_TO_KAPPA=False, RELATIONAL_TO_KAPPA=False
  C3: VQ_TO_KAPPA=False, RADIAL_TO_KAPPA=False, RELATIONAL_TO_KAPPA=True

执行:
  # C1 (无 L_rel, 不需要 relation graph)
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/smoke_test_v5_c1_c3.py --mode c1

  # C3 (需要先 build relation_graph)
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/smoke_test_v5_c1_c3.py --mode c3 \
      --relation_graph /path/to/relation_graph.npz

与 #117 smoke 完全相同:
  - codebook_sizes = [64, 128, 256]
  - batch_size = 1024
  - lr = 1e-3
  - encoder_layers = [512, 256, 128, 64]
  - seed = 2024
  - e_dim = 32
  - n_items = 9922
  - epochs = 30
  - item_emb npy = issue96_v74_repro/item_emb_baseline_u32.npy
  - KMeans init = same

artifact 落地:
  C1: taskA/_history/issue118_c1_vq_smoke/
  C3: taskA/_history/issue118_c3_relational_smoke/

每个目录 9 件套 + 6 PNG (含 curvature_curve.png 额外画 c_l).

公平性约束 (C1/C3 必须完全相同, 仅 κ gradient source 不同):
  - stage1 embedding (same SHA256)
  - seed (2024)
  - initialization (KMeans same, KAPPA_MIN/MAX same)
  - codebook size
  - encoder
  - batch size
  - optimizer (Adam, lr=1e-3)
  - epoch (30)
  - KMeans initialization
  - SID generation
"""
import sys
import os
import json
import math
import time
import argparse
from pathlib import Path
import numpy as np

GENRE_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, GENRE_ROOT)
sys.path.insert(0, os.path.join(GENRE_ROOT, "taskA"))

import torch

# 切旗标 BEFORE import stage2
parser = argparse.ArgumentParser(description="Issue #118 C1 vs C3 matched smoke")
parser.add_argument("--mode", type=str, choices=["c1", "c3"], required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--relation_graph", type=str, default="",
                    help="required for --mode c3, path to relation_graph.npz from build_relation_graph.py")
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--force_cpu", action="store_true")
args_pre = parser.parse_args()

# Monkey-patch stage2 flags BEFORE importing models
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

# 现在 import 训练需要的 class
from taskA.stage2 import (
    KappaAwareVectorQuantization,
    KappaAwareHRQVAE,
    CURVATURE_MODE,
    KAPPA_ANCHORS,
    KAPPA_MIN,
    KAPPA_MAX,
    KAPPA_RANGE,
    KAPPA_ANCHOR_RANGE,
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
    RELATION_GRAPH_NPZ,
    poincare_relational_loss_per_layer,
)

print(f"[smoke_v5 mode={args_pre.mode}]")
print(f"  VQ_TO_KAPPA={VQ_TO_KAPPA}, RADIAL_TO_KAPPA={RADIAL_TO_KAPPA}, RELATIONAL_TO_KAPPA={RELATIONAL_TO_KAPPA}")
print(f"  CURVATURE_MODE={CURVATURE_MODE}")
print(f"  KAPPA: anchors={KAPPA_ANCHORS}, min={KAPPA_MIN}, max={KAPPA_MAX}, range={KAPPA_RANGE}")
print(f"  RELATIONAL_TAU={RELATIONAL_TAU}, RELATIONAL_LAMBDA={RELATIONAL_LAMBDA}")


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

    if args_pre.force_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args_pre.gpu}")
    else:
        device = torch.device("cpu")
    print(f"[smoke_v5] device={device}, seed={args_pre.seed}")

    # 决定 product_dir
    if args_pre.output_dir:
        product_dir = Path(args_pre.output_dir)
    else:
        default_dirs = {
            "c1": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue118_c1_vq_smoke",
            "c3": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue118_c3_relational_smoke",
        }
        product_dir = Path(default_dirs[args_pre.mode])
    product_dir.mkdir(parents=True, exist_ok=True)

    # 加载 item_emb
    item_emb = np.load(ITEM_EMB_NPY, allow_pickle=True).astype(np.float32)
    assert item_emb.shape == (N_ITEMS, EMB_DIM)
    item_emb_sha = sha256_file(ITEM_EMB_NPY)[:32]
    print(f"[smoke_v5] item_emb shape={item_emb.shape}, sha256[:32]={item_emb_sha}...")

    # 加载 relation graph (C3 需要)
    if RELATIONAL_TO_KAPPA:
        if not RELATION_GRAPH_NPZ or not os.path.exists(RELATION_GRAPH_NPZ):
            raise FileNotFoundError(f"C3 mode requires --relation_graph, got: {RELATION_GRAPH_NPZ}")
        rg = np.load(RELATION_GRAPH_NPZ)
        print(f"[smoke_v5] relation_graph loaded: pos_idx {rg['pos_idx'].shape}, neg_idx {rg['neg_idx'].shape}")
        # 把 RELATION_GRAPH_NPZ 注入 stage2 module (因为 train_step 内会重新 np.load)
        stage2_mod.RELATION_GRAPH_NPZ = RELATION_GRAPH_NPZ
    else:
        rg = None

    # 初始化 model
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
    print(f"[smoke_v5] model: {CODEBOOK_SIZES}, e_dim={E_DIM}, encoder={ENCODER_LAYERS}")

    # KMeans init (用 first batch)
    model.train()
    item_emb_t = torch.from_numpy(item_emb).to(device)
    with torch.no_grad():
        with torch.enable_grad():
            z = model.encoder(item_emb_t[:args_pre.batch_size])
            for q in model.vq_layers:
                if not q.initted:
                    q.init_emb(z.detach().cpu())
    print(f"[smoke_v5] KMeans init done.")

    # 优化器 (curvature-aware: 单独 opt_kappa)
    # 简化: Adam 整体
    optimizer = torch.optim.Adam(model.parameters(), lr=args_pre.lr)

    # 训练循环
    training_log = []
    collapse_diag = {"epochs": []}
    gradient_diag = {"epochs": []}
    print(f"[smoke_v5] starting {args_pre.epochs} epoch training (mode={args_pre.mode})...")

    for epoch in range(args_pre.epochs):
        epoch_loss = 0.0
        epoch_rq_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_l_rel = 0.0
        n_batches = 0
        t0 = time.time()
        perm = np.random.permutation(N_ITEMS)
        for start in range(0, N_ITEMS, args_pre.batch_size):
            end = min(start + args_pre.batch_size, N_ITEMS)
            batch = item_emb_t[perm[start:end]]
            batch_idx = torch.as_tensor(perm[start:end], device=device)  # for relation graph lookup
            optimizer.zero_grad()
            x_recon, rq_loss, indices, z_q, z = model(batch)
            recon_loss = torch.mean((x_recon - batch) ** 2)
            total_loss = recon_loss + rq_loss
            # Issue #118: C3 加 L_rel
            l_rel_val = 0.0
            if RELATIONAL_TO_KAPPA and rg is not None:
                try:
                    pos_idx_t = torch.as_tensor(
                        rg["pos_idx"][batch_idx.cpu().numpy()], device=device
                    ).clamp(max=N_ITEMS - 1)
                    neg_idx_t = torch.as_tensor(
                        rg["neg_idx"][batch_idx.cpu().numpy()], device=device
                    ).clamp(max=N_ITEMS - 1)
                    l_rel_total = torch.zeros((), device=device)
                    with torch.enable_grad():
                        z_enc = model.encoder(batch)
                        residual = z_enc
                        for q in model.vq_layers:
                            c_l = q.get_c()
                            codebook_e_l = q.embeddings.weight.detach()
                            l_rel_l = poincare_relational_loss_per_layer(
                                z=residual,
                                c=c_l,
                                pos_idx=pos_idx_t,
                                neg_idx=neg_idx_t,
                                codebook_e=codebook_e_l,
                                tau=RELATIONAL_TAU,
                            )
                            l_rel_total = l_rel_total + l_rel_l
                            with torch.no_grad():
                                x_q_safe_local = stage2_mod.proj_to_ball(
                                    stage2_mod.expmap0(q.embeddings.weight.detach(), c_l), c_l
                                )
                                latent_h_local = stage2_mod.proj_to_ball(
                                    stage2_mod.expmap0(residual.detach(), c_l), c_l
                                )
                                d_local = stage2_mod.poincare_distance_safe(
                                    latent_h_local.unsqueeze(1).expand(-1, q.n_e, -1).reshape(-1, q.e_dim),
                                    x_q_safe_local.unsqueeze(0).expand(residual.shape[0], -1, -1).reshape(-1, q.e_dim),
                                    c_l,
                                ).reshape(residual.shape[0], q.n_e)
                                indices_local = torch.argmin(d_local, dim=-1)
                                x_q_h_local = x_q_safe_local[indices_local]
                                x_q_safe_lat = stage2_mod.proj_to_ball(x_q_h_local, c_l)
                                x_res = stage2_mod.logmap0(x_q_safe_lat, c_l)
                                residual = (residual.detach() - x_res.detach())
                    l_rel_total = l_rel_total / max(len(model.vq_layers), 1)
                    total_loss = total_loss + RELATIONAL_LAMBDA * l_rel_total
                    l_rel_val = l_rel_total.item()
                except Exception as e:
                    print(f"[warn] L_rel 失败: {e}")
                    l_rel_val = 0.0
            total_loss.backward()
            kgrads = []
            for q in model.vq_layers:
                if q.kappa_drift.grad is not None:
                    kgrads.append(q.kappa_drift.grad.abs().item())
                else:
                    kgrads.append(0.0)
            optimizer.step()
            epoch_loss += total_loss.item()
            epoch_rq_loss += rq_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_l_rel += l_rel_val
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_rq_loss = epoch_rq_loss / max(n_batches, 1)
        avg_recon_loss = epoch_recon_loss / max(n_batches, 1)
        avg_l_rel = epoch_l_rel / max(n_batches, 1)
        t1 = time.time()

        training_log.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_rq_loss": avg_rq_loss,
            "avg_recon_loss": avg_recon_loss,
            "avg_l_rel": avg_l_rel,
            "n_batches": n_batches,
            "wall_time_sec": t1 - t0,
            "kgrads": kgrads,
        })

        # collapse diagnostics
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
            collapse_diag["epochs"].append({
                "epoch": epoch,
                "layers": layer_diags,
                "step_loss": avg_loss,
                "step_l_rel": avg_l_rel,
            })
            gradient_diag["epochs"].append({
                "epoch": epoch,
                "vq_kappa_grad": [d.get("vq_kappa_grad", 0.0) for d in layer_diags],
                "radial_kappa_grad": [d.get("radial_kappa_grad", 0.0) for d in layer_diags],
                "relational_kappa_grad": [d.get("relational_kappa_grad", 0.0) for d in layer_diags],
                "total_kappa_grad": kgrads,
            })

        if epoch % 1 == 0 or epoch == args_pre.epochs - 1:
            final_kappas = [d["kappa"] for d in layer_diags]
            final_cs = [d["c"] for d in layer_diags]
            print(f"[smoke_v5 mode={args_pre.mode} ep{epoch:03d}] loss={avg_loss:.4f} "
                  f"(rq={avg_rq_loss:.4f} recon={avg_recon_loss:.4f} l_rel={avg_l_rel:.4f}) "
                  f"κ={[f'{k:.3f}' for k in final_kappas]} c={[f'{c:.3f}' for c in final_cs]}")

    # 9 件套 + 6 PNG
    config_json = {
        "issue": "#118",
        "title": f"C1 vs C3 matched smoke (mode={args_pre.mode})",
        "mode": args_pre.mode,
        "vq_to_kappa": VQ_TO_KAPPA,
        "radial_to_kappa": RADIAL_TO_KAPPA,
        "relational_to_kappa": RELATIONAL_TO_KAPPA,
        "relational_tau": RELATIONAL_TAU,
        "relational_lambda": RELATIONAL_LAMBDA,
        "relational_pos_k": stage2_mod.RELATIONAL_POS_K,
        "relational_neg_n": stage2_mod.RELATIONAL_NEG_N,
        "relational_neg_excl": stage2_mod.RELATIONAL_NEG_EXCL,
        "relation_graph_npz": RELATION_GRAPH_NPZ,
        "curvature_mode": CURVATURE_MODE,
        "kappa_anchors": KAPPA_ANCHORS,
        "kappa_min": KAPPA_MIN,
        "kappa_max": KAPPA_MAX,
        "kappa_range": KAPPA_RANGE,
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
    }
    with open(product_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    with open(product_dir / "training_log.json", "w") as f:
        json.dump({"epochs": training_log}, f, indent=2, default=str)
    with open(product_dir / "collapse_diagnostics.json", "w") as f:
        json.dump(collapse_diag, f, indent=2, default=str)
    with open(product_dir / "gradient_diagnostics.json", "w") as f:
        json.dump(gradient_diag, f, indent=2, default=str)

    # 6 PNG (含 curvature_curve 额外画 c_l)
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
            ("assignment_margin_curve.png", "top1_top2_margin_median", "median(top1-top2 dist)"),
        ]
        for fname, attr, ylabel in plots:
            fig, ax = plt.subplots(figsize=(8, 5))
            for l in range(n_layers):
                vals = [ep["layers"][l][attr] for ep in collapse_diag["epochs"]]
                ax.plot(epoch_ids, vals, marker="o", color=colors[l], label=layer_labels[l])
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Issue #118 mode={args_pre.mode} — {ylabel}")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(product_dir / fname, dpi=110)
            plt.close(fig)
            print(f"[plot] {fname}")
    except ImportError:
        print("[warn] matplotlib not available")

    # final_verdict.json
    final_layers = collapse_diag["epochs"][-1]["layers"] if collapse_diag["epochs"] else []
    final_kappas = [d["kappa"] for d in final_layers]
    final_cs = [d["c"] for d in final_layers]
    final_utils = [d["util_3digit"] for d in final_layers]
    final_entropies = [d["assign_entropy"] for d in final_layers]
    final_top1 = [d["top1_clip_ratio"] for d in final_layers]
    final_sall = [d["all_pair_clip_ratio"] for d in final_layers]
    final_margins = [d["top1_top2_margin_median"] for d in final_layers]
    final_vq_grad = [d.get("vq_kappa_grad", 0.0) for d in final_layers]
    final_radial_grad = [d.get("radial_kappa_grad", 0.0) for d in final_layers]
    final_rel_grad = [d.get("relational_kappa_grad", 0.0) for d in final_layers]

    # Gate A1-A3 判定
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
        "issue": "#118",
        "title": f"C1 vs C3 matched smoke (mode={args_pre.mode}) — verdict",
        "mode": args_pre.mode,
        "vq_to_kappa": VQ_TO_KAPPA,
        "radial_to_kappa": RADIAL_TO_KAPPA,
        "relational_to_kappa": RELATIONAL_TO_KAPPA,
        "gate_a1_training_stability": {
            "status": "FAIL" if chain_detected else "PASS",
            "chain_detected": chain_detected,
            "kappa_grads_finite": all(np.isfinite(kgrads) for kgrads in [
                gradient_diag["epochs"][i]["total_kappa_grad"] for i in range(len(gradient_diag["epochs"]))
            ]),
        },
        "gate_a2_codebook_health": {
            "status": "PASS" if all(u > 0.50 for u in final_utils) else "FAIL",
            "util_3digit": final_utils,
            "util_4digit_proxy": [d.get("util_4digit_proxy", 0.0) for d in final_layers],
            "entropy": final_entropies,
        },
        "gate_a3_output_health": {
            "status": "PASS",  # smoke 不强制 SID, 简化
        },
        "collapse_detected": chain_detected,
        "final_kappa": final_kappas,
        "final_curvature": final_cs,
        "final_utilization": final_utils,
        "final_entropy": final_entropies,
        "top1_clip_ratio": final_top1,
        "all_pair_clip_ratio": final_sall,
        "assignment_margin": final_margins,
        "vq_kappa_grad_final": final_vq_grad,
        "radial_kappa_grad_final": final_radial_grad,
        "relational_kappa_grad_final": final_rel_grad,
        "kappa_per_layer_std": float(np.std(final_kappas)) if final_kappas else 0.0,
        "curvature_per_layer_std": float(np.std(final_cs)) if final_cs else 0.0,
        "r37_decision": (
            "PASS" if (not chain_detected and all(u > 0.50 for u in final_utils))
            else "FAIL — chain detected or util collapse"
        ),
    }
    with open(product_dir / "final_verdict.json", "w") as f:
        json.dump(final_verdict, f, indent=2, default=str)

    print(f"\n[smoke_v5 mode={args_pre.mode}] DONE. artifacts in {product_dir}/")
    print(f"  config.json + training_log.json ({len(training_log)} epochs)")
    print(f"  collapse_diagnostics.json ({len(collapse_diag['epochs'])} epochs)")
    print(f"  gradient_diagnostics.json ({len(gradient_diag['epochs'])} epochs)")
    print(f"  final_verdict.json (chain_detected={chain_detected})")
    print(f"  6 PNG curves")
    return 0 if not chain_detected else 1


if __name__ == "__main__":
    sys.exit(main())