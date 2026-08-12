"""Issue #145 Stage 2 trainer — Per-layer shared curvature RQ-VAE (A or B arm).

Usage:
  python3 stage2_train.py --arm control   # A: tangent residual
  python3 stage2_train.py --arm treatment # B: Möbius residual

Both arms share identical:
  - PerLayerCurvatureHRQVAE architecture (3 layers, K=[64,128,256])
  - Per-layer shared learnable curvature c_l = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_l)
  - Distance metric, Sinkhorn/argmin, anchor, optimizer, scheduler, seed

唯一差异: residual/reconstruction algebra (mobius_residual flag).
"""

import os
import sys
import json
import math
import time
import argparse
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

# R44: 引用本任务 _lib (utils.py 在 arm/_lib/, per_layer_curvature_quantizer.py 在 task/_lib/)
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ARM = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "control"
ARM_DIR = os.path.join(TASK_DIR, ARM)
sys.path.insert(0, os.path.join(ARM_DIR, "_lib"))  # utils.py
sys.path.insert(0, os.path.join(TASK_DIR, "_lib"))  # per_layer_curvature_quantizer.py

from utils import (
    proj_to_ball, mobius_add, expmap0, logmap0, poincare_distance,
    MLP, sinkhorn_algorithm, kmeans, _eps, EmbDataset
)
from per_layer_curvature_quantizer import (
    PerLayerCurvatureHRQVAE, C_MIN, C_MAX, init_theta_for_c, THETA_INIT
)

# ──────────────────────────────────────────────────────────────
# 常量 (硬编码, 禁 CLI 传数值超参, R30 + R43)
# ──────────────────────────────────────────────────────────────
SEED = 42
N_ITEMS = 9922
EMB_DIM = 768
E_DIM = 32
CODEBOOK_SIZES = [64, 128, 256]
ENCODER_LAYERS = [512, 256, 128, 64]
BATCH_SIZE = 512
NUM_EPOCHS = 1000
LR = 1e-3
WEIGHT_DECAY = 0.0
BETA = 0.25
SK_EPS = [0.003] * 3
SK_ITERS = 3
KMEANS_ITERS = 10
GRAD_CLIP = 1.0
LOG_EVERY = 50
SAVE_EVERY = 100

ITEM_EMB_PARQUET = os.path.join(TASK_DIR, "stage1", "item_emb.parquet")  # R44 共享 Stage1
STAGE2_DIR = os.path.join(TASK_DIR, ARM, "stage2")
CKPT_PATH = os.path.join(STAGE2_DIR, "hrqvae_kappa_sync.ckpt")
SID_PATH = os.path.join(STAGE2_DIR, "sid_output.npy")
VERDICT_PATH = os.path.join(STAGE2_DIR, "verdict.json")
MANIFEST_PATH = os.path.join(TASK_DIR, ARM, "manifest.json")
CURVATURE_TRACE_PATH = os.path.join(STAGE2_DIR, "curvature_trace.csv")
RESIDUAL_ENERGY_PATH = os.path.join(STAGE2_DIR, "residual_energy_per_layer.csv")
TRANSPORT_AUDIT_PATH = os.path.join(STAGE2_DIR, "exp_log_transport_roundtrip.csv")

os.makedirs(STAGE2_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr):
    return hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()


def make_model(mobius_residual: bool):
    return PerLayerCurvatureHRQVAE(
        in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
        layers=ENCODER_LAYERS, beta=BETA, kmeans_init=True,
        kmeans_iters=KMEANS_ITERS, sk_eps=SK_EPS, sk_iters=SK_ITERS,
        fix_c=False, mobius_residual=mobius_residual
    ).to(DEVICE)


def train(model, dataloader, num_epochs, log_every=LOG_EVERY):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    model.train()
    history = []
    for epoch in range(num_epochs):
        ep_loss = 0.0
        ep_recon = 0.0
        ep_vq = 0.0
        n_batch = 0
        for batch in dataloader:
            x = batch.to(DEVICE)
            optimizer.zero_grad()
            out, rq_loss, indices = model(x, use_sk=True)
            # recon loss (MSE)
            recon_loss = F.mse_loss(out, x)
            total = recon_loss + rq_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ep_loss += float(total.item())
            ep_recon += float(recon_loss.item())
            ep_vq += float(rq_loss.item())
            n_batch += 1
        scheduler.step()

        # 记录曲率
        cs = [float(q.get_c().item()) for q in model.hrq.vq_layers]
        thetas = [float(q.theta.item()) for q in model.hrq.vq_layers]
        avg_loss = ep_loss / max(n_batch, 1)
        avg_recon = ep_recon / max(n_batch, 1)
        avg_vq = ep_vq / max(n_batch, 1)
        if epoch % log_every == 0 or epoch == num_epochs - 1:
            print(f"  epoch {epoch:4d} | loss={avg_loss:.4f} | recon={avg_recon:.4f} | vq={avg_vq:.4f} | c={cs}")
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "recon": avg_recon,
            "vq": avg_vq,
            "c": cs,
            "theta": thetas,
            "residual_norms": model.hrq._last_residual_norms,
            "roundtrip_errors": model.hrq._last_roundtrip_errors,
            "transport_errors": model.hrq._last_transport_errors,
            "boundary_hit": model.hrq._last_boundary_hit,
        })

    return history


def infer_sid(model, item_emb: torch.Tensor, batch_size=BATCH_SIZE):
    model.eval()
    n = item_emb.shape[0]
    all_indices = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = item_emb[i:i+batch_size].to(DEVICE)
            idx = model.get_indices(batch, use_sk=True)
            all_indices.append(idx.cpu())
    return torch.cat(all_indices, dim=0).numpy()


def check_collision(all_str):
    return len(set(all_str)) != len(all_str)


def resolve_4digit(sid_3digit, K_l2=CODEBOOK_SIZES[-1]):
    """生成第 4 位: L2 token + 1 个 dedup 数字, 保证 4 位唯一."""
    n = sid_3digit.shape[0]
    seen = {}
    sid_4digit = np.zeros((n, 4), dtype=np.int64)
    sid_4digit[:, :3] = sid_3digit
    for i in range(n):
        key = tuple(sid_3digit[i].tolist())
        if key not in seen:
            seen[key] = 0
        else:
            seen[key] += 1
        sid_4digit[i, 3] = seen[key]
    # 4-digit 唯一性: 如果仍有冲突, 进一步扩展
    all_str = ["_".join(map(str, sid_4digit[i])) for i in range(n)]
    if check_collision(all_str):
        # 退化为 4 位 (k_l2 * 1) + 1 (4th digit 已是 sequential, 唯一性靠 K_l2 内 sequential 不会撞)
        # 实际上 3 位相同 + sequential 4th 不会撞 (因为 sequential 是 unique 的)
        # 检查: 如果仍然撞, 说明问题在 3-digit 之外
        unique = len(set(all_str))
        print(f"  ⚠️ 4-digit collisions remain: {n - unique}/{n}")
    return sid_4digit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True, choices=["control", "treatment"])
    args = parser.parse_args()
    arm = args.arm
    print(f"\n{'='*70}\nIssue #145 Stage 2 trainer — arm={arm}\n{'='*70}\n")

    mobius_residual = (arm == "treatment")

    # 1. Stage1 item_emb 验证
    print("[1/6] Stage1 item_emb 验证 ...")
    if not os.path.exists(ITEM_EMB_PARQUET):
        print(f"  ⚠️ Stage1 item_emb.parquet 不存在: {ITEM_EMB_PARQUET}")
        print(f"  → 将先运行 Stage1 共享 ...")
        # Run stage1 inline
        stage1_script = os.path.join(TASK_DIR, arm, "stage1.py")
        if not os.path.exists(stage1_script):
            stage1_script = os.path.join(TASK_DIR, "control", "stage1.py")
        os.system(f"python3 {stage1_script}")
    item_emb_sha = sha256_file(ITEM_EMB_PARQUET)
    print(f"  item_emb sha256: {item_emb_sha}")

    # 2. 加载数据
    print("[2/6] 加载数据 ...")
    dataset = EmbDataset(ITEM_EMB_PARQUET)
    print(f"  dataset size: {len(dataset)}, dim: {dataset.dim}")
    assert len(dataset) == N_ITEMS, f"Expected {N_ITEMS}, got {len(dataset)}"
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=False)

    # 3. 构建模型
    print(f"[3/6] 构建模型 (mobius_residual={mobius_residual}) ...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = make_model(mobius_residual=mobius_residual)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params}")

    # 4. 训练
    print(f"[4/6] 训练 {NUM_EPOCHS} epochs ...")
    t0 = time.time()
    history = train(model, dataloader, NUM_EPOCHS)
    train_time = time.time() - t0
    print(f"  训练耗时: {train_time:.1f}s ({train_time/60:.1f} min)")

    # 5. 推断 SID
    print("[5/6] 推断 SID ...")
    item_emb_all = torch.from_numpy(np.stack(
        pd.read_parquet(ITEM_EMB_PARQUET)['embedding'].values, axis=0
    ).astype(np.float32))
    sid_3digit = infer_sid(model, item_emb_all)
    print(f"  sid_3digit shape: {sid_3digit.shape}, dtype: {sid_3digit.dtype}")
    print(f"  unique 3-digit: {len(np.unique(sid_3digit, axis=0))}/{N_ITEMS}")

    # 加 4th dedup digit
    sid_4digit = resolve_4digit(sid_3digit)
    print(f"  sid_4digit shape: {sid_4digit.shape}")
    print(f"  unique 4-digit: {len(np.unique(sid_4digit, axis=0))}/{N_ITEMS}")

    np.save(SID_PATH, sid_4digit)
    print(f"  SID saved: {SID_PATH}")

    # 6. 保存 checkpoint + 分析产物
    print("[6/6] 保存产物 ...")
    # Issue #145: 去掉 "hrq." 前缀以兼容 Stage3 期望 (baseline HRQVAE 直接 vq_layers.X.embeddings.weight)
    sd = model.state_dict()
    sd_flat = {}
    for k, v in sd.items():
        if k.startswith("hrq."):
            sd_flat[k[len("hrq."):]] = v
        else:
            sd_flat[k] = v
    torch.save({
        "model_state_dict": sd_flat,
        "epoch": NUM_EPOCHS,
        "final_cs": [float(q.get_c().item()) for q in model.hrq.vq_layers],
        "final_thetas": [float(q.theta.item()) for q in model.hrq.vq_layers],
        "mobius_residual": mobius_residual,
        "arm": arm,
        "sid_sha256": sha256_array(sid_4digit),
        "item_emb_sha256": item_emb_sha,
        "seed": SEED,
    }, CKPT_PATH)
    print(f"  ckpt saved: {CKPT_PATH}")

    # curvature_trace.csv
    trace_rows = []
    for h in history:
        trace_rows.append({
            "epoch": h["epoch"],
            "loss": h["loss"],
            "recon": h["recon"],
            "vq": h["vq"],
            "c_l0": h["c"][0], "c_l1": h["c"][1], "c_l2": h["c"][2],
            "theta_l0": h["theta"][0], "theta_l1": h["theta"][1], "theta_l2": h["theta"][2],
        })
    pd.DataFrame(trace_rows).to_csv(CURVATURE_TRACE_PATH, index=False)

    # residual_energy_per_layer.csv
    energy_rows = []
    for h in history:
        rn = h["residual_norms"]
        rt = h["roundtrip_errors"]
        te = h["transport_errors"]
        bh = h["boundary_hit"]
        for l in range(3):
            row = {"epoch": h["epoch"], "layer": l}
            if l < len(rn): row["residual_norm"] = rn[l]
            if l < len(rt): row["roundtrip_error"] = rt[l]
            if l < len(te): row["transport_error"] = te[l]
            if l < len(bh): row["boundary_hit_rate"] = bh[l]
            energy_rows.append(row)
    pd.DataFrame(energy_rows).to_csv(RESIDUAL_ENERGY_PATH, index=False)

    # exp_log_transport_roundtrip.csv
    trans_rows = []
    for h in history:
        te = h["transport_errors"]
        for l, v in enumerate(te):
            trans_rows.append({"epoch": h["epoch"], "layer": l, "transport_error": v})
    pd.DataFrame(trans_rows).to_csv(TRANSPORT_AUDIT_PATH, index=False)

    # verdict.json
    final_cs = [float(q.get_c().item()) for q in model.hrq.vq_layers]
    final_thetas = [float(q.theta.item()) for q in model.hrq.vq_layers]
    final_residual_norms = model.hrq._last_residual_norms
    final_rt_errors = model.hrq._last_roundtrip_errors
    final_te_errors = model.hrq._last_transport_errors
    final_boundary = model.hrq._last_boundary_hit

    cs_in_bounds = all(C_MIN <= c <= C_MAX for c in final_cs)
    any_nan = any(
        (math.isnan(v) or math.isinf(v))
        for v in final_cs + final_thetas + final_residual_norms + final_rt_errors + final_te_errors
    )

    verdict = {
        "issue": "#145",
        "arm": arm,
        "mobius_residual": mobius_residual,
        "gate2_pass": cs_in_bounds and not any_nan,
        "cs_in_bounds": cs_in_bounds,
        "any_nan_or_inf": any_nan,
        "final_cs": final_cs,
        "final_thetas": final_thetas,
        "final_residual_norms": final_residual_norms,
        "final_roundtrip_errors": final_rt_errors,
        "final_transport_errors": final_te_errors,
        "final_boundary_hit_rate": final_boundary,
        "n_unique_3digit": int(len(np.unique(sid_3digit, axis=0))),
        "n_unique_4digit": int(len(np.unique(sid_4digit, axis=0))),
        "sid_sha256": sha256_array(sid_4digit),
        "item_emb_sha256": item_emb_sha,
        "ckpt_path": CKPT_PATH,
        "sid_path": SID_PATH,
        "train_time_s": train_time,
        "epochs": NUM_EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    print(f"  verdict saved: {VERDICT_PATH}")

    # manifest.json
    manifest = {
        "issue": "#145",
        "arm": arm,
        "mobius_residual": mobius_residual,
        "stage2_dir": STAGE2_DIR,
        "ckpt": CKPT_PATH,
        "sid": SID_PATH,
        "verdict": VERDICT_PATH,
        "stage1_item_emb": ITEM_EMB_PARQUET,
        "config": {
            "n_items": N_ITEMS, "emb_dim": EMB_DIM, "e_dim": E_DIM,
            "codebook_sizes": CODEBOOK_SIZES,
            "encoder_layers": ENCODER_LAYERS,
            "n_epochs": NUM_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "beta": BETA, "sk_eps": SK_EPS, "sk_iters": SK_ITERS,
            "C_MIN": C_MIN, "C_MAX": C_MAX,
            "seed": SEED,
        },
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  manifest saved: {MANIFEST_PATH}")

    print(f"\n✓ Stage 2 ({arm}) 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
