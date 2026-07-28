#!/usr/bin/env python3
"""
Task #191 — L0 量化误差 ‖z − e_L0‖ 随 epoch 变化 (vs collision 曲线)
从 task188 hrqvae_save_limit50 存的 30+ 个 epoch_*_collision_*.pth 直接算.
不重训. 验证机制:
  - 单调↑ 且与 collision 同步 → 机制坐实
  - 基本不变 → 假设二也不成立, 要重新找

输出:
  - logs/task191/l0_quant_error_per_epoch.tsv (epoch, collision, L0_err_mean, L0_err_p50, L0_err_p90)
  - verdicts/task191_l0_quant_error_result.md

用法:
  python3 scripts/task191_l0_quant_error_per_epoch.py
"""
import os, sys, glob, json
import numpy as np
import torch
from torch.utils.data import DataLoader

# R11: 不绕开共享 checkout (R13 禁止 worktree), 临时改 cwd 让 model.utils 可 import
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/HG-Rec")
sys.path.insert(0, os.getcwd())

from model.utils import EmbDataset
from model.hrqvae import HRQVAE

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
# glob [64,128,256] is interpreted as char set in Python glob, use os.listdir instead
HRQVAE_ROOT = f"{REPO}/products/task188/hrqvae_save_limit50"
DATA_PATH = f"{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet"
OUT_TSV = f"{REPO}/logs/task191/l0_quant_error_per_epoch.tsv"
OUT_JSON = f"{REPO}/verdicts/task191_l0_quant_error_per_epoch.json"

os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)


def compute_l0_quant_error(model, dataloader, device, e_dim):
    """返回 (N,) per-item Euclidean L0 quantization error ‖z − e_L0‖."""
    errs = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)  # (B, in_dim)
            z = model.encoder(x)  # (B, e_dim) 切空间 latent
            # 取出 vq_layers[0] 单独跑 (不要跑整个 hrq, 只要 L0)
            quantizer_0 = model.hrq.vq_layers[0]
            x_res_0, _, _ = quantizer_0(z, use_sk=False)  # (B, e_dim) = e_L0
            err = torch.norm(z - x_res_0, dim=-1)  # (B,) Euclidean
            errs.append(err.detach().cpu().numpy())
    return np.concatenate(errs, axis=0)


def main():
    # 1) os.listdir 找目录 (避免 glob [] 字符集陷阱)
    subdirs = sorted(os.listdir(HRQVAE_ROOT))
    if not subdirs:
        raise RuntimeError(f"No subdirs in {HRQVAE_ROOT}")
    CKPT_DIR = os.path.join(HRQVAE_ROOT, subdirs[0])
    print(f"[Task #191] CKPT_DIR = {CKPT_DIR}")
    all_files = sorted(os.listdir(CKPT_DIR))
    ckpt_paths = sorted([os.path.join(CKPT_DIR, f) for f in all_files
                         if f.startswith("epoch_") and "_collision_" in f and f.endswith(".pth")])
    print(f"[Task #191] Found {len(ckpt_paths)} epoch_collision ckpts")
    if not ckpt_paths:
        raise RuntimeError(f"No epoch_collision_*.pth in {CKPT_DIR}")

    # 2) 加载数据一次 (不会变)
    data = EmbDataset(DATA_PATH)
    print(f"[Task #191] Dataset size: {len(data)}, dim: {data.dim}")
    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=0)

    device = torch.device("cuda:0")

    # 3) 对每个 ckpt 算 L0 quant error
    results = []
    for ckpt_path in ckpt_paths:
        fname = os.path.basename(ckpt_path)
        # epoch_104_collision_0.1062_model.pth -> epoch=104, collision=0.1062
        # parts: ['epoch', '104', 'collision', '0.1062', 'model']
        parts = fname.replace(".pth", "").split("_")
        epoch = int(parts[1])  # parts[0]='epoch', parts[1]='104'
        collision = float(parts[3])  # parts[2]='collision', parts[3]='0.1062'

        ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device("cpu"))
        train_args = ckpt["args"]

        # R11.3: 复建一个 fresh model with same args (除了 save_limit/ckpt_dir/epoch)
        model = HRQVAE(
            in_dim=data.dim,
            num_emb_list=train_args.num_emb_list,
            e_dim=train_args.e_dim,
            layers=train_args.layers,
            dropout_prob=train_args.dropout_prob,
            bn=train_args.bn,
            loss_type=train_args.loss_type,
            quant_loss_weight=train_args.quant_loss_weight,
            beta=train_args.beta,
            kmeans_init=train_args.kmeans_init,
            kmeans_iters=train_args.kmeans_iters,
            sk_eps=train_args.sk_epsilons,
            sk_iters=train_args.sk_iters,
            euclidean_qloss=train_args.euclidean_qloss,
            loss_mult_codebook=train_args.loss_mult_codebook,
            radii=train_args.radii,
            rho_reg_weight=train_args.rho_reg_weight,
            product_manifold=train_args.product_manifold,
            angular_dim=train_args.angular_dim,
            radial_dim=train_args.radial_dim,
            alpha=train_args.alpha,
            beta_radial=train_args.beta_radial,
        )
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)

        # 4) 算 L0 quant error (per item)
        errs = compute_l0_quant_error(model, loader, device, train_args.e_dim)
        mean_err = float(np.mean(errs))
        p50_err = float(np.percentile(errs, 50))
        p90_err = float(np.percentile(errs, 90))
        p99_err = float(np.percentile(errs, 99))

        results.append({
            "epoch": epoch,
            "collision": collision,
            "L0_err_mean": mean_err,
            "L0_err_p50": p50_err,
            "L0_err_p90": p90_err,
            "L0_err_p99": p99_err,
            "ckpt": fname,
        })
        print(f"  ep={epoch:4d} coll={collision:.4f} L0_err: mean={mean_err:.4f} p50={p50_err:.4f} p90={p90_err:.4f} p99={p99_err:.4f}")

    # 5) 按 epoch 排序
    results.sort(key=lambda r: r["epoch"])

    # 6) 写 TSV
    with open(OUT_TSV, "w") as f:
        f.write("epoch\tcollision\tL0_err_mean\tL0_err_p50\tL0_err_p90\tL0_err_p99\tckpt\n")
        for r in results:
            f.write(f"{r['epoch']}\t{r['collision']:.6f}\t{r['L0_err_mean']:.6f}\t{r['L0_err_p50']:.6f}\t{r['L0_err_p90']:.6f}\t{r['L0_err_p99']:.6f}\t{r['ckpt']}\n")
    print(f"\n[Task #191] TSV saved: {OUT_TSV}")

    # 7) 写 JSON
    with open(OUT_JSON, "w") as f:
        json.dump({
            "task": "task191_l0_quant_error",
            "ckpt_dir": CKPT_DIR,
            "n_ckpts": len(results),
            "per_epoch": results,
        }, f, indent=2)
    print(f"[Task #191] JSON saved: {OUT_JSON}")

    # 8) 一行摘要
    print("\n=== Task #191 摘要 ===")
    print(f"  epoch 范围: {results[0]['epoch']} → {results[-1]['epoch']}")
    epochs = np.array([r["epoch"] for r in results])
    colls = np.array([r["collision"] for r in results])
    errs_mean = np.array([r["L0_err_mean"] for r in results])
    print(f"  L0_err (mean) 范围: {errs_mean.min():.4f} → {errs_mean.max():.4f} (Δ={(errs_mean.max()-errs_mean.min())/errs_mean.min():+.1%})")
    # 简单相关 / 单调性判据
    # Pearson 跟 collision 相关
    if len(epochs) > 2:
        # 把 epoch 当时间, 看 mean_err 是否单调↑
        diffs = np.diff(errs_mean)
        n_ups = (diffs > 0).sum()
        n_downs = (diffs < 0).sum()
        pct_up = n_ups / (n_ups + n_downs) if (n_ups + n_downs) > 0 else 0
        # Pearson 跟 collision
        if colls.std() > 1e-9 and errs_mean.std() > 1e-9:
            cor = float(np.corrcoef(colls, errs_mean)[0, 1])
        else:
            cor = float('nan')
        print(f"  跟 collision 相关系数: r = {cor:.3f}")
        print(f"  趋势: {n_ups}/{n_ups+n_downs} 步↑ ({pct_up:.0%})")


if __name__ == "__main__":
    main()
