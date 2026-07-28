#!/usr/bin/env python3
"""
Task #194 — K0 码本容量 × L0_err 诊断
复用 task191 骨架, 但遍历 4 个 K0 臂 (K0={32, 64, 128, 256}).
对每个 K0 臂算 L0_err vs epoch + collision vs epoch + L0_err vs collision (按 K0 分组).

输出:
- logs/task194/k0_l0_err.tsv (epoch, K0, collision, L0_err_mean, L0_err_p50, L0_err_p90)
- verdicts/task194_k0_capacity_diagnose.json (per-K0 corr(L0_err, collision))

用法:
  python3 scripts/task194_k0_l0_err_diagnose.py
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/HG-Rec")
sys.path.insert(0, os.getcwd())

from model.utils import EmbDataset
from model.hrqvae import HRQVAE

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
DATA_PATH = f"{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet"
OUT_TSV = f"{REPO}/logs/task194/k0_l0_err.tsv"
OUT_JSON = f"{REPO}/verdicts/task194_k0_capacity_diagnose.json"

os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)


def compute_l0_quant_error(model, dataloader, device):
    errs = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            z = model.encoder(x)
            quantizer_0 = model.hrq.vq_layers[0]
            x_res_0, _, _ = quantizer_0(z, use_sk=False)
            err = torch.norm(z - x_res_0, dim=-1)
            errs.append(err.detach().cpu().numpy())
    return np.concatenate(errs, axis=0)


def main():
    K0_LIST = [32, 64, 128, 256]
    all_results = []
    summary = {}

    # 共享 dataset (各 K0 臂共用)
    data = EmbDataset(DATA_PATH)
    print(f"[Task #194] Dataset size: {len(data)}, dim: {data.dim}")
    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=0)
    device = torch.device("cuda:0")

    for K0 in K0_LIST:
        hrqvae_root = f"{REPO}/products/task194/hrqvae_k0{K0}"
        if not os.path.isdir(hrqvae_root):
            print(f"[Task #194] K0={K0}: no dir {hrqvae_root}, skip")
            continue

        subdirs = sorted([d for d in os.listdir(hrqvae_root)
                          if os.path.isdir(os.path.join(hrqvae_root, d))])
        if not subdirs:
            print(f"[Task #194] K0={K0}: no subdir, skip")
            continue
        ckpt_dir = os.path.join(hrqvae_root, subdirs[0])

        all_files = sorted(os.listdir(ckpt_dir))
        ckpt_paths = sorted([os.path.join(ckpt_dir, f) for f in all_files
                             if f.startswith("epoch_") and "_collision_" in f and f.endswith(".pth")])
        print(f"[Task #194] K0={K0}: found {len(ckpt_paths)} ckpts")

        if not ckpt_paths:
            continue

        K0_results = []
        for ckpt_path in ckpt_paths:
            fname = os.path.basename(ckpt_path)
            parts = fname.replace(".pth", "").split("_")
            try:
                epoch = int(parts[1])
                collision = float(parts[3])
            except (IndexError, ValueError):
                continue

            # Load model
            try:
                model = HRQVAE.load_from_checkpoint(ckpt_path)
                model.freeze()
            except Exception as e:
                print(f"  ⚠️ epoch {epoch} load failed: {e}")
                continue

            errs = compute_l0_quant_error(model, loader, device)
            l0_err_mean = float(errs.mean())
            l0_err_p50 = float(np.percentile(errs, 50))
            l0_err_p90 = float(np.percentile(errs, 90))

            row = {
                "epoch": epoch,
                "K0": K0,
                "collision": collision,
                "l0_err_mean": l0_err_mean,
                "l0_err_p50": l0_err_p50,
                "l0_err_p90": l0_err_p90,
            }
            K0_results.append(row)
            all_results.append(row)

        if K0_results:
            K0_results.sort(key=lambda r: r["epoch"])
            epochs = np.array([r["epoch"] for r in K0_results])
            colls = np.array([r["collision"] for r in K0_results])
            errs = np.array([r["l0_err_mean"] for r in K0_results])
            if len(errs) > 2:
                corr = float(np.corrcoef(errs, colls)[0, 1])
            else:
                corr = None
            summary[f"K0_{K0}"] = {
                "n_epochs": len(K0_results),
                "collision_min": float(colls.min()),
                "collision_final": float(colls[-1]),
                "collision_final_over_min": float(colls[-1] / max(colls.min(), 1e-9)),
                "l0_err_min": float(errs.min()),
                "l0_err_final": float(errs[-1]),
                "l0_err_final_over_min": float(errs[-1] / max(errs.min(), 1e-9)),
                "corr_l0_err_collision": corr,
            }
            print(f"  K0={K0}: collision {colls.min():.3f} → {colls[-1]:.3f} "
                  f"(ratio={colls[-1]/max(colls.min(),1e-9):.3f}), "
                  f"L0_err {errs.min():.4f} → {errs[-1]:.4f} "
                  f"(ratio={errs[-1]/max(errs.min(),1e-9):.3f}), "
                  f"corr={corr:.3f}" if corr is not None else "")

    # Write TSV
    if all_results:
        with open(OUT_TSV, "w") as f:
            f.write("epoch\tK0\tcollision\tl0_err_mean\tl0_err_p50\tl0_err_p90\n")
            for r in sorted(all_results, key=lambda r: (r["K0"], r["epoch"])):
                f.write(f"{r['epoch']}\t{r['K0']}\t{r['collision']:.4f}\t"
                        f"{r['l0_err_mean']:.6f}\t{r['l0_err_p50']:.6f}\t{r['l0_err_p90']:.6f}\n")
        print(f"[Task #194] Wrote {OUT_TSV}")

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Task #194] Wrote {OUT_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()