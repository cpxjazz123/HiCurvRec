"""Issue #151 Stage2 train — Assignment-coupled GW Curvature.

A/B controlled by --arm {control, treatment}:
- control:    GW_LAMBDA=0 (regular RQ-VAE training)
- treatment:  GW_LAMBDA=1.0 (alternating RQ + GW step)

Usage:
    cd control/ && python3 stage2_train.py --arm control
    cd treatment/ && python3 stage2_train.py --arm treatment

(R43: only --arm flag; all numeric hyperparams hardcoded.)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# R44 baseline self-contained: local _lib first
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "_lib"))
sys.path.insert(0, str(HERE.parent / "_lib"))  # shared gw_quantizer at parent level

# Pick the correct _lib for this arm
ARM_DIR = HERE.name  # control or treatment
sys.path.insert(0, str(HERE))  # local _lib copy
from gw_quantizer import (  # noqa: E402
    EMB_DIM, E_DIM, CODEBOOK_SIZES, BATCH_SIZE, N_EPOCHS, SEED,
    LR_PHI, LR_THETA, GW_LAMBDA, REC_LAMBDA, REC_TAU, REC_NEG_N,
    GWHRQVAE, poincare_recon_loss, gromov_wasserstein_loss, C_MIN, C_MAX,
)


_argparser = argparse.ArgumentParser()
_argparser.add_argument("--arm", type=str, choices=["control", "treatment"], required=True)
_args = _argparser.parse_args()
ARM = _args.arm


# Paths (R44 self-contained)
ITEM_EMB_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/baseline/stage1/item_emb.parquet"
STAGE2_DIR = HERE / "stage2"
STAGE2_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[stage2/{ARM}] {msg}", flush=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_items():
    df = pd.read_parquet(ITEM_EMB_PARQUET)
    embs = np.stack([np.asarray(e, dtype=np.float32) for e in df["embedding"].values])
    return torch.from_numpy(embs)


def pre_init_codebooks(model: GWHRQVAE, item_emb: torch.Tensor):
    """Encode all items, run kmeans on z_full for each VQ layer."""
    model.eval()
    with torch.no_grad():
        z = model.encoder(item_emb.cuda())
        for q in model.vq_layers:
            q.initted = False
        residual = z
        for q in model.vq_layers:
            q.init_emb(residual)
            x_res, _, _ = q(residual, use_sk=False)
            residual = residual - x_res
    model.train()


def main():
    set_seed(SEED)
    device = torch.device("cuda:0")
    log(f"loading {ITEM_EMB_PARQUET}")
    item_emb = load_items()
    log(f"item_emb: {tuple(item_emb.shape)}")

    log("building GWHRQVAE")
    model = GWHRQVAE().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params: {n_params}, GW_LAMBDA={GW_LAMBDA if ARM=='treatment' else 0.0}")

    log("pre-init codewords with kmeans on z_full")
    pre_init_codebooks(model, item_emb)

    # Optimizers
    phi_params = [p for n, p in model.named_parameters() if "theta" not in n]
    theta_params = [p for n, p in model.named_parameters() if "theta" in n]
    opt_phi = torch.optim.Adam(phi_params, lr=LR_PHI)
    opt_theta = torch.optim.Adam(theta_params, lr=LR_THETA)

    n = item_emb.shape[0]
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    g = torch.Generator(device=device).manual_seed(SEED)

    use_gw = (ARM == "treatment")
    n_steps = N_EPOCHS * (n // BATCH_SIZE)
    log(f"start training: epochs={N_EPOCHS}, batch={BATCH_SIZE}, n_steps={n_steps}, use_gw={use_gw}")

    step = 0
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        idx_perm = rng.permutation(n)
        ep_loss = 0.0
        ep_count = 0
        for s in range(0, n, BATCH_SIZE):
            idx = idx_perm[s:s + BATCH_SIZE]
            batch = item_emb[idx].to(device)
            opt_phi.zero_grad()
            if use_gw:
                opt_theta.zero_grad()
            out, rq_loss, indices, z_q, z = model(batch)
            recon = poincare_recon_loss(out, batch, c=1.0)
            total_loss = recon + rq_loss
            total_loss.backward()
            opt_phi.step()

            # GW step (treatment only, after phi step — codebook/encoder have been updated)
            gw_loss_val = None
            if use_gw:
                opt_theta.zero_grad()
                gw_loss = gromov_wasserstein_loss(model, batch, g)
                (GW_LAMBDA * gw_loss).backward()
                opt_theta.step()
                gw_loss_val = gw_loss.item()

            ep_loss += total_loss.item() * batch.shape[0]
            ep_count += batch.shape[0]
            step += 1

        # End epoch log
        avg_loss = ep_loss / max(ep_count, 1)
        cs = [q.get_c().item() for q in model.vq_layers]
        # Compute SID uniqueness
        with torch.no_grad():
            sid = model.get_indices(item_emb.to(device)).cpu().numpy()
            unique_3 = len(set(map(tuple, sid[:, :3].tolist())))
        msg = f"epoch {epoch:3d} | loss={avg_loss:.4f} | c=[{cs[0]:.4f}, {cs[1]:.4f}, {cs[2]:.4f}] | SID3={unique_3}"
        if use_gw:
            msg += f" | gw={gw_loss_val:.4f}" if gw_loss_val is not None else ""
        log(msg)

    elapsed = time.time() - t0
    log(f"training done in {elapsed:.1f}s")

    # Save ckpt + SID
    ckpt_path = STAGE2_DIR / "hrqvae_kappa_sync.ckpt"
    final_cs = [q.get_c().item() for q in model.vq_layers]
    final_cs_global = final_cs
    state = {
        "model_state": model.state_dict(),
        "final_cs": final_cs,
        "final_cs_global": final_cs_global,
        "epoch": N_EPOCHS,
        "arm": ARM,
    }
    torch.save(state, ckpt_path)
    log(f"ckpt saved: {ckpt_path}")

    sid_path = STAGE2_DIR / "sid_output.npy"
    np.save(sid_path, sid)
    log(f"sid saved: {sid_path}, shape={sid.shape}, SID3 unique={len(set(map(tuple, sid[:, :3].tolist())))}/{n}")

    verdict = {
        "arm": ARM,
        "n_params": n_params,
        "use_gw": use_gw,
        "final_cs": final_cs,
        "final_cs_global": final_cs_global,
        "n_epochs": N_EPOCHS,
        "elapsed_s": elapsed,
        "ckpt": str(ckpt_path),
        "sid_path": str(sid_path),
        "sid_shape": list(sid.shape),
        "sid_unique_3_of_9922": len(set(map(tuple, sid[:, :3].tolist()))),
        "gw_lambda": GW_LAMBDA if use_gw else 0.0,
    }
    verdict_path = STAGE2_DIR / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2))
    log(f"verdict saved: {verdict_path}")


if __name__ == "__main__":
    main()