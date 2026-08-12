"""Issue #149 — Stage2 训练脚本 (A vs B signed κ-stereographic)

用法:
  python3 stage2_train.py --arm control    # A: signed κ 只能负曲率 c_l ∈ [C_MIN, C_MAX]
  python3 stage2_train.py --arm treatment  # B: signed κ ∈ (-κ_max, +κ_max), init κ=-1.25

输出:
  {TASK_DIR}/{arm}/stage2/
    ├── hrqvae_kappa_sync.ckpt  — Stage2 ckpt (含 final_cs_global + final_cs)
    ├── sid_output.npy          — (N_ITEMS, 3) SID 整数
    ├── verdict.json            — Stage2 训练摘要
    └── train_log.jsonl         — 每 epoch loss/c/SID unique
"""

import os, sys, json, time, argparse
from pathlib import Path

import numpy as np
import torch

TASK_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue149_signed_k_stereographic"
sys.path.insert(0, os.path.join(TASK_DIR, "_lib"))

from utils import EmbDataset, poincare_distance, expmap0, proj_to_ball
from signed_kappa_quantizer import SignedKappaHRQVAE

# === R30: 全部配置硬编码 ===
EMB_DIM = 768
E_DIM = 32
CODEBOOK_SIZES = (64, 128, 256)
ENCODER_LAYERS = (512, 256, 128, 64)
N_ITEMS = 9922
BATCH_SIZE = 256
LR = 1e-3
N_EPOCHS = 200
LOG_EVERY = 10
SEED = 42
STAGE1_PARQUET = f"{TASK_DIR}/stage1/item_emb.parquet"


def poincare_recon_loss(out, target):
    """recon = mean poincare_distance(out, target)^2 (target 在 ball 上)"""
    return torch.mean(poincare_distance(out, target, c=1.0) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["control", "treatment"], required=True)
    args = parser.parse_args()
    arm = args.arm
    signed_kappa = (arm == "treatment")
    out_dir = Path(TASK_DIR) / arm / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    item_emb = torch.tensor(EmbDataset(STAGE1_PARQUET).embeddings,
                             dtype=torch.float32).to(device)
    print(f"[stage2/{arm}] item_emb: {item_emb.shape}, signed_kappa={signed_kappa}")

    model = SignedKappaHRQVAE(
        in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
        layers=ENCODER_LAYERS, kmeans_init=True, kmeans_iters=1000,
        signed_kappa=signed_kappa,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage2/{arm}] model params: {n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps_per_epoch = max(1, N_ITEMS // BATCH_SIZE)
    print(f"[stage2/{arm}] start training: {N_EPOCHS} epochs, {n_steps_per_epoch} steps/epoch")

    log_records = []
    t_start = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        perm = np.random.permutation(N_ITEMS)
        epoch_losses = []
        for step in range(n_steps_per_epoch):
            idx = perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            batch = item_emb[idx]
            opt.zero_grad()
            out, rq_loss, indices, zq, z = model(batch, use_sk=False)
            recon = poincare_recon_loss(out, batch)
            loss = recon + rq_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss.item()))
        avg_loss = float(np.mean(epoch_losses))
        with torch.no_grad():
            if signed_kappa:
                cs = [float(q.get_c_or_kappa()[0].item()) for q in model.vq_layers]
            else:
                cs = [float(q.get_c_or_kappa()[0].item()) for q in model.vq_layers]
            n_unique_3digit = int(np.unique(np.concatenate([
                model.get_indices(item_emb[i:i + 1024], use_sk=False).cpu().numpy()
                for i in range(0, N_ITEMS, 1024)
            ]), axis=0).shape[0])
        rec = {
            "epoch": epoch,
            "loss": avg_loss,
            "cs_or_kappa_global": cs,
            "n_unique_3digit": n_unique_3digit,
        }
        log_records.append(rec)
        if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
            print(f"  epoch {epoch:4d} | loss={avg_loss:.4f} | c/κ={cs} | SID3={n_unique_3digit}")
    train_time = time.time() - t_start
    print(f"[stage2/{arm}] training done in {train_time:.1f}s")

    # 推断 SID (全量 9922)
    model.eval()
    all_indices = []
    with torch.no_grad():
        for i in range(0, N_ITEMS, 1024):
            batch = item_emb[i:i + 1024]
            all_indices.append(model.get_indices(batch, use_sk=False).cpu().numpy())
    sid = np.concatenate(all_indices, axis=0)
    assert sid.shape == (N_ITEMS, 3)
    sid_path = out_dir / "sid_output.npy"
    np.save(sid_path, sid)
    n_unique_3digit = int(np.unique(sid, axis=0).shape[0])
    print(f"[stage2/{arm}] SID unique: {n_unique_3digit}/{N_ITEMS}")

    # 提取 c/κ
    final_cs_global = [float(q.get_c_or_kappa()[0].item()) for q in model.vq_layers]

    # 保存 ckpt (含 final_cs_global + final_cs — Stage3/4 兼容)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "signed_kappa": signed_kappa,
        "final_cs_global": final_cs_global,
        "final_cs": final_cs_global,  # Stage4 hyperbolic_attention_bias 兼容
        "n_unique_3digit": n_unique_3digit,
        "epoch": N_EPOCHS,
    }
    ckpt_path = out_dir / "hrqvae_kappa_sync.ckpt"
    torch.save(ckpt, ckpt_path)
    print(f"[stage2/{arm}] ckpt saved: {ckpt_path}")

    # 保存 train log
    with open(log_path, "w") as f:
        for rec in log_records:
            f.write(json.dumps(rec) + "\n")

    # 保存 verdict
    verdict = {
        "arm": arm,
        "signed_kappa": signed_kappa,
        "n_epochs": N_EPOCHS,
        "final_cs_global": final_cs_global,
        "n_unique_3digit": n_unique_3digit,
        "train_time_s": train_time,
        "ckpt_path": str(ckpt_path),
        "sid_path": str(sid_path),
    }
    verdict_path = out_dir / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    print(f"[stage2/{arm}] verdict: {verdict_path}")


if __name__ == "__main__":
    main()
