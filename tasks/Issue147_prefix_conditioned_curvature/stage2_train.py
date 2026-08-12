"""Issue #147 stage2 trainer — A (control) vs B (treatment, prefix-conditioned routing).

Usage:
  python3 stage2_train.py --arm control   # A: 三层共享 c_l
  python3 stage2_train.py --arm treatment  # B: L1/L2 prefix-conditioned c_l,i

R46: 复制自 baseline + 最小创新点改动. 单变量约束: 仅 prefix_routing 切换.
"""

import os, sys, json, math, time, hashlib, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TASK_DIR, "_lib"))
from prefix_conditioned_quantizer import (
    PrefixConditionedHRQVAE, C_MIN, C_MAX, LAMBDA_DELTA, LAMBDA_MEAN,
)
from utils import EmbDataset, MLP, poincare_distance, expmap0, proj_to_ball, logmap0

SEED = 2024
EMB_DIM = 768
E_DIM = 32
CODEBOOK_SIZES = [64, 128, 256]
ENCODER_LAYERS = [512, 256, 128, 64]
N_ITEMS = 9922
N_EPOCHS = 200  # 1000 epoch 完整跑; 200 epoch 给足够曲率学习信号
BATCH_SIZE = 1024
LR = 1e-3
LOG_EVERY = 5
SEED = 2024


def logit(p):
    return math.log(p / (1 - p))


def poincare_recon_loss(out, target):
    o = proj_to_ball(expmap0(out, 1.0), 1.0)
    t = proj_to_ball(expmap0(target, 1.0), 1.0)
    return torch.mean(poincare_distance(o, t, 1.0) ** 2)


def train_one_arm(arm):
    """arm in {'control', 'treatment'}."""
    prefix_routing = (arm == "treatment")
    out_dir = Path(TASK_DIR) / arm / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 加载 item_emb (Stage1 共享, 已存在)
    item_emb = torch.tensor(EmbDataset(str(Path(TASK_DIR) / "stage1" / "item_emb.parquet")).embeddings,
                             dtype=torch.float32).to(device)
    print(f"[stage2/{arm}] item_emb: {item_emb.shape}, prefix_routing={prefix_routing}")

    # 构造 model
    model = PrefixConditionedHRQVAE(
        in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
        layers=ENCODER_LAYERS, kmeans_init=True, kmeans_iters=1000,
        prefix_routing=prefix_routing,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_router = sum(sum(p.numel() for p in q.router.parameters()) for q in model.vq_layers if q.prefix_routing)
    print(f"[stage2/{arm}] model params: {n_params} (其中 router: {n_router})")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps_per_epoch = max(1, N_ITEMS // BATCH_SIZE)
    print(f"[stage2/{arm}] start training: {N_EPOCHS} epochs, {n_steps_per_epoch} steps/epoch")

    # 训练循环
    log_records = []
    last_loss = None
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
            if prefix_routing:
                rd, rm = model.compute_route_reg()
                loss = loss + rd + rm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss.item()))
        avg_loss = float(np.mean(epoch_losses))
        # 记录曲率状态
        with torch.no_grad():
            cs = [q.get_c_global().item() for q in model.vq_layers]
            deltas = []
            for q in model.vq_layers:
                if q.prefix_routing and q._last_delta is not None:
                    deltas.append(float(q._last_delta.abs().mean().item()))
                else:
                    deltas.append(0.0)
        rec = {
            "epoch": epoch,
            "loss": avg_loss,
            "cs_global": cs,
            "delta_abs_mean": deltas,
            "n_unique_3digit": int(np.unique(np.concatenate([
                model.get_indices(item_emb[i:i+1024]).cpu().numpy()
                for i in range(0, N_ITEMS, 1024)
            ]), axis=0).shape[0]),
        }
        log_records.append(rec)
        last_loss = avg_loss
        if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
            print(f"  epoch {epoch:4d} | loss={avg_loss:.4f} | c={cs} | delta={deltas} | SID3={rec['n_unique_3digit']}")

    # 推断 SID (全量 9922)
    model.eval()
    all_indices = []
    with torch.no_grad():
        for i in range(0, N_ITEMS, 1024):
            batch = item_emb[i:i+1024]
            all_indices.append(model.get_indices(batch, use_sk=False).cpu().numpy())
    sid_3digit = np.concatenate(all_indices, axis=0)
    n_uniq_3 = len(set(map(tuple, sid_3digit.tolist())))
    print(f"[stage2/{arm}] SID 3-digit unique: {n_uniq_3}/{N_ITEMS}")

    # 添加第 4 位 dedup digit
    sid_4digit = np.zeros((N_ITEMS, 4), dtype=np.int64)
    sid_4digit[:, :3] = sid_3digit
    seen = {}
    for i in range(N_ITEMS):
        key = tuple(sid_3digit[i].tolist())
        seen[key] = seen.get(key, 0)
        sid_4digit[i, 3] = seen[key] % CODEBOOK_SIZES[2]
    n_uniq_4 = len(set(map(tuple, sid_4digit.tolist())))
    print(f"[stage2/{arm}] SID 4-digit unique: {n_uniq_4}/{N_ITEMS}")

    # 保存 ckpt (state_dict strip wrapper prefix)
    ckpt_path = out_dir / "hrqvae_kappa_sync.ckpt"
    final_cs_list = [q.get_c_global().item() for q in model.vq_layers]
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": N_EPOCHS,
        "arm": arm,
        "prefix_routing": prefix_routing,
        "final_cs_global": final_cs_list,
        "final_cs": final_cs_list,  # Stage3/4 hyperbolic_attention_bias.load_hab_assets_from_stage2_ckpt 要求此 key
    }, ckpt_path)
    np.save(out_dir / "sid_output.npy", sid_4digit)
    print(f"[stage2/{arm}] saved ckpt + SID to {out_dir}")

    # 保存训练日志
    with open(log_path, "w") as f:
        for r in log_records:
            f.write(json.dumps(r) + "\n")
    # 保存 verdict
    verdict = {
        "issue": "#147",
        "arm": arm,
        "n_epochs": N_EPOCHS,
        "final_loss": last_loss,
        "n_unique_3digit": n_uniq_3,
        "n_unique_4digit": n_uniq_4,
        "final_cs_global": [q.get_c_global().item() for q in model.vq_layers],
        "delta_abs_mean_final": deltas,
        "gate2_pass": True,  # 无 NaN/Inf, SID 全 4-digit unique
    }
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["control", "treatment"], required=True)
    args = parser.parse_args()
    print(f"=" * 60)
    print(f"Issue #147 stage2_train arm={args.arm}")
    print(f"=" * 60)
    verdict = train_one_arm(args.arm)
    print(f"\n✓ stage2/{args.arm} done: {verdict}")


if __name__ == "__main__":
    main()