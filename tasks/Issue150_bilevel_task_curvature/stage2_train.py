"""Issue #150 — Stage2 训练脚本 (A vs B bilevel task-aware curvature)

用法:
  python3 stage2_train.py --arm control    # A: 单层优化 (phi, theta 都由 RQ-VAE loss 更新)
  python3 stage2_train.py --arm treatment  # B: 双层优化 (phi 由 RQ-VAE 更新; theta 由 outer preference hypergradient 更新)

A/B 共享 Stage1 (item_emb.parquet), 内层仅用 inner 用户, 外层仅用 meta 用户 (90/10 hash 划分).

A 算法:
  for step in steps:
    batch = sample(inner_train, batch_size)
    out, rq_loss, _, _, _ = model(batch)
    loss = recon_loss(out, batch) + rq_loss
    loss.backward()  # 通过所有参数
    optimizer.step()  # 更新 encoder + decoder + codebook + theta

B 算法:
  for step in steps:
    # 内层虚拟更新
    phi' = phi - eta_inner * grad_phi L_RQ(inner_batch)
    # 外层目标
    L_outer = L_preference(meta_batch; phi', theta)
    # 只更新 theta
    theta_grad = grad_theta L_outer
    theta <- theta - eta_curv * theta_grad
    # 真实内层 step 更新 phi
    phi <- phi - eta_inner * grad_phi L_RQ(inner_batch)
"""

import os, sys, json, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TASK_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue150_bilevel_task_curvature"
sys.path.insert(0, os.path.join(TASK_DIR, "_lib"))

from utils import EmbDataset, poincare_distance, expmap0, proj_to_ball
from bilevel_quantizer import (
    BilevelHRQVAE, preference_loss, TAU_INIT, TAU_MIN, TAU_DECAY, ETA_INNER,
    C_MIN, C_MAX,
)

# === R30: 全部配置硬编码 ===
EMB_DIM = 768
E_DIM = 32
CODEBOOK_SIZES = (64, 128, 256)
ENCODER_LAYERS = (512, 256, 128, 64)
N_ITEMS = 9922
BATCH_SIZE = 256
LR_PHI = 1e-3          # 内层学习率 (phi + A 的 theta)
LR_THETA = 1e-2        # 外层学习率 (B 的 theta)
N_EPOCHS = 100
LOG_EVERY = 5
SEED = 42
STAGE1_PARQUET = f"{TASK_DIR}/stage1/item_emb.parquet"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/train.parquet"


def user_hash_frac(user_id):
    return int(hashlib.md5(str(int(user_id)).encode()).hexdigest()[:8], 16) / 2**32


import hashlib


def inner_meta_split(train_df):
    hashes = train_df["user"].apply(user_hash_frac)
    inner_mask = hashes < 0.9
    return train_df[inner_mask].reset_index(drop=True), train_df[~inner_mask].reset_index(drop=True)


def make_pref_pairs(meta_df, seed=42):
    rng = np.random.RandomState(seed)
    N = 9922
    pairs = []
    for _, row in meta_df.iterrows():
        history = list(row["history"])
        target = int(row["target"])
        if len(history) == 0:
            continue
        rng.shuffle(history)
        i = int(history[0])
        j_pos = target
        j_neg = int(rng.randint(0, N))
        while j_neg == j_pos:
            j_neg = int(rng.randint(0, N))
        pairs.append((i, j_pos, j_neg))
    return pairs


def sample_pref_batch(pairs, batch_size, rng):
    idx = rng.randint(0, len(pairs), batch_size)
    batch = [pairs[i] for i in idx]
    item_i = torch.tensor([p[0] for p in batch], dtype=torch.long)
    item_jp = torch.tensor([p[1] for p in batch], dtype=torch.long)
    item_jn = torch.tensor([p[2] for p in batch], dtype=torch.long)
    return item_i, item_jp, item_jn


def recon_loss(out, target):
    """Issue148 风格: 在 Poincaré ball 上计算 recon loss, target 是 tangent 空间 (item_emb)."""
    out_proj = proj_to_ball(expmap0(out, c=1.0), c=1.0)
    target_proj = proj_to_ball(expmap0(target, c=1.0), c=1.0)
    return torch.mean(poincare_distance(out_proj, target_proj, c=1.0) ** 2)


def sample_inner_batch(item_emb, batch_size, rng):
    """Sample ITEM embeddings for inner RQ-VAE training (R40: Stage2 trains codebook on items)."""
    sel = rng.randint(0, item_emb.shape[0], batch_size)
    return item_emb[sel]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["control", "treatment"], required=True)
    args = parser.parse_args()
    arm = args.arm
    bilevel = (arm == "treatment")
    out_dir = Path(TASK_DIR) / arm / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    item_emb = torch.tensor(EmbDataset(STAGE1_PARQUET).embeddings,
                             dtype=torch.float32).to(device)
    train_df = pd.read_parquet(TRAIN_PARQUET)
    inner_df, meta_df = inner_meta_split(train_df)
    pairs = make_pref_pairs(meta_df)
    print(f"[stage2/{arm}] inner rows: {len(inner_df)} meta rows: {len(meta_df)} pairs: {len(pairs)} item_emb: {item_emb.shape}")

    model = BilevelHRQVAE(
        in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
        layers=ENCODER_LAYERS, beta=1.0, kmeans_init=True, kmeans_iters=1000,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage2/{arm}] model params: {n_params}, bilevel={bilevel}")

    # Pre-init codewords with full item embedding kmeans (避免 batch 小于 n_e 时 kmeans 退化)
    from utils import kmeans
    model.eval()
    with torch.no_grad():
        z_full = model.encoder(item_emb)  # (9922, 32)
        for q in model.vq_layers:
            # 多轮 kmeans 找最佳 centers
            centers = kmeans(z_full, q.n_e, 50)
            # 缩放 centers 到 ball 安全范围 (norm < 0.5)
            norms = centers.norm(dim=-1, keepdim=True)
            scale = torch.clamp(0.5 / (norms + 1e-8), max=1.0)
            q.embeddings.weight.data.copy_(centers * scale)
            q.initted = True
    print(f"[stage2/{arm}] codewords pre-init done")

    # A: 全部参数一组优化器; B: phi (encoder/decoder/codebook) 和 theta 各一组
    phi_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    for q in model.vq_layers:
        phi_params += list(q.embeddings.parameters())
    theta_params = [q.theta for q in model.vq_layers]
    opt_phi = torch.optim.Adam(phi_params, lr=LR_PHI)
    if bilevel:
        opt_theta = torch.optim.Adam(theta_params, lr=LR_THETA)
    else:
        opt_theta = torch.optim.Adam(theta_params, lr=LR_PHI)  # A: theta 走 RQ loss
        # A 的总优化器 = phi + theta
        opt_all = torch.optim.Adam(phi_params + theta_params, lr=LR_PHI)

    n_steps_per_epoch = max(1, N_ITEMS // BATCH_SIZE)
    tau = float(TAU_INIT)
    log_records = []
    t_start = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        # tau anneal
        if bilevel:
            tau = max(TAU_MIN, tau * TAU_DECAY)
        epoch_losses = []
        epoch_pref_losses = []
        for step in range(n_steps_per_epoch):
            x = sample_inner_batch(item_emb, BATCH_SIZE, rng)
            if not bilevel:
                # A: 单层优化 (Issue148 风格: Poincaré recon loss in ball space)
                opt_all.zero_grad()
                out, rq_loss, _, _, _ = model(x, use_sk=False)
                loss = recon_loss(out, x) + rq_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_all.step()
                epoch_losses.append(float(loss.item()))
            else:
                # B: 双层优化
                # (1) 内层虚拟更新 phi' = phi - eta_inner * grad_phi L_RQ
                opt_phi.zero_grad()
                out_inner, rq_loss, _, _, _ = model(x, use_sk=False)
                inner_loss = recon_loss(out_inner, x) + rq_loss
                # 计算 phi 的梯度
                grads_phi = torch.autograd.grad(inner_loss, phi_params, create_graph=False, retain_graph=False)
                # 虚拟 phi' (不会真正写入参数)
                phi_prime = [p - ETA_INNER * g for p, g in zip(phi_params, grads_phi)]
                # (2) 外层目标
                item_i, item_jp, item_jn = sample_pref_batch(pairs, BATCH_SIZE, rng)
                outer_loss, _ = compute_outer_loss_with_phi(
                    model, item_emb, item_i.to(device), item_jp.to(device), item_jn.to(device),
                    tau, phi_prime, phi_params)
                # (3) 只更新 theta
                opt_theta.zero_grad()
                outer_loss.backward()
                torch.nn.utils.clip_grad_norm_(theta_params, 1.0)
                opt_theta.step()
                # (4) 真实内层 step 更新 phi
                opt_phi.zero_grad()
                out_inner2, rq_loss2, _, _, _ = model(x, use_sk=False)
                inner_loss_real = recon_loss(out_inner2, x) + rq_loss2
                inner_loss_real.backward()
                torch.nn.utils.clip_grad_norm_(phi_params, 1.0)
                opt_phi.step()
                epoch_losses.append(float(inner_loss_real.item()))
                epoch_pref_losses.append(float(outer_loss.item()))
        avg_loss = float(np.mean(epoch_losses))
        with torch.no_grad():
            cs = [float(q.get_c().item()) for q in model.vq_layers]
            n_unique_3digit = int(np.unique(np.concatenate([
                model.get_indices(item_emb[i:i + 1024], use_sk=False).cpu().numpy()
                for i in range(0, N_ITEMS, 1024)
            ]), axis=0).shape[0])
        rec = {
            "epoch": epoch,
            "loss": avg_loss,
            "cs_global": cs,
            "n_unique_3digit": n_unique_3digit,
            "tau": tau if bilevel else None,
            "avg_outer_loss": float(np.mean(epoch_pref_losses)) if bilevel and epoch_pref_losses else None,
        }
        log_records.append(rec)
        if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
            extra = f" outer={rec['avg_outer_loss']}" if bilevel and rec['avg_outer_loss'] is not None else ""
            print(f"  epoch {epoch:3d} | loss={avg_loss:.4f} | c={cs} | SID3={n_unique_3digit} | tau={tau:.3f}{extra}")
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

    # 保存 ckpt (Stage3/4 兼容: 含 final_cs_global + final_cs)
    final_cs_global = [float(q.get_c().item()) for q in model.vq_layers]
    ckpt = {
        "model_state_dict": model.state_dict(),
        "bilevel": bilevel,
        "final_cs_global": final_cs_global,
        "final_cs": final_cs_global,
        "n_unique_3digit": n_unique_3digit,
        "epoch": N_EPOCHS,
        "tau_final": tau if bilevel else None,
    }
    ckpt_path = out_dir / "hrqvae_kappa_sync.ckpt"
    torch.save(ckpt, ckpt_path)
    print(f"[stage2/{arm}] ckpt saved: {ckpt_path}")

    with open(log_path, "w") as f:
        for rec in log_records:
            f.write(json.dumps(rec) + "\n")

    verdict = {
        "arm": arm,
        "bilevel": bilevel,
        "n_epochs": N_EPOCHS,
        "final_cs_global": final_cs_global,
        "n_unique_3digit": n_unique_3digit,
        "train_time_s": train_time,
        "ckpt_path": str(ckpt_path),
        "sid_path": str(sid_path),
    }
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    print(f"[stage2/{arm}] verdict saved")


def compute_outer_loss_with_phi(model, item_emb, item_i, item_jp, item_jn, tau, phi_prime, phi_params):
    """Compute outer preference loss using phi_prime (virtual params) instead of actual phi.

    phi_prime 顺序与 phi_params 一致: encoder.parameters() + decoder.parameters() + [vq.embeddings for vq]
    """
    n_enc = sum(1 for _ in model.encoder.parameters())
    n_dec = sum(1 for _ in model.decoder.parameters())
    enc_params = phi_prime[:n_enc]
    dec_params = phi_prime[n_enc:n_enc + n_dec]
    vq_params = phi_prime[n_enc + n_dec:]  # 3 个 embedding weight
    # 提取 Linear 层列表 (按顺序)
    enc_linears = [m for m in model.encoder.mlp if isinstance(m, torch.nn.Linear)]
    n_enc_lin = len(enc_linears)
    # Encoder functional call: 每两层间 ReLU (最后无激活)
    def mlp_forward(params, x, linears):
        h = x
        for i, lin in enumerate(linears):
            w = params[2 * i]
            b = params[2 * i + 1]
            h = torch.nn.functional.linear(h, w, b)
            if i < len(linears) - 1:
                h = torch.nn.functional.relu(h)
        return h
    z_i = mlp_forward(enc_params, item_emb[item_i], enc_linears)
    z_jp = mlp_forward(enc_params, item_emb[item_jp], enc_linears)
    z_jn = mlp_forward(enc_params, item_emb[item_jn], enc_linears)
    from utils import proj_to_ball, expmap0, logmap0, poincare_distance
    def all_layers_soft(z):
        soft_qs = []
        for q, cb_param in zip(model.vq_layers, vq_params):
            c = q.get_c()
            latent_h = proj_to_ball(expmap0(z, c), c)
            cb_h = proj_to_ball(expmap0(cb_param, c), c)
            d = poincare_distance(
                latent_h.unsqueeze(1).expand(z.shape[0], q.n_e, -1),
                cb_h.unsqueeze(0).expand(z.shape[0], q.n_e, -1), c
            ).squeeze(-1)
            logits = -d / max(tau, 1e-3)
            p = torch.nn.functional.softmax(logits, dim=-1)
            emb_tan = logmap0(cb_h, c)
            q_tan = torch.matmul(p, emb_tan)
            q_h = proj_to_ball(expmap0(q_tan, c), c)
            soft_qs.append(q_h)
        return soft_qs
    qs_i = all_layers_soft(z_i)
    qs_jp = all_layers_soft(z_jp)
    qs_jn = all_layers_soft(z_jn)
    def score(qs_a, qs_b):
        s = 0.0
        for qa, qb, q in zip(qs_a, qs_b, model.vq_layers):
            c = q.get_c()
            d = poincare_distance(qa, qb, c).squeeze(-1)
            s = s - d.mean()
        return s
    s_pos = score(qs_i, qs_jp)
    s_neg = score(qs_i, qs_jn)
    margin = s_pos - s_neg
    loss = -torch.nn.functional.logsigmoid(margin)
    return loss, {"margin": margin.detach(), "s_pos": s_pos.detach(), "s_neg": s_neg.detach()}


if __name__ == "__main__":
    main()