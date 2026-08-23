"""R99 Scale-Decoupled Hyperbolic Quantization.

继承 v97 Radially-Diverse Codebook 参数化 (a_ℓk → q → z → r → v = r·u/|u|, |v|≈1).
新增: 每层 EMA residual norm m_ℓ → γ_ℓ = 1/(m_ℓ + ε).
  - assignment: 在 γ_ℓ·r_ℓ hyperbolic space (与 |v|≈1 同尺度, curvature 真工作)
  - subtraction: 用 v_ℓk/γ_ℓ 回到 residual scale (reconstruction 精确)

c=0.5 固定 (不 learnable). 10k steps 训练 + 每 1k ckpt.

5 gates:
  1. L1 recon_improve > 0  (从 v97 -13.73 转正)
  2. |r_ℓ| 严格递减
  3. L1/L2 unique codes > 10
  4. margin 显著下降 (< 3.10 / 2.40)
  5. curvature sensitivity 保留 (NN overlap < 100% across c)

启动 (DDP 4 卡, R51+ 确定性):
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=":4096:8" \
  /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun --nproc_per_node=4 --master_port=29530 \
      /home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment_scale_decoupled_v99/train_rqvae_instruments.py
"""
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/curvature_base/modules")
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode, QuantizeOutput
from modules.hyperbolic import _expmap0_t, _poincare_distance_t
from data.schemas import SeqBatch

from curvature_config import ITEM_EMB_NPY as EMB_NPY, RQVAE_OUT_DIR as OUT_DIR


# === 超参 (硬编码 R30/R43) ===
SEED = 42

INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
EMBED_DIM = 32
CODEBOOK_SIZE = 256
N_LAYERS = 3
COMMITMENT_WEIGHT = 1.0

# === 训练硬约束 ===
NUM_EPOCHS = 10000
MAX_GLOBAL_STEPS = 10_000        # R99: 10k steps (与 v97 对齐)
CKPT_EVERY = 1_000

# === F3 Spread Loss ===
USE_SPREAD_LOSS = True
SPREAD_LOSS_WEIGHT = 0.005
SPREAD_LOSS_MARGIN = 2.5
CODEBOOK_COLLAPSE_THRESHOLD = 0.10

# === v97 Radially-Diverse Codebook 参数化 (继承) ===
USE_RADIALLY_DIVERSE = True
RADIAL_MU_WARMUP_START = 0.25
RADIAL_MU_WARMUP_END = 1.0
RADIAL_MU_WARMUP_STEPS = 10_000
RADIAL_SIGMA_RATIO = 0.05
RADIAL_EPS = 1e-5

# === R99 Scale-Decoupling (新) ===
USE_SCALE_DECOUPLED = True
SD_TARGET_NORM = 0.3            # R99 v4: μ = 0.3 (从 1.0 → 0.3)
                                 # 让 x_scaled norm = 0.3 (在 ball 内部), codebook |v|=μ=0.25→1.0 在 ρ=0.13→0.42
                                 # 关键: x_scaled 在内, codebook 在外 → curvature 显著 (v3 时两者都在边界 ρ=0.4 → NN overlap=1.0)
SD_EMA_DECAY = 0.999             # EMA 衰减 (从 0.99 → 0.999, 抗 encoder 早期抖动)
SD_EPS = 1e-6                    # 防 0 除
SD_GAMMA_MAX = 5.0              # v109 BUG_FIX (R36o audit HIGH 置信): v99 γ_max=1024 太松, EMA norm=0.01 → γ=400 (实测 39.50/24.07/18.96) 让 x_scaled 远超 codebook norm, collapse. 修复为 5.0 (sane upper bound).
SD_GAMMA_MIN = 1.0               # γ 至少 1.0 (避免过度耦合)
SD_USE_COMMITMENT = True         # v109 BUG_FIX (R36o audit HIGH 置信): v99 关 commitment loss 让 encoder 缩到 0 失去梯度信号. 修复为 True 提供梯度稳定.
SD_CODEBOOK_LOSS_W = 1.0         # codebook_loss 权重

# === v97 Curvature 固定 ===
C_TRAIN = 0.5

# === HG-Rec 实现 ===
GRAD_CLIP_NORM = 1.0
USE_M3_TRANSPORT = True
USE_M2_INTRINSIC = False          # R99 关 M2 (M3 transport 跨层, 用 raw res)
USE_TCU = False
TCU_ALPHA = 0.05
TCU_ETA = 0.1
USE_MCDQ = False
MCDQ_ALPHA_INIT = 0.5
USE_SCS = False
SCS_EPS_SCALE = 1.0
MARGIN_REG_WEIGHT = 0.0
MARGIN_TARGET = 0.6

BATCH_SIZE = 640
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
COMPILE = False


# =================================================================
# Radially-Diverse Codebook (继承 v97)
# =================================================================
class RadiallyDiverseCodebook(nn.Module):
    def __init__(self, K, D, init_seed=42):
        super().__init__()
        torch.manual_seed(init_seed)
        self.K = K
        self.D = D
        w = torch.randn(K, D)
        w = F.normalize(w, dim=-1, eps=1e-12)
        self._w = nn.Parameter(w.clone())
        self._a = nn.Parameter(torch.randn(K) * 0.1)
        self.register_buffer('_c_fixed', torch.tensor(C_TRAIN))
        self._current_mu = RADIAL_MU_WARMUP_START
        self._current_sigma = RADIAL_SIGMA_RATIO * RADIAL_MU_WARMUP_START

    @property
    def weight(self):
        u = F.normalize(self._w, dim=-1, eps=1e-12)
        q = torch.tanh(self._a)
        q_mean = q.mean()
        q_std = q.std() + RADIAL_EPS
        z = (q - q_mean) / q_std
        r = self._current_mu + self._current_sigma * z
        v = r.unsqueeze(-1) * u
        return v

    def forward(self, ids=None):
        if ids is None:
            return self.weight
        return self.weight[ids]

    def get_c(self):
        return self._c_fixed

    def get_r_stats(self):
        q = torch.tanh(self._a)
        q_std = q.std() + RADIAL_EPS
        z = (q - q.mean()) / q_std
        r = self._current_mu + self._current_sigma * z
        return {
            "mean": float(r.mean().item()),
            "std": float(r.std().item()),
            "min": float(r.min().item()),
            "max": float(r.max().item()),
        }

    def set_radial_schedule(self, mu, sigma):
        self._current_mu = mu
        self._current_sigma = sigma

    @torch.no_grad()
    def kmeans_init_from_centroids(self, centroids: torch.Tensor) -> None:
        """继承 v97: _w = centroid direction, _a = small random."""
        eps = 1e-6
        r = centroids.norm(dim=-1).clamp(min=eps)
        u = centroids / r.unsqueeze(-1)
        self._w.data.copy_(u.to(self._w.device, dtype=self._w.dtype))
        self._a.data.copy_((torch.randn_like(self._a) * 0.1).to(self._a.device, dtype=self._a.dtype))


def compute_radial_mu_sigma(global_step):
    if global_step < RADIAL_MU_WARMUP_STEPS:
        t = global_step / RADIAL_MU_WARMUP_STEPS
        mu = RADIAL_MU_WARMUP_START + (RADIAL_MU_WARMUP_END - RADIAL_MU_WARMUP_START) * t
    else:
        mu = RADIAL_MU_WARMUP_END
    sigma = RADIAL_SIGMA_RATIO * mu
    return mu, sigma


# =================================================================
# R99 Scale-Decoupled Quantize Forward
# =================================================================
def r99_layer_forward(self, x, temperature, prefix_emb=None):
    """Scale-Decoupled Hyperbolic Quantization.

    Args:
        x: (B, D) 输入 residual (reconstruction scale)
    Returns:
        QuantizeOutput(embeddings=ê, ids, loss, margin, spread_loss)
        其中 ê = v[ids]/γ 在 residual scale, cascade subtraction 直接用.
    """
    # === 1. EMA residual norm + γ ===
    cur_norm = x.detach().norm(dim=-1).mean().item()
    if not hasattr(self, '_sd_ema_norm') or self._sd_ema_norm is None:
        self._sd_ema_norm = cur_norm
        self._sd_gamma = SD_TARGET_NORM / (cur_norm + SD_EPS)
    else:
        self._sd_ema_norm = SD_EMA_DECAY * self._sd_ema_norm + (1 - SD_EMA_DECAY) * cur_norm
        self._sd_gamma = SD_TARGET_NORM / (self._sd_ema_norm + SD_EPS)
    gamma = self._sd_gamma
    # === R99 v3: γ clamp 抗死亡螺旋 (encoder 缩到 0 时 EMA 跟到 0 → γ 爆炸) ===
    gamma = float(max(SD_GAMMA_MIN, min(SD_GAMMA_MAX, gamma)))

    # === 2. Scale residual to geometry space ===
    x_scaled = x * gamma  # (B, D) E|x_scaled| ≈ 1

    # === 3. Hyperbolic argmin ===
    codebook = self.out_proj(self.embedding.weight)  # (K, D) = v (tangent)
    c = self.get_c_per_item(prefix_emb)
    if c.dim() == 0:
        c_exp = c.view(1, 1, 1)
    else:
        c_exp = c.view(-1, 1, 1)
    B = x_scaled.shape[0]; K = codebook.shape[0]
    latent_h = _expmap0_t(x_scaled.unsqueeze(1), c_exp).squeeze(1)        # (B, D)
    cb_v_exp = codebook.unsqueeze(0).expand(B, K, -1)                     # (B, K, D) tangent
    codebook_h = _expmap0_t(cb_v_exp, c_exp)                              # (B, K, D) Poincaré
    d_poincare = _poincare_distance_t(
        latent_h.unsqueeze(1).expand(B, K, -1),
        codebook_h,
        c_exp
    ).squeeze(-1)                                                          # (B, K)

    # Hard assignment (STE)
    ids = d_poincare.argmin(dim=-1)                                        # (B,)

    # === 4. Selected codeword in tangent space, scale back ===
    q_tangent = codebook[ids]                                              # (B, D) tangent
    emb_recon = q_tangent / gamma                                          # (B, D) 在 residual scale

    # === 5. Codebook loss (在 residual scale, emb_recon 与 x 同空间) ===
    codebook_loss = F.mse_loss(emb_recon, x.detach())
    if SD_USE_COMMITMENT:
        commitment_loss = F.mse_loss(x, emb_recon.detach())
        loss = SD_CODEBOOK_LOSS_W * codebook_loss + self.quantize_loss.commitment_weight * commitment_loss
    else:
        # R99 v3: drop commitment loss — 在 scale-decoupled 下 commitment_loss 会把 encoder 拉向 0
        # (因为 ê=v/γ 是 residual scale 的小量, gradient 倾向缩小 |x|)
        # decoder recon_loss 已足够驱动 encoder 学习有意义的 representation
        loss = SD_CODEBOOK_LOSS_W * codebook_loss

    # === 6. Margin (在 geometry scale, distance-based) ===
    d1 = d_poincare.gather(1, ids.unsqueeze(-1)).squeeze(-1)
    d_poincare_scattered = d_poincare.scatter(1, ids.unsqueeze(-1), float('inf'))
    d2 = d_poincare_scattered.min(dim=-1).values
    margin = d2 - d1

    # === 7. Spread loss (在 geometry scale, codebook expmapped) ===
    spread_loss = torch.zeros((), device=x.device, dtype=x.dtype)
    if self.use_spread_loss and self.training and not getattr(self, 'hypervq', False):
        c_scalar = float(c.mean().item())
        c_t = torch.tensor(c_scalar, device=codebook.device, dtype=codebook.dtype)
        cb_h_full = _expmap0_t(codebook, c_t)                              # (K, D)
        d_pair = _poincare_distance_t(
            cb_h_full.unsqueeze(0).expand(K, K, -1),
            cb_h_full.unsqueeze(1).expand(K, K, -1),
            c_t
        ).squeeze(-1)
        mask_off = ~torch.eye(K, dtype=torch.bool, device=codebook.device)
        d_off = d_pair[mask_off]
        spread_loss = F.relu(self.spread_loss_margin - d_off).mean()
    self._last_spread_loss = spread_loss.detach()

    return QuantizeOutput(
        embeddings=emb_recon,    # 关键: scale back 让 cascade subtraction 直接用
        ids=ids,
        loss=loss,
        margin=margin,
        spread_loss=spread_loss,
    )


# =================================================================
# DDP / Data
# =================================================================
class EmbeddingDataset(Dataset):
    def __init__(self, emb_path):
        arr = np.load(emb_path).astype(np.float32)
        self.embeddings = torch.from_numpy(arr)
        if dist.get_rank() == 0:
            print(f"[data] loaded {emb_path}: shape={tuple(self.embeddings.shape)}, dtype={self.embeddings.dtype}", flush=True)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def collate_items(batch):
    return torch.stack(batch, dim=0)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def save_ckpt(model, optimizer, global_step, out_dir, tag):
    if dist.get_rank() != 0:
        return
    path = os.path.join(out_dir, f"rqvae_{tag}.pt")
    state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "mu": float(getattr(model.module.layers[0].embedding, '_current_mu', 0.0)),
        "sigma": float(getattr(model.module.layers[0].embedding, '_current_sigma', 0.0)),
    }
    # 保存 EMA state
    ema_state = {}
    for li, layer in enumerate(model.module.layers):
        ema_state[li] = {
            'ema_norm': getattr(layer, '_sd_ema_norm', None),
            'gamma': getattr(layer, '_sd_gamma', None),
        }
    state['ema_state'] = ema_state
    torch.save(state, path)
    print(f"[ckpt] saved {path} (step={global_step}, μ={state['mu']:.4f}, σ={state['sigma']:.4f})", flush=True)


def inject_radially_diverse(model, init_seed=42):
    """v97 inject + R99 scale-decoupled forward override."""
    import init.kmeans as _kmeans_mod
    kmeans_runner = _kmeans_mod.Kmeans

    for li, layer in enumerate(model.layers):
        K, D = layer.embedding.weight.shape
        dev = layer.embedding.weight.device
        if 'weight' in layer.embedding._parameters:
            del layer.embedding._parameters['weight']
        cb = RadiallyDiverseCodebook(K=K, D=D, init_seed=init_seed + li).to(dev)
        layer.embedding = cb
        layer.get_c = types.MethodType(lambda self: self.embedding.get_c(), layer)
        layer.get_c_per_item = types.MethodType(lambda self, prefix_emb=None: self.embedding.get_c(), layer)

        def _v97_kmeans_init(self_layer, x):
            with torch.no_grad():
                km_out = kmeans_runner(k=self_layer.embedding.K).run(x)
                self_layer.embedding.kmeans_init_from_centroids(km_out.centroids)
            self_layer.kmeans_initted = True

        layer._kmeans_init = types.MethodType(_v97_kmeans_init, layer)

        # R99: 替换 forward 为 scale-decoupled 版本
        layer.forward = types.MethodType(r99_layer_forward, layer)

        # 初始化 EMA state
        layer._sd_ema_norm = None
        layer._sd_gamma = 1.0


def set_radial_schedule_all(model, mu, sigma):
    target = model.module if hasattr(model, 'module') else model
    for layer in target.layers:
        layer.embedding.set_radial_schedule(mu, sigma)


def get_radial_mu_sigma_all(model):
    target = model.module if hasattr(model, 'module') else model
    return target.layers[0].embedding._current_mu, target.layers[0].embedding._current_sigma


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(SEED + rank)
    np.random.seed(SEED + rank)

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"=== R99 Scale-Decoupled Hyperbolic Quantization DDP (world_size={world_size}) ===", flush=True)
        print(f"target: MAX_GLOBAL_STEPS={MAX_GLOBAL_STEPS} | ckpt every {CKPT_EVERY}", flush=True)
        print(f"v97 radial: μ: {RADIAL_MU_WARMUP_START}→{RADIAL_MU_WARMUP_END} over {RADIAL_MU_WARMUP_STEPS} steps, σ=0.05·μ", flush=True)
        print(f"R99 scale-decoupled: EMA({SD_EMA_DECAY}) on |r|, γ = {SD_TARGET_NORM}/m_ℓ", flush=True)
        print(f"c_train={C_TRAIN} fixed", flush=True)

    ds = EmbeddingDataset(EMB_NPY)
    n_items = len(ds)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED, drop_last=True)
    train_loader = DataLoader(
        ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=True, collate_fn=collate_items,
    )

    if rank == 0:
        print(f"[data] {n_items} items | {len(train_loader)} batches/rank", flush=True)

    model = RqVae(
        input_dim=INPUT_DIM, embed_dim=EMBED_DIM, hidden_dims=HIDDEN_DIMS,
        codebook_size=CODEBOOK_SIZE, codebook_kmeans_init=True,
        codebook_normalize=False, codebook_sim_vq=False,
        codebook_mode=QuantizeForwardMode.STE, n_layers=N_LAYERS, n_cat_features=0,
        commitment_weight=COMMITMENT_WEIGHT,
        gate_M2_intrinsic=USE_M2_INTRINSIC, gate_M3_transport=USE_M3_TRANSPORT,
        hyperbolic_distance=True, sk_eps=0.0, prefix_router_layers=None,
        margin_reg_weight=MARGIN_REG_WEIGHT, margin_target=MARGIN_TARGET,
        spread_loss_weight=SPREAD_LOSS_WEIGHT if USE_SPREAD_LOSS else 0.0,
        spread_loss_margin=SPREAD_LOSS_MARGIN,
        use_tcu=False, tcu_alpha=TCU_ALPHA, tcu_eta=TCU_ETA,
        use_mcdq=False, mcdq_alpha_init=MCDQ_ALPHA_INIT,
        use_scs=False, scs_eps_scale=SCS_EPS_SCALE,
        use_fixed_curvature=True, c_fixed=C_TRAIN,
        use_curriculum_curvature=False,
        c_start=0.05, c_end=0.7, curriculum_steps=50_000,
    ).to(device)

    if USE_RADIALLY_DIVERSE:
        inject_radially_diverse(model)
        if rank == 0:
            print(f"[R99] v97 RadiallyDiverseCodebook + Scale-Decoupled forward injected", flush=True)

    if COMPILE:
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    global_step = 0
    global_step_sync = 0
    t_start = time.time()
    last_log_t = t_start

    if rank == 0:
        print(f"[train] start, target={MAX_GLOBAL_STEPS} global steps", flush=True)
    model.train()
    done = False
    while not done:
        sampler.set_epoch(global_step)
        for batch in train_loader:
            x = batch.to(device, non_blocking=True)
            bsz = x.shape[0]
            seq_batch = SeqBatch(
                user_ids=torch.zeros(bsz, dtype=torch.long, device=device),
                ids=torch.arange(bsz, dtype=torch.long, device=device),
                ids_fut=torch.zeros(bsz, dtype=torch.long, device=device),
                x=x, x_fut=x,
                seq_mask=torch.ones(bsz, dtype=torch.long, device=device),
            )
            mu, sigma = compute_radial_mu_sigma(global_step_sync)
            set_radial_schedule_all(model, mu, sigma)

            optimizer.zero_grad()
            out = model(seq_batch, gumbel_t=0.2)
            loss = out.loss
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            global_step += 1

            step_tensor = torch.tensor([global_step], dtype=torch.long, device=device)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            global_step_sync = int(step_tensor.item())

            now = time.time()
            if rank == 0 and (now - last_log_t >= 5.0):
                elapsed = now - t_start
                ips_global = global_step_sync / elapsed
                ips_per_gpu = (global_step_sync / world_size) / elapsed
                eta_s = (MAX_GLOBAL_STEPS - global_step_sync) / ips_global if ips_global > 0 else float("inf")
                usage = [int(u) for u in out.per_layer_usage.tolist()]
                usage_str = "/".join(str(u) for u in usage)
                low_usage = [li for li, u in enumerate(usage) if u < CODEBOOK_SIZE * CODEBOOK_COLLAPSE_THRESHOLD]
                curvs = [float(l.get_c().item()) for l in model.module.layers]
                curv_str = "/".join(f"{c:.3f}" for c in curvs)
                # R99: γ per layer + EMA norm
                gammas = []
                for l in model.module.layers:
                    g = getattr(l, '_sd_gamma', 1.0)
                    gammas.append(float(g))
                gamma_str = "/".join(f"{g:.2f}" for g in gammas)
                ema_norms = []
                for l in model.module.layers:
                    e = getattr(l, '_sd_ema_norm', None)
                    ema_norms.append(float(e) if e is not None else 0.0)
                ema_str = "/".join(f"{e:.3f}" for e in ema_norms)
                # radial
                r_stats = [l.embedding.get_r_stats() for l in model.module.layers]
                r_str = "/".join(f"[μ{s['mean']:.3f},σ{s['std']:.3f}]" for s in r_stats)
                # margin
                margs = out.per_layer_margin.tolist()
                marg_str = "/".join(f"{m:.4f}" for m in margs)
                print(
                    f"  ep {global_step_sync // 10:5d}/{NUM_EPOCHS} | step {global_step_sync:6d}/{MAX_GLOBAL_STEPS} "
                    f"| loss={float(loss.item()):.4f} "
                    f"| rl={float(out.reconstruction_loss.item()):.4f} "
                    f"| vl={float(out.rqvae_loss.item()):.4f} "
                    f"| codes={usage_str}/{CODEBOOK_SIZE} "
                    f"| c={curv_str} "
                    f"| γ={gamma_str} "
                    f"| |r|={ema_str} "
                    f"| r={r_str} "
                    f"| marg={marg_str} "
                    f"| ips_g={ips_global:.1f} "
                    f"| elapsed={elapsed:.1f}s | eta={eta_s:.1f}s",
                    flush=True,
                )
                if low_usage:
                    print(f"  [CODEBOOK WARNING] layer {low_usage} usage={usage_str}/{CODEBOOK_SIZE} (< 10%)", flush=True)
                last_log_t = now

            if global_step_sync >= MAX_GLOBAL_STEPS or (
                global_step_sync > 0 and global_step_sync % CKPT_EVERY == 0
            ):
                tag = f"step{global_step_sync}" if global_step_sync < MAX_GLOBAL_STEPS else "final"
                save_ckpt(model, optimizer, global_step_sync, OUT_DIR, tag)
                dist.barrier()

            if global_step_sync >= MAX_GLOBAL_STEPS:
                done = True
                break

    if rank == 0:
        print(f"[train] done at global_step={global_step_sync}, total time={(time.time()-t_start):.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()