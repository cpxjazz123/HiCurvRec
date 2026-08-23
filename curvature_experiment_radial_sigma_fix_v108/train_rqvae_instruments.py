"""v97 Stage-1 RQ-VAE training with Radially-Diverse Curvature-Identifiable Codebook.

用户设计 v97 (数学根因 v96 修复版):
  a_ℓk = raw learnable scalar           (K,)
  q_ℓk = tanh(a_ℓk)                     (K,), 限 [-1, 1]
  z_ℓk = (q_ℓk - mean_k(q_ℓk)) / (std_k(q_ℓk) + ε)   (K,), 每层内标准化 mean=0, std=1
  r_ℓk = μ + σ · z_ℓk                   (K,),  结构保证 std(r)=σ>0 (不能 collapse)
  v_ℓk = r_ℓk · u_ℓk / |u_ℓk|           (K, D), |v|=r
  e_ℓk = exp_0^{c_ℓ}(v_ℓk)              (K, D), Poincaré ball

μ warmup: 0.25 → 1.0 over 10k steps (linear), σ(t) = 0.05 · μ(t).
10k 后: μ=1.0, σ=0.05 固定 (不 learnable, 避免 c ↔ (μ,σ) 互相补偿).
c_0=c_1=c_2=0.5 固定前 10k.
10k ckpt 后做 post-hoc c sweep {0.1, 0.3, 0.5, 0.7, 1.0, 1.5}.

关键: 与 v96 最大区别是结构保证 std(r)=σ>0, optimizer 不能把所有 codeword 压到同一球壳.
不 learnable μ,σ (避免与 c_ℓ 形成 trivial 补偿); 不加 D loss / relational loss / forced 异质 c.

启动 (DDP 4 卡, R51+ 确定性):
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=":4096:8" \
  /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun --nproc_per_node=4 --master_port=29525 \
      /home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment_radially_diverse_v97/train_rqvae_instruments.py
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
from modules.quantize import QuantizeForwardMode
from modules.hyperbolic import _expmap0_t
from data.schemas import SeqBatch

from curvature_config import ITEM_EMB_NPY as EMB_NPY, RQVAE_OUT_DIR as OUT_DIR


# === R51+ 确定性约束 (硬编码, 6 项) ===
SEED = 42
INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
EMBED_DIM = 32
CODEBOOK_SIZE = 256
N_LAYERS = 3
COMMITMENT_WEIGHT = 1.0

# === 训练硬约束 (R30/R43) ===
NUM_EPOCHS = 10000
MAX_GLOBAL_STEPS = 10_000        # v97: 10k steps (用户要求 10k 后 sweep)
CKPT_EVERY = 1_000               # v97: 每 1k 保存 (10 ckpts)

# === F3 v87 baseline: Spread Loss ===
USE_SPREAD_LOSS = True
SPREAD_LOSS_WEIGHT = 0.005
SPREAD_LOSS_MARGIN = 2.5
CODEBOOK_COLLAPSE_THRESHOLD = 0.10

# === v97 Radially-Diverse Codebook 参数化 ===
USE_RADIALLY_DIVERSE = True
RADIAL_MU_WARMUP_START = 0.25
RADIAL_MU_WARMUP_END = 1.0
RADIAL_MU_WARMUP_STEPS = 10_000
RADIAL_SIGMA_RATIO = 0.30        # v108 BUG_FIX: v97 用 0.05 太小, L1/L2 collapse 到 1 code (R36o audit HIGH 置信 BUG_FOUND); 修复为 0.30 (6x 多样性) 让 σ(t) = 0.30 · μ(t)
RADIAL_EPS = 1e-5

# === v97 Curvature 控制 (前 10k 固定 0.5, 10k 后 post-hoc sweep) ===
C_TRAIN = 0.5                    # 训练期 c_0=c_1=c_2=0.5 fixed

# === θ 曲率学习检查 ===
CURV_INIT_C = C_TRAIN
CURV_LEARNED_TOL = 0.01

# === HG-Rec 实现: gradient clipping + M3 transport ===
GRAD_CLIP_NORM = 1.0
USE_M3_TRANSPORT = True
USE_M2_INTRINSIC = False
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
# Radially-Diverse Curvature-Identifiable Codebook (v97)
# =================================================================
class RadiallyDiverseCodebook(nn.Module):
    """参数化:
        a_ℓk = raw learnable scalar           (K,)
        q_ℓk = tanh(a_ℓk)                     (K,), 限 [-1, 1]
        z_ℓk = (q_ℓk - mean_k(q_ℓk)) / (std_k(q_ℓk) + ε)   (K,), mean=0, std=1
        r_ℓk = μ + σ · z_ℓk                   (K,),  std(r)=σ>0 (结构保证)
        v_ℓk = r_ℓk · u_ℓk / |u_ℓk|           (K, D), |v|=r
        e_ℓk = exp_0^{c_ℓ}(v_ℓk)              (K, D), Poincaré ball

    c_ℓ 在训练期固定为 C_TRAIN=0.5, 10k 后 post-hoc sweep.

    关键: z_ℓk 每层内部标准化, 保证 std(r)=σ>0, optimizer 不能 collapse 成单球壳.
    μ, σ 不 learnable, 由外部 schedule 控制.
    """

    def __init__(self, K, D, init_seed=42):
        super().__init__()
        torch.manual_seed(init_seed)
        self.K = K
        self.D = D

        # raw direction u — random init on sphere
        w = torch.randn(K, D)
        w = F.normalize(w, dim=-1, eps=1e-12)
        self._w = nn.Parameter(w.clone())

        # raw radial a — 初始化为 N(0, 0.1), 让 q=tanh(a)≈a (小 a) → z_ℓk ≈ a / std(a) → z 接近正态
        self._a = nn.Parameter(torch.randn(K) * 0.1)

        # 固定 c_param (训练期固定 c=C_TRAIN=0.5, 不学习)
        self.register_buffer('_c_fixed', torch.tensor(C_TRAIN))

        # μ, σ 由外部 schedule 控制 (前向时调用)
        self._current_mu = RADIAL_MU_WARMUP_START
        self._current_sigma = RADIAL_SIGMA_RATIO * RADIAL_MU_WARMUP_START

    @property
    def weight(self):
        """返回 tangent 空间向量 v = r·u (K, D), 与 baseline 接口对齐
        让上层 hyperbolic_distance 路径自然调用 _expmap0_t(v, c) 推到 Poincaré 球.
        这样 spread_loss 路径 (_expmap0_t(codebook, c_t) → cb_h) 不会双重 expmap.
        """
        u = F.normalize(self._w, dim=-1, eps=1e-12)
        q = torch.tanh(self._a)                               # (K,), in [-1, 1]
        q_mean = q.mean()
        q_std = q.std() + RADIAL_EPS
        z = (q - q_mean) / q_std                              # (K,), mean=0, std=1
        r = self._current_mu + self._current_sigma * z       # (K,)
        v = r.unsqueeze(-1) * u                                # (K, D)
        return v

    def forward(self, ids=None):
        if ids is None:
            return self.weight
        return self.weight[ids]

    def get_c(self):
        return self._c_fixed

    def get_r_stats(self):
        """返回当前 r 的 mean/std/min/max (诊断用)."""
        q = torch.tanh(self._a)
        q_mean = q.mean()
        q_std = q.std() + RADIAL_EPS
        z = (q - q_mean) / q_std
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
        """把 KMeans centroids (K, D) 拆解为 _w (direction) + _a (radial):
          u_k = centroid_k / ||centroid_k||          # 方向: KMeans 给出, 标准化到单位球
          _w.data := u_k
          _a.data := randn(K) * 0.1                  # 径向: 重置为小随机, 让 |v|_k ≈ μ + σ·small

        设计取舍:
          KMeans centroids 的 ||centroid_k|| 通常 ≈ 输入 x 的 norm 量级 (e.g. ~5),
          而我们的 |v|_k 范围被 [μ-σ, μ+σ] = [0.2375, 0.2625] 限制 (μ=0.25, σ=0.0125).
          把 r 直接编码进 _a 会全部 clamp 到 ±0.999, 等价于把 |v| 锁死在 μ+0.999σ 单球壳.
          所以方向用 KMeans (informed), 径向重置为小随机 (让 μ schedule 自然施加).
        """
        eps = 1e-6
        r = centroids.norm(dim=-1).clamp(min=eps)               # (K,)
        u = centroids / r.unsqueeze(-1)                          # (K, D) 单位向量
        self._w.data.copy_(u.to(self._w.device, dtype=self._w.dtype))
        self._a.data.copy_((torch.randn_like(self._a) * 0.1).to(self._a.device, dtype=self._a.dtype))


def compute_radial_mu_sigma(global_step):
    """Warmup schedule: μ(t): 0.25 → 1.0 over 10k steps, σ(t) = 0.05 · μ(t)."""
    if global_step < RADIAL_MU_WARMUP_STEPS:
        t = global_step / RADIAL_MU_WARMUP_STEPS
        mu = RADIAL_MU_WARMUP_START + (RADIAL_MU_WARMUP_END - RADIAL_MU_WARMUP_START) * t
    else:
        mu = RADIAL_MU_WARMUP_END
    sigma = RADIAL_SIGMA_RATIO * mu
    return mu, sigma


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
    mu, sigma = get_radial_mu_sigma_all(model)
    state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "mu": float(mu),
        "sigma": float(sigma),
    }
    torch.save(state, path)
    print(f"[ckpt] saved {path} (step={global_step}, μ={state['mu']:.4f}, σ={state['sigma']:.4f})", flush=True)


def inject_radially_diverse(model, init_seed=42):
    """替换 layer.embedding 为 RadiallyDiverseCodebook, 同时接管 _kmeans_init
    让 KMeans centroids 拆解写入 _w (direction) + _a (radial), 而非写到 property 临时 tensor.
    """
    for li, layer in enumerate(model.layers):
        K, D = layer.embedding.weight.shape
        dev = layer.embedding.weight.device
        if 'weight' in layer.embedding._parameters:
            del layer.embedding._parameters['weight']
        cb = RadiallyDiverseCodebook(K=K, D=D, init_seed=init_seed + li).to(dev)
        layer.embedding = cb
        layer.get_c = types.MethodType(lambda self: self.embedding.get_c(), layer)
        layer.get_c_per_item = types.MethodType(lambda self, prefix_emb=None: self.embedding.get_c(), layer)

        # 接管 _kmeans_init: 用 KMeans 出 centroids 后, 走我们自己的拆解方法
        import init.kmeans as _kmeans_mod
        kmeans_runner = _kmeans_mod.Kmeans

        def _v97_kmeans_init(self_layer, x):
            with torch.no_grad():
                K_local = self_layer.embedding.K
                km_out = kmeans_runner(k=K_local).run(x)
                self_layer.embedding.kmeans_init_from_centroids(km_out.centroids)
            self_layer.kmeans_initted = True

        layer._kmeans_init = types.MethodType(_v97_kmeans_init, layer)


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
        print(f"=== v97 RQ-VAE Musical_Instruments DDP (world_size={world_size}) ===", flush=True)
        print(f"target: MAX_GLOBAL_STEPS={MAX_GLOBAL_STEPS} | ckpt every {CKPT_EVERY}", flush=True)
        print(f"v97 radial: μ: {RADIAL_MU_WARMUP_START}→{RADIAL_MU_WARMUP_END} over {RADIAL_MU_WARMUP_STEPS} steps, σ=0.05·μ", flush=True)
        print(f"v97 c_train={C_TRAIN} fixed (10k 后 post-hoc sweep {{0.1, 0.3, 0.5, 0.7, 1.0, 1.5}})", flush=True)

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
        use_fixed_curvature=True, c_fixed=C_TRAIN,   # 训练期固定 c=0.5
        use_curriculum_curvature=False,
        c_start=0.05, c_end=0.7, curriculum_steps=50_000,
    ).to(device)

    # === v97: 注入 RadiallyDiverseCodebook ===
    if USE_RADIALLY_DIVERSE:
        inject_radially_diverse(model)
        if rank == 0:
            print(f"[v97] RadiallyDiverseCodebook injected (c fixed=0.5, μ=0.25 init)", flush=True)

    if COMPILE:
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    global_step = 0
    global_step_sync = 0  # 同步后的全局 step (用于 radial schedule 计算)
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
            # v97: 用同步后的 GLOBAL step 算 radial schedule (R51+ 同步一致)
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
                r_stats = [l.embedding.get_r_stats() for l in model.module.layers]
                r_str = "/".join(f"[μ{s['mean']:.3f},σ{s['std']:.3f}]" for s in r_stats)
                rho_means = []
                for l in model.module.layers:
                    e = l.embedding.weight.detach()
                    c = l.get_c().detach()
                    rho = (c ** 0.5) * e.norm(dim=-1)
                    rho_means.append(float(rho.mean().item()))
                rho_str = "/".join(f"{r:.3f}" for r in rho_means)
                margs = out.per_layer_margin.tolist()
                marg_str = "/".join(f"{m:.4f}" for m in margs)
                print(
                    f"  ep {global_step_sync // 10:5d}/{NUM_EPOCHS} | step {global_step_sync:6d}/{MAX_GLOBAL_STEPS} "
                    f"| loss={float(loss.item()):.4f} "
                    f"| rl={float(out.reconstruction_loss.item()):.4f} "
                    f"| vl={float(out.rqvae_loss.item()):.4f} "
                    f"| codes={usage_str}/{CODEBOOK_SIZE} "
                    f"| c={curv_str} "
                    f"| μ={mu:.4f}/σ={sigma:.4f} "
                    f"| r={r_str} "
                    f"| ρ={rho_str} "
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