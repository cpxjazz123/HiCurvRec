"""
R153: v133 checkpoint curvature viability diagnostic
在可复现的 v133 baseline 上，验证 true-Poincaré curvature sweep 是否能动 SID assignment。

4 项诊断:
  1. 三层 normalized radius: ρ_{ℓk} = sqrt(c) * ||e_{ℓk}||  (c=0.6, C_END)
  2. Curvature sweep c ∈ {0.1, 0.3, 0.6, 1.0, 1.5}: 用 true Poincaré distance 重做 SID
  3. 每层 SID change rate + pairwise-distance Spearman
  4. Code usage / perplexity

Poincaré distance: d_c(x,y) = (1/√c) * arccosh(1 + 2c·||x-y||² / ((1-c||x||²)(1-c||y||²)))
Mahalanobis distance: d²(x,y) = Σ_d (x_d - y_d)² / var_d
"""
import sys, os, json, math, random
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment_crvq_mahalanobis_c_end_06_v133")
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode, QuantizeDistance
from data.schemas import SeqBatch

BASE = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment_crvq_mahalanobis_c_end_06_v133"
CKPT_PATH = f"{BASE}/out/rqvae/instruments/rqvae_final.pt"
EMB_NPY = f"{BASE}/dataset/Instruments/item_emb.npy"
OUT_JSON = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/r153_diagnostic.json"

C_END = 0.6          # v133 curriculum final curvature
C_SWEEP = [0.1, 0.3, 0.6, 1.0, 1.5]

# === Poincaré distance (item-wise) ===
def poincare_dist(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """
    x: [N, D] embeddings
    y: [K, D] codebook entries
    Returns: [N, K] Poincaré distances
    """
    x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)          # [N, 1]
    y_norm_sq = (y ** 2).sum(dim=1, keepdim=True)          # [K, 1]
    diff_sq = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=2)  # [N, K]

    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq.T)    # [N, K]
    denom = denom.clamp(min=1e-8)
    z = 1 + 2 * c * diff_sq / denom
    z = z.clamp(min=1.0 + 1e-8)
    return (1 / math.sqrt(c)) * torch.acosh(z)

def mahalanobis_dist(x: torch.Tensor, codebook: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    x: [N, D], codebook: [K, D], log_var: [K, D]
    Returns: [N, K] squared Mahalanobis distances
    """
    var = log_var.exp()   # [K, D]
    diff = x.unsqueeze(1) - codebook.unsqueeze(0)           # [N, K, D]
    weighted = diff ** 2 / var.unsqueeze(0)                  # [N, K, D]
    return weighted.sum(dim=2)                               # [N, K]

def assign_by_dist(dist: torch.Tensor) -> torch.Tensor:
    return dist.argmin(dim=1)  # [N]

def code_usage(sids: torch.Tensor) -> dict:
    unique, counts = torch.unique(sids, return_counts=True)
    coverage = len(unique)
    probs = counts.float() / counts.sum()
    perplexity = torch.exp(-(probs * torch.log(probs.clamp(min=1e-10))).sum()).item()
    return {"coverage": coverage, "perplexity": perplexity}

def sid_change_rate(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))

def spearman_flat(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman between two flat distance vectors (upper triangle or full)."""
    # Use full flatten
    r, _ = spearmanr(a.flatten(), b.flatten())
    return float(r) if not np.isnan(r) else 0.0

# === Load checkpoint ===
print("Loading v133 checkpoint...", flush=True)
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state = ckpt["model"]
global_step = ckpt.get("global_step", "N/A")
print(f"  global_step={global_step}", flush=True)

# === Load embeddings (768-dim) ===
emb_768 = torch.from_numpy(np.load(EMB_NPY).astype(np.float32))  # [N, 768]
N, D768 = emb_768.shape
print(f"  embeddings: {N} items, raw_dim={D768}", flush=True)

# === Build model and load weights ===
print("Building model...", flush=True)
model = RqVae(
    input_dim=768, embed_dim=32, hidden_dims=[512, 256, 128],
    codebook_size=256, n_layers=3,
    codebook_kmeans_init=True, codebook_normalize=False,
    codebook_sim_vq=False, codebook_mode=QuantizeForwardMode.STE,
    n_cat_features=0, commitment_weight=1.0,
    gate_M2_intrinsic=False, gate_M3_transport=True,
    hyperbolic_distance=False, sk_eps=0.0,
    distance_mode=QuantizeDistance.MAHALANOBIS,
    rbf_bandwidth=1.0, mahalanobis_init_var=1.0,
    mahalanobis_commit_weight=0.05,
    prefix_router_layers=None, margin_reg_weight=0.0,
    margin_target=0.0, spread_loss_weight=0.0,
    spread_loss_margin=2.5,
    use_tcu=False, tcu_alpha=0.05, tcu_eta=0.1,
    use_mcdq=False, mcdq_alpha_init=0.5,
    use_scs=False, scs_eps_scale=1.0,
    use_fixed_curvature=True, c_fixed=C_END,
    use_curriculum_curvature=True,
    c_start=0.05, c_end=C_END, curriculum_steps=50000,
)
model.load_state_dict(state, strict=False)
model.eval()

# Set model to have c=C_END (curriculum finished)
model.set_curriculum_step(50000)

# === Encode 768-dim → 32-dim latent ===
print("Encoding embeddings to latent space...", flush=True)
with torch.no_grad():
    seq_batch = SeqBatch(
        user_ids=torch.zeros(N, dtype=torch.long),
        ids=torch.arange(N, dtype=torch.long),
        ids_fut=torch.zeros(N, dtype=torch.long),
        x=emb_768,
        x_fut=emb_768,
        seq_mask=torch.ones(N, dtype=torch.long),
    )
    # Get encoder output (before quantization)
    z = model.encoder(seq_batch.x)   # [N, 32]
print(f"  latent shape: {z.shape}", flush=True)

# === Extract per-layer codebooks + log_vars ===
codebooks = []
log_vars = []
for li in range(3):
    cb = state[f"layers.{li}.embedding.weight"].float()   # [256, 32]
    lv = state[f"layers.{li}.log_var"].float()            # [256]
    codebooks.append(cb)
    log_vars.append(lv)
    print(f"  L{li}: codebook {cb.shape}, log_var {lv.shape}", flush=True)

# === DIAG 1: Normalized radius ===
print("\n=== DIAG 1: Normalized radius (c=0.6) ===", flush=True)
diag1 = {}
for li in range(3):
    cb = codebooks[li]
    rho = math.sqrt(C_END) * cb.norm(dim=1)   # [256]
    diag1[f"L{li}"] = {
        "rho_mean": round(float(rho.mean()), 6),
        "rho_std": round(float(rho.std()), 6),
        "rho_min": round(float(rho.min()), 6),
        "rho_max": round(float(rho.max()), 6),
        "n_outside_ball": int((rho >= 1.0 / math.sqrt(C_END)).sum()),
    }
    print(f"  L{li}: ρ={rho.mean():.4f}±{rho.std():.4f} "
          f"[{rho.min():.4f}, {rho.max():.4f}], "
          f"outside_ball={(rho >= 1/math.sqrt(C_END)).sum()}")

# === DIAG 2: Curvature sweep ===
print("\n=== DIAG 2: Curvature sweep → code assignment ===", flush=True)

# Reference: Mahalanobis assignment (what v133 training used)
ref_maha_sids = {}   # li → [N]
ref_poincare_sids = {}  # li → [N]

for li in range(3):
    dist_m = mahalanobis_dist(z, codebooks[li], log_vars[li])
    sids_m = assign_by_dist(dist_m)
    ref_maha_sids[li] = sids_m.numpy()
    usage = code_usage(sids_m)
    print(f"  REF Mahalanobis L{li}: cov={usage['coverage']}, perp={usage['perplexity']:.2f}", flush=True)

    # Also Poincaré at c=C_END for reference
    dist_p = poincare_dist(z, codebooks[li], C_END)
    sids_p = assign_by_dist(dist_p)
    ref_poincare_sids[li] = sids_p.numpy()
    usage_p = code_usage(sids_p)
    print(f"  REF Poincaré c={C_END} L{li}: cov={usage_p['coverage']}, perp={usage_p['perplexity']:.2f}", flush=True)

# Sweep c values
sweep_results = {}
for c in C_SWEEP:
    sweep_results[c] = {}
    print(f"\n  --- c = {c} ---", flush=True)

    for li in range(3):
        cb = codebooks[li]
        lv = log_vars[li]

        # Poincaré assignment at sweep c
        dist_p = poincare_dist(z, cb, c)
        sids_p = assign_by_dist(dist_p).numpy()

        # Mahalanobis (constant across c)
        dist_m = mahalanobis_dist(z, cb, lv)
        sids_m = assign_by_dist(dist_m).numpy()

        # Reference at c=C_END
        dist_p_ref = poincare_dist(z, cb, C_END)
        sids_p_ref = assign_by_dist(dist_p_ref).numpy()

        usage_p = code_usage(torch.from_numpy(sids_p))
        change_vs_maha = sid_change_rate(ref_maha_sids[li], sids_p)
        change_vs_poincare_ref = sid_change_rate(ref_poincare_sids[li], sids_p)

        # Spearman: compare Mahalanobis vs Poincaré(c) distance rankings
        # Sample 300 items for speed
        N_s = min(300, N)
        idx = np.random.choice(N, N_s, replace=False)
        d_m_flat = dist_m[idx].cpu().numpy().flatten()      # [N_s*256]
        d_p_flat = dist_p[idx].cpu().numpy().flatten()
        spearman_mp = spearman_flat(d_m_flat, d_p_flat)

        # Also Poincaré(c) vs Poincaré(c=0.6)
        d_p_ref_flat = dist_p_ref[idx].cpu().numpy().flatten()
        spearman_pp = spearman_flat(d_p_ref_flat, d_p_flat)

        sweep_results[c][f"L{li}"] = {
            "coverage": usage_p["coverage"],
            "perplexity": round(usage_p["perplexity"], 3),
            "change_vs_maha_pct": round(change_vs_maha * 100, 2),
            "change_vs_poincare_c06_pct": round(change_vs_poincare_ref * 100, 2),
            "spearman_maha_vs_poincare_c": round(spearman_mp, 4),
            "spearman_poincare06_vs_poincare_c": round(spearman_pp, 4),
        }

        print(f"    L{li}: cov={usage_p['coverage']}, perp={usage_p['perplexity']:.1f}, "
              f"ΔMahalanobis={change_vs_maha*100:.1f}%, "
              f"ΔPoincaré_c06={change_vs_poincare_ref*100:.1f}%, "
              f"ρ_Maha_vs_Pc={spearman_mp:.4f}, "
              f"ρ_P06_vs_Pc={spearman_pp:.4f}", flush=True)

# === DIAG 3: Sensitivity summary ===
print("\n=== DIAG 3: Per-layer sensitivity ===", flush=True)
sensitivity = {}
for li in range(3):
    changes = [sweep_results[c][f"L{li}"]["change_vs_maha_pct"] for c in C_SWEEP]
    sensitivity[f"L{li}"] = {
        "min_change_pct": round(min(changes), 2),
        "max_change_pct": round(max(changes), 2),
        "range_pct": round(max(changes) - min(changes), 2),
        "mean_change_pct": round(float(np.mean(changes)), 2),
    }
    print(f"  L{li}: change range=[{min(changes):.1f}%, {max(changes):.1f}%], "
          f"mean={np.mean(changes):.2f}%", flush=True)

# === DIAG 4: Ball constraint check ===
print("\n=== DIAG 4: Ball constraint safety ===", flush=True)
diag4 = {}
for c in C_SWEEP:
    for li in range(3):
        cb = codebooks[li]
        max_rho = math.sqrt(c) * cb.norm(dim=1).max()
        diag4[f"c{c}_L{li}"] = {
            "max_rho": round(float(max_rho), 6),
            "ball_radius": round(1.0 / math.sqrt(c), 6),
            "safe": bool(max_rho < 1.0 / math.sqrt(c)),
        }
        if li == 0:
            print(f"  c={c}: L0 max_ρ={max_rho:.4f}, ball_r={1/math.sqrt(c):.4f}, safe={max_rho < 1/math.sqrt(c)}", flush=True)

# === Save results ===
output = {
    "checkpoint": CKPT_PATH,
    "global_step": int(global_step),
    "c_end": C_END,
    "c_sweep": C_SWEEP,
    "n_items": N,
    "latent_dim": 32,
    "diag1_normalized_radius": diag1,
    "diag2_sweep": {str(c): sweep_results[c] for c in C_SWEEP},
    "diag3_sensitivity": sensitivity,
    "diag4_ball_safety": diag4,
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {OUT_JSON}", flush=True)
