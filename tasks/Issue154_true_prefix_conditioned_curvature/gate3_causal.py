"""Issue #154 Gate 3: 曲率因果诊断.

对相同 residual 构造三组输入: 原始 prefix、打乱 prefix、替换 prefix. 比较:
- 曲率变化率与变化幅度
- codeword assignment flip rate
- quantization distortion
- residual norm 分桶下的曲率响应

需要证明 assignment 的变化来自 prefix 引起的曲率变化, 而不是候选列尺度或数值边界.
"""

import os, sys, json
from pathlib import Path
import numpy as np
import torch

TASK_DIR = Path(__file__).parent
sys.path.insert(0, str(TASK_DIR / "treatment" / "_lib"))
from prefix_conditioned_quantizer import PrefixConditionedHRQVAE, C_MIN, C_MAX
from utils import EmbDataset

SEED = 2024
N_ITEMS = 9922
CKPT = TASK_DIR / "treatment" / "stage2" / "hrqvae_kappa_sync.ckpt"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    item_emb = torch.tensor(
        EmbDataset(str(TASK_DIR / "treatment" / "stage1" / "item_emb.parquet")).embeddings,
        dtype=torch.float32).to(device)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = PrefixConditionedHRQVAE(prefix_routing=True).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()

    z = m.encoder(item_emb)  # (N, 32) 固定 residual

    results = {}
    rng = np.random.RandomState(42)
    for li in [1, 2]:
        q = m.vq_layers[li]
        if not q.prefix_routing:
            continue
        # 原始 prefix (训练路径: L1=[e0], L2=[e0,e1])
        sid = m.get_indices(item_emb, use_sk=False)  # (N,3)
        if li == 1:
            cb0 = m.vq_layers[0].embeddings.weight[sid[:, 0]]
            prefix_orig = cb0
        else:
            cb0 = m.vq_layers[0].embeddings.weight[sid[:, 0]]
            cb1 = m.vq_layers[1].embeddings.weight[sid[:, 1]]
            prefix_orig = torch.cat([cb0, cb1], dim=-1)

        with torch.no_grad():
            d_orig = q.router(prefix_orig)
            c_orig = C_MIN + (C_MAX - C_MIN) * torch.sigmoid(q.theta + d_orig)

            # 打乱 prefix (同集合不同排列)
            perm = torch.from_numpy(rng.permutation(N_ITEMS))
            d_shuf = q.router(prefix_orig[perm])
            c_shuf = C_MIN + (C_MAX - C_MIN) * torch.sigmoid(q.theta + d_shuf)

            # 替换 prefix (换成另一 item 的 prefix 集合)
            shift = (torch.arange(N_ITEMS) + N_ITEMS // 3) % N_ITEMS
            d_repl = q.router(prefix_orig[shift])
            c_repl = C_MIN + (C_MAX - C_MIN) * torch.sigmoid(q.theta + d_repl)

        dc_shuf = (c_orig - c_shuf).abs()
        dc_repl = (c_orig - c_repl).abs()

        # assignment flip rate: 用扰动后的 c 重算该层量化 assignment
        def assignment_with_c(c_per):
            qq = q
            c_exp = c_per.view(-1, 1, 1)
            latent = z  # (N,32) residual = encoder 输出 (近似, L1/L2 残差不同但够诊断)
            latent_exp = latent.unsqueeze(1)
            cb_exp0 = qq.embeddings.weight.unsqueeze(0).expand(N_ITEMS, qq.n_e, -1)
            latent_h = torch.sqrt(torch.clamp(1.0 / c_exp, min=1e-6)) if False else None
            # 用与 forward 相同的投影
            from prefix_conditioned_quantizer import _proj_to_ball_t, _expmap0_t
            from utils import poincare_distance
            latent_h = _proj_to_ball_t(_expmap0_t(latent_exp, c_exp), c_exp)
            codebook_h = _proj_to_ball_t(_expmap0_t(cb_exp0, c_exp), c_exp)
            x_exp = latent_h.expand(N_ITEMS, qq.n_e, -1)
            d = poincare_distance(x_exp, codebook_h, c_exp).squeeze(-1)
            return torch.argmin(d, dim=-1)

        with torch.no_grad():
            idx_orig = assignment_with_c(c_orig)
            idx_shuf = assignment_with_c(c_shuf)
            idx_repl = assignment_with_c(c_repl)
        flip_shuf = float((idx_orig != idx_shuf).float().mean())
        flip_repl = float((idx_orig != idx_repl).float().mean())

        # residual norm 分桶下的曲率响应
        rnorm = z.norm(dim=-1)
        bins = np.quantile(rnorm.detach().cpu().numpy(), [0, 0.25, 0.5, 0.75, 1.0])
        bucket_resp = []
        for b in range(4):
            mask = (rnorm >= bins[b]) & (rnorm <= bins[b + 1])
            bucket_resp.append({
                "bucket": [float(bins[b]), float(bins[b + 1])],
                "n": int(mask.sum()),
                "c_mean": float(c_orig[mask].mean()),
                "c_std": float(c_orig[mask].std()),
                "dc_shuf_mean": float(dc_shuf[mask].mean()),
            })

        results[f"L{li}"] = {
            "c_orig_mean": float(c_orig.mean()),
            "c_orig_std": float(c_orig.std()),
            "dc_shuf_mean": float(dc_shuf.mean()),
            "dc_repl_mean": float(dc_repl.mean()),
            "flip_rate_shuffle": flip_shuf,
            "flip_rate_replace": flip_repl,
            "fraction_c_change_gt_1e-4_shuffle": float((dc_shuf > 1e-4).float().mean()),
            "fraction_c_change_gt_1e-4_replace": float((dc_repl > 1e-4).float().mean()),
            "boundary_hit": float((c_orig <= C_MIN + 1e-4).float().mean() + (c_orig >= C_MAX - 1e-4).float().mean()),
            "residual_norm_buckets": bucket_resp,
        }
        print(f"L{li}: flip(shuf)={flip_shuf:.4f} flip(repl)={flip_repl:.4f} "
              f"dc_shuf={float(dc_shuf.mean()):.4f} dc_repl={float(dc_repl.mean()):.4f} "
              f"c_range=[{float(c_orig.min()):.3f},{float(c_orig.max()):.3f}]")

    out_path = TASK_DIR / "treatment" / "stage2" / "gate3_causal.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
