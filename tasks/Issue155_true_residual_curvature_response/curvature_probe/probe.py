"""Issue #155 — Curvature Chain 1/5: 真实逐层 Residual 的曲率响应与作用边界.

从真实 L0/L1/L2 residual 出发, 固定 residual + codebook + 非曲率参数,
只改变该商品该层使用的曲率 c, 测量:
- Poincaré distance 与 d(distance)/dc (有限差分 vs 自动微分)
- top1-top2 margin 与最近邻排序
- assignment flip
- quantization distortion
- 邻域关系保持
- 不同 residual norm 区域的曲率响应

Gate M1 (真实路径): 用 control stage2 ckpt 的 HRQVAE.forward hook 捕获真实
z/e0/e1/r0/r1/r2, 与独立公式 r0=z, r1=z-e0, r2=z-e0-e1 逐元素一致验证.

判定 → chain_decision.json:
  MARGIN_LOCKED / CURVATURE_INSENSITIVE / ASSIGNMENT_ACTIVE / IMPLEMENTATION_INVALID
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import torch

TASK_DIR = Path(__file__).parent.parent  # tasks/Issue155_.../
CONTROL_LIB = TASK_DIR / "control" / "_lib"
sys.path.insert(0, str(CONTROL_LIB))
from hrqvae import HRQVAE
from utils import EmbDataset, proj_to_ball, expmap0, poincare_distance

C_CANDIDATES = [0.5, 0.75, 1.0, 1.5, 2.0]
ITEM_EMB = TASK_DIR / "control" / "stage1" / "item_emb.parquet"
CKPT = TASK_DIR / "control" / "stage2" / "hrqvae_kappa_sync.ckpt"
OUT_DIR = TASK_DIR / "curvature_probe"
N_ITEMS = 9922
SEED = 2024


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    item_emb = torch.tensor(
        EmbDataset(str(ITEM_EMB)).embeddings, dtype=torch.float32).to(device)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    from prefix_conditioned_quantizer import PrefixConditionedHRQVAE
    model = PrefixConditionedHRQVAE(
        in_dim=768, num_emb_list=[64, 128, 256], e_dim=32,
        layers=[512, 256, 128, 64], kmeans_init=True, kmeans_iters=1000,
        prefix_routing=False,
    ).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[155] ckpt loaded, missing={len(missing)} unexpected={len(unexpected)}")

    # ── Gate M1: forward hook 捕获真实路径 residual (r0/r1/r2) ──
    captured = {}

    def make_hook(layer_idx):
        def hook(module, args, out):
            captured[f"layer{layer_idx}"] = {
                "input": args[0].detach().cpu(),  # 该层 residual 输入 (B,32)
                "indices": out[2].detach().cpu(),
                "x_q": out[0].detach().cpu(),
            }
        return hook

    handles = []
    for li, vq in enumerate(model.vq_layers):
        handles.append(vq.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        z = model.encoder(item_emb)  # (N,32) = r0
        out, rq_loss, indices = model._rq_forward(z, use_sk=False)
    for h in handles:
        h.remove()

    # r0/r1/r2 独立公式 (用捕获的 x_q 逐层累减)
    e0 = captured["layer0"]["x_q"].numpy()
    e1 = captured["layer1"]["x_q"].numpy()
    e2 = captured["layer2"]["x_q"].numpy()
    r0 = z.cpu().numpy()
    r1 = r0 - e0
    r2 = r0 - e0 - e1
    # 与 hook 捕获的层输入一致?
    hook_r1 = captured["layer1"]["input"].numpy()
    hook_r2 = captured["layer2"]["input"].numpy()
    err_r1 = float(np.abs(r1 - hook_r1).max())
    err_r2 = float(np.abs(r2 - hook_r2).max())
    print(f"[155] real residual: err(r1 vs hook)={err_r1:.2e} err(r2 vs hook)={err_r2:.2e}")

    # 保存 residual hash / norm
    rnorm = np.linalg.norm(r0, axis=-1)
    residual_report = {
        "r0_hash": hashlib.sha256(r0.tobytes()).hexdigest()[:16],
        "r1_hash": hashlib.sha256(r1.tobytes()).hexdigest()[:16],
        "r2_hash": hashlib.sha256(r2.tobytes()).hexdigest()[:16],
        "err_r1_vs_hook": err_r1,
        "err_r2_vs_hook": err_r2,
        "r0_norm": {"mean": float(rnorm.mean()), "std": float(rnorm.std()),
                    "min": float(rnorm.min()), "max": float(rnorm.max())},
    }

    # ── 曲率响应: 固定 residual + codebook, 只改 c ──
    # 每层用真实 codebook (未投影) + 该层真实 residual, 不同 c 下重算距离/assignment/distortion
    codebooks = [vq.embeddings.weight.detach().cpu().numpy() for vq in model.vq_layers]
    layers_residual = [r0, r1, r2]

    layer_report = {}
    for li in range(3):
        resid = torch.tensor(layers_residual[li], dtype=torch.float32).to(device)  # (N,32)
        cb = torch.tensor(codebooks[li], dtype=torch.float32).to(device)  # (K,32)
        c0 = 1.0  # baseline 训练曲率
        report = {}
        dist_at_c0 = None
        margin_at_c0 = None
        for c in C_CANDIDATES:
            c = float(c)
            with torch.no_grad():
                latent_h = proj_to_ball(expmap0(resid, c), c)
                codebook_h = proj_to_ball(expmap0(cb, c), c)
                d = poincare_distance(
                    latent_h.unsqueeze(1).expand(-1, cb.shape[0], -1),
                    codebook_h.unsqueeze(0).expand(resid.shape[0], -1, -1),
                    c).squeeze(-1)  # (N,K)
            top2 = torch.topk(d, 2, dim=-1, largest=False)
            margin = (top2.values[:, 1] - top2.values[:, 0])
            asg = top2.indices[:, 0]
            # distortion: 量化后 x_q 与 residual 的欧氏 + Poincaré 距离
            x_q = cb[asg]
            dist_e = torch.norm(x_q - resid, dim=-1)
            with torch.no_grad():
                d_p = poincare_distance(
                    proj_to_ball(expmap0(x_q, c), c), proj_to_ball(expmap0(resid, c), c), c)
            if c == c0:
                dist_at_c0 = d
                margin_at_c0 = margin
                asg0 = asg
                report["baseline_assignment"] = asg.cpu().numpy().tolist()[:50]
            # d(distance)/dc 有限差分 (用距离矩阵均值)
            report[f"c{c}"] = {
                "d_matrix_mean": float(d.mean()),
                "margin_mean": float(margin.mean()),
                "distortion_euc": float(dist_e.mean()),
                "distortion_poincare": float(d_p.mean()),
            }
        # flip vs baseline c=1.0
        flips = []
        for c in C_CANDIDATES:
            if c == c0:
                flips.append(0.0)
                continue
            with torch.no_grad():
                latent_h = proj_to_ball(expmap0(resid, c), c)
                codebook_h = proj_to_ball(expmap0(cb, c), c)
                d = poincare_distance(
                    latent_h.unsqueeze(1).expand(-1, cb.shape[0], -1),
                    codebook_h.unsqueeze(0).expand(resid.shape[0], -1, -1),
                    c).squeeze(-1)
                asg_c = torch.topk(d, 1, dim=-1, largest=False).indices[:, 0]
            flips.append(float((asg_c != asg0).float().mean()))
        report["assignment_flip_vs_c1"] = {f"c{c}": f for c, f in zip(C_CANDIDATES, flips)}
        # d(distance)/dc: 有限差分 vs 自动微分 (在 c0 处, 用一小批样本)
        sub = resid[:512]
        cb_sub = cb
        c_t = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=device)
        latent_h = proj_to_ball(expmap0(sub, c_t), c_t)
        codebook_h = proj_to_ball(expmap0(cb_sub, c_t), c_t)
        d_aut = poincare_distance(
            latent_h.unsqueeze(1).expand(-1, cb_sub.shape[0], -1),
            codebook_h.unsqueeze(0).expand(sub.shape[0], -1, -1),
            c_t).squeeze(-1)
        d_aut.mean().backward()
        grad_auto = float(c_t.grad.item())
        eps = 1e-4
        with torch.no_grad():
            latent_h2 = proj_to_ball(expmap0(sub, 1.0 + eps), 1.0 + eps)
            codebook_h2 = proj_to_ball(expmap0(cb_sub, 1.0 + eps), 1.0 + eps)
            d2 = poincare_distance(
                latent_h2.unsqueeze(1).expand(-1, cb_sub.shape[0], -1),
                codebook_h2.unsqueeze(0).expand(sub.shape[0], -1, -1),
                1.0 + eps).squeeze(-1)
        grad_fd = float((d2.mean() - dist_at_c0[:512].mean()) / eps)
        report["d_dist_dc"] = {"autograd": grad_auto, "finite_diff": grad_fd,
                               "consistent": bool(abs(grad_auto - grad_fd) < 1e-3 * max(1, abs(grad_auto)))}
        # residual norm 分桶下的曲率响应
        rnorm_li = np.linalg.norm(layers_residual[li], axis=-1)
        bins = np.quantile(rnorm_li, [0, 0.25, 0.5, 0.75, 1.0])
        buckets = []
        with torch.no_grad():
            d1 = poincare_distance(
                proj_to_ball(expmap0(resid, 1.0), 1.0).unsqueeze(1).expand(-1, cb.shape[0], -1),
                proj_to_ball(expmap0(cb, 1.0), 1.0).unsqueeze(0).expand(resid.shape[0], -1, -1),
                1.0).squeeze(-1).mean(dim=-1).cpu().numpy()
            d2v = poincare_distance(
                proj_to_ball(expmap0(resid, 2.0), 2.0).unsqueeze(1).expand(-1, cb.shape[0], -1),
                proj_to_ball(expmap0(cb, 2.0), 2.0).unsqueeze(0).expand(resid.shape[0], -1, -1),
                2.0).squeeze(-1).mean(dim=-1).cpu().numpy()
        for b in range(4):
            mask = (rnorm_li >= bins[b]) & (rnorm_li <= bins[b + 1])
            buckets.append({
                "bucket": [float(bins[b]), float(bins[b + 1])],
                "n": int(mask.sum()),
                "d_mean_c1": float(d1[mask].mean()),
                "d_mean_c2": float(d2v[mask].mean()),
                "d_ratio_c2_c1": float(d2v[mask].mean() / (d1[mask].mean() + 1e-12)),
            })
        report["residual_norm_buckets"] = buckets
        layer_report[f"L{li}"] = report
        print(f"[155] L{li}: d_dist/dc autograd={grad_auto:.4e} fd={grad_fd:.4e} "
              f"consistent={report['d_dist_dc']['consistent']}")
        print(f"  flips: {flips}")

    # ── 判定 ──
    flip_max = max(max(layer_report[f"L{li}"]["assignment_flip_vs_c1"].values())
                   for li in range(3))
    margin_change = max(abs(layer_report[f"L{li}"][f"c{c}"]["margin_mean"]
                            - layer_report[f"L{li}"][f"c1.0"]["margin_mean"])
                        for li in range(3) for c in C_CANDIDATES)
    dist_change = max(abs(layer_report[f"L{li}"][f"c{c}"]["d_matrix_mean"]
                          - layer_report[f"L{li}"][f"c1.0"]["d_matrix_mean"])
                      for li in range(3) for c in C_CANDIDATES)

    if max(err_r1, err_r2) > 1e-4:
        decision = "IMPLEMENTATION_INVALID"
    elif flip_max < 0.01 and dist_change < 1e-6:
        decision = "CURVATURE_INSENSITIVE"
    elif flip_max < 0.1:
        decision = "MARGIN_LOCKED"
    else:
        decision = "ASSIGNMENT_ACTIVE"

    result = {
        "issue": "#155",
        "decision": decision,
        "evidence": {
            "real_residual": residual_report,
            "flip_max_across_layers_c": flip_max,
            "margin_max_change": margin_change,
            "distance_max_change": dist_change,
            "layer_report": layer_report,
        },
        "mechanism_verdicts": {
            "implementation_verdict": "PASS" if max(err_r1, err_r2) <= 1e-4 else "FAIL",
            "mechanism_activation_verdict": "PASS" if dist_change > 1e-6 else "INACTIVE",
            "causal_effect_verdict": decision,
        },
    }
    out = OUT_DIR / "chain_decision.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[155] decision={decision} flip_max={flip_max:.4f} "
          f"dist_change={dist_change:.4e} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
