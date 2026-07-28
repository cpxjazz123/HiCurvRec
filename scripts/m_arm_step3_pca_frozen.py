"""Direction H — PCA-init hyp codeword directions + freeze, only log_r learnable.

Architecture:
  1. Monkey-patch HVectorQuantization.init_emb to use PCA on encoded z_hyp
     instead of kmeans. For product_manifold: hyp part = PCA directions on (B,HYP_DIM),
     euc part = KMeans on z_euc (kept from v6 recipe).
  2. After init: freeze self.embeddings.weight (direction fixed), keep log_r learnable.
  3. Replicate train_hrqvae main flow but with the patch + freeze hook.

Why: 用户 2026-07-27 方向 H — "码字方向不再自由学习或强制打散, 钉死在真实数据 PCA 主方向,
只让 radius 自由学习". 假设: collision 爆炸是因为 w_angular 把码字推到了 residual vector
密度低的地方, 跟真实数据分布打架. 钉死方向后, 应该可以让 radius 自然分化但不冲撞.

Usage: 直接传给 v6 recipe 的全套 CLI flags (跟 m_arm_step3_50ep_wdiv100_wangular.sh 一样).
"""
import os
import sys
import torch
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, os.path.join(REPO, "HG-Rec"))

# ---- 1) Monkey-patch HVectorQuantization.init_emb ----
from model.utils import HVectorQuantization  # noqa: E402

_ORIG_INIT_EMB = HVectorQuantization.init_emb


def _pca_init_emb(self, data):
    """用户 2026-07-27 方向 H: PCA 主方向初始化 hyp 部分, euc 部分 kmeans (跟 v6 一致)."""
    if self.dual_codebook or not (self.product_manifold and self.hyp_dim > 0):
        return _ORIG_INIT_EMB(self, data)

    from model.utils import kmeans as _kmeans
    data_hyp = data[:, :self.hyp_dim]
    data_euc = data[:, self.hyp_dim:]

    # ---- PCA on hyp ----
    z_centered = data_hyp - data_hyp.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(z_centered, full_matrices=False)
    n_dir_needed = self.n_e
    if self.hyp_dim < n_dir_needed:
        n_avail = V.shape[1]
        dirs = V.t().contiguous()  # (n_avail, hyp_dim)
        extra = torch.randn(n_dir_needed - n_avail, self.hyp_dim,
                            device=data.device, dtype=data.dtype)
        # Gram-Schmidt 简化版: 依次减已有方向投影 + 单位化.
        for k in range(extra.shape[0]):
            for j in range(dirs.shape[0]):
                proj = (extra[k] * dirs[j]).sum() * dirs[j]
                extra[k] = extra[k] - proj
            extra[k] = F.normalize(extra[k], dim=-1, eps=1e-8)
        dirs = torch.cat([dirs, extra], dim=0)
    else:
        dirs = V[:n_dir_needed].t().contiguous()
    dirs = F.normalize(dirs, dim=-1, eps=1e-8)
    if self.r_target_norm is None and self.rho is not None:
        r_target = self.rho / 2.0
    elif self.r_target_norm is not None:
        r_target = self.r_target_norm
    else:
        r_target = 1.0
    centers_hyp = r_target * dirs

    # ---- KMeans on euc (跟 v6 一致) ----
    centers_euc = _kmeans(data_euc, self.n_e, self.kmeans_iters)

    centers = torch.cat([centers_hyp, centers_euc], dim=-1)
    self.embeddings.weight.data.copy_(centers)
    self.initted = True

    cos_matrix = F.normalize(data_hyp, dim=-1, eps=1e-8) @ F.normalize(centers_hyp, dim=-1, eps=1e-8).t()
    cos_max = cos_matrix.max(dim=-1).values
    print(f"[Direction H PCA-init K={self.n_e} hyp_dim={self.hyp_dim}] "
          f"r_target={r_target}  cos_max_avg={cos_max.mean().item():.4f}  "
          f"cos_max_max={cos_max.max().item():.4f}")


HVectorQuantization.init_emb = _pca_init_emb


# ---- 2) Import train_hrqvae main flow ----
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_hrqvae_main", os.path.join(REPO, "HG-Rec/train_hrqvae.py"))
th = importlib.util.module_from_spec(spec)
spec.loader.exec_module(th)


def _freeze_codeword_directions(model):
    """用户 2026-07-27 方向 H: 冻结 embeddings.weight (方向), 保留 log_r."""
    frozen = 0
    learnable = 0
    for li, vq in enumerate(model.hrq.vq_layers):
        if not isinstance(vq, HVectorQuantization):
            continue
        vq.embeddings.weight.requires_grad = False
        frozen += vq.embeddings.weight.numel()
        if hasattr(vq, "log_r"):
            vq.log_r.requires_grad = True
            learnable += vq.log_r.numel()
    print(f"[Direction H freeze] embeddings.weight frozen={frozen}  log_r learnable={learnable}")


# Patch HRQVAE.__init__ → model 创建后立刻 freeze (但 weight 仍 uniform_-0.01,
# 等第一次 forward 时 init_emb 跑 PCA 覆盖 weight 为 PCA 方向, 之后被 freeze 保护).
from model.hrqvae import HRQVAE
_orig_hrqvae_init = HRQVAE.__init__


def _patched_hrqvae_init(self, *aargs, **kwargs):
    _orig_hrqvae_init(self, *aargs, **kwargs)
    _freeze_codeword_directions(self)


HRQVAE.__init__ = _patched_hrqvae_init


def main():
    """复刻 train_hrqvae main flow (line 232-425), 加上我们的 patch."""
    # 解析 CLI args.
    args = th.parse_args()

    # 走跟 train_hrqvae 一致的 pre-init validations (lines 281-339).
    # 简化: 直接调 train_hrqvae 内部的处理.
    # 实际我们只关心 (a) data, (b) model, (c) trainer, (d) fit.
    from model.utils import EmbDataset
    from torch.utils.data import DataLoader

    # 跑 main flow 仿照 train_hrqvae.py line 234 起 (简化版).
    import random, numpy as np
    seed = 2024
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 提取跟 train_hrqvae line 343-416 完全一样的 kwargs.
    # 为了避免重写所有参数解析, 我们直接 exec train_hrqvae main body
    # 但让它在 model 创建后被 patch hook 自动 freeze.
    # 做法: 让 main() 直接调 train_hrqvae.main 但跳过其 __name__ guard.
    # train_hrqvae.py 的 main() 不存在 (直接 if __name__ == "__main__"), 所以
    # 我们用 runpy 跑它.

    import runpy
    # runpy 会执行 __main__ 块, 其中 train_hrqvae 自己建 model, 然后 trainer.fit().
    # 我们的 patch 已 import + 激活 → 它跑 model = HRQVAE(...) 时, __init__ 会触发 freeze.
    runpy.run_path(os.path.join(REPO, "HG-Rec/train_hrqvae.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()