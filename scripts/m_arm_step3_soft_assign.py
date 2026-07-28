"""方向 J — 软分配 (soft/differentiable assignment).

训练时: 不用硬 argmin, 改用 softmax(-d/τ) 加权所有码字
        x_q_soft = Σπ_k · e_k
        loss 用 x_q_soft 算 (跟硬 argmin 同一 loss 公式)
评估时: 保持硬 argmin (跟之前所有变体可比)
τ 退火: τ_start=1.0 → τ_end=0.05 over 50 epochs (linear)

J1 = v6 recipe + 软分配 + NO w_angular
J2 = v6 recipe + 软分配 + w_angular=10 (条件启动)

Why: 用户 2026-07-27 /goal 方向 J — "赢家通吃 → 所有码字都拿梯度, 训练动态
更平滑, 也许天然就能长出 cos_std>0.3 (不需要硬拉)."
新风险: 码字互相拉近 → utilization 崩溃 (5cond 条件2).
"""
import os
import sys
import torch
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, os.path.join(REPO, "HG-Rec"))

from model.utils import HVectorQuantization  # noqa: E402
from model.utils import proj_to_ball, expmap0, poincare_distance  # noqa: E402

_ORIG_FORWARD = HVectorQuantization.forward

# τ 退火参数
_TAU_START = 1.0
_TAU_END = 0.05
_CURRENT_TAU = _TAU_START
_TOTAL_EPOCHS = 50


def set_soft_tau_for_epoch(epoch):
    """线性退火 τ_start → τ_end (用户 2026-07-27 方向 J)."""
    global _CURRENT_TAU
    if _TOTAL_EPOCHS <= 1:
        _CURRENT_TAU = _TAU_END
    else:
        progress = min(epoch / (_TOTAL_EPOCHS - 1), 1.0)
        _CURRENT_TAU = _TAU_START + (_TAU_END - _TAU_START) * progress


def _hyp_layernorm_safe(self, w_hyp, tangent_norm):
    """软分配时: 优先 tangent_norm, 不依赖 per-codeword radii.

    Why: 软分配 x_q_soft = Σπ_k·e_k 是 K 码字的加权均值, 没有 per-sample 'winner',
    per-codeword radii (r_per_k) 在这里失去 index → 用全局 tangent_norm 是更稳定的近似.
    """
    if tangent_norm is not None:
        return self._hyp_layernorm_normalize(w_hyp, tangent_norm)
    # 退化: 用 mean(r_per_k) 当 fallback
    r_per_k = self._hyp_radii_per_codeword()
    if r_per_k is not None:
        return self._hyp_layernorm_normalize(w_hyp, r_per_k.mean())
    return w_hyp  # last resort: 不归一化


def _patched_forward(self, x, use_sk=True):
    """训练时 (model.train()) 用软分配, 评估时仍走原 argmin 路径.

    关键: 只对 product_manifold 路径注入; 其他路径 (pure hyp / dual / euclidean) 走原始.
    评估时 (`self.training=False`) 也走原始, 保证 metrics 可比.
    """
    # 非训练态 OR 非 product_manifold OR τ 已冻结到接近 0 → 走原始
    if not (self.training and _CURRENT_TAU > 1e-2
            and self.product_manifold and self.hyp_dim > 0):
        return _ORIG_FORWARD(self, x, use_sk=use_sk)

    # 训练 + product_manifold + τ 大于 0.01 → 软分配路径
    latent = x.view(-1, self.e_dim)
    codebook = self.embeddings.weight  # (K, e_dim)

    if self.theta is not None:
        self.c = self.theta.exp().clamp(max=self.c_max)

    latent_hyp = latent[:, :self.hyp_dim]
    latent_euc = latent[:, self.hyp_dim:]
    codebook_hyp = codebook[:, :self.hyp_dim]
    codebook_euc = codebook[:, self.hyp_dim:]

    B = latent.shape[0]
    K = codebook.shape[0]

    # 距离 d = α * d_hyp + β_radial * d_euc (跟 _ORIG_FORWARD 内部 ~1290 一致, 切空间 MSE)
    d_hyp = ((latent_hyp.unsqueeze(1).expand(B, K, -1)
              - codebook_hyp.unsqueeze(0).expand(B, K, -1)) ** 2).mean(-1)
    d_euc = ((latent_euc.unsqueeze(1).expand(B, K, -1)
              - codebook_euc.unsqueeze(0).expand(B, K, -1)) ** 2).mean(-1)
    d = self.alpha * d_hyp + self.beta_radial * d_euc

    # Soft assignment
    tau = max(_CURRENT_TAU, 1e-3)
    pi = F.softmax(-d / tau, dim=-1)  # (B, K), 所有码字都拿梯度

    # x_q_soft = π @ codebook (切空间)
    xq_soft_hyp = (pi.unsqueeze(-1) * codebook_hyp.unsqueeze(0)).sum(dim=1)  # (B, hyp_dim)
    xq_soft_euc = (pi.unsqueeze(-1) * codebook_euc.unsqueeze(0)).sum(dim=1)  # (B, euc_dim)
    xq_soft = torch.cat([xq_soft_hyp, xq_soft_euc], dim=-1)  # (B, e_dim)

    # 把 hyp part 走 product_manifold 同样的归一化 (切空间 → ball)
    tangent_norm = self.r_target_norm if self.r_target_norm is not None else (
        self.rho / 2.0 if self.rho is not None else None)
    xq_soft_hyp_n = _hyp_layernorm_safe(self, xq_soft_hyp, tangent_norm)
    xq_soft_hyp_h = proj_to_ball(expmap0(xq_soft_hyp_n, self.c), self.c)

    # Loss (跟 _ORIG_FORWARD 内部 line 1337-1338 一致)
    latent_hyp_h = proj_to_ball(expmap0(latent_hyp, self.c), self.c)
    cl_hyp = torch.mean(poincare_distance(xq_soft_hyp_h.detach(), latent_hyp_h, self.c) ** 2)
    ql_hyp = torch.mean(poincare_distance(xq_soft_hyp_h, latent_hyp_h.detach(), self.c) ** 2)
    cl_euc = F.mse_loss(xq_soft_euc.detach(), latent_euc)
    ql_euc = F.mse_loss(xq_soft_euc, latent_euc.detach())
    loss = (self.alpha * (cl_hyp + self.beta * ql_hyp)
            + self.beta_radial * (cl_euc + self.beta * ql_euc * self.loss_mult_codebook))

    # Straight-through estimator: x_q = x + (xq_soft - x).detach()
    # 让 x_q 在前向传播时跟 x_q_soft 一致, 但 backward 时梯度走 x (encoder) 这条线
    x_q = x + (xq_soft - x).detach()

    # 保留硬 argmin 给 HRQVAE (residual VQ 需要 integer index)
    indices = torch.argmin(d, dim=-1).view(x.shape[:-1])

    return x_q, loss, indices


HVectorQuantization.forward = _patched_forward


# ---- τ 退火钩到 Trainer.fit ----
from model.hrqvae_trainer import Trainer  # noqa: E402
_ORIG_FIT = Trainer.fit


def _patched_fit(self, data):
    """每个 epoch 开始时调 set_soft_tau_for_epoch."""
    global _TOTAL_EPOCHS
    _TOTAL_EPOCHS = self.epochs
    # Pre-compute epochs list (we wrap _train_epoch to inject τ update)
    _orig_train_epoch = self._train_epoch

    def _wrapped_train_epoch(*args, **kw):
        epoch_idx = kw.get('epoch_idx', args[1] if len(args) > 1 else None)
        if epoch_idx is not None:
            set_soft_tau_for_epoch(epoch_idx)
            self.logger.info(f"[Direction J soft-assign] epoch={epoch_idx} tau={_CURRENT_TAU:.4f}")
        return _orig_train_epoch(*args, **kw)

    self._train_epoch = _wrapped_train_epoch
    return _ORIG_FIT(self, data)


Trainer.fit = _patched_fit


def main():
    """复刻 train_hrqvae main flow."""
    import runpy
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a not in ('--tau_start', '--tau_end')]
    runpy.run_path(os.path.join(REPO, "HG-Rec/train_hrqvae.py"), run_name="__main__")


if __name__ == "__main__":
    main()
