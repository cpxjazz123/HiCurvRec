"""v91: Full Riemannian Adam for Poincaré ball (Bécigneul & Ganea 2019).

Reference:
  - Bécigneul, Ganea. "Riemannian Adaptive Optimization Methods." ICLR 2019.
  - Nickel, Kiela. "Poincaré Embeddings for Learning Hierarchical Representations." NeurIPS 2017.

设计:
  对参数 p 标记 p.riemannian_ball=True 时, 用 Riemannian gradient + Möbius expmap retraction.
  其他参数用标准 Euclidean Adam (无 manifold constraint).

Riemannian gradient conversion (Poincaré ball, 曲率 c):
    grad_R = (1 - c * ||x||²)² / 4 * grad_E
  (Chami et al. 2019, "Hyperbolic Neural Networks")

Retraction (严格 expmap from x, Bécigneul & Ganea 2019 Eq. 2):
    expmap_x(v, c) = x ⊕_c (1/λ_x * expmap_0(λ_x * v, c))
    where λ_x = 2 / (1 - c*||x||²)
  where:
    expmap_0(u, c) = tanh(sqrt(c) * ||u||) * u / (sqrt(c) * ||u||)
    x ⊕_c y = ((1 + 2c<x,y> + c||y||²) x + (1 - c||x||²) y) / (1 + 2c<x,y> + c²||x||²||y||²)

R36c Round 2 修复: 严格 λ_x scaling, 避免 ||x|| > 0 时 retraction 不沿 geodesic.
"""
import torch
from torch.optim import Optimizer


class RiemannianAdam(Optimizer):
    """Riemannian Adam for Poincaré ball (Bécigneul & Ganea 2019).

    对标记 param.riemannian_ball=True 的参数:
      - Adam moment 在 Euclidean 空间维护
      - step direction 乘 Riemannian scaling (1 - c||x||²)² / 4
      - Retraction via Möbius expmap

    其他参数: 标准 Euclidean Adam.

    设置 param.riemannian_c (float) 必须在 optimizer.step() 之前, 用于更新曲率.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RiemannianAdam 不支持 sparse gradient")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # weight decay (Euclidean)
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Adam moment update (Euclidean tangent space)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Adam step direction (Euclidean)
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                step_dir = exp_avg / bias_correction1 / denom

                if getattr(p, 'riemannian_ball', False):
                    # === Riemannian branch (Poincaré ball) ===
                    c = float(getattr(p, 'riemannian_c', 1.0))
                    sqrt_c = c ** 0.5

                    # 1. Riemannian gradient conversion: grad_R = (1 - c||x||²)² / 4 * grad_E
                    x_norm_sq = (p * p).sum(dim=-1, keepdim=True)
                    scaling = (1 - c * x_norm_sq).clamp(min=1e-8) ** 2 / 4
                    grad_R = step_dir * scaling

                    # 2. 严格 expmap from x (Bécigneul & Ganea 2019 Eq. 2)
                    #    λ_x = 2 / (1 - c*||x||²)
                    #    expmap_x(v, c) = x ⊕_c (1/λ_x * expmap_0(λ_x * v, c))
                    u = -group['lr'] * grad_R
                    lambda_x = (2.0 / (1 - c * x_norm_sq).clamp(min=1e-8))
                    u_scaled = u * lambda_x  # tangent → origin tangent space
                    u_scaled_norm = u_scaled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    # expmap_0(u_scaled, c) = tanh(sqrt(c) * ||u_scaled||) * u_scaled / (sqrt(c) * ||u_scaled||)
                    expmap_0_us = torch.tanh(sqrt_c * u_scaled_norm) * u_scaled / (sqrt_c * u_scaled_norm)
                    expmap_v = expmap_0_us / lambda_x  # origin tangent → x tangent (parallel transport rescale)

                    # 3. Möbius addition: x ⊕_c y
                    xy_dot = (p * expmap_v).sum(dim=-1, keepdim=True)
                    y_norm_sq = (expmap_v * expmap_v).sum(dim=-1, keepdim=True)
                    num = (1 + 2*c*xy_dot + c*y_norm_sq) * p + (1 - c*x_norm_sq) * expmap_v
                    denom_mob = (1 + 2*c*xy_dot + c**2 * x_norm_sq * y_norm_sq).clamp(min=1e-12)
                    p_new = num / denom_mob

                    # 4. Numerical safety: clip to ball
                    radius = (1 - 1e-5) / sqrt_c
                    new_norm = p_new.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    scale = torch.where(new_norm > radius, radius / new_norm, torch.ones_like(new_norm))
                    p.data.copy_(p_new * scale)
                else:
                    # === Euclidean branch (standard Adam) ===
                    p.data.add_(step_dir, alpha=-group['lr'])
        return loss
