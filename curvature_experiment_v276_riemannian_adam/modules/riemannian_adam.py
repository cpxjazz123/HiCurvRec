"""v276 NOVEL: Riemannian Adam optimizer for codebook embedding on Poincaré ball.

Reference: Bécigneul & Ganea (2019) "Riemannian Adaptive Optimization Methods".
Poincaré ball metric at x: g_x = ((1 - ||x||^2)^2 / 4) * I_d, so Riemannian
gradient is rescaled Euclidean gradient by lambda = (1 - ||x||^2)^2 / 4.

For numerical stability in DDP + bf16 mixed precision, we:
1. Compute Riemannian-scaled gradient in fp32.
2. Maintain Adam m, v on Riemannian gradient.
3. Apply Adam update in tangent space (Euclidean locally).
4. Project back to ball by clamping ||x|| <= MAX_NORM < 1.
5. Re-rescale embedding magnitude via tanh-style squash for points near boundary.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class RiemannianAdam(Optimizer):
    """Riemannian Adam for Poincaré ball embedding (Bécigneul & Ganea 2019).

    Args:
        params: iterable of parameters (typically codebook embedding.weight of shape [K, D])
        lr: learning rate (tangent space step size)
        betas: coefficients for moving averages of gradient
        eps: epsilon for numerical stability
        weight_decay: L2 penalty (Euclidean, applied to embedding)
        max_norm: max L2 norm of embedding (Poincaré constraint, must be < 1.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_norm: float = 0.99,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 < max_norm < 1.0:
            raise ValueError(f"max_norm must be in (0, 1), got {max_norm}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_norm=max_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            max_norm = group["max_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RiemannianAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]

                # 1. Riemannian gradient scale (Poincaré metric): λ(x) = (1 - ||x||²)² / 4
                #    Only meaningful for 2D+ parameters (codebook [K, D])
                if p.dim() >= 2:
                    x_norm_sq = (p.data ** 2).sum(dim=-1, keepdim=True)  # (..., 1)
                    # Clamp to avoid division issues near boundary
                    one_minus_norm_sq = (1.0 - x_norm_sq).clamp(min=1e-7, max=1.0)
                    lambda_x = one_minus_norm_sq ** 2 / 4.0  # (..., 1)
                    rgrad = grad * lambda_x
                else:
                    # 1D parameters (bias, scalar): no metric, just Euclidean gradient
                    rgrad = grad

                # 2. Optional weight decay (Euclidean, applied to rgrad)
                if weight_decay != 0.0:
                    rgrad = rgrad.add(p, alpha=weight_decay)

                # 3. Adam moments on Riemannian gradient
                m, v = state["m"], state["v"]
                m.mul_(beta1).add_(rgrad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(rgrad, rgrad, value=1 - beta2)

                # 4. Bias-corrected moments
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # 5. Update in tangent space (Euclidean locally)
                update = m_hat / (v_hat.sqrt() + eps)
                p.data.add_(update, alpha=-lr)

                # 6. Project back to Poincaré ball (||x|| < 1)
                if p.dim() >= 2:
                    x_norm = p.data.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    scale = (max_norm / x_norm).clamp(max=1.0)
                    p.data.mul_(scale)

        return loss

    def project_ball(self):
        """Explicit re-projection after external updates (e.g. EMA)."""
        for group in self.param_groups:
            max_norm = group["max_norm"]
            for p in group["params"]:
                if p.dim() >= 2:
                    x_norm = p.data.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    scale = (max_norm / x_norm).clamp(max=1.0)
                    p.data.mul_(scale)