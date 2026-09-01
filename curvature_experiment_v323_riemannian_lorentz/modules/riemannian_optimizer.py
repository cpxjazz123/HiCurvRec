"""v323 NOVEL: Riemannian Adam with Lorentz retraction.

R36n (c) + (d) 联合: 用 Lorentz manifold retraction 更新 codebook.

论文支撑:
  - Bécigneul & Ganea "Riemannian Adaptive Optimization Methods" ICLR 2019
    (RiemannianSGD/RiemannianAdam 通用框架)
  - Law et al. NeurIPS 2019 "Lorentzian Distance Learning for Hyperbolic Representations"
    (Lorentz manifold 的 acosh 距离数值稳定性)

v323 关键设计:
  - codebook 仍是 Euclidean nn.Embedding (R^d) — 不存 Lorentz 坐标
  - 每个 update step 用 Lorentz exp_0 retraction 把 Adam update 方向映射回 manifold
  - 简化版 Riemannian Adam: 在切空间做 Adam momentum, 然后 exp_0 retraction

历史证伪:
  - v305/v128/v285/v290/v307 之前都用 Poincaré retraction, 全部 R37 FAIL
  - v323 用 Lorentz retraction (acosh 数值稳定) 是 (c)+(d) 全新组合
"""
import torch


class RiemannianAdam(torch.optim.Optimizer):
    """简化版 Riemannian Adam — 用 Lorentz exp_0 retraction 约束 codebook 更新方向.

    更新公式:
      m_t = β1 · m_{t-1} + (1 - β1) · g_t             (Euclidean Adam momentum)
      v_t = β2 · v_{t-1} + (1 - β2) · g_t²            (Euclidean Adam variance)
      update_euclidean = m_t / (sqrt(v_t) + eps)        (Euclidean update direction)
      update_lorentz = exp_0(scale · update_euclidean, c) (Lorentz retraction)
      p ← p ⊖ (lr · tangent_part(update_lorentz))      (Euclidean step in ball)

    关键:
      - m_t/v_t 在 Euclidean space (适合 Adam 框架)
      - update 方向用 Lorentz exp_0 retraction, 避免 boundary saturate
      - 最终 step 仍在 Euclidean (codebook 是 Euclidean tensor)
      - Lorentz retraction 提供 Riemannian norm scaling, 让 update 数值稳定
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        c: float = 1.0,
        use_riemannian: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            c=c, use_riemannian=use_riemannian,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                update_euclidean = exp_avg / denom  # (..., D)

                if group['use_riemannian']:
                    # Lorentz exp_0 retraction: 把 Euclidean update 方向映射到 Lorentz manifold
                    # y_0 = cosh(√c · ||u||) / √c
                    # y_i = sinh(√c · ||u||) · u_i / (√c · ||u||)
                    # 然后取空间分量 y_i 作为新的 Euclidean update (Riemannian norm scaling)
                    c = group['c']
                    sqrt_c = c ** 0.5
                    norm = update_euclidean.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    sinh_term = torch.sinh(sqrt_c * norm) / (sqrt_c * norm)
                    # 简化: 只用空间分量作为 Euclidean step direction
                    p.data.add_(update_euclidean * sinh_term, alpha=-step_size)
                else:
                    # 标准 Adam (Euclidean fallback)
                    p.data.add_(update_euclidean, alpha=-step_size)

        return loss
