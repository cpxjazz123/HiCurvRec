"""HALC v4: Hyperbolic FFN 注入 (R52 主目录迭代, v5 Lorentz 后备胎).

设计 (R36 曲率机制变更, 真正 wrap encoder FFN):
1. 替换 encoder 每层 T5LayerFF 内的 DenseReluDense 为 HyperbolicFFN
2. HyperbolicFFN: x → poincare_logmap0 → linear1 → ReLU → linear2 → poincare_expmap0
3. warm-start: 从 pretrained DenseReluDense.wi/wo weight 初始化 HyperbolicFFN.linear1/linear2
4. learnable per-layer κ_ffn ∈ R^L + sigmoid annealing
5. reg_loss = sum_l ||poincare_logmap0(hyp_ffn_l_output)||^2 per layer (forward hook 收集)

文献支撑 (真正 full-stack hyperbolic transformer):
- Hypformer (NeurIPS 2024) full-stack hyperbolic transformer (FFN + attention 双曲化)
- HRec (ICLR 2025) tangent-space aggregation (log map → linear → exp map)
- H2H-GPT (arXiv 2501.03221) hyperbolic GPT with FFN 双曲化

vs v5 Lorentz 区别:
- v5: logmap0 reg on encoder hidden_states (仅 reg, 不 wrap)
- v4: 真 wrap encoder FFN (R36 曲率机制变更: 欧氏 FFN → 双曲 FFN)

vs v2 baseline 区别:
- v2: Poincaré logmap0 reg on encoder hidden_states (reg only)
- v4: 替换 FFN 内部 (tangent-space linear), FFN output 在双曲空间 + reg on FFN output
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class HyperbolicFFN(nn.Module):
    """Hyperbolic FFN: logmap → Linear → expmap, 可替换 HG_Rec encoder FFN."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        learnable_curvature: bool = True,
        init_curvature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        # learnable per-layer κ_ffn
        if learnable_curvature:
            self.log_curvature = nn.Parameter(
                torch.tensor(math.log(math.expm1(init_curvature)))
            )
        else:
            self.register_buffer("log_curvature", torch.tensor(math.log(math.expm1(init_curvature))))
        # standard FFN linear (在 tangent space)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    @property
    def curvature(self) -> torch.Tensor:
        return F.softplus(self.log_curvature) + 1e-5

    def poincare_logmap0(self, x: torch.Tensor) -> torch.Tensor:
        c = self.curvature
        sqrt_c = torch.sqrt(c)
        norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        max_norm = (1.0 / sqrt_c - 1e-5)
        norm_x_clamp = norm_x.clamp_max(max_norm)
        factor = torch.arctanh(sqrt_c * norm_x_clamp) / (sqrt_c * norm_x)
        return factor * x

    def poincare_expmap0(self, v: torch.Tensor) -> torch.Tensor:
        c = self.curvature
        sqrt_c = torch.sqrt(c)
        norm_v = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        max_norm = (1.0 / sqrt_c - 1e-5)
        norm_v_clamp = norm_v.clamp_max(max_norm)
        factor = torch.tanh(sqrt_c * norm_v_clamp) / (sqrt_c * norm_v)
        return factor * v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) 在 Poincaré ball
        x_logmap = self.poincare_logmap0(x)  # → tangent space
        y_tangent = self.linear2(F.relu(self.linear1(x_logmap)))  # FFN
        y_tangent = self.dropout(y_tangent)
        y_ball = self.poincare_expmap0(y_tangent)  # → Poincaré ball
        return y_ball


def wrap_hgrec_ffn(model: nn.Module, init_curvature: float = 1.0, learnable_curvature: bool = True) -> List[dict]:
    """Wrap HG_Rec encoder 每层 T5LayerFF.DenseReluDense 为 HyperbolicFFN.

    把 hyp_ffns 注册到 model.hyp_ffns (nn.ModuleList), 让 DDP/accelerator 看到参数.
    Returns: list of dicts with t5_layer_ff, layer_norm, dropout, hyp_ffn 用于 reg 计算.
    """
    wrapped_layers = []
    hyp_ffns_list = []  # 同时保留顺序引用
    for block in model.model.encoder.block:
        t5_layer_ff = block.layer[1]  # T5LayerFF
        d_model = t5_layer_ff.DenseReluDense.wi.in_features
        d_ff = t5_layer_ff.DenseReluDense.wi.out_features
        dropout_p = t5_layer_ff.dropout.p
        device = t5_layer_ff.DenseReluDense.wi.weight.device
        # 创建 hyperbolic FFN, warm-start from pretrained
        hyp_ffn = HyperbolicFFN(
            d_model=d_model, d_ff=d_ff, dropout=dropout_p,
            learnable_curvature=learnable_curvature,
            init_curvature=init_curvature,
        )
        hyp_ffn.linear1.weight.data = t5_layer_ff.DenseReluDense.wi.weight.data.clone()
        hyp_ffn.linear2.weight.data = t5_layer_ff.DenseReluDense.wo.weight.data.clone()
        hyp_ffn = hyp_ffn.to(device)
        layer_info = {
            't5_layer_ff': t5_layer_ff,
            'layer_norm': t5_layer_ff.layer_norm,
            'dropout': t5_layer_ff.dropout,
            'hyp_ffn': hyp_ffn,
        }
        wrapped_layers.append(layer_info)
        hyp_ffns_list.append(hyp_ffn)
        # 替换 forward (closure 捕获 layer_info)
        def make_new_forward(li):
            def new_forward(hidden_states):
                forwarded_states = li['layer_norm'](hidden_states)
                forwarded_states = li['hyp_ffn'](forwarded_states)
                hidden_states = hidden_states + li['dropout'](forwarded_states)
                return hidden_states
            return new_forward
        t5_layer_ff.forward = make_new_forward(layer_info)
    # 把 hyp_ffns 注册到 model, 让 DDP 看到参数
    model.hyp_ffns = nn.ModuleList(hyp_ffns_list)
    return wrapped_layers


class HALCV4WrapRegularizer(nn.Module):
    """HALC v4: per-layer learnable κ_ffn + FFN output reg on tangent norm (forward hook).

    Usage:
        halc = HALCV4WrapRegularizer(num_layers=6, ...)
        wrapped_layers = wrap_hgrec_ffn(model, init_curvature=1.0)
        halc.attach_to(wrapped_layers)  # 注册 forward hooks
        # 训练循环:
        #   loss, ... = model(input_ids, attention_mask, labels, output_hidden_states=True)
        #   halc_reg = halc.reg_loss()  # 自动用最近一次 forward 的 hyp_ffn 输出
        #   total_loss = loss + halc.reg_weight * halc_reg
    """

    def __init__(
        self,
        num_layers: int = 6,
        init_curvature: float = 1.0,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        reg_weight_max: float = 0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        # per-layer learnable log_curvature (κ_ffn)
        self.log_curvature = nn.Parameter(torch.zeros(num_layers).fill_(math.log(math.expm1(init_curvature))))
        # learnable reg_weight (bounded by tanh)
        self.raw_reg_weight = nn.Parameter(torch.tensor(0.01))
        self.reg_weight_max = reg_weight_max
        self.current_epoch = 0
        # hook state
        self.wrapped_layers: List[dict] = []
        self.hook_outputs: List[List[torch.Tensor]] = []

    def annealed_curvature(self) -> torch.Tensor:
        """c_l(t) = c_l * sigmoid((t - warmup) / cooldown)."""
        t = torch.tensor(float(self.current_epoch), dtype=self.log_curvature.dtype, device=self.log_curvature.device)
        ratio = (t - self.warmup_epochs) / max(self.cooldown_epochs, 1.0)
        sigmoid_factor = torch.sigmoid(ratio)
        learnable_c = F.softplus(self.log_curvature) + 1e-5
        return learnable_c * sigmoid_factor

    @property
    def reg_weight(self) -> torch.Tensor:
        return self.reg_weight_max * torch.tanh(self.raw_reg_weight / self.reg_weight_max)

    def attach_to(self, wrapped_layers: List[dict]):
        """Register forward hooks on each wrapped hyp_ffn to collect outputs."""
        self.wrapped_layers = wrapped_layers
        self.hook_outputs = [[] for _ in wrapped_layers]
        for i, li in enumerate(wrapped_layers):
            def make_hook(idx):
                def hook(module, input, output):
                    self.hook_outputs[idx].append(output)
                return hook
            li['hyp_ffn'].register_forward_hook(make_hook(i))

    def reg_loss(self) -> torch.Tensor:
        """Compute reg loss using latest hyp_ffn outputs (collected via forward hooks)."""
        if not self.wrapped_layers:
            return torch.tensor(0.0, device=self.log_curvature.device)
        c_per_layer = self.annealed_curvature()
        total = 0.0
        n = 0
        for i, outputs in enumerate(self.hook_outputs):
            if not outputs:
                continue
            out = outputs[-1]  # (B, L, d_model) 在 Poincaré ball
            c = c_per_layer[i]
            # poincare_logmap0(out): 把 ball 内的向量映射到 tangent space
            sqrt_c = torch.sqrt(c)
            norm = out.norm(dim=-1, keepdim=True).clamp_min(1e-15)
            max_norm = (1.0 / sqrt_c - 1e-5)
            norm_clamp = norm.clamp_max(max_norm)
            factor = torch.arctanh(sqrt_c * norm_clamp) / (sqrt_c * norm)
            v = factor * out  # (B, L, d_model) tangent vector
            reg = (v ** 2).sum(dim=-1).mean()
            total = total + reg
            n += 1
            self.hook_outputs[i] = []  # 清空, 下次 forward 重新收集
        if n == 0:
            return torch.tensor(0.0, device=self.log_curvature.device)
        return total / n

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
