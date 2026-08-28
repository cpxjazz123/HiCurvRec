"""PoincaréFFN — v30 备胎 (R36 曲率机制变更)

在 HG-Rec T5 FFN 输出 wrap Poincaré ball 几何变换:
  ffn_out → project_to_poinc (Euclidean→Ball) → MöbiusGeometricTransform (c_learnable) → project_to_euclidean (Ball→Euclidean)
  → dropout  → residual

实现要点:
- PoincareGeometricTransform(nn.Module): d_model 维输入, 输出 d_model 维
- install_poinc_ffn(hg_rec, c_init, proj_dropout): monkey-patch T5Block.forward, 在 T5LayerFF 后插入 transform
- 用 object.__setattr__ 绕过 nn.Module.__setattr__ (防止 _apply 递归, 类似 v29 HAB)

R36 曲率机制变更 explicit allowed — 与 v22 Stage1 Möbius 减法同源, 作用于 Stage 3 FFN 层,
而非 Stage 1 codebook. 期望: 利用 RQ-VAE 的双曲几何信号但保持 T5 主干完整性.

References:
- Poincaré ball: Nickel & Kiela 2017
- Möbius operations: Ungar 2008
- Lorentz/Poincaré equivariant NN: Chami et al. 2019
"""
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoincareProjector(nn.Module):
    """Euclidean ↔ Poincaré ball projector (numerically stable).

    Euclidean → Ball: x_ball = x / (1 + sqrt(1 + ||x||^2))   (Nickel & Kiela 2017, Eq. 2 inverse)
    Ball → Euclidean: x_eucl = 2 * x_ball / (1 - ||x_ball||^2)  (Eq. 2 forward inverse)
    """

    def forward(self, x_eucl: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """x_eucl: (..., d) tensor. inverse=False → Euclidean to Ball; True → Ball to Euclidean."""
        if inverse:
            # Ball → Euclidean
            norm_sq = (x_eucl * x_eucl).sum(dim=-1, keepdim=True)
            # 防止 norm → 1 时数值爆炸 (clip 到 1-1e-5)
            norm_sq = torch.clamp(norm_sq, max=1.0 - 1e-5)
            return 2.0 * x_eucl / (1.0 - norm_sq)
        else:
            # Euclidean → Ball
            norm_sq = (x_eucl * x_eucl).sum(dim=-1, keepdim=True)
            sqrt_term = torch.sqrt(1.0 + norm_sq)
            return x_eucl / (1.0 + sqrt_term)


class PoincareGeometricTransform(nn.Module):
    """在 Poincaré ball 内的局部几何变换.

    输入: Euclidean vector x ∈ R^d
    处理:
      1. Euclidean → Poincaré Ball: x_b = x / (1 + sqrt(1 + ||x||^2))
      2. 局部 Möbius 缩放 (curvature c_learnable): x_b' = x_b * sigmoid(c_param)  (缩放因子 <1)
      3. 残差可学习 affine: x_b'' = x_b' + alpha * tanh(W * x_b')  (小扰动, alpha init ≈ 0)
      4. Poincaré Ball → Euclidean
    输出: Euclidean vector y ∈ R^d

    Parameters (per-instance):
      - c_param: scalar nn.Parameter (init = logit(c_init)) — 控制 ball 内缩放
      - alpha: scalar nn.Parameter (init = -5.0, sigmoid ≈ 0.0067) — 残差扰动强度
      - W: (d, d) nn.Parameter (init = small)
    """

    def __init__(self, d_model: int, c_init: float = 0.5, alpha_init: float = -5.0):
        super().__init__()
        self.d_model = d_model
        # logit(c_init) ≈ 0 if c_init=0.5
        self.c_param = nn.Parameter(torch.tensor(math.log(c_init / (1.0 - c_init)), dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        # Small init W to avoid disrupting T5 main
        self.W = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.normal_(self.W, mean=0.0, std=0.01)
        self.projector = PoincareProjector()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) → y: (..., d_model)"""
        # 1. Euclidean → Ball
        x_b = self.projector(x, inverse=False)
        # 2. 缩放 (curvature effect): |c| → 缩放因子. c in (0,1), 用 sigmoid 映射到 (0,0.5)
        scale = torch.sigmoid(self.c_param) * 0.5  # 0~0.5
        x_b_scaled = x_b * scale
        # 3. 残差扰动
        perturbation = torch.tanh(F.linear(x_b_scaled, self.W))  # (..., d)
        alpha_eff = torch.sigmoid(self.alpha)
        x_b_perturbed = x_b_scaled + alpha_eff * perturbation
        # 4. Ball → Euclidean
        y = self.projector(x_b_perturbed, inverse=True)
        return y


def _wrap_t5_block_with_poinc_ffn(t5_block, transform: PoincareGeometricTransform):
    """Monkey-patch T5LayerFF.forward 在 FFN 输出后注入 PoincareGeometricTransform.

    比 monkey-patch T5Block.forward 更安全 (不破坏 T5Stack 内部 unpack 逻辑).

    Wrap 逻辑: 在 T5LayerFF.forward 末尾 (dropout 后) 注入:
        forwarded_states = transform(forwarded_states)  # 残差式注入 (避免大数值改变主路径)
    """
    # 取最后一层 (T5LayerFF): encoder block 是 [SelfAttn, FFN], decoder block 是 [SelfAttn, CrossAttn, FFN]
    # 必须用 layer[-1], 不能用 layer[1] (decoder 中是 CrossAttention!)
    layer_ff = t5_block.layer[-1]  # T5LayerFF (encoder layer[-1]=layer[1]; decoder layer[-1]=layer[2])
    if not isinstance(layer_ff, type(layer_ff).__mro__[0]):  # skip identity/empty modules
        pass
    if hasattr(layer_ff, "_poinc_ffn_installed"):
        return  # 已经注入过 (避免重复)
    object.__setattr__(layer_ff, "_poinc_ffn_installed", True)

    def poinc_ffn_layer_forward(self, hidden_states, **kwargs):
        """HF T5LayerFF.forward 的镜像实现, 在 FFN 输出后注入 transform.

        原版: forwarded_states = self.layer_norm(hidden_states)
              forwarded_states = self.DenseReluDense(forwarded_states)
              forwarded_states = self.dropout(forwarded_states)
        注: **kwargs 兜住可能的额外 kwarg (HF 上游偶发传入 key_value_states 等),
            但 FFN 实际只读 hidden_states.
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        forwarded_states = self.dropout(forwarded_states)
        # === v30 注入: FFN 输出后, residual 式几何注入 ===
        forwarded_states = forwarded_states + transform(hidden_states)
        # === 结束注入 ===
        return forwarded_states

    object.__setattr__(layer_ff, "forward", types.MethodType(poinc_ffn_layer_forward, layer_ff))


def install_poinc_ffn(hg_rec, c_init: float = 0.5, alpha_init: float = -5.0):
    """在 HG_Rec 的 encoder 和 decoder 每一层 FFN 后注入 PoincareGeometricTransform.

    hg_rec: HG_Rec instance (含 self.model = T5ForConditionalGeneration)
    c_init: 曲率初始值 (Poincaré ball 缩放因子)
    alpha_init: 残差扰动强度 init (sigmoid(alpha_init) ≈ 0)
    """
    d_model = hg_rec.model.config.d_model
    num_encoder_layers = hg_rec.model.config.num_layers
    num_decoder_layers = hg_rec.model.config.num_decoder_layers

    # Encoder: 一层 PoincareGeometricTransform, 共享给所有 encoder block (参数效率)
    # Decoder: 独立一层 (encoder/decoder 几何特性可能不同)
    encoder_transform = PoincareGeometricTransform(d_model=d_model, c_init=c_init, alpha_init=alpha_init)
    decoder_transform = PoincareGeometricTransform(d_model=d_model, c_init=c_init, alpha_init=alpha_init)
    hg_rec.add_module("_poinc_ffn_encoder_transform", encoder_transform)
    hg_rec.add_module("_poinc_ffn_decoder_transform", decoder_transform)

    # Wrap encoder blocks
    encoder = hg_rec.model.encoder
    if hasattr(encoder, "block"):
        blocks = encoder.block
    else:
        # 某些版本有 encoder.encoder.block
        blocks = encoder.encoder.block if hasattr(encoder, "encoder") else None
    if blocks is None:
        raise RuntimeError("Cannot locate T5 encoder blocks for PoincareFFN injection")

    for block in blocks:
        _wrap_t5_block_with_poinc_ffn(block, encoder_transform)

    # Wrap decoder blocks (类似处理, 优先用 decoder.block)
    decoder = hg_rec.model.decoder
    if hasattr(decoder, "block"):
        dblocks = decoder.block
    else:
        dblocks = decoder.decoder.block if hasattr(decoder, "decoder") else None
    if dblocks is None:
        raise RuntimeError("Cannot locate T5 decoder blocks for PoincareFFN injection")

    for block in dblocks:
        _wrap_t5_block_with_poinc_ffn(block, decoder_transform)

    print(f"[PoincareFFN] installed on {len(blocks)} encoder + {len(dblocks)} decoder blocks, d_model={d_model}, c_init={c_init}, alpha_init={alpha_init}", flush=True)
    return encoder_transform, decoder_transform


def count_poinc_ffn_params(hg_rec) -> int:
    """统计 PoincareGeometricTransform 参数数量."""
    n = 0
    if hasattr(hg_rec, "_poinc_ffn_encoder_transform"):
        n += sum(p.numel() for p in hg_rec._poinc_ffn_encoder_transform.parameters())
    if hasattr(hg_rec, "_poinc_ffn_decoder_transform"):
        n += sum(p.numel() for p in hg_rec._poinc_ffn_decoder_transform.parameters())
    return n