"""Issue #64: 双曲码字距离作为 T5 Encoder Self-Attention Bias.

设计: 保持 SID token embedding 不变, 把同层码字之间的真实双曲距离作为
encoder self-attention 的 additive bias. 几何只影响"历史 token 应该关注谁",
而不改变"每个 token 本身是什么".

关键设计:
1. z_lk = Proj_{c_l}(Exp_0^{c_l}(e_lk)): 把切空间 codebook 映射到曲率 c_l = -kappa_l 的 Poincaré 球
2. D_l[k,k'] = d_{c_l}(z_lk, z_lk'): 三层双曲距离矩阵 (64x64, 128x128, 256x256), 预计算冻结
3. Dbar_l = D_l / median_nonzero(D_l): 每层非零距离中位数归一化, 消除量纲差异
4. lambda_l = lambda_max * tanh(lambda_raw_l / lambda_max), lambda_raw 初始 0 → 严格等价 #61
5. B_geo_ij = -lambda_l * Dbar_l[k_i, k_j] 仅当 i,j 同属层 l; 跨层/PAD/L3 → 0
6. 注入位置: encoder 第一次构造 position_bias 时附加一次, 后续 6 个 block 复用 (HF T5 行为)
7. decoder self-attention + cross-attention 完全不受影响 (只在 encoder forward 内注入)

Stage3 train + Stage4 eval 必须 import 这个模块, 禁止复制两套类.
"""
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Issue #64 固定偏移: L0=[1,64], L1=[65,192], L2=[193,448], L3=[449]
HAB_CODEWORD_OFFSETS = [1, 65, 193, 449]
HAB_CODEWORD_K = [64, 128, 256, 1]
# Issue #64 λ_max 默认 0.20 (spec 推荐, 本 issue 不允许 sweep)
HAB_LAMBDA_MAX = 0.20


def load_hab_assets_from_stage2_ckpt(stage2_ckpt_path):
    """从 #61 Stage2 ckpt 读 codebook (e_l) + final_kappas (κ_l).

    Returns:
        codebook_list: list of (K_l, d_tangent) numpy float32
        final_kappas: list of 4 floats (L3 占位 0.0)

    Note: #61 ckpt 里 vq_layers.{i}.kappa 是 drift 量 (训练后 = 0), 真实 κ 在 final_kappas.
    """
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    codebook_list = []
    for l in range(3):
        cb = sd[f"vq_layers.{l}.embeddings.weight"].numpy().astype(np.float32)
        codebook_list.append(cb)
    final_kappas = list(ckpt["final_kappas"])
    if len(final_kappas) < 4:
        # L3 占位 0 (dedup, 无几何信息)
        final_kappas = list(final_kappas) + [0.0] * (4 - len(final_kappas))
    return codebook_list, final_kappas


def _artanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _expmap0(u, c):
    """Poincaré expmap0: 切空间 u → 球面 x. 复用 HG-Rec/model/utils.py 语义."""
    sqrt_c = c ** 0.5
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    factor = torch.tanh(sqrt_c * norm_u) / (sqrt_c * norm_u)
    return factor * u


def _proj_to_ball(x, c, eps=1e-5):
    """Poincaré 球硬截断 (norm < R=1/sqrt(c)). 距离计算用, 不影响训练稳定性."""
    r = (1.0 / c) ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) * r
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def _mobius_add(x, y, c):
    """Poincaré 加法, 复用 HG-Rec/model/utils.py 实现 (避免重写带来的非对称数值问题)."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / den.clamp_min(1e-30)


def _poincare_distance(x, y, c):
    """Poincaré 距离, 用 mobius_add(-x, y) + norm 形式 (严格对称)."""
    diff = _mobius_add(-x, y, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return (2.0 / sqrt_c) * _artanh((sqrt_c * norm).clamp(max=1 - 1e-10))


def precompute_distance_matrices(codebook_list, final_kappas):
    """预计算三层双曲距离矩阵 D_l (K_l, K_l) + 归一化 Dbar_l.

    关键: c_l = -kappa_l (Poincaré 流形要求 c > 0, 而 #61 κ 为负值, 取 c = -κ).
    每层独立 c, 不允许跨层距离混合.

    Returns:
        D_list: list of (K_l, K_l) tensor (torch.float32), 未归一化
        Dbar_list: list of (K_l, K_l) tensor (torch.float32), 归一化 (D / median_nonzero)
        stats_list: list of dict 每层 {finite, sym_err, diag_max, med, p95}
    """
    D_list = []
    Dbar_list = []
    stats_list = []
    for l in range(len(codebook_list)):
        cb = torch.as_tensor(codebook_list[l], dtype=torch.float32)  # (K_l, d_tangent)
        kappa_l = float(final_kappas[l])
        c_l = max(-kappa_l, 1e-6)  # κ 为负, c = -κ > 0; 若 κ ≥ 0 用极小正值保护
        # 映射到 Poincaré 球
        z = _proj_to_ball(_expmap0(cb, c_l), c_l)  # (K_l, d_tangent)
        # pairwise 距离
        K = z.shape[0]
        z_exp = z.unsqueeze(1).expand(K, K, -1)
        z_pair = z.unsqueeze(0).expand(K, K, -1)
        D = _poincare_distance(z_exp, z_pair, c_l).squeeze(-1)  # (K_l, K_l)
        # 审计
        finite = torch.isfinite(D).all().item()
        sym_err = (D - D.T).abs().max().item()
        diag_max = D.diagonal().abs().max().item()
        # 非零距离 (排除对角线)
        mask = ~torch.eye(K, dtype=torch.bool)
        D_nonzero = D[mask]
        if D_nonzero.numel() > 0:
            med = D_nonzero.median().item()
            p95 = D_nonzero.quantile(0.95).item()
        else:
            med = 0.0
            p95 = 0.0
        # 归一化
        Dbar = D / (med + 1e-10) if D_nonzero.numel() > 0 else D
        D_list.append(D)
        Dbar_list.append(Dbar)
        stats_list.append({
            "layer": l, "c": c_l, "kappa": kappa_l,
            "finite": finite, "sym_err": sym_err, "diag_max": diag_max,
            "median": med, "p95": p95,
        })
    return D_list, Dbar_list, stats_list


class HyperbolicAttentionBias(nn.Module):
    """Issue #64 核心模块: 持有三层归一化双曲距离矩阵 + 三层 learnable lambda.

    Args:
        Dbar_list: list of (K_l, K_l) tensor (预计算冻结, register_buffer)
        lambda_max: λ 上限 (默认 0.20, Issue #64 spec)
        force_zero_layers: 强制 lambda_l = 0 的层列表 (默认 [3] L3 dedup)

    Note:
        lambda_raw init 0 → lambda_eff = 0 → 严格等价 #61 (Gate3 强制 < 1e-6)
        lambda_l = lambda_max * tanh(lambda_raw_l / lambda_max) 数学保证 ±lambda_max
    """
    def __init__(self, Dbar_list, lambda_max=HAB_LAMBDA_MAX, force_zero_layers=()):
        super().__init__()
        # num_layers = 实际 Dbar 数量 (#64 = 3, 因为 L3 是 dedup 无距离矩阵)
        self.num_layers = len(Dbar_list)
        self.lambda_max = float(lambda_max)
        for l in range(self.num_layers):
            self.register_buffer(f"Dbar_{l}", Dbar_list[l].clone())
        self.K = [Dbar_list[l].shape[0] for l in range(self.num_layers)]
        # lambda_raw 只对前 num_layers 个层有意义; force_zero_layers 超出范围跳过
        lambda_raw = torch.zeros(self.num_layers)
        for l in force_zero_layers:
            if l < self.num_layers:
                lambda_raw[l] = 0.0
        self.lambda_raw = nn.Parameter(lambda_raw)

    @property
    def lambda_eff(self):
        """Effective lambda per layer: lambda_max * tanh(lambda_raw / lambda_max)."""
        return self.lambda_max * torch.tanh(self.lambda_raw / self.lambda_max)

    def get_B_geo(self, input_ids, layer_id_lut_tensor, attention_mask_2d=None):
        """计算 B_geo additive bias (bsz, 1, L, L) 用于 encoder self-attention.

        Args:
            input_ids: (B, L) long tensor
            layer_id_lut_tensor: (vocab_size,) long, token → 层位 (-1=PAD/L3=0)
            attention_mask_2d: (B, L) long, 1=valid 0=padding (用于 mask PAD 对 bias 贡献)

        Returns:
            B_geo: (B, 1, L, L) additive bias, 同层 (L0/L1/L2) → -lambda_l * Dbar_l[k_i, k_j];
                   跨层/PAD/L3 → 0

        Note: 不对 L3 应用几何 bias (L3 是 dedup 1 个码字, k=0 → Dbar 全 0 → 不影响).
              PAD (-1) 不影响, 因为 mask 屏蔽 attention.
              cross-layer (L0 vs L1 等) → 严格 0 (不允许跨层距离).
        """
        B, L = input_ids.shape
        device = input_ids.device
        layer_ids = layer_id_lut_tensor[input_ids]  # (B, L) long: 0/1/2/3 或 -1
        B_geo = torch.zeros(B, L, L, device=device, dtype=torch.float32)
        lambda_eff = self.lambda_eff.detach()  # (num_layers,)
        # 关键: 对每层独立算 k_for_layer (该层专用), 不复用全局 k_in_layer (避免跨层越界)
        for l in range(self.num_layers):  # L0/L1/L2: 三层有距离矩阵
            mask_l = (layer_ids == l)  # (B, L) 该层 token
            if not mask_l.any():
                continue
            # k_for_layer = id - offset[l], 仅在该层 token 处计算 (其他层/PAD 保持 0)
            k_for_layer = torch.where(mask_l, input_ids - HAB_CODEWORD_OFFSETS[l],
                                       torch.zeros_like(input_ids))
            k_for_layer = k_for_layer.clamp(0, self.K[l] - 1)
            Dbar_l = getattr(self, f"Dbar_{l}")  # (K_l, K_l)
            mask_pair_l = mask_l.unsqueeze(2) & mask_l.unsqueeze(1)  # (B, L_i, L_j) 同层 pair
            Dbar_ij = Dbar_l[k_for_layer.unsqueeze(2), k_for_layer.unsqueeze(1)]  # (B, L, L)
            B_geo_l = -lambda_eff[l] * Dbar_ij
            B_geo = torch.where(mask_pair_l, B_geo_l, B_geo)
        # PAD 屏蔽: 任何 token 为 PAD, 该行/列 bias 设为 0 (跟 attention_mask 一致)
        if attention_mask_2d is not None:
            valid = attention_mask_2d.bool()  # (B, L)
            mask_pad = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, L_i, L_j) 两个都有效
            B_geo = B_geo * mask_pad.float()
        return B_geo.unsqueeze(1)  # (B, 1, L, L) — HF T5 4D bias 形状


def install_hab(hg_rec, hab_module, layer_id_lut_array):
    """Monkey-patch HG_Rec.model.encoder.forward: 在 attention_mask 4D bias 上附加 B_geo 一次,
    后续 6 个 encoder block 复用同一 attention_mask (符合 HF T5 行为 + spec "注入计数 = 1").

    关键约束 (Issue #64 spec):
    - 只在 encoder (is_decoder=False) 路径注入; decoder 完全走 _original_forward
    - attention_mask 修补只在 forward 开始处一次, 后续 encoder block 复用 (HF T5 行为)
    - decoder self-attention + cross-attention 不受影响
    - hg_rec._hab_inject_count 计数器验证注入 = 1 per encoder forward

    Args:
        hg_rec: HG_Rec 实例
        hab_module: HyperbolicAttentionBias 实例 (含 Dbar + lambda_raw)
        layer_id_lut_array: numpy (vocab_size,) long, token → 层位 (-1=PAD, 0/1/2/3=L0/L1/L2/L3)
    """
    import types
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
    from transformers.models.t5.modeling_t5 import create_bidirectional_mask
    device = next(hg_rec.parameters()).device
    hab_module = hab_module.to(device)
    layer_id_lut_tensor = torch.as_tensor(layer_id_lut_array, dtype=torch.long).to(device)
    hg_rec.add_module("hab_module", hab_module)
    # 注入计数 (Issue #64 Gate3 强制: 每个 encoder forward 注入调用 = 1)
    hg_rec._hab_inject_count = 0
    # 保存原 encoder.forward 引用 (decoder 路径需要 fallback)
    hg_rec.model.encoder._original_forward = hg_rec.model.encoder.forward

    def hab_encoder_forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                              encoder_hidden_states=None, *args, **kwargs):
        """Issue #64: monkey-patch 的 encoder.forward, 只在 is_decoder=False 时修补 attention_mask."""
        if self.is_decoder:
            # Decoder 路径 (Issue #64 明确不允许注入): 直接调原 forward
            return self._original_forward(input_ids=input_ids, attention_mask=attention_mask,
                                          inputs_embeds=inputs_embeds,
                                          encoder_hidden_states=encoder_hidden_states, *args, **kwargs)
        # ===== Encoder 路径 =====
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids_flat = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input_ids_flat = None
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids_flat)
        batch_size, seq_length = input_shape

        # HF T5 构造 attention_mask (4D additive bias) — encoder 路径下走 bidirectional
        attention_mask_4d = create_bidirectional_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # Issue #64: 计算 B_geo 并附加一次 (后续 6 block 复用同一 attention_mask_4d)
        # 关键: 当 lambda_eff 全 0 (训练前/λ 强制 0) 时, 严格走 _original_forward, bitwise 等价 #61
        if input_ids_flat is not None:
            hg_rec._hab_inject_count += 1  # 注入计数 (即使 λ=0 也算 = "注入调用发生, λ 为 0")
            lambda_eff = hab_module.lambda_eff.detach()
            lambda_is_zero = bool((lambda_eff.abs().sum() == 0).item())
            if lambda_is_zero:
                # λ=0 → 严格等价 #61, 直接调原 forward 避免任何重写引入的 bit-level 差异
                # 注意: 不要传 inputs_embeds (如果 input_ids 不为 None 会冲突)
                return self._original_forward(input_ids=input_ids, attention_mask=attention_mask,
                                                *args, **kwargs)
            # λ>0: 计算 B_geo 并附加到 attention_mask_4d
            ids_view = input_ids_flat.view(batch_size, seq_length)
            B_geo = hab_module.get_B_geo(ids_view, layer_id_lut_tensor,
                                          attention_mask_2d=attention_mask)
            if attention_mask_4d is not None:
                if attention_mask_4d.dtype != B_geo.dtype:
                    attention_mask_4d = attention_mask_4d.to(B_geo.dtype)
                attention_mask_4d = attention_mask_4d + B_geo.to(attention_mask_4d.device)

        # Encoder block loop (6 层共用修补后的 attention_mask_4d)
        output_hidden_states = kwargs.get("output_hidden_states", False)
        output_attentions = kwargs.get("output_attentions", False)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = self.dropout(inputs_embeds)
        for layer_module in self.block:
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, attention_mask_4d, None, None, None, None,
                past_key_values=None, use_cache=False,
                output_attentions=output_attentions, return_dict=True)
            hidden_states = layer_outputs[0]
        hidden_states = self.final_layer_norm(hidden_states)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    hg_rec.model.encoder.forward = types.MethodType(hab_encoder_forward, hg_rec.model.encoder)
    return hg_rec


def make_hab_layer_id_lut():
    """构造 layer_id LUT (vocab_size=1025): 0=PAD(-1), 1-64=L0(0), 65-192=L1(1),
    193-448=L2(2), 449=L3(3), 其它=−1.
    """
    lut = np.full(1025, -1, dtype=np.int64)
    lut[1:65] = 0
    lut[65:193] = 1
    lut[193:449] = 2
    lut[449:450] = 3
    return lut