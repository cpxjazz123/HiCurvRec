"""HG-Rec v2: Hyperbolic Attention Bias (HAB) at encoder self-attention.

思路 (R36 #3: Stage 3 加几何 bias):
  把 Stage2 码本 (Euclidean, K_l × d_tangent) 映射到 Poincaré ball (c=1), 预计算
  码字间双曲距离 D_l (K_l × K_l). 注入到 HG_Rec encoder self-attention 4D bias:
    B_geo_ij = -lambda_l * D_l[k_i, k_j]    (i, j 同属层 l)
    B_geo_ij = 0                              (跨层 / PAD / L3)

HG-Rec vocab layout (from data/dataset.py):
  PAD=0, L0=[1,64], L1=[65,192], L2=[193,448], L3=[449]
  (与 baseline HAB_CODEWORD_OFFSETS = [1, 65, 193, 449] 一致)

Stage2 ckpt (HG-Rec/train_hrqvae.py loss_type='poincare', c=1 固定):
  best_collision_model.pth → state_dict['hrq.vq_layers.{0,1,2}.embeddings.weight']
  → (K_l=64/128/256, d_tangent=32)

只启用 frozen Dbar + learnable lambda (Issue #64 v4 模式, baseline 已知 PASS).
不引入 U/V learnable (v6b FAIL), 不引入 residual (v71), 严格走 v4 路径保证 R37 PASS 路径.
"""
import math
import numpy as np
import torch
import torch.nn as nn


# HG-Rec vocab token → 层位 映射 (PAD=0→-1, L0=[1,64]→0, L1=[65,192]→1, L2=[193,448]→2, L3=[449]→3)
HAB_VOCAB_OFFSETS = [1, 65, 193, 449]
HAB_VOCAB_K = [64, 128, 256, 1]


def _artanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _expmap0(u, c):
    """Poincaré expmap0: 切空间 u → 球面 x."""
    sqrt_c = c ** 0.5
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    factor = torch.tanh(sqrt_c * norm_u) / (sqrt_c * norm_u)
    return factor * u


def _proj_to_ball(x, c, eps=1e-5):
    """Poincaré 球硬截断 (norm < R = 1/sqrt(c))."""
    r = (1.0 / c) ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) * r
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def _mobius_add(x, y, c):
    """Poincaré 加法."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / den.clamp_min(1e-30)


def _poincare_distance(x, y, c):
    """严格对称双曲距离, 通过 mobius_add(-x, y)."""
    diff = _mobius_add(-x, y, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return (2.0 / sqrt_c) * _artanh((sqrt_c * norm).clamp(max=1 - 1e-10))


def precompute_hab_distances(stage2_ckpt_path, c_list=(1.0, 1.0, 1.0)):
    """从 HG-Rec Stage2 ckpt 读码本, 预计算三层双曲距离矩阵.

    Args:
        stage2_ckpt_path: best_collision_model.pth 路径
        c_list: 3 个曲率 (HG-Rec HRQ-VAE 用 c=1 固定)

    Returns:
        D_list: list of 3 tensors, (K_l, K_l) 双曲距离
        Dbar_list: list of 3 tensors, (K_l, K_l) 归一化距离 (D / median_nonzero)
        stats_list: list of 3 dicts 每层统计
    """
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    # HG-Rec Stage2 ckpt 用 state_dict key (与 baseline 不同, baseline 用 model_state_dict)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    D_list = []
    Dbar_list = []
    stats_list = []
    for l in range(3):
        cb_key = f"hrq.vq_layers.{l}.embeddings.weight"
        if cb_key not in sd:
            raise KeyError(f"Stage2 ckpt 缺少 {cb_key}, keys: {list(sd.keys())[:5]}")
        cb = sd[cb_key].float()  # (K_l, d_tangent)
        c = float(c_list[l])
        # 映射到 Poincaré 球
        z = _proj_to_ball(_expmap0(cb, c), c)  # (K_l, d_tangent)
        K = z.shape[0]
        # pairwise 距离 (严格对称, 因为 poincare_distance 内部用 mobius_add)
        z_exp = z.unsqueeze(1).expand(K, K, -1)
        z_pair = z.unsqueeze(0).expand(K, K, -1)
        D = _poincare_distance(z_exp, z_pair, c).squeeze(-1)  # (K_l, K_l)
        # 统计 + 归一化
        finite = torch.isfinite(D).all().item()
        sym_err = (D - D.T).abs().max().item()
        diag_max = D.diagonal().abs().max().item()
        mask = ~torch.eye(K, dtype=torch.bool)
        D_nonzero = D[mask]
        med = D_nonzero.median().item() if D_nonzero.numel() > 0 else 1.0
        Dbar = D / (med + 1e-10)
        D_list.append(D)
        Dbar_list.append(Dbar)
        stats_list.append({
            "layer": l, "c": c,
            "finite": finite, "sym_err": sym_err,
            "diag_max": diag_max, "median": med,
            "K": K, "shape": tuple(D.shape),
        })
    return D_list, Dbar_list, stats_list


def make_layer_id_lut(vocab_size=1025, offsets=HAB_VOCAB_OFFSETS, K=HAB_VOCAB_K):
    """构造 (vocab_size,) 长数组: token → 层位 (0=L0, 1=L1, 2=L2, 3=L3, -1=PAD/其他).

    HG-Rec vocab: PAD=0, L0=[1..64], L1=[65..192], L2=[193..448], L3=[449].
    L3 K=1 是 dedup, Dbar 全 0 → bias=0, 故不注入.
    """
    lut = np.full(vocab_size, -1, dtype=np.int64)
    for l, (off, k) in enumerate(zip(offsets, K)):
        if k <= 0:
            continue
        end = off + k
        lut[off:end] = l
    return lut


class HABModule(nn.Module):
    """v2 HAB: frozen Dbar per layer + learnable λ (Issue #64 v4 模式).

    Forward 给定 (input_ids B×L, layer_id_lut, attention_mask) → 输出 (B, 1, L, L) bias.
    """

    def __init__(self, Dbar_list, lambda_max=0.20, force_zero_layers=(3,)):
        super().__init__()
        self.num_layers = len(Dbar_list)
        self.lambda_max = float(lambda_max)
        self.K = [D.shape[0] for D in Dbar_list]
        # 注册 frozen Dbar (requires_grad=False, 但参与 state_dict)
        self.Dbar_buffers = nn.ParameterList()
        for l in range(self.num_layers):
            Dbar_l = Dbar_list[l].detach().cpu().to(torch.float32)
            self.Dbar_buffers.append(nn.Parameter(Dbar_l, requires_grad=False))
        # lambda_raw init = 0.5 * lambda_max → lambda_eff ≈ 0.10
        _lambda_init = 0.5 * self.lambda_max
        lambda_raw = torch.full((self.num_layers,), _lambda_init)
        for l in force_zero_layers:
            if l < self.num_layers:
                lambda_raw[l] = 0.0
        self.lambda_raw = nn.Parameter(lambda_raw)
        # offsets: 用于 input_id → k_in_layer 转换
        self.register_buffer("offsets", torch.tensor(HAB_VOCAB_OFFSETS[:self.num_layers], dtype=torch.long))

    @property
    def lambda_eff(self):
        return self.lambda_max * torch.tanh(self.lambda_raw / self.lambda_max)

    def get_B_geo(self, input_ids, layer_id_lut_tensor, attention_mask_2d=None):
        """计算 B_geo additive bias (bsz, L, L) → 用于 encoder self-attention.

        Args:
            input_ids: (B, L) long
            layer_id_lut_tensor: (vocab_size,) long, token → 层位 (-1=PAD/L3=0)
            attention_mask_2d: (B, L) long, 1=valid 0=padding

        Returns:
            B_geo: (B, 1, L, L) additive bias
        """
        B, L = input_ids.shape
        device = input_ids.device
        if layer_id_lut_tensor.device != device:
            layer_id_lut_tensor = layer_id_lut_tensor.to(device)
        layer_ids = layer_id_lut_tensor[input_ids]  # (B, L)
        B_geo = torch.zeros(B, L, L, device=device, dtype=torch.float32)
        lambda_eff = self.lambda_eff
        for l in range(self.num_layers):
            mask_l = (layer_ids == l)
            if not mask_l.any():
                continue
            k_for_layer = torch.where(mask_l, input_ids - self.offsets[l],
                                       torch.zeros_like(input_ids))
            k_for_layer = k_for_layer.clamp(0, self.K[l] - 1)
            Dbar_l = self.Dbar_buffers[l]  # (K_l, K_l)
            Dbar_ij = Dbar_l[
                k_for_layer.unsqueeze(-1).expand(-1, -1, k_for_layer.shape[1]).clamp(0, self.K[l] - 1),
                k_for_layer.unsqueeze(-2).expand(-1, k_for_layer.shape[1], -1).clamp(0, self.K[l] - 1),
            ]  # (B, L, L)
            mask_pair_l = mask_l.unsqueeze(2) & mask_l.unsqueeze(1)  # (B, L, L)
            B_geo_l = -lambda_eff[l] * Dbar_ij
            B_geo = torch.where(mask_pair_l, B_geo_l, B_geo)
        # PAD 屏蔽
        if attention_mask_2d is not None:
            valid = attention_mask_2d.bool()
            mask_pad = valid.unsqueeze(2) & valid.unsqueeze(1)
            B_geo = B_geo * mask_pad.float()
        return B_geo.unsqueeze(1)  # (B, 1, L, L)


def install_hab(hg_rec, hab_module, layer_id_lut_array):
    """Monkey-patch HG_Rec.model.encoder.forward: 在 encoder 路径注入 HAB bias.

    与 baseline install_hab 一致: 手动实现 encoder block 循环 (6 层), 在 attention_mask 4D bias 上附加 B_geo 一次,
    后续所有 encoder layer 复用 (符合 HF T5 行为).
    """
    import types
    try:
        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
    except ImportError:
        BaseModelOutputWithPastAndCrossAttentions = None
    try:
        from transformers.models.t5.modeling_t5 import create_bidirectional_mask
    except ImportError:
        def create_bidirectional_mask(config, inputs_embeds, attention_mask):
            if attention_mask is None:
                attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device,
                                             dtype=torch.long)
            extended = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
            extended = (1.0 - extended) * torch.finfo(inputs_embeds.dtype).min
            return extended

    device = next(hg_rec.parameters()).device
    hab_module = hab_module.to(device)
    layer_id_lut_tensor = torch.as_tensor(layer_id_lut_array, dtype=torch.long).to(device)
    hg_rec.add_module("hab_module", hab_module)
    hg_rec._hab_inject_count = 0
    # 保存原 encoder.forward (decoder 路径需要 fallback)
    encoder = hg_rec.model.encoder
    encoder._original_forward = encoder.forward

    def hab_encoder_forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                              encoder_hidden_states=None, *args, **kwargs):
        """只在 encoder (is_decoder=False) 路径注入 HAB bias. Decoder 路径直接走原 forward."""
        if self.is_decoder:
            return self._original_forward(input_ids=input_ids, attention_mask=attention_mask,
                                          inputs_embeds=inputs_embeds,
                                          encoder_hidden_states=encoder_hidden_states,
                                          *args, **kwargs)
        # ===== Encoder 路径: 手动实现 (与 baseline install_hab 语义一致) =====
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

        # HF T5 默认构造 4D attention_mask (bidirectional, encoder 路径)
        attention_mask_4d = create_bidirectional_mask(
            self.config, inputs_embeds, attention_mask)

        # λ=0 → 严格等价原 forward (避免任何重写差异)
        lambda_eff = hab_module.lambda_eff.detach()
        lambda_is_zero = bool((lambda_eff.abs().sum() == 0).item())
        if lambda_is_zero or input_ids_flat is None:
            return self._original_forward(input_ids=input_ids, attention_mask=attention_mask,
                                            inputs_embeds=inputs_embeds,
                                            encoder_hidden_states=encoder_hidden_states,
                                            *args, **kwargs)

        # 计算 B_geo 并附加到 attention_mask_4d (1 次, 后续 block 复用)
        hg_rec._hab_inject_count += 1
        ids_view = input_ids_flat.view(batch_size, seq_length)
        B_geo = hab_module.get_B_geo(ids_view, layer_id_lut_tensor,
                                      attention_mask_2d=attention_mask)
        if attention_mask_4d.dtype != B_geo.dtype:
            B_geo = B_geo.to(attention_mask_4d.dtype)
        attention_mask_4d = attention_mask_4d + B_geo

        # Encoder block 循环 (6 层共用修补后的 attention_mask_4d)
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
                past_key_value=None, use_cache=False,
                cache_position=torch.arange(seq_length, device=inputs_embeds.device),
                output_attentions=output_attentions, return_dict=True)
            hidden_states = layer_outputs[0]
        hidden_states = self.final_layer_norm(hidden_states)
        if BaseModelOutputWithPastAndCrossAttentions is not None:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )
        return (hidden_states, all_hidden_states, all_attentions)

    encoder.forward = types.MethodType(hab_encoder_forward, encoder)
    return hg_rec
