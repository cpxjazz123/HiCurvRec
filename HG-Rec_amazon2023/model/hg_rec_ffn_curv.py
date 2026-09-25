"""v9: Stage 3 T5 FFN learnable curvature (no HAB, no Lorentz).

机制 (R36 #1 类 — 全新机制, 不与 v3-v8 重复):
  T5 encoder 每个 block 的 FFN 输出后, 加一个 FFNCurvature 模块:
  - learnable log_c 参数 (init=0 → c=1)
  - forward: tanh(√c * ||x||) / (√c * ||x||) 压缩 magnitude
  - 大 c → 强压缩 (紧凑表示), 小 c → 弱压缩 (展开表示)

为什么不与 v3 重复:
  v3 在 attention 层加 HAB (影响 Q/K/V); v9 在 FFN 层加 curvature (影响 hidden magnitude).

为什么不与 v4 重复:
  v4 是 HAB + Poincaré embedding + 正则 (组合机制); v9 是单 FFN curvature 模块 (无 HAB).

正则项: lambda_c * sum(log_c^2) 防止 c 漂移.
不加 λ 调参 (R36 禁止), 用 fixed lambda_c = 1e-4.

代码改动: monkey-patch T5LayerFF.forward (HF transformers 4.x 兼容).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5LayerFF


class FFNCurvature(nn.Module):
    """Poincaré ball magnitude compression with learnable curvature c.

    forward(x): x_out_i = x_i * tanh(√c * ||x_i||) / (√c * ||x_i||)
    """
    def __init__(self, c_init=1.0, c_min=0.05, c_max=20.0):
        super().__init__()
        self.c_min = c_min
        self.c_max = c_max
        # log_c: init=0 → c=1 (与 baseline 一致)
        self.log_c = nn.Parameter(torch.tensor(math.log(c_init), dtype=torch.float32))

    def get_c(self):
        return self.log_c.exp().clamp(min=self.c_min, max=self.c_max)

    def forward(self, x):
        c = self.get_c()
        sqrt_c = c.sqrt()
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        factor = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)
        return x * factor


def attach_ffn_curvature(model: T5ForConditionalGeneration, c_init=1.0):
    """Monkey-patch 所有 T5LayerFF.forward (encoder + decoder 都改)."""
    new_modules = {}
    for layer_idx, block in enumerate(model.encoder.block):
        ffn: T5LayerFF = block.layer[1]  # T5Block.layer = (SelfAttention, FFN, ...)

        curv = FFNCurvature(c_init=c_init)
        # wrap ffn forward
        original_forward = ffn.forward
        def make_forward(curv_module, orig_fwd):
            def patched(hidden_states):
                out = orig_fwd(hidden_states)
                out = curv_module(out)
                return out
            return patched, curv

        new_fwd, curv = make_forward(curv, original_forward)
        ffn.forward = new_fwd
        # 替换 layer[1].layer_norm 后的 residual connection 也算 curvature 影响 (这里只改 ffn 输出)
        # 注册 curvature module 到 block, 方便 optimizer 发现
        block.curvature_ffn = curv
        new_modules[f"encoder.block.{layer_idx}.curvature_ffn"] = curv

    # decoder 也加 (对称)
    for layer_idx, block in enumerate(model.decoder.block):
        ffn: T5LayerFF = block.layer[2]  # decoder.block.layer = (SelfAttention, CrossAttention, FFN)
        curv = FFNCurvature(c_init=c_init)
        original_forward = ffn.forward
        def make_forward(curv_module, orig_fwd):
            def patched(hidden_states):
                out = orig_fwd(hidden_states)
                out = curv_module(out)
                return out
            return patched, curv

        new_fwd, curv = make_forward(curv, original_forward)
        ffn.forward = new_fwd
        block.curvature_ffn = curv
        new_modules[f"decoder.block.{layer_idx}.curvature_ffn"] = curv

    return new_modules


def curvature_reg_loss(model: T5ForConditionalGeneration, lambda_c: float = 1e-4):
    """log_c^2 正则项: 防止 c 漂到极端值."""
    reg = 0.0
    n = 0
    for name, p in model.named_parameters():
        if 'curvature_ffn.log_c' in name:
            reg = reg + (p ** 2).sum()
            n += 1
    if n == 0:
        return torch.tensor(0.0)
    return lambda_c * reg


class HG_Rec_FFN_Curv(nn.Module):
    """HG_Rec with FFN learnable curvature (v9)."""
    def __init__(self, config: dict, c_init=1.0, lambda_c=1e-4):
        super().__init__()
        t5config = T5Config(
            num_layers=config['num_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            vocab_size=config['vocab_size'],
            pad_token_id=config['pad_token_id'],
            eos_token_id=config['eos_token_id'],
            decoder_start_token_id=config['pad_token_id'],
            feed_forward_proj=config['feed_forward_proj'],
        )
        self.model = T5ForConditionalGeneration(t5config)
        self.curvature_modules = attach_ffn_curvature(self.model, c_init=c_init)
        self.lambda_c = lambda_c

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        ce_loss = outputs.loss
        curv_reg = curvature_reg_loss(self.model, self.lambda_c)
        total_loss = ce_loss + curv_reg
        return total_loss, outputs.logits

    def get_curvatures(self):
        return {name: mod.get_c().item() for name, mod in self.curvature_modules.items()}

    def generate(self, input_ids, attention_mask=None, num_beams=20, **kwargs):
        return self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_length=5, num_beams=num_beams, num_return_sequences=num_beams, **kwargs
        )