"""HG_Rec: HG-Rec 标准 T5ForConditionalGeneration wrapper (合并自 C33 Issue181).

v102 增强 (R36 v3.6 explicit relaxation 一次): Curvature-Aware SID Embedding.
- standard: 原始 T5 输入 (无注入)
- euclidean: g_l = [codeword, ||codeword||, 0], 几何信号只来自 codeword (无 curvature)
- curvature: g_l = [logmap0(codeword), ||codeword||, log c_l], 完整 Poincaré 几何信号

h_{l,k} = E(SID_{l,k}) + α_l · P_l(g_{l,k}), α_l = σ(a_l), a_l init ≈ -2.94 (σ ≈ 0.05)
P_l: ℝ^{d+2} → ℝ^{d_T5}, 每层独立 Linear

架构 = 序列级 CE + 平铺 vocab, 不需要 SemanticIdTokenizer/SEP token/per-hierarchy head.

差异 vs modules/model.py 的 EncoderDecoderRetrievalModel:
- 自定义 T5EncoderModel+T5Stack (per-hierarchy CE + SEP) → 标准 T5ForConditionalGeneration (序列级 CE)
- num_hierarchies=3 + SEP token 标志 → num_layers=6 + num_decoder_layers=4
- 自定义 generate_next_sem_id → 标准 model.generate(num_beams=20)

vocab_size=769 = 1 (PAD) + 256×3 (3 层 codebook 平铺).
labels shape: (B, 4) = (B, target item 4 位 vocab id), 含 PAD.

合并来源: tasks/Issue181_c33_hg_rec_stage3_arch_t5_ce/_lib/model/hg_rec.py
验证: C33 verdict test R@10=0.1093, C33 rerun test R@10=0.1058 (R37 PASS vs baseline 0.0972).
v102: Stage 3 端 curvature injection, 冻结 Stage 1 (v87 baseline).
"""
import math
import torch
from transformers import T5Config, T5ForConditionalGeneration
from typing import Optional, Dict, Any
import torch.nn as nn


def _logmap0_poincare_t(x: torch.Tensor, c_scalar: float) -> torch.Tensor:
    """logmap0: Poincaré ball → tangent space at origin.

    Args:
        x: (..., d) points on Poincaré ball (||x|| < 1/sqrt(c))
        c_scalar: curvature (Python float)

    Returns:
        y: (..., d) points in tangent space at origin
    """
    eps = 1e-6
    sqrt_c = math.sqrt(max(c_scalar, 1e-9))
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    arg = (sqrt_c * x_norm).clamp(max=1.0 - 1e-4)
    u_norm = torch.atanh(arg) / (sqrt_c + eps)
    y = x * (u_norm / (x_norm + eps))
    return y


class HG_Rec(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(HG_Rec, self).__init__()
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

        # === v102: Curvature-Aware SID Embedding 配置 ===
        self.injection_mode = config.get('injection_mode', 'standard')
        assert self.injection_mode in ('standard', 'euclidean', 'curvature'), \
            f"injection_mode must be standard/euclidean/curvature, got {self.injection_mode!r}"

        # n_codebook_layers = RQ-VAE 层数 (=3), 与 T5 num_layers 区分
        self.n_codebook_layers = int(config.get('n_codebook_layers', 3))
        self.codebook_size = int(config.get('codebook_size', 256))
        self.codeword_dim = int(config.get('codeword_dim', 32))

        # Token ID → codeword index 偏移: offset[l] = sum(codebook_size[0:l]) + 1
        # PAD=0, L0 tokens 1-256, L1 tokens 257-512, L2 tokens 513-768
        token_offsets = config.get('token_offsets', None)
        if token_offsets is None:
            token_offsets = [1 + sum([self.codebook_size] * i) for i in range(self.n_codebook_layers)]
        self.register_buffer(
            'token_offsets',
            torch.tensor(token_offsets, dtype=torch.long),
        )

        if self.injection_mode != 'standard':
            # Stage 1 codebooks (frozen, buffer): (n_layers, K, d)
            codebooks = config['stage1_codebooks']  # (n_layers, K, codeword_dim)
            curvatures = config['stage1_curvatures']  # (n_layers,)
            assert codebooks.dim() == 3 and codebooks.shape[0] == self.n_codebook_layers, \
                f"codebooks shape must be (n_layers={self.n_codebook_layers}, K, d), got {codebooks.shape}"
            assert codebooks.shape[1] == self.codebook_size and codebooks.shape[2] == self.codeword_dim, \
                f"codebooks last 2 dims must be (K={self.codebook_size}, d={self.codeword_dim})"
            self.register_buffer('codebooks', codebooks.detach().clone())
            self.register_buffer('curvatures', curvatures.detach().clone())

            # === 预计算 g_per_layer: (n_layers, K, d+2) ===
            # curvature mode: g_l = [logmap0(codeword_l), ||codeword_l||, log c_l]
            # euclidean mode: g_l = [codeword_l,       ||codeword_l||, 0         ]
            g_per_layer = []
            for l in range(self.n_codebook_layers):
                cb_l = codebooks[l]  # (K, d)
                c_l = float(curvatures[l].item()) if torch.is_tensor(curvatures[l]) else float(curvatures[l])
                if self.injection_mode == 'curvature':
                    cb_log = _logmap0_poincare_t(cb_l, c_l)  # (K, d)
                    last = math.log(max(c_l, 1e-9))
                else:  # euclidean
                    cb_log = cb_l  # raw codeword (Euclidean)
                    last = 0.0
                rho_l = cb_l.norm(dim=-1, keepdim=True)  # (K, 1)
                last_feat = torch.full(
                    (self.codebook_size, 1), last,
                    dtype=cb_l.dtype, device=cb_l.device,
                )
                g_l = torch.cat([cb_log, rho_l, last_feat], dim=-1)  # (K, d+2)
                g_per_layer.append(g_l)
            g_per_layer = torch.stack(g_per_layer, dim=0)  # (n_layers, K, d+2)
            self.register_buffer('g_per_layer', g_per_layer)

            # === 每层独立投影: ℝ^{d+2} → ℝ^{d_T5} ===
            d_t5 = int(config['d_model'])
            d_plus_2 = self.codeword_dim + 2
            self.proj_layers = nn.ModuleList([
                nn.Linear(d_plus_2, d_t5) for _ in range(self.n_codebook_layers)
            ])

            # === 每层 gate: σ(a_l), 初始化 a_l = -2.94 (σ ≈ 0.05) ===
            init_a = float(config.get('gate_init_logit', -2.94))
            self.gate_params = nn.Parameter(
                torch.full((self.n_codebook_layers,), init_a)
            )

    def _compute_modified_inputs_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """计算 curvature-injected 输入 embeddings.

        input_ids: (B, max_len * n_codebook_layers)
          position i → layer_idx = i % n_codebook_layers
          token_id 在该层 vocab 段: codeword_idx = token_id - offset[layer_idx]

        Returns:
            (B, max_len * n_codebook_layers, d_t5) inputs_embeds
        """
        B, T = input_ids.shape
        e = self.model.shared(input_ids)  # (B, T, d_t5)

        positions = torch.arange(T, device=input_ids.device)
        layer_idx = positions % self.n_codebook_layers  # (T,)

        alpha = torch.sigmoid(self.gate_params)  # (n_layers,)

        h_modified = e.clone()
        for l in range(self.n_codebook_layers):
            mask = (layer_idx == l)  # (T,) bool
            if not mask.any():
                continue
            token_ids_l = input_ids[:, mask]  # (B, T_l)
            # clamp PAD=0 (offset 之外) 到 [0, K-1], 防止索引越界
            codeword_idx_l = (token_ids_l - self.token_offsets[l]).clamp(0, self.codebook_size - 1)
            g_l = self.g_per_layer[l][codeword_idx_l]  # (B, T_l, d+2)
            proj_l = self.proj_layers[l](g_l)  # (B, T_l, d_t5)
            h_modified[:, mask, :] = e[:, mask, :] + alpha[l] * proj_l

        return h_modified

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, output_hidden_states: bool = False):
        """Forward pass. Returns (loss, logits) tuple (or with hidden_states if requested).

        Args:
            input_ids: (B, max_len * n_codebook_layers) flat token sequence (history 展平)
            attention_mask: (B, max_len * n_codebook_layers) 1 where real token, 0 for PAD
            labels: (B, n_codebook_layers) target item code sequence (含 PAD)
            output_hidden_states: 是否返回各层 hidden_states (HALC 创新需要)

        Returns:
            loss: 序列级 cross-entropy
            logits: (B, n_codebook_layers, vocab_size) decoder 输出 logits
            (可选) encoder_hidden_states + decoder_hidden_states: T5 各层 hidden_states
        """
        if self.injection_mode == 'standard':
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states,
            )
        else:
            inputs_embeds = self._compute_modified_inputs_embeds(input_ids)
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states,
            )
        if output_hidden_states:
            return outputs.loss, outputs.logits, outputs.encoder_hidden_states, outputs.decoder_hidden_states
        return outputs.loss, outputs.logits

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, num_beams: int = 20, **kwargs):
        """Generate recommendations using beam search.

        Args:
            input_ids: (B, max_len * n_codebook_layers) flat history token sequence
            attention_mask: (B, max_len * n_codebook_layers)
            num_beams: beam search 宽度 (R35 硬约束 = 20)

        Returns:
            preds: (B * num_beams, n_codebook_layers + 1) — 含 PAD/eos token, [1:] 才是有效 SID
        """
        if self.injection_mode == 'standard':
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=5,  # target = 4 token + 1 eos/pad
                num_beams=num_beams,
                num_return_sequences=num_beams,
                **kwargs
            )
        else:
            inputs_embeds = self._compute_modified_inputs_embeds(input_ids)
            return self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=5,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                **kwargs
            )