"""HG_Rec: HG-Rec 标准 T5ForConditionalGeneration wrapper (合并自 C33 Issue181).

架构 = 序列级 CE + 平铺 vocab, 不需要 SemanticIdTokenizer/SEP token/per-hierarchy head.

差异 vs modules/model.py 的 EncoderDecoderRetrievalModel:
- 自定义 T5EncoderModel+T5Stack (per-hierarchy CE + SEP) → 标准 T5ForConditionalGeneration (序列级 CE)
- num_hierarchies=3 + SEP token 标志 → num_layers=6 + num_decoder_layers=4
- 自定义 generate_next_sem_id → 标准 model.generate(num_beams=20)

vocab_size=769 = 1 (PAD) + 256×3 (3 层 codebook 平铺).
labels shape: (B, 4) = (B, target item 4 位 vocab id), 含 PAD.

合并来源: tasks/Issue181_c33_hg_rec_stage3_arch_t5_ce/_lib/model/hg_rec.py
验证: C33 verdict test R@10=0.1093, C33 rerun test R@10=0.1058 (R37 PASS vs baseline 0.0972).
"""
import torch
from transformers import T5Config, T5ForConditionalGeneration
from typing import Optional, Dict, Any
import torch.nn as nn


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
        # Initialize T5 model with the specified configuration
        self.model = T5ForConditionalGeneration(t5config)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, output_hidden_states: bool = False):
        """Forward pass of the model. Returns (loss, logits) tuple.

        Args:
            input_ids: (B, max_len * n_layers) flat token sequence (history 展平)
            attention_mask: (B, max_len * n_layers) 1 where real token, 0 for PAD
            labels: (B, n_layers) target item code sequence (含 PAD)
            output_hidden_states: 是否返回各层 hidden_states (HALC 创新需要)

        Returns:
            loss: 序列级 cross-entropy (整个 target 序列 1 个标量)
            logits: (B, n_layers, vocab_size) decoder 输出 logits
            (可选) encoder_hidden_states + decoder_hidden_states: T5 各层 hidden_states
        """
        outputs = self.model(
            input_ids=input_ids,
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
            input_ids: (B, max_len * n_layers) flat history token sequence
            attention_mask: (B, max_len * n_layers)
            num_beams: beam search 宽度 (R35 硬约束 = 20)

        Returns:
            preds: (B * num_beams, n_layers + 1) — 含 PAD/eos token, [1:] 才是有效 SID
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,  # target = 4 token + 1 eos/pad
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs
        )