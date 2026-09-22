"""T5 wrapper used by HG-Rec's vanilla TIGER-compatible runner."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration


class HG_Rec(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        t5config = T5Config(
            num_layers=config["num_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_heads=config["num_heads"],
            d_kv=config["d_kv"],
            dropout_rate=config["dropout_rate"],
            activation_function=config.get("activation_function", "relu"),
            vocab_size=config["vocab_size"],
            pad_token_id=config["pad_token_id"],
            eos_token_id=config["eos_token_id"],
            decoder_start_token_id=config["decoder_start_token_id"],
            feed_forward_proj=config["feed_forward_proj"],
            n_positions=config.get("max_token_seq_len", 512),
        )
        self.sid_length = int(config.get("sid_length", 4))
        self.model = T5ForConditionalGeneration(t5config)

    @property
    def n_parameters(self):
        num_params = lambda params: sum(p.numel() for p in params if p.requires_grad)
        total_params = num_params(self.parameters())
        embedding_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f"#Embedding parameters: {embedding_params}\n"
            f"#Non-embedding parameters: {total_params - embedding_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 50,
        **kwargs,
    ):
        """Generate ``SID width + EOS`` tokens with RecBole's beam setting."""
        generation_kwargs = dict(kwargs)
        generation_kwargs.setdefault("max_new_tokens", self.sid_length + 1)
        generation_kwargs.setdefault("use_cache", True)
        generation_kwargs.setdefault("output_scores", False)
        generation_kwargs.setdefault("return_dict_in_generate", False)
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **generation_kwargs,
        )

