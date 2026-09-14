"""HG_Rec_Curv_V7 — Stage 2 HRQ-VAE codebook → T5 SID embedding init transfer.

v6 关键发现: 单纯改 embedding 几何 (Poincaré/Lorentz) 没有贡献, T5 会自适应 magnitude.
v7 必须改信息传递路径: 把 Stage 2 学到的双曲 codebook 直接初始化 Stage 3 T5 SID embeddings.

机制 (R36 #1: Stage 2 → Stage 3 信息传递):
  Stage 2 ckpt: hrq.vq_layers.{0,1,2}.embeddings.weight ∈ (K_l, d_tangent=32)
    K_0=64, K_1=128, K_2=256 (HG-Rec vocab layout: L0=[1..64], L1=[65..192], L2=[193..448])
  v7 init: T5 shared.embedding[SID token range] ← Stage 2 codebook (32→128 dim repeat)
  非 SID token (PAD, L3 dedup, [450..1024]): 保持 T5 default init

预期收益:
  Stage 2 在 d_tangent=32 Euclidean 空间学到 codebook 位置 (隐含双曲结构)
  直接 init T5 SID embedding 节省 T5 重新学 SID 表示的容量, 期望 test R@10 ≥ 0.1074 baseline

约束:
  - 不调超参 (R36): 只换 init scheme
  - 不动训练/valid/test 数据 (R40)
  - 不调 Stage 2 (直接读现有 ckpt)
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any


def init_t5_with_codebook(vocab_size: int, d_model: int, stage2_ckpt_path: str,
                          scale_to_t5_default: bool = True,
                          c: float = 1.0):
    """从 Stage 2 HRQ-VAE ckpt 读 codebook, 构造 T5 shared embedding init.

    Args:
        vocab_size: T5 vocab (1025)
        d_model: T5 d_model (128)
        stage2_ckpt_path: best_collision_model.pth 路径
        scale_to_t5_default: 是否缩放到 T5 default magnitude (||v||≈11.3)
        c: Stage 2 HRQ-VAE 曲率 (HG-Rec 用 c=1 固定)

    Returns:
        init_emb: (vocab_size, d_model) tensor, 可直接 copy 到 T5 shared.weight
    """
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    init_emb = torch.zeros(vocab_size, d_model)

    # HG-Rec vocab layout (与 hab.py HAB_VOCAB_OFFSETS / HAB_VOCAB_K 一致)
    vocab_offsets = [1, 65, 193]  # L0, L1, L2
    K_list = [64, 128, 256]

    for l in range(3):
        cb_key = f"hrq.vq_layers.{l}.embeddings.weight"
        if cb_key not in sd:
            raise KeyError(f"Stage 2 ckpt 缺少 {cb_key}")
        cb = sd[cb_key].float()  # (K_l, d_tangent=32)
        K_l, d_tan = cb.shape
        assert K_l == K_list[l], f"L{l} codebook size mismatch: {K_l} vs {K_list[l]}"

        # Stage 2 forward 时会做 expmap0 (tangent → Poincaré ball), 但 ckpt 存的是 raw tangent.
        # 我们 init 用 raw tangent 还是 Poincaré ball? 用 raw tangent (避免 saturation 风险).
        # raw tangent (Euclidean d=32) → 重复 4x 到 d=128
        repeats = d_model // d_tan
        assert repeats * d_tan == d_model, f"d_model={d_model} must be divisible by d_tan={d_tan}"
        cb_padded = cb.repeat(1, repeats)  # (K_l, 128)

        # Scale 到 T5 default magnitude, 避免 SID 与其他 token magnitude 失衡
        if scale_to_t5_default:
            cur_norm = cb_padded.norm(dim=-1).mean().item()
            target_norm = 11.3  # T5 default init std=1, ||v|| ≈ sqrt(128) = 11.3
            scale = target_norm / max(cur_norm, 1e-7)
            cb_padded = cb_padded * scale

        offset = vocab_offsets[l]
        init_emb[offset:offset + K_l, :] = cb_padded

    # PAD [0], L3 [449] (dedup), 其他 [450..1024] 保持 0 init (T5 default 通常也是 small)
    # T5 default init 用 std=1 for non-PAD tokens. 这里我们只覆盖 SID, 其他留 0.
    # T5 forward 中 vocab [450..1024] 几乎不被命中 (训练数据只到 [449]), 0 init OK.

    print(f"[V7 init] codebook loaded: L0 cb.norm={sd['hrq.vq_layers.0.embeddings.weight'].norm(dim=-1).mean().item():.4f}")
    print(f"[V7 init] after padding: ||v||_L0={init_emb[1:65].norm(dim=-1).mean().item():.4f}")
    print(f"[V7 init] after padding: ||v||_L1={init_emb[65:193].norm(dim=-1).mean().item():.4f}")
    print(f"[V7 init] after padding: ||v||_L2={init_emb[193:449].norm(dim=-1).mean().item():.4f}")
    print(f"[V7 init] PAD [0] norm: {init_emb[0].norm().item():.4f}")

    return init_emb


class HG_Rec_Curv_V7(nn.Module):
    """HG_Rec + Stage 2 codebook init for T5 SID embedding — for v7.

    Curvature mechanism: Stage 2 双曲 codebook → Stage 3 T5 init transfer.
    不需要 HAB, 不需要 regularizer. 纯 CE training.
    """

    def __init__(self, config: Dict[str, Any], stage2_ckpt_path: str = None):
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

        # v7 核心: 用 Stage 2 codebook init T5 shared embedding
        if stage2_ckpt_path is not None:
            init_emb = init_t5_with_codebook(
                vocab_size=t5config.vocab_size,
                d_model=t5config.d_model,
                stage2_ckpt_path=stage2_ckpt_path,
                scale_to_t5_default=True,
                c=1.0,
            )
            with torch.no_grad():
                self.model.shared.weight.data.copy_(init_emb)
            # 同步 encoder/decoder embed_tokens
            self.model.encoder.embed_tokens = self.model.shared
            self.model.decoder.embed_tokens = self.model.shared
            self.model.lm_head.weight = self.model.shared.weight
            print("[V7] T5 SID embeddings initialized from Stage 2 codebook", flush=True)
        else:
            print("[V7] WARNING: no stage2_ckpt_path provided, using T5 default init", flush=True)

    @property
    def n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total = num_params(self.parameters())
        emb = num_params(self.model.shared.parameters())
        return (f"#Embedding params: {emb}\n"
                f"#Non-embedding params: {total - emb}\n"
                f"#Total trainable params: {total}\n")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits

    def generate(self, input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 num_beams: int = 20, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs,
        )