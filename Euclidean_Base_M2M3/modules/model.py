import torch
import torch.nn as nn
import torch.nn.functional as F

from data.schemas import TokenizedSeqBatch
from typing import NamedTuple, Optional
from torch import Tensor
from transformers import T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Config, T5Stack
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

torch.set_float32_matmul_precision("high")


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


class CurvatureStateInjector(nn.Module):
    """Issue #161+#162 迁移: response-confidence gate + layer curvature mixer.

    gate = sigmoid(MLP(s)), s = [c, boundary_distance, margin, residual_norm].
    mixer: a_t = softmax(W[q_pool, c0,c1,c2] / tau) → per-layer 注入权重 (L0/L1/L2).
    h_token = Emb(SID_l) + alpha * a_t[layer_l] * g_l,i * phi_l(c_l,i).
    alpha 零初始化 → 训练起点无注入 (与 Control 严格等价).
    """

    def __init__(self, d_model, feat_dim=4, hidden=16, alpha_init=0.0, tau=1.0, n_layers=3):
        super().__init__()
        self.proj = nn.Linear(1, d_model)  # phi: c → d_model
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Issue #162: layer mixer — 输入 [q_pool(2), c0,c1,c2(3)] → 3 层权重
        self.mixer = nn.Linear(5, n_layers)
        self.tau = tau
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self._last_gate_mean = None
        self._last_mixer_mean = None

    def forward(self, curvature, response, layer_ids):
        # curvature: (B, L), response: (B, L, feat_dim), layer_ids: (B, L) 0-2
        g = torch.sigmoid(self.gate_mlp(response))  # (B, L, 1)
        self._last_gate_mean = float(g.mean())
        c_feat = response[:, :, :3].mean(dim=1)  # (B, 3)
        q_pool = torch.full((response.shape[0], 2), 0.5, device=response.device)
        m_in = torch.cat([q_pool, c_feat], dim=-1)  # (B, 5)
        a = torch.softmax(self.mixer(m_in) / self.tau, dim=-1)  # (B, 3)
        self._last_mixer_mean = float(a.mean())
        a_t = a.gather(1, layer_ids.clamp(0, 2))  # (B, L) 每 token 层权重
        delta = self.alpha * a_t.unsqueeze(-1) * g * self.proj(curvature.unsqueeze(-1))
        return delta


def _strip_dedup_col(
    tensor: torch.Tensor, sem_ids_dim: int, n_layers: int
) -> torch.Tensor:
    """Strip the deduplication column appended by SemanticIdTokenizer.

    Args:
        tensor:      [B, N * sem_ids_dim]  where sem_ids_dim = n_layers + 1
        sem_ids_dim: tokens per item including the dedup column
        n_layers:    number of RQ-VAE codebook levels

    Returns:
        [B, N * n_layers]
    """
    B, total = tensor.shape
    N = total // sem_ids_dim
    return (
        tensor.view(B, N, sem_ids_dim)[:, :, :n_layers]
        .contiguous()
        .view(B, N * n_layers)
    )


class EncoderDecoderRetrievalModel(nn.Module):
    """HuggingFace T5 encoder-decoder for sequential recommendation.

    Uses T5EncoderModel for encoding and T5Stack for decoding. Per-hierarchy
    linear output heads project decoder hidden states to codebook logits.
    Beam search uses multinomial sampling with log-probability accumulation
    and a float("-inf") mask for invalid SID prefixes.
    """

    def __init__(
        self,
        codebooks: torch.Tensor,
        num_hierarchies: int,
        num_embeddings_per_hierarchy: int,
        t5_d_model: int = 128,
        t5_num_heads: int = 6,
        t5_d_ff: int = 1024,
        t5_num_layers: int = 4,
        t5_num_decoder_layers: Optional[int] = None,  # C31: HG-Rec 风格 encoder=6 decoder=4 (None=decoder=encoder)
        top_k_for_generation: int = 10,
        should_add_sep_token: bool = True,
        num_user_bins: Optional[int] = None,
        curv_state_path: Optional[str] = None,   # Issue #159: curvature_state.npy (n_items, D)
        response_path: Optional[str] = None,     # Issue #161: response.npy (n_items, D, 4)
        hab_rqvae_ckpt: Optional[str] = None,    # Issue #64/#162: HAB (stage2 RQ-VAE ckpt → Dbar)
    ):
        super().__init__()

        self.num_hierarchies = num_hierarchies
        self.num_embeddings_per_hierarchy = num_embeddings_per_hierarchy
        self.top_k_for_generation = top_k_for_generation
        self.register_buffer("codebooks", codebooks)

        # Issue #159/#161/#162: 曲率状态注入 (None = 关闭注入, 纯 baseline)
        if curv_state_path is not None and response_path is not None:
            import numpy as np
            curv = torch.from_numpy(np.load(curv_state_path).astype(np.float32))  # (n_items, D)
            resp = torch.from_numpy(np.load(response_path).astype(np.float32))    # (n_items, D, 4)
            self.register_buffer("curvature_state", curv)
            self.register_buffer("response_state", resp)
            self.curv_injector = CurvatureStateInjector(
                d_model=t5_d_model, feat_dim=4, hidden=16, alpha_init=0.0, tau=1.0,
                n_layers=num_hierarchies,
            )
            print(f"[model] curvature injection ON: curv={tuple(curv.shape)} resp={tuple(resp.shape)}")
        else:
            self.curv_injector = None
            self.curvature_state = None
            self.response_state = None

        encoder_config = T5Config(
            vocab_size=num_embeddings_per_hierarchy * num_hierarchies,
            d_model=t5_d_model,
            num_heads=t5_num_heads,
            d_ff=t5_d_ff,
            num_layers=t5_num_layers,
            is_decoder=False,
        )
        self.encoder = T5EncoderModel(encoder_config)

        # Issue #64/#162: HAB (Hyperbolic Attention Bias) — 从 stage2 ckpt 预计算 Dbar
        self.hab_module = None
        self._hab_layer_id_lut = None
        if hab_rqvae_ckpt is not None:
            from modules.hab import (
                load_hab_assets_from_stage2_ckpt as _load_hab,
                precompute_distance_matrices as _precompute,
                HyperbolicAttentionBias, install_hab, make_hab_layer_id_lut,
            )
            codebook_list, final_cs = _load_hab(hab_rqvae_ckpt)
            _D_list, Dbar_list, stats_list = _precompute(codebook_list, final_cs)
            print(f"[HAB] Dbar layers={len(Dbar_list)} final_cs={final_cs}")
            for _l, _st in enumerate(stats_list):
                print(f"  [HAB] L{_l}: c={_st['c']:.3f} finite={_st['finite']} med={_st['median']:.4f}")
            self.hab_module = HyperbolicAttentionBias(
                Dbar_list=Dbar_list, lambda_max=0.20, force_zero_layers=(),
                bias_rank=16, enable_residual=True, residual_alpha_init=-20.0,
            )
            # C7 (Issue #224 曲率扰动学习): Stage3 把 Stage2 冻结 final_cs 的 Dbar 加 per-layer 可微扰动
            # c_perturb_raw init 0 → 起点与 frozen baseline 严格等价; 启用后 Stage3 反向微调有效曲率 (R36)
            self.hab_module.c_perturb_enabled = True
            self.hab_module.c_perturb_raw.requires_grad_(True)
            self._hab_layer_id_lut = torch.from_numpy(make_hab_layer_id_lut(768)).long()
            install_hab(self, self.hab_module, self._hab_layer_id_lut.numpy())
            self._hab_inject_count = 0
            self._hab_attn_entropy_loss = None

        decoder_config = T5Config(
            vocab_size=num_embeddings_per_hierarchy * num_hierarchies,
            d_model=t5_d_model,
            num_heads=t5_num_heads,
            d_ff=t5_d_ff,
            num_layers=t5_num_decoder_layers if t5_num_decoder_layers is not None else t5_num_layers,  # C31: HG-Rec decoder=4, encoder=6
            is_decoder=True,
            is_encoder_decoder=False,
        )
        self.t5_decoder = T5Stack(decoder_config)
        self.bos_token = nn.Parameter(torch.randn(1, t5_d_model), requires_grad=True)
        self.decoder_mlp = nn.ModuleList(
            [
                nn.Linear(t5_d_model, num_embeddings_per_hierarchy, bias=False)
                for _ in range(num_hierarchies)
            ]
        )

        # Shared embedding table; hierarchy h, token t maps to index h * codebook_size + t.
        self.item_sid_embedding_table = nn.Embedding(
            num_embeddings=num_embeddings_per_hierarchy * num_hierarchies,
            embedding_dim=t5_d_model,
        )

        self.user_embedding = (
            nn.Embedding(num_user_bins, t5_d_model) if num_user_bins else None
        )
        self.sep_token = (
            nn.Parameter(torch.randn(1, t5_d_model), requires_grad=True)
            if should_add_sep_token
            else None
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _is_cache_valid(self, kv) -> bool:
        if isinstance(kv, (EncoderDecoderCache, DynamicCache)):
            return len(kv) > 0
        return isinstance(kv, tuple)

    def _add_repeating_offset_to_rows(
        self,
        input_sids: torch.Tensor,
        codebook_size: int,
        num_hierarchies: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add per-hierarchy offsets so a single embedding table covers all hierarchies."""
        if input_sids.ndim != 2:
            raise ValueError("Input tensor must be 2-dimensional.")
        _, num_cols = input_sids.shape
        offsets = (
            torch.arange(num_hierarchies, device=input_sids.device) * codebook_size
        )
        num_repeats = (num_cols + num_hierarchies - 1) // num_hierarchies
        repeated_offsets = offsets.repeat(num_repeats)[:num_cols]
        result = input_sids + repeated_offsets
        if attention_mask is not None:
            result = result * attention_mask
        return result

    def _inject_sep_token_between_sids(
        self,
        id_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sep_token: torch.Tensor,
        num_hierarchies: int,
    ):
        """Inject a separator embedding after each item's token group."""
        batch_size, seq_len, emb_dim = id_embeddings.size()
        item_count = seq_len // num_hierarchies
        reshaped_emb = id_embeddings.view(batch_size, item_count, num_hierarchies, -1)
        reshaped_mask = attention_mask.view(batch_size, item_count, num_hierarchies)
        sep = sep_token.unsqueeze(0).expand(batch_size, item_count, -1).unsqueeze(-2)
        id_embeddings = torch.cat([reshaped_emb, sep], dim=-2)
        attention_mask = torch.cat([reshaped_mask, reshaped_mask[:, :, [-1]]], dim=-1)
        return id_embeddings.reshape(batch_size, -1, emb_dim), attention_mask.reshape(
            batch_size, -1
        )

    def _check_valid_prefix(
        self, prefix: torch.Tensor, batch_size: int = 100000
    ) -> torch.Tensor:
        """Return a boolean mask indicating which prefixes exist in the corpus codebook."""
        if prefix.device != self.codebooks.device:
            self.codebooks = self.codebooks.to(prefix.device)
        trimmed = self.codebooks[:, : prefix.shape[1]]
        results = []
        for i in range(0, prefix.shape[0], batch_size):
            batch = prefix[i : i + batch_size]
            results.append(
                (trimmed.unsqueeze(1) == batch.unsqueeze(0)).all(dim=2).any(dim=0)
            )
        return torch.cat(results)

    def encoder_forward_pass(self, attention_mask, input_ids, user_id=None, item_ids=None):
        shifted = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds = self.item_sid_embedding_table(shifted)

        # Issue #159/#161/#162 迁移: 曲率状态注入 (α 零初始化 → 起点与 Control 严格等价)
        if self.curv_injector is not None and item_ids is not None:
            B, ND = input_ids.shape
            D = self.num_hierarchies
            N = ND // D
            curv = self.curvature_state[item_ids]  # (B, N, D)
            resp = self.response_state[item_ids]   # (B, N, D, 4)
            curv_flat = curv.reshape(B, ND)
            resp_flat = resp.reshape(B, ND, 4)
            # flat vocab token → 层号: token = h*codebook_size + 1 + c (offset 后)
            layer_ids = ((shifted - 1) // self.num_embeddings_per_hierarchy).clamp(0, D - 1)
            delta = self.curv_injector(curv_flat, resp_flat, layer_ids)
            inputs_embeds = inputs_embeds + delta

        if self.sep_token is not None:
            inputs_embeds, attention_mask = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds,
                attention_mask=attention_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        if user_id is not None and self.user_embedding is not None:
            user_embeds = self.user_embedding(
                torch.remainder(user_id[:, 0], self.user_embedding.num_embeddings)
            )
            inputs_embeds = torch.cat([user_embeds.unsqueeze(1), inputs_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones(attention_mask.size(0), 1, device=attention_mask.device),
                    attention_mask,
                ],
                dim=1,
            )

        # Issue #64 HAB: 外部预计算 B_geo (SID 部分) + 扩展 sep/user 行列 (0 偏置)
        if self.hab_module is not None:
            B_geo_sub = self.hab_module.get_B_geo(
                shifted, self._hab_layer_id_lut.to(shifted.device),
                attention_mask_2d=attention_mask[:, : shifted.shape[1]],
            )  # (B, 1, N*D, N*D)
            L_full = inputs_embeds.shape[1]
            L_sub = shifted.shape[1]
            B_geo_full = torch.zeros(
                B_geo_sub.shape[0], 1, L_full, L_full,
                device=B_geo_sub.device, dtype=torch.float32,
            )
            B_geo_full[:, :, :L_sub, :L_sub] = B_geo_sub
            self._hab_external_B_geo = B_geo_full
        else:
            self._hab_external_B_geo = None

        encoder_output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        return encoder_output, attention_mask

    def decoder_forward_pass(
        self,
        attention_mask=None,
        future_ids=None,
        encoder_output=None,
        attention_mask_for_encoder=None,
        use_cache=False,
        past_key_values=None,
    ):
        if future_ids is not None:
            shifted = self._add_repeating_offset_to_rows(
                input_sids=future_ids,
                codebook_size=self.num_embeddings_per_hierarchy,
                num_hierarchies=self.num_hierarchies,
                attention_mask=torch.ones_like(future_ids)
                if attention_mask is None
                else attention_mask,
            )
            inputs_embeds = self.item_sid_embedding_table(shifted)

            if not self._is_cache_valid(past_key_values):
                bos = self.bos_token.unsqueeze(0).expand(future_ids.size(0), 1, -1)
                inputs_embeds = torch.cat([bos, inputs_embeds], dim=1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            torch.ones(future_ids.size(0), 1, device=future_ids.device),
                            attention_mask,
                        ],
                        dim=1,
                    )
            else:
                inputs_embeds = inputs_embeds[:, -1:, :]
        else:
            inputs_embeds = self.bos_token.unsqueeze(0).expand(
                encoder_output.size(0), 1, -1
            )

        out = self.t5_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=attention_mask_for_encoder,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        if use_cache:
            return out.last_hidden_state, out.past_key_values
        return out.last_hidden_state

    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        sem_ids_dim = self.num_hierarchies + 1
        input_ids = _strip_dedup_col(batch.sem_ids, sem_ids_dim, self.num_hierarchies)
        attention_mask = _strip_dedup_col(
            batch.seq_mask.long(), sem_ids_dim, self.num_hierarchies
        )
        fut_ids = batch.sem_ids_fut[:, : self.num_hierarchies]

        encoder_output, attention_mask_for_encoder = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=batch.user_ids,
            item_ids=getattr(batch, "item_ids", None),
        )
        decoder_output = self.decoder_forward_pass(
            future_ids=fut_ids,
            encoder_output=encoder_output,
            attention_mask_for_encoder=attention_mask_for_encoder,
            use_cache=False,
        )[:, :-1]  # [B, num_hierarchies, d_model]

        total_loss = torch.tensor(0.0, device=decoder_output.device)
        loss_d = []
        for h in range(self.num_hierarchies):
            logits = self.decoder_mlp[h](decoder_output[:, h])
            h_loss = F.cross_entropy(logits, fut_ids[:, h].long())
            total_loss = total_loss + h_loss
            loss_d.append(h_loss.detach())

        return ModelOutput(loss=total_loss, logits=None, loss_d=torch.stack(loss_d))

    @torch.no_grad()
    def generate(self, attention_mask, input_ids, user_id=None, item_ids=None):
        """Generate top-k semantic IDs using deterministic beam search.

        For each hierarchy level, expands the top-n_cands tokens by probability
        (deterministic, 非 multinomial 采样), scores them using cumulative
        log-probabilities with a float("-inf") mask for invalid SID prefixes,
        and keeps the top-k highest-scoring candidates (R35 同口径 beam).

        Returns:
            generated_ids: [B, top_k, num_hierarchies]
            log_probas:    [B, top_k]
        """
        B = input_ids.size(0)
        k = self.top_k_for_generation
        n_cands = min(64, self.num_embeddings_per_hierarchy)

        enc_out, enc_mask = self.encoder_forward_pass(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
            item_ids=item_ids if self.curv_injector is not None else None,
        )
        rep_enc = enc_out.repeat_interleave(k, dim=0)
        rep_mask = enc_mask.repeat_interleave(k, dim=0)

        generated = None  # [B, k, h] grows with each hierarchy step
        log_probas = 0
        past_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())

        for h in range(self.num_hierarchies):
            if generated is not None:
                cur_enc, cur_mask = rep_enc, rep_mask
                squeezed = generated.reshape(-1, h)
            else:
                cur_enc, cur_mask = enc_out, enc_mask
                squeezed = None

            dec_out, past_kv = self.decoder_forward_pass(
                future_ids=squeezed,
                encoder_output=cur_enc,
                attention_mask_for_encoder=cur_mask,
                use_cache=True,
                past_key_values=past_kv,
            )

            probas = F.softmax(self.decoder_mlp[h](dec_out[:, -1, :]), dim=-1)
            # B3: 确定性 beam 展开 (top-n_cands 概率, 非 multinomial 采样) — R35 同口径
            samples = probas.topk(n_cands, dim=-1).indices
            samp_log_p = torch.log(torch.gather(probas, 1, samples))

            if generated is None:
                is_valid = self._check_valid_prefix(samples.reshape(-1, 1)).reshape(
                    B, n_cands
                )
                scores, idx = samp_log_p.masked_fill(~is_valid, float("-inf")).sort(
                    -1, descending=True
                )
                top_k_idx = idx[:, :k]
                generated = torch.gather(samples, 1, top_k_idx).unsqueeze(
                    -1
                )  # [B, k, 1]
                log_probas = scores[:, :k]
                past_kv = EncoderDecoderCache(DynamicCache(), DynamicCache())
            else:
                prev = generated.reshape(-1, h).repeat_interleave(n_cands, dim=0)
                prefix = torch.cat([prev, samples.reshape(-1, 1)], dim=1)
                is_valid = self._check_valid_prefix(prefix).reshape(B, k * n_cands)
                scores, idx = (
                    (
                        samp_log_p.reshape(B, k * n_cands)
                        + log_probas.repeat_interleave(n_cands, dim=1)
                    )
                    .masked_fill(~is_valid, float("-inf"))
                    .sort(-1, descending=True)
                )

                top_k_idx = idx[:, :k]
                parent_beam_idx = top_k_idx // n_cands
                parent_global = (
                    parent_beam_idx
                    + torch.arange(B, device=parent_beam_idx.device).unsqueeze(1) * k
                ).flatten()
                past_kv.reorder_cache(parent_global)

                parent_ids = torch.gather(
                    generated, 1, parent_beam_idx.unsqueeze(-1).expand(-1, -1, h)
                )
                new_ids = torch.gather(
                    samples.reshape(B, k * n_cands), 1, top_k_idx
                ).unsqueeze(-1)
                generated = torch.cat([parent_ids, new_ids], dim=-1)  # [B, k, h+1]
                log_probas = scores[:, :k]

        return generated, log_probas

    @torch.no_grad()
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        top_k: bool = True,
        temperature: int = 1,
    ) -> GenerationOutput:
        sem_ids_dim = self.num_hierarchies + 1
        input_ids = _strip_dedup_col(batch.sem_ids, sem_ids_dim, self.num_hierarchies)
        attention_mask = _strip_dedup_col(
            batch.seq_mask.long(), sem_ids_dim, self.num_hierarchies
        )
        generated_ids, log_probas = self.generate(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=batch.user_ids,
            item_ids=getattr(batch, "item_ids", None),
        )
        return GenerationOutput(sem_ids=generated_ids, log_probas=log_probas)
