"""
Decoder-only variant of SemanticIDGenerativeRecommender for the GRID paper's
Section 4.2 Table 5 ablation (TIGER Encoder-Decoder vs Decoder-Only).

This module replaces the T5 encoder-decoder backbone in
SemanticIDEncoderDecoder with a single causal transformer
(transformers.GPT2Model). The forward path concatenates a user history
prefix with a [BOS] + future_id (teacher-forced) suffix; generation runs
hierarchically by appending one predicted token at a time and re-feeding
the entire sequence through GPT-2.

The shape contract of `forward` matches SemanticIDEncoderDecoder so that
the inherited `model_step` (`model_output[:, :-1]` then per-hierarchy
projection -> CE loss) works unchanged:

    forward(...) -> (batch, num_hierarchies + 1, embed_dim)

where the trailing `+1` is the encoder-side BOS slot the base class
expects.
"""

import logging
from typing import Any, Optional, Tuple

import torch
import transformers
from torch import nn
from transformers.cache_utils import DynamicCache

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.interfaces import OneKeyPerPredictionOutput
from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDGenerativeRecommender,
)
from src.utils.utils import reset_parameters


class SemanticIDDecoderOnly(SemanticIDGenerativeRecommender):
    """
    Decoder-only implementation of GRID/TIGER, mirroring TIGER paper Figure 2.b
    but with the encoder removed: a single GPT-2 causal transformer attends
    over the user history prefix and then auto-regressively emits the
    num_hierarchies SID tokens.

    Args:
        top_k_for_generation (int): beam width at generation time.
        codebooks (torch.Tensor): (num_hierarchies, vocab_per_codebook) codebook
            table; used by the base class.
        embedding_dim (int): hidden dim; must match GPT-2 config n_embd.
        num_hierarchies (int): how many SID tokens per item (4 for our setup).
        num_embeddings_per_hierarchy (int): vocabulary size per hierarchy (256).
        num_user_bins (Optional[int]): number of user embeddings to spawn;
            None disables user conditioning.
        should_check_prefix (bool): enable codebook prefix pruning during beam
            search (passed through to base class beam search).
        should_add_sep_token (bool): whether to insert a learned SEP token
            between items in the history sequence.
        prediction_key_name, prediction_value_name: keys for the output dict.
    """

    def __init__(
        self,
        top_k_for_generation: int = 10,
        codebooks: torch.Tensor = None,
        embedding_dim: int = None,
        num_hierarchies: int = None,
        num_embeddings_per_hierarchy: int = None,
        num_user_bins: Optional[int] = None,
        should_check_prefix: bool = False,
        should_add_sep_token: bool = True,
        prediction_key_name: str = "user_id",
        prediction_value_name: str = "semantic_ids",
        **kwargs,
    ) -> None:
        # Derive hierarchy/vocab from codebook shape if not provided.
        if num_hierarchies is None or num_embeddings_per_hierarchy is None:
            num_hierarchies, num_embeddings_per_hierarchy = (
                codebooks.shape[0],
                codebooks.max().item() + 1,
            )
        # GPT-2 has no T5-style q.in_features probe, so embedding_dim must be
        # passed explicitly via Hydra (== huggingface_model.config.n_embd).
        if embedding_dim is None:
            embedding_dim = kwargs["huggingface_model"].config.n_embd

        # Drop T5-specific kwargs that may travel through Hydra so we don't
        # forward them to the base TransformerBaseModule, which would raise.
        kwargs.pop("mlp_layers", None)
        kwargs.pop("decoder", None)

        super().__init__(
            codebooks=codebooks,
            num_hierarchies=num_hierarchies,
            num_embeddings_per_hierarchy=num_embeddings_per_hierarchy,
            embedding_dim=embedding_dim,
            top_k_for_generation=top_k_for_generation,
            should_check_prefix=should_check_prefix,
            **kwargs,
        )

        # TransformerBaseModule.__init__ assigned `self.encoder = huggingface_model`.
        # For decoder-only we alias the GPT-2 backbone and do NOT use a separate
        # decoder module, so we clear `self.decoder` if it was set.
        self.gpt2 = self.encoder
        self.decoder = None
        # SDPA's causal+padding mask conversion triggers CUDA asserts on some
        # configurations; using the deterministic eager attention path avoids
        # them entirely at negligible perf cost at our batch sizes.
        if hasattr(self.gpt2, "config"):
            self.gpt2.config._attn_implementation = "eager"
        reset_parameters(self.gpt2)

        # Learned BOS token (analogous to SemanticIDEncoderDecoder.decoder.bos_token).
        # Initialised with a small scale so logits stay well-behaved early in training.
        self.bos_token = torch.nn.Parameter(
            torch.randn(1, self.embedding_dim) * 0.02, requires_grad=True
        )

        # Per-hierarchy logits projection. Mirrors decoder_mlp in enc-dec.
        self.hierarchy_mlp: torch.nn.ModuleList = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.embedding_dim,
                    self.num_embeddings_per_hierarchy,
                    bias=False,
                )
                for _ in range(self.num_hierarchies)
            ]
        )

        # Shared item SID embedding table: SID[t, h] is stored at offset
        # t * num_embeddings_per_hierarchy + offset[h], matching
        # SemanticIDEncoderDecoder.item_sid_embedding_table_encoder.
        self.item_sid_embedding_table = self._spawn_embedding_tables(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies,
            embedding_dim=self.embedding_dim,
        )

        # Optional user embedding (gated by num_user_bins).
        self.user_embedding: Optional[torch.nn.Embedding] = (
            self._spawn_embedding_tables(
                num_embeddings=num_user_bins,
                embedding_dim=self.embedding_dim,
            )
            if num_user_bins
            else None
        )

        # Optional learned separator between items in the history sequence.
        self.sep_token: Optional[torch.nn.Parameter] = (
            torch.nn.Parameter(
                torch.randn(1, self.embedding_dim) * 0.02, requires_grad=True
            )
            if should_add_sep_token
            else None
        )

        self.prediction_key_name = prediction_key_name
        self.prediction_value_name = prediction_value_name

    # ------------------------------------------------------------------ #
    # History embedding construction
    # ------------------------------------------------------------------ #
    def build_history_prefix(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the per-sequence history prefix that will be prepended to
        the next-item prediction positions.

        Args:
            attention_mask: (batch, seq_len) over the flat SID sequence.
            input_ids: (batch, seq_len) SID ids.
            user_id: (batch, ...) optional user ids.

        Returns:
            inputs_embeds: (batch, prefix_len, embed_dim).
            new_mask: (batch, prefix_len) attention mask for GPT-2.
        """
        shifted_sids = self._add_repeating_offset_to_rows(
            input_sids=input_ids,
            codebook_size=self.num_embeddings_per_hierarchy,
            num_hierarchies=self.num_hierarchies,
            attention_mask=attention_mask,
        )
        inputs_embeds = self.item_sid_embedding_table(shifted_sids)
        # Zero out padded positions so they don't contribute gradient/loss.
        inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1).to(
            inputs_embeds.dtype
        )

        new_mask = attention_mask
        if self.sep_token is not None:
            inputs_embeds, new_mask = self._inject_sep_token_between_sids(
                id_embeddings=inputs_embeds,
                attention_mask=new_mask,
                sep_token=self.sep_token,
                num_hierarchies=self.num_hierarchies,
            )

        if user_id is not None and self.user_embedding is not None:
            user_id = user_id[:, 0]
            user_embeds = self.user_embedding(
                torch.remainder(user_id, self.user_embedding.num_embeddings)
            )
            inputs_embeds = torch.cat(
                [user_embeds.unsqueeze(1), inputs_embeds], dim=1
            )
            new_mask = torch.cat(
                [
                    torch.ones(
                        new_mask.size(0), 1, device=new_mask.device
                    ),
                    new_mask,
                ],
                dim=1,
            )

        return inputs_embeds, new_mask

    # ------------------------------------------------------------------ #
    # Forward / generation
    # ------------------------------------------------------------------ #
    def _assemble_full_sequence(
        self,
        history_embeds: torch.Tensor,
        history_mask: torch.Tensor,
        future_ids: Optional[torch.Tensor],
        repeat_factor: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Concatenate history prefix with the [BOS] + future_ids[-1] suffix
        used to derive per-hierarchy hidden states.

        Returns (full_inputs_embeds, full_attention_mask, history_length).
        The returned full sequence length is history_length + num_hierarchies + 1.
        """
        if repeat_factor > 1:
            history_embeds = history_embeds.repeat_interleave(repeat_factor, dim=0)
            history_mask = history_mask.repeat_interleave(repeat_factor, dim=0)

        if future_ids is not None:
            # Teacher-forced training path: prepend BOS, drop last fut token
            # so total suffix length == num_hierarchies (the model returns
            # num_hierarchies + 1 and the +1 is sliced off by model_step).
            shifted_future = self._add_repeating_offset_to_rows(
                input_sids=future_ids,
                codebook_size=self.num_embeddings_per_hierarchy,
                num_hierarchies=self.num_hierarchies,
                attention_mask=torch.ones_like(future_ids),
            )
            future_embeds = self.item_sid_embedding_table(shifted_future)
            # Keep first (num_hierarchies - 1) tokens; model_step drops the last.
            suffix_embeds = future_embeds[:, : self.num_hierarchies - 1, :]
        else:
            # Inference path: only the BOS token is fed at this stage.
            suffix_embeds = None

        bos_emb = self.bos_token.unsqueeze(0).expand(
            history_embeds.size(0), 1, -1
        )
        if suffix_embeds is None:
            decoder_prefix = bos_emb  # length 1
        else:
            decoder_prefix = torch.cat([bos_emb, suffix_embeds], dim=1)

        full_embeds = torch.cat([history_embeds, decoder_prefix], dim=1)
        full_mask = torch.cat(
            [
                history_mask,
                torch.ones(
                    history_embeds.size(0),
                    decoder_prefix.size(1),
                    device=history_mask.device,
                    dtype=history_mask.dtype,
                ),
            ],
            dim=1,
        )
        return full_embeds, full_mask, history_embeds.size(1)

    def forward(
        self,
        attention_mask_encoder: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
        future_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder-only model.

        Returns the last (num_hierarchies + 1) hidden states from GPT-2,
        matching SemanticIDEncoderDecoder's contract so the inherited
        `model_step` (drop-last slice + per-hierarchy projection + CE loss)
        can be applied unchanged.
        """
        history_embeds, history_mask = self.build_history_prefix(
            attention_mask=attention_mask_encoder,
            input_ids=input_ids,
            user_id=user_id,
        )

        full_embeds, full_mask, _ = self._assemble_full_sequence(
            history_embeds=history_embeds,
            history_mask=history_mask,
            future_ids=future_ids,
        )


        gpt2_output = self.gpt2(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
        )
        all_hidden = gpt2_output.last_hidden_state
        suffix_len = self.num_hierarchies
        decoder_hiddens = all_hidden[:, -suffix_len:]
        pad_row = decoder_hiddens[:, -1:, :]
        return torch.cat([decoder_hiddens, pad_row], dim=1)

    # ------------------------------------------------------------------ #
    # Beam-search generation
    # ------------------------------------------------------------------ #
    def _beam_search_one_step_dec_only(
        self,
        candidate_logits: torch.Tensor,
        generated_ids: Optional[torch.Tensor],
        marginal_log_prob: Optional[torch.Tensor],
        past_key_values: DynamicCache,
        hierarchy: int,
        batch_size: int,
    ):
        """
        Override of `_beam_search_one_step` for the decoder-only path.

        Differs from the base class in two ways:
        1. On the first hierarchy step (`generated_ids is None`), we KEEP the
           DynamicCache we built up by encoding the history prefix. The base
           class's enc-dec path replaces the cache with a fresh
           EncoderDecoderCache because enc-dec keeps self-attn and cross-attn
           caches separate and only the cross-attn cache is meaningful for
           the first prediction. For dec-only, our self-attn cache IS the
           only cache and we want to preserve it.
        2. We use `DynamicCache.reorder_cache` (no EncoderDecoderCache
           wrapper), and we feed `beam_idx` to extend the cache after picking
           the top-k surviving beams.
        """
        if self.should_check_prefix:
            if generated_ids is None:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.arange(
                        self.num_embeddings_per_hierarchy,
                        device=candidate_logits.device,
                    ).unsqueeze(1)
                )
                candidate_logits[:, ~valid_prefix_mask] = float("-inf")
            else:
                valid_prefix_mask = self._check_valid_prefix(
                    torch.cat(
                        [
                            generated_ids.reshape(-1, hierarchy).repeat_interleave(
                                self.num_embeddings_per_hierarchy, dim=0
                            ),
                            torch.arange(
                                self.num_embeddings_per_hierarchy,
                                device=candidate_logits.device,
                            )
                            .repeat(self.top_k_for_generation * batch_size)
                            .unsqueeze(1),
                        ],
                        dim=1,
                    )
                ).reshape(-1, self.num_embeddings_per_hierarchy)
            candidate_logits[~valid_prefix_mask] = float("-inf")

        candidate_logits = torch.nn.functional.softmax(candidate_logits, dim=-1)
        proba, indices = torch.sort(candidate_logits, descending=True)

        if generated_ids is None:
            proba_topk, indices_topk = (
                proba[:, : self.top_k_for_generation],
                indices[:, : self.top_k_for_generation],
            )
            generated_ids = indices_topk.unsqueeze(-1)
            replace_indices = None
        else:
            proba, indices = (
                proba[:, : self.num_embeddings_per_hierarchy],
                indices[:, : self.num_embeddings_per_hierarchy],
            )
            proba = proba.reshape(
                -1, self.top_k_for_generation * self.num_embeddings_per_hierarchy
            )
            indices = indices.reshape(
                -1, self.top_k_for_generation * self.num_embeddings_per_hierarchy
            )
            proba = torch.mul(
                marginal_log_prob.repeat_interleave(
                    self.num_embeddings_per_hierarchy, dim=-1
                ),
                proba,
            )
            topk_results = torch.topk(
                torch.nan_to_num(proba, nan=-1), k=self.top_k_for_generation, dim=-1
            )
            proba_topk, indices_topk = topk_results.values, topk_results.indices
            replace_indices = (
                (indices_topk // self.num_embeddings_per_hierarchy)
                + torch.arange(indices_topk.size(0), device=proba.device).unsqueeze(1)
                * self.top_k_for_generation
            ).flatten()
            if past_key_values is not None:
                past_key_values.reorder_cache(replace_indices)
            indices_topk = torch.gather(indices, 1, indices_topk)

        if replace_indices is not None:
            generated_ids = torch.cat(
                [
                    generated_ids.reshape(-1, hierarchy)[replace_indices].reshape(
                        -1, self.top_k_for_generation, hierarchy
                    ),
                    indices_topk.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            generated_ids = indices_topk.unsqueeze(-1)

        return generated_ids, proba_topk, past_key_values

    def generate(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        user_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchically generate SIDs using the base class's
        `_beam_search_one_step` over `DynamicCache` (replacing the
        `EncoderDecoderCache` the base class assumes).

        Note: GPT-2's modeling code in transformers v5.8 still uses the legacy
        `Tuple[Tuple[Tensor]]` cache format internally. We therefore bridge
        between DynamicCache (used for `reorder_cache`/`batch_repeat_interleave`
        during beam search) and the legacy tuple (consumed/produced by GPT-2).
        """
        history_embeds, history_mask = self.build_history_prefix(
            attention_mask=attention_mask,
            input_ids=input_ids,
            user_id=user_id,
        )

        # Encode the entire history prefix once. First call: no cache.
        history_output = self.gpt2(
            inputs_embeds=history_embeds,
            attention_mask=history_mask,
            past_key_values=None,
            use_cache=True,
        )
        # Convert legacy tuple cache -> DynamicCache for beam-search helpers.
        past_key_values = DynamicCache.from_legacy_cache(
            history_output.past_key_values
        )

        bos_emb = self.bos_token.unsqueeze(0).expand(
            history_embeds.size(0), 1, -1
        )
        bos_output = self.gpt2(
            inputs_embeds=bos_emb,
            past_key_values=past_key_values.to_legacy_cache(),
            use_cache=True,
        )
        past_key_values = DynamicCache.from_legacy_cache(
            bos_output.past_key_values
        )

        generated_ids: Optional[torch.Tensor] = None
        marginal_log_prob: Optional[torch.Tensor] = None

        for hierarchy in range(self.num_hierarchies):
            if generated_ids is not None:
                squeezed_generated_ids = generated_ids.reshape(
                    -1, hierarchy
                ).to(history_embeds.device)

                # Only expand the cache from bsz -> bsz*top_k ONCE, at h=1.
                # After h=1 the cache is already at bsz*top_k and the inputs
                # to the next step are also at bsz*top_k; re-expanding would
                # produce bsz*top_k*top_k which mismatches the input batch dim.
                if hierarchy == 1:
                    past_key_values.batch_repeat_interleave(
                        self.top_k_for_generation
                    )

                latest_token = squeezed_generated_ids[:, -1:]
                shifted_latest = self._add_repeating_offset_to_rows(
                    input_sids=latest_token,
                    codebook_size=self.num_embeddings_per_hierarchy,
                    num_hierarchies=1,
                    attention_mask=torch.ones_like(latest_token),
                )
                latest_emb = self.item_sid_embedding_table(shifted_latest)

                step_output = self.gpt2(
                    inputs_embeds=latest_emb,
                    past_key_values=past_key_values.to_legacy_cache(),
                    use_cache=True,
                )
                past_key_values = DynamicCache.from_legacy_cache(
                    step_output.past_key_values
                )
                latest_output_representation = step_output.last_hidden_state[
                    :, -1, :
                ]
            else:
                latest_output_representation = bos_output.last_hidden_state[
                    :, -1, :
                ]

            candidate_logits = self.hierarchy_mlp[hierarchy](
                latest_output_representation
            )

            (
                generated_ids,
                marginal_log_prob,
                past_key_values,
            ) = self._beam_search_one_step_dec_only(
                candidate_logits=candidate_logits,
                generated_ids=generated_ids,
                marginal_log_prob=marginal_log_prob,
                past_key_values=past_key_values,
                hierarchy=hierarchy,
                batch_size=input_ids.size(0),
            )

        return generated_ids, marginal_log_prob

    # ------------------------------------------------------------------ #
    # Lightning plumbing required by the base class
    # ------------------------------------------------------------------ #
    def get_embedding_table(self, table_name: str = None, hierarchy: int = None):
        # Override the base class so weight_tying does not reach into the
        # GPT-2 token embedding. We always use our SID embedding table.
        if hierarchy is not None:
            return self.item_sid_embedding_table(
                torch.arange(
                    hierarchy * self.num_embeddings_per_hierarchy,
                    (hierarchy + 1) * self.num_embeddings_per_hierarchy,
                ).to(self.device)
            )
        return self.item_sid_embedding_table

    def predict_step(self, batch: SequentialModelInputData):
        generated_sids, _ = self.model_step(batch)
        ids = [
            id.item() if isinstance(id, torch.Tensor) else id
            for id in batch.user_id_list
        ]
        model_output = OneKeyPerPredictionOutput(
            keys=ids,
            predictions=generated_sids,
            key_name=self.prediction_key_name,
            prediction_name=self.prediction_value_name,
        )
        return model_output

    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if label_data is None:
            generated_ids, marginal_probs = self.generate(
                attention_mask=model_input.mask,
                **{
                    self.feature_to_model_input_map.get(k, k): v
                    for k, v in model_input.transformed_sequences.items()
                },
            )
            return generated_ids, torch.tensor(0.0, device=model_input.mask.device)

        fut_ids = None
        for label in label_data.labels:
            curr_label = label_data.labels[label]
            fut_ids = curr_label.reshape(model_input.mask.size(0), -1)
        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=fut_ids,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )
        model_output = model_output[:, :-1]

        loss = torch.tensor(0.0, device=model_output.device)
        for hierarchy in range(self.num_hierarchies):
            head_input = self.hierarchy_mlp[hierarchy](model_output[:, hierarchy])
            loss = loss + self.loss_function(
                input=head_input,
                target=fut_ids[:, hierarchy].long(),
            )
        return model_output, loss

    def _make_deterministic(self, is_training: bool) -> None:
        if is_training:
            self.gpt2.train()
        else:
            self.gpt2.eval()

    def on_predict_start(self):
        super().on_predict_start()
        self._make_deterministic(is_training=False)

    def on_predict_end(self):
        super().on_predict_end()
        self._make_deterministic(is_training=True)

    def on_validation_start(self):
        super().on_validation_start()
        self._make_deterministic(is_training=False)

    def on_validation_end(self):
        super().on_validation_end()
        self._make_deterministic(is_training=True)

    def on_test_start(self):
        super().on_test_start()
        self._make_deterministic(is_training=False)

    def on_test_end(self):
        super().on_test_end()
        self._make_deterministic(is_training=True)

    def on_train_start(self):
        super().on_train_start()
        self._make_deterministic(is_training=True)
