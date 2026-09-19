from __future__ import annotations

from functools import cached_property
from typing import List, NamedTuple

import torch
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from torch import nn, Tensor

from data.schemas import SeqBatch
from modules.encoder import MLP
from modules.hyperbolic import (
    _expmap0_t,
    _logmap0_t,
    _mobius_add_t,
    _transport_between_t,
)
from modules.loss import QuantizeLoss, ReconstructionLoss
from modules.normalize import l2norm
from modules.quantize import Quantize
from modules.step_checks import (
    check_step1_config,
    check_step4_model,
    check_step6_batch,
    check_step7_forward,
    check_step8_loss,
)


torch.set_float32_matmul_precision("high")


class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor
    per_layer_usage: Tensor
    sem_ids_shape: tuple = ()  # 仅记录 forward 内部 sem_ids 形状, 避免重跑 get_semantic_ids


class RqVae(nn.Module, PyTorchModelHubMixin):
    """按 Step1-Step8 组织的 Poincaré RQ-VAE。"""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        n_layers: int = 3,
        commitment_weight: float = 1.0,
        sk_eps: float = 0.05,
        sk_iters: int = 3,
        c_cyclic_min: float = 0.3,
        c_cyclic_max: float = 1.0,
        c_cyclic_period: int = 50_000,
        midpoint_layer_mask: List[bool] | None = None,
        per_layer_phase_offset_steps: List[int] | None = None,
    ) -> None:
        super().__init__()
        if midpoint_layer_mask is None:
            midpoint_layer_mask = [True] + [False] * (n_layers - 1)
        # iter27 (R36n b): per_layer_phase_offset_steps 默认 [0, 0, 0] 与 v318 baseline 等价
        # (所有层共享 phase). 启用 USE_PER_LAYER_HETERO_C 时传 [0, T/3, 2T/3] 让 3 层 c(t) 异质.
        if per_layer_phase_offset_steps is None:
            per_layer_phase_offset_steps = [0] * n_layers
        if len(per_layer_phase_offset_steps) != n_layers:
            raise ValueError(
                f"R36n b per_layer_phase_offset_steps 长度 {len(per_layer_phase_offset_steps)} ≠ n_layers {n_layers}"
            )
        for off in per_layer_phase_offset_steps:
            if off < 0:
                raise ValueError(f"R36n b per_layer_phase_offset_steps 不能为负: {off}")
        check_step1_config(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            codebook_size=codebook_size,
            n_layers=n_layers,
            c_min=c_cyclic_min,
            c_max=c_cyclic_max,
            c_period=c_cyclic_period,
            sk_eps=sk_eps,
            sk_iters=sk_iters,
        )
        if len(midpoint_layer_mask) != n_layers:
            raise ValueError("Step1: midpoint_layer_mask 长度必须等于 n_layers")

        self._config = {
            "input_dim": input_dim,
            "embed_dim": embed_dim,
            "hidden_dims": hidden_dims,
            "codebook_size": codebook_size,
            "n_layers": n_layers,
            "commitment_weight": commitment_weight,
            "sk_eps": sk_eps,
            "sk_iters": sk_iters,
            "c_cyclic_min": c_cyclic_min,
            "c_cyclic_max": c_cyclic_max,
            "c_cyclic_period": c_cyclic_period,
            "midpoint_layer_mask": midpoint_layer_mask,
        }
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.midpoint_layer_mask = [bool(value) for value in midpoint_layer_mask]

        self.layers = nn.ModuleList(
            [
                Quantize(
                    embed_dim=embed_dim,
                    n_embed=codebook_size,
                    do_kmeans_init=codebook_kmeans_init,
                    commitment_weight=commitment_weight,
                    sk_eps=sk_eps,
                    sk_iters=sk_iters,
                    c_cyclic_min=c_cyclic_min,
                    c_cyclic_max=c_cyclic_max,
                    c_cyclic_period=c_cyclic_period,
                    phase_offset_steps=per_layer_phase_offset_steps[layer_idx],
                )
                for layer_idx in range(n_layers)
            ]
        )
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=False,
        )
        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=False,
        )
        self.reconstruction_loss = ReconstructionLoss()
        check_step4_model(
            self,
            input_dim=input_dim,
            embed_dim=embed_dim,
            codebook_size=codebook_size,
            n_layers=n_layers,
            device=self.device,
        )

    @cached_property
    def config(self) -> dict:
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "model" not in state:
            raise KeyError("checkpoint 缺少 model 字段")
        self.load_state_dict(state["model"], strict=False)
        iter_value = state.get("iter", state.get("global_step", "unknown"))
        print(f"---Loaded RQVAE Iter {iter_value}---")
        check_step4_model(
            self,
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            codebook_size=self.codebook_size,
            n_layers=self.n_layers,
            device=self.device,
        )

    def set_curriculum_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("Step11: curriculum step 必须非负")
        for layer in self.layers:
            layer.set_curriculum_step(step)

    def encode(self, x: Tensor) -> Tensor:
        """Step2：Encoder 将输入 embedding 映射到 latent。"""
        check_step6_batch(x, self.input_dim, self.device)
        latent = self.encoder(x)
        if latent.shape != (x.shape[0], self.embed_dim):
            raise RuntimeError("Step2: Encoder 输出形状错误")
        if not torch.isfinite(latent).all().item():
            raise RuntimeError("Step2: Encoder 输出含 NaN 或 Inf")
        return latent

    def decode(self, x: Tensor) -> Tensor:
        """Step7：Decoder 从量化 embedding 重建输入。"""
        if x.ndim != 2 or x.shape[1] != self.embed_dim:
            raise ValueError("Step7: Decoder 输入形状错误")
        reconstruction = self.decoder(x)
        if reconstruction.shape != (x.shape[0], self.input_dim):
            raise RuntimeError("Step7: Decoder 输出形状错误")
        if not torch.isfinite(reconstruction).all().item():
            raise RuntimeError("Step7: Decoder 输出含 NaN 或 Inf")
        return reconstruction

    def _step3_quantize(self, residual: Tensor, layer_index: int):
        """Step3：对指定层 residual 执行 Poincaré/Sinkhorn/STE 量化。"""
        quantized = self.layers[layer_index](residual)
        if quantized.embeddings.shape != residual.shape:
            raise RuntimeError(f"Step3 layer {layer_index}: embedding 形状错误")
        if quantized.ids.shape != (residual.shape[0],):
            raise RuntimeError(f"Step3 layer {layer_index}: ids 形状错误")
        if not torch.isfinite(quantized.loss).all().item():
            raise RuntimeError(f"Step3 layer {layer_index}: loss 含 NaN 或 Inf")
        return quantized

    def _step4_m2_residual(
        self, residual: Tensor, embedding: Tensor, layer_index: int
    ) -> Tensor:
        """Step4：在 Poincaré 空间执行 M2 intrinsic residual。"""
        curvature = self.layers[layer_index].get_c().view(1, 1)
        if self.midpoint_layer_mask[layer_index]:
            log_residual = _logmap0_t(residual, curvature)
            log_embedding = _logmap0_t(embedding, curvature)
            next_residual = _expmap0_t(
                (log_residual + log_embedding) / 2.0, curvature
            )
        else:
            residual_h = _expmap0_t(residual, curvature)
            embedding_h = _expmap0_t(embedding, curvature)
            next_residual = _logmap0_t(
                _mobius_add_t(-embedding_h, residual_h, curvature), curvature
            )
        if next_residual.shape != residual.shape:
            raise RuntimeError(f"Step4 layer {layer_index}: residual 形状错误")
        if not torch.isfinite(next_residual).all().item():
            raise RuntimeError(f"Step4 layer {layer_index}: residual 含 NaN 或 Inf")
        return next_residual

    def _step5_transport(self, residual: Tensor, layer_index: int) -> Tensor:
        """Step5：通过原点执行 M3 跨层曲率传输。"""
        if layer_index == self.n_layers - 1:
            return residual
        current_curvature = self.layers[layer_index].get_c().view(1, 1)
        next_curvature = self.layers[layer_index + 1].get_c().view(1, 1)
        transported = _transport_between_t(
            residual, current_curvature, next_curvature
        )
        if not torch.isfinite(transported).all().item():
            raise RuntimeError(f"Step5 layer {layer_index}: transport 输出含 NaN 或 Inf")
        return transported

    def get_semantic_ids(self, x: Tensor) -> RqVaeOutput:
        """Step2-Step6：编码、逐层量化、残差更新并汇总 semantic IDs。"""
        x = x.to(self.device, dtype=next(self.encoder.parameters()).dtype)
        check_step6_batch(x, self.input_dim, self.device)
        residual = self.encode(x)
        quantize_loss = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        embeddings, residuals, semantic_ids = [], [], []

        for layer_index in range(self.n_layers):
            residuals.append(residual)
            quantized = self._step3_quantize(residual, layer_index)
            quantize_loss = quantize_loss + quantized.loss
            embedding = quantized.embeddings
            residual = self._step4_m2_residual(residual, embedding, layer_index)
            residual = self._step5_transport(residual, layer_index)
            semantic_ids.append(quantized.ids)
            embeddings.append(embedding)

        stacked_embeddings = torch.stack(embeddings, dim=0)            # (n_layers, batch, embed_dim)
        stacked_residuals = torch.stack(residuals, dim=0)              # (n_layers, batch, embed_dim)
        stacked_sem_ids = torch.stack(semantic_ids, dim=0)             # (n_layers, batch)
        output = RqVaeOutput(
            embeddings=stacked_embeddings.permute(0, 2, 1).contiguous(),  # (n_layers, embed_dim, batch)
            residuals=stacked_residuals.permute(0, 2, 1).contiguous(),    # (n_layers, embed_dim, batch)
            sem_ids=stacked_sem_ids.transpose(0, 1).contiguous(),         # (batch, n_layers)
            quantize_loss=quantize_loss,
        )
        check_step7_forward(
            output,
            x,
            n_layers=self.n_layers,
            codebook_size=self.codebook_size,
            embed_dim=self.embed_dim,
            device=self.device,
        )
        return output

    def _step6_sum_embeddings(self, quantized: RqVaeOutput) -> Tensor:
        """Step6：汇总所有层的量化 embedding。"""
        embeddings = quantized.embeddings.sum(dim=0).transpose(0, 1)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embed_dim:
            raise RuntimeError("Step6: 汇总 embedding 形状错误")
        if not torch.isfinite(embeddings).all().item():
            raise RuntimeError("Step6: 汇总 embedding 含 NaN 或 Inf")
        return embeddings

    def forward(self, batch: SeqBatch) -> RqVaeComputedLosses:
        """Step8：重建、计算总 loss 并检查计算图。"""
        check_step6_batch(batch.x, self.input_dim, self.device)
        quantized = self.get_semantic_ids(batch.x)
        summed_embeddings = self._step6_sum_embeddings(quantized)
        x_hat = l2norm(self.decode(summed_embeddings))
        # 当前所有量化层共享同一个 curriculum c(t)，重构项必须使用该曲率。
        reconstruction_curvature = self.layers[0].get_c().view(1, 1)
        reconstruction_loss = self.reconstruction_loss(
            x_hat, batch.x, c=reconstruction_curvature
        )
        rqvae_loss = quantized.quantize_loss
        loss = (reconstruction_loss + rqvae_loss).mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Step8: total loss 含 NaN 或 Inf")

        with torch.no_grad():
            embs_norm = quantized.embeddings.norm(dim=1)
            p_unique_ids = (
                ~torch.triu(
                    (
                        rearrange(quantized.sem_ids, "b d -> b 1 d")
                        == rearrange(quantized.sem_ids, "b d -> 1 b d")
                    ).all(axis=-1),
                    diagonal=1,
                )
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]
            per_layer_usage = torch.tensor(
                [quantized.sem_ids[:, i].unique().numel() for i in range(self.n_layers)],
                dtype=torch.long,
                device=quantized.sem_ids.device,
            )

        losses = RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstruction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
            per_layer_usage=per_layer_usage,
            sem_ids_shape=tuple(quantized.sem_ids.shape),
        )
        check_step8_loss(losses, self.training)
        return losses
