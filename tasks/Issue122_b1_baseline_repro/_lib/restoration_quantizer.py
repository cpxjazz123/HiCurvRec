# Task #292 — Restoration (EMA + dead code revival) wrapper for FreeCurvHRQVAE
# 2026-07-29
#
# 设计:
#   - 包装现有 FreeCurvResidualVectorQuantization (不污染 src/)
#   - EMA codebook update (跟 Task #291 一样)
#   - + 自适应 dead code revival (修复 hrqvae_trainer.py:271 latent_gravy=empty no-op bug)
#   - 强制 dead code (usage=0) 用 batch 中随机 latent 替换 → 保持 L0 高
#
# 来源:
#   - Restoration paper: Yu et al. 2022 (https://arxiv.org/abs/2203.11522)
#   - 修复 Task #283 发现的 dead_revive hook no-op bug
#
# 跟 Task #291 EMA 区别:
#   - Task #291 只 EMA
#   - Task #292 EMA + dead code revival (定期强制替换 dead codes)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .hrqvae_free_curv import FreeCurvResidualVectorQuantization
from .ema_codebook_quantizer import EMAQuantizerWrapper


class RestorationQuantizerWrapper(nn.Module):
    """EMA + adaptive dead code revival wrapper.

    Dead code (usage=0) is replaced with random batch latents every `revive_freq` epochs.
    """

    def __init__(self, base_rq: FreeCurvResidualVectorQuantization,
                 decay: float = 0.99,
                 eps: float = 1e-5,
                 revive_threshold: int = 1,  # usage < threshold → dead
                 revive_ratio: float = 0.1,  # replace ratio
                 revive_freq_epochs: int = 5):
        super().__init__()
        # Reuse EMA wrapper
        self.ema_wrapper = EMAQuantizerWrapper(base_rq, decay, eps)
        self.base_rq = base_rq
        self.revive_threshold = revive_threshold
        self.revive_ratio = revive_ratio
        self.revive_freq_epochs = revive_freq_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Called by trainer to update current_epoch."""
        self.current_epoch = epoch

    @torch.no_grad()
    def _revive_dead_codes(self, x_flat: torch.Tensor):
        """Replace dead codes (usage < threshold) with random batch latents."""
        for m, vq in enumerate(self.base_rq.vq_layers):
            K = vq.n_e  # FreeCurvVQ 用 n_e
            e_dim = vq.e_dim
            # cluster_size from EMA buffer
            cluster_size = self.ema_wrapper.ema_cluster_size[m, :K]
            dead_mask = cluster_size < self.revive_threshold  # (K,)
            n_dead = dead_mask.sum().item()
            if n_dead == 0:
                continue
            # Cap by revive_ratio
            n_replace = min(n_dead, int(K * self.revive_ratio))
            if n_replace == 0:
                continue
            # Pick n_replace dead codeword indices
            dead_indices = torch.where(dead_mask)[0]
            if len(dead_indices) > n_replace:
                perm = torch.randperm(len(dead_indices), device=dead_indices.device)[:n_replace]
                dead_indices = dead_indices[perm]
            # Random batch latents (sample without replacement)
            n_samples = min(len(x_flat), n_replace)
            if n_samples < n_replace:
                # reuse samples if batch too small
                idx_samples = torch.randint(0, len(x_flat), (n_replace,), device=x_flat.device)
            else:
                perm = torch.randperm(len(x_flat), device=x_flat.device)[:n_replace]
                idx_samples = perm
            # Replace dead codes with batch latents
            vq.embeddings.weight.data[dead_indices] = x_flat[idx_samples]

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        """Forward + EMA + (every revive_freq_epochs) dead code revival."""
        x_q_st, total_loss, all_indices = self.ema_wrapper(x, use_sk=use_sk)

        if self.training:
            # Revive dead codes every revive_freq_epochs
            if self.current_epoch > 0 and self.current_epoch % self.revive_freq_epochs == 0:
                x_flat = x.view(-1, self.base_rq.vq_layers[0].e_dim)
                self._revive_dead_codes(x_flat)

        return x_q_st, total_loss, all_indices

    def get_codebook_usage(self, indices: torch.Tensor):
        return self.base_rq.get_codebook_usage(indices)