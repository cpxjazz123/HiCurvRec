"""ReviveDeadCodes — Task #67 落地的 Dead Code Revival Callback。

设计动机: snap-research/GRID 的 RQ-VAE 训练在 Toy 数据上常因 Sparse Update 坍缩:
- Layer 0 的 256 个桶在 BatchSize=128 下每 step 至少被命中 ~128 次,但很多桶
  永远拿不到 gradient 永远不会更新。Result: L0 coverage <50%,下游 SID 偏 unique,
  TIGER R@5 跌穿 baseline。

修复:
- 在 RQ.forward 末尾 stashed `last_cluster_ids` (B, n_layers) 和 `last_embeddings` (B, F)。
- Callback 累积每个 batch 命中次数 usage_count[i] = sum(cluster_ids[layer] == i)。
- 每 N step 检查: usage < mean * threshold_ratio 的桶视为"dead",用当前 batch 的
  random embedding sample + 噪声替换它们 data。

注意:
- 这是 `src/utils/callbacks/__init__.py` 下一个新的 Lightning Callback。
- 注册: rqvae_train_flat.yaml 加 `callbacks.revive_dead_codes._target_: src.utils.callbacks.revive_dead_codes.ReviveDeadCodesCallback`
- 也可通过 CLI override 注册: `+callbacks.revive_dead_codes._args_={...}`。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class ReviveDeadCodesCallback(Callback):
    """每 N step 检查并重置低使用率 codebook 桶。"""

    def __init__(
        self,
        revive_every_n_steps: int = 250,
        threshold_ratio: float = 0.005,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.revive_every_n_steps = revive_every_n_steps
        # 桶使用率 < 平均值 * threshold_ratio 视为 dead
        # 0.005 是平均值的 0.5%
        self.threshold_ratio = threshold_ratio
        # 注入 hot residual 时的额外扰动
        self.noise_std = noise_std

        # per-layer usage count, initialized in setup
        self.usage_count: Optional[list] = None
        self.n_layers: Optional[int] = None
        self.n_clusters: Optional[int] = None
        self.n_features: Optional[int] = None

        # 运行统计
        self.revived_total = 0
        self.revive_events = 0

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if hasattr(pl_module, "quantization_layer_list"):
            self.n_layers = len(pl_module.quantization_layer_list)
            self.n_clusters = pl_module.quantization_layer_list[0].n_clusters
            self.n_features = pl_module.quantization_layer_list[0].n_features
            device = pl_module.device
            self.usage_count = [
                torch.zeros(self.n_clusters, device=device)
                for _ in range(self.n_layers)
            ]

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # 仅 rank 0 上工作（避免 DDP 重复 trigger）
        if trainer.world_size > 1 and trainer.global_rank != 0:
            return

        cluster_ids = getattr(pl_module, "last_cluster_ids", None)
        if cluster_ids is None or self.usage_count is None:
            return

        cluster_ids = cluster_ids.view(-1, self.n_layers).long()
        device = cluster_ids.device

        # 累积 usage_count
        for layer_idx in range(self.n_layers):
            ids = cluster_ids[:, layer_idx]
            self.usage_count[layer_idx] = self.usage_count[layer_idx].scatter_add(
                0,
                ids,
                torch.ones_like(ids, dtype=torch.float, device=device),
            )

        # 每 N step 触发 revival
        if (trainer.global_step + 1) % self.revive_every_n_steps == 0:
            self._revive(pl_module, batch)
            for layer_idx in range(self.n_layers):
                self.usage_count[layer_idx].zero_()

    def _revive(self, pl_module: LightningModule, batch: Any) -> None:
        # 用 pl_module.last_layer_inputs[layer_idx] 作为每层 hot sample 来源。
        # 这是 RQ.forward 末尾 stash 的 per-layer residual-input pool:
        #   layer 0: encoder(x) 输出
        #   layer l>=1: encoder(x) - sum_{i<l} centroid[cluster_id_i]
        # 与 centroids 的 subspace 精准匹配 (latent_dim = codebook feature dim)。
        layer_inputs_pool = getattr(pl_module, "last_layer_inputs", None)
        if not layer_inputs_pool:
            return

        device = pl_module.device
        layer_inputs_pool = [
            t.detach().to(device).reshape(-1, t.shape[-1]) if t.dim() > 2 else t.detach().to(device)
            for t in layer_inputs_pool
        ]

        n_revived = 0
        with torch.no_grad():
            for layer_idx, q_layer in enumerate(pl_module.quantization_layer_list):
                if layer_idx >= len(layer_inputs_pool):
                    break
                hot_pool = layer_inputs_pool[layer_idx]
                if hot_pool is None or hot_pool.numel() == 0:
                    continue
                usage = self.usage_count[layer_idx]
                total = usage.sum().clamp(min=1)
                mean_usage = total / self.n_clusters
                threshold = mean_usage * self.threshold_ratio

                dead_mask = usage < threshold
                n_dead = int(dead_mask.sum().item())
                if n_dead == 0:
                    continue

                # 取最近 batch 中 sample 替换 dead 桶
                n_take = min(n_dead, hot_pool.shape[0])
                if n_take <= 0:
                    continue

                # random sample (with replacement if needed)
                idx = torch.randint(
                    0, hot_pool.shape[0], (n_take,), device=device
                )
                hot = hot_pool[idx].clone()
                if self.noise_std > 0:
                    hot = hot + torch.randn_like(hot) * self.noise_std

                dead_indices = torch.where(dead_mask)[0][:n_take]
                # Cast to centroids' storage dtype to avoid
                # bf16-mixed precision dtype mismatch (activations are bf16,
                # parameters are fp32).
                # Build the assignment explicitly via .copy_() to avoid
                # advanced-indexing broadcast shape ambiguity when n_take == 1
                # and the hot tensor happens to be squeezed.
                n_assign = int(min(len(dead_indices), hot.shape[0]))
                if n_assign > 0:
                    target_value = hot[:n_assign].to(
                        q_layer.centroids.data.dtype
                    ).reshape(n_assign, -1)
                    sel = dead_indices[:n_assign].long().reshape(-1)
                    centroids_view = q_layer.centroids.data
                    if centroids_view.shape[0] == sel.numel() == 1:
                        # Edge case: 1 centroid to replace
                        centroids_view[sel[0]] = target_value[0]
                    else:
                        centroids_view[sel] = target_value
                    n_revived += int(sel.numel())

        if n_revived > 0:
            self.revived_total += n_revived
            self.revive_events += 1
            # 落到 Lightning logger
            pl_module.log(
                "revive/n_dead_replaced",
                float(n_revived),
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(
                "revive/total_replaced",
                float(self.revived_total),
                on_step=False,
                on_epoch=True,
            )
            print(
                f"[ReviveDeadCodes] step={pl_module.global_step} "
                f"replaced {n_revived} dead centroids "
                f"(event #{self.revive_events}, total_revived={self.revived_total})"
            )
