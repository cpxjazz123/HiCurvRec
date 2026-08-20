"""v49 (Issue270): HALC + linear decay multiplier (R36 曲率调度变更).

机制: 在 HALCAnnealingRegularizer.annealed_curvature 输出之上, 叠加一个 linear decay
multiplier, 让训练后期 c_l 衰减 (与 C27 反向).

  new_c_l(t) = annealed_curvature(t) * decay_factor(t)
  decay_factor(t) = max(DECAY_END, 1 - (t / DECAY_STEPS) * (1 - DECAY_END))

效果:
  - t < DECAY_STEPS: decay_factor ≈ 1.0 (保持 HALC v2 sigmoid 行为)
  - t > DECAY_STEPS: decay_factor 线性下降到 DECAY_END (默认 0.3)
  - 训练后期 c_max ≈ c_l * DECAY_END (比 HALC v2 的 c_l * 1.0 小)

R36 严格允许的曲率调度变更, 区别于:
  - v32 cosine restart (重启 schedule)
  - v36 per-token input-dependent (per-token 调度)
  - v49 是 per-epoch linear decay, 单调不重启

使用:
  import os
  if os.environ.get("USE_HALC_V49", "0") == "1":
      from _lib.halc_v49 import HALCWithLinearDecay
      halc = HALCWithLinearDecay(num_layers=7, init_curvature=1.0, ...)
"""
import math
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from _lib.halc import HALCAnnealingRegularizer


# v49 默认值 (R36 不调参, 硬编码)
DECAY_START_EPOCH = int(os.environ.get("V49_DECAY_START_EPOCH", "20"))
DECAY_STEPS = int(os.environ.get("V49_DECAY_STEPS", "60"))
DECAY_END = float(os.environ.get("V49_DECAY_END", "0.3"))


class HALCWithLinearDecay(HALCAnnealingRegularizer):
    """HALCAnnealingRegularizer 子类, 在 annealed_curvature 输出上叠加 linear decay multiplier.

    用法与 HALCAnnealingRegularizer 完全一致, 但 c_l 会随训练 epoch 线性下降:
      c_final ≈ c_orig * DECAY_END (在 epoch = DECAY_START + DECAY_STEPS 时)
    """

    def __init__(
        self,
        num_layers: int,
        init_curvature: float = 1.0,
        c_max: float = 1.0,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        encoder_warmup: int = 3,
        encoder_cooldown: int = 8,
        decoder_warmup: int = 7,
        decoder_cooldown: int = 12,
        reg_weight_max: float = 0.05,
        reg_weight_max_per_layer: Optional[list] = None,
        decay_start_epoch: Optional[int] = None,
        decay_steps: Optional[int] = None,
        decay_end: Optional[float] = None,
    ):
        super().__init__(
            num_layers=num_layers,
            init_curvature=init_curvature,
            c_max=c_max,
            warmup_epochs=warmup_epochs,
            cooldown_epochs=cooldown_epochs,
            encoder_warmup=encoder_warmup,
            encoder_cooldown=encoder_cooldown,
            decoder_warmup=decoder_warmup,
            decoder_cooldown=decoder_cooldown,
            reg_weight_max=reg_weight_max,
            reg_weight_max_per_layer=reg_weight_max_per_layer,
        )
        # v49: linear decay schedule (epoch 起始, 步长, 终值)
        self.decay_start_epoch = int(decay_start_epoch) if decay_start_epoch is not None else DECAY_START_EPOCH
        self.decay_steps = int(decay_steps) if decay_steps is not None else DECAY_STEPS
        self.decay_end = float(decay_end) if decay_end is not None else DECAY_END
        assert 0.0 < self.decay_end < 1.0, f"decay_end {self.decay_end} 必须在 (0, 1) 之间"
        assert self.decay_steps > 0, f"decay_steps {self.decay_steps} 必须 > 0"

    def _decay_factor(self) -> torch.Tensor:
        """Linear decay multiplier on top of annealed_curvature.

        Returns:
            scalar tensor (decay 是 epoch 级, 与 layer 无关)
        """
        t = float(self.current_epoch)
        if t < self.decay_start_epoch:
            return torch.tensor(1.0, dtype=self.log_curvature.dtype, device=self.log_curvature.device)
        # 训练后期: 线性下降到 decay_end
        elapsed = t - self.decay_start_epoch
        frac = min(1.0, elapsed / self.decay_steps)
        factor = 1.0 - frac * (1.0 - self.decay_end)
        return torch.tensor(factor, dtype=self.log_curvature.dtype, device=self.log_curvature.device)

    def annealed_curvature(self) -> torch.Tensor:
        """v49: 在 HALC v2 sigmoid 输出上叠加 linear decay multiplier.

        Returns:
            (num_layers,) tensor of annealed + decayed curvature
        """
        c_orig = super().annealed_curvature()  # (num_layers,)
        decay = self._decay_factor()  # scalar
        return c_orig * decay