import math

import torch


class WarmupCosineSchedulerNonzeroMin(torch.optim.lr_scheduler.LambdaLR):
    """Cosine schedule with warmup and decay to a possibly nonzero min learning rate."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        scheduler_steps: int,
        min_ratio: float = 0.1,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        **kwargs,
    ):
        """
        Create a schedule with a learning rate that decreases following the values of
        the cosine function between the initial lr set in the optimizer to 0, after a
        warmup period during which it increases linearly between 0 and the initial lr
        set in the optimizer.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            The optimizer for which to schedule the learning rate.
        warmup_steps: int
            The number of steps for the warmup phase.
        scheduler_steps: int
            The total number of training steps.
        min_lr_ratio: int
            Minimum learning rate divided by initial learning rate.
        num_cycles: float
            The number of waves in the cosine schedule (the default is to just decrease
            from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        """
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.num_cycles = num_cycles
        super(WarmupCosineSchedulerNonzeroMin, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(
        self,
        step: int,
    ) -> float:
        """Return the factor to mulitply the initial learning rate with."""
        if step < self.warmup_steps:  # linear warm-up
            return float(step) / float(max(1, self.warmup_steps))
        if step <= self.scheduler_steps:
            # cosine decay
            decay_ratio = float(step - self.warmup_steps) / float(
                max(1, self.scheduler_steps - self.warmup_steps)
            )
            coeff = 0.5 * (
                1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * decay_ratio)
            )
            return max(self.min_ratio, self.min_ratio + coeff * (1 - self.min_ratio))
        else:  # current_step > self.scheduler_steps
            return self.min_ratio


class InverseSqrtScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Inverse-square-root (a.k.a. T5X "rsqrt") learning rate schedule.

    lr = base_lr * scale(step)

    where

        scale(step) = 1/sqrt(max(step, warmup_steps))   if step >= warmup_steps
        scale(step) = step / warmup_steps                if step <  warmup_steps

    (i.e. linear warmup for the first `warmup_steps` steps, then 1/sqrt decay.)
    This matches the T5 / T5X reference schedule (see Rajput et al. 2023 "TIGER"
    Appendix / T5.1.1 §3.2.2 "inverse square root schedule, with 10000-step
    linear warmup").

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for which to schedule the learning rate.
    warmup_steps : int
        Number of linear-warmup steps. After this many steps the lr decays as
        1/sqrt(step). Per TIGER, defaults to 10000.
    last_epoch : int, optional
        The index of the last epoch (passed through to LambdaLR; -1 means "start
        from step 0"). The internal ``_step_count`` of LambdaLR tracks the
        actual ``optimizer.step()`` call count, so restart-from-checkpoint is
        safe — no manual drift.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 10000,
        last_epoch: int = -1,
        **kwargs,
    ):
        self.warmup_steps = int(warmup_steps)
        super(InverseSqrtScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step: int) -> float:
        """Return multiplicative factor applied to the base lr at ``step``."""
        # LambdaLR passes the absolute step (0-indexed) here. LambdaLR's own
        # _step_count is the source of truth; we never manually offset.
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # Inverse-square-root decay. Note: do NOT use step**-0.5 — that's broken
        # for step==0 (would also be replaced by sympy/sympy issue), and the
        # canonical T5 form scales *down* by 1/sqrt(step), not by sqrt(warmup/step).
        return float(self.warmup_steps) ** 0.5 / float(step) ** 0.5
