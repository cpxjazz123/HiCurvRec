"""Small pure helpers for crowding metrics and run-configuration provenance."""
from __future__ import annotations


def assemble_losses(base_loss, behavior_loss, behavior_weight: float, crowding_loss,
                    crowding_weight: float, has_crowding: bool):
    if not has_crowding:
        if behavior_weight == 0.0:
            return base_loss
        return base_loss + behavior_weight * behavior_loss
    return base_loss + behavior_weight * behavior_loss + crowding_weight * crowding_loss


def crowding_ramp_weight(epoch: int, config: dict | None, ramp_start_epoch: int,
                         epochs: int) -> float:
    if config is None:
        return 0.0
    span = max(epochs - ramp_start_epoch, 1)
    alpha = min(max((epoch - ramp_start_epoch) / span, 0.0), 1.0)
    return float(config["strength"]) * alpha


def crowding_record(config: dict | None, epochs: int, ramp_start_epoch: int,
                    parameter_count: int) -> dict | None:
    if config is None:
        return None
    return {
        "mechanism": config["mechanism"],
        "strength": float(config["strength"]),
        "tau": float(config["tau"]),
        "concentration": float(config["concentration"]),
        "depth": int(config["depth"]),
        "k": int(config["k"]),
        "ramp": {"start_epoch": int(ramp_start_epoch), "end_epoch": int(epochs)},
        "parameter_count": int(parameter_count),
    }


def crowding_train_fields(config: dict | None, crowding_loss: float,
                          crowding_weight: float) -> dict[str, float]:
    if config is None:
        return {}
    return {
        "crowding_loss": float(crowding_loss),
        "crowding_weight": float(crowding_weight),
    }
