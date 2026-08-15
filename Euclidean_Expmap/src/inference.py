from typing import Any, Dict

import hydra
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig

_orig_torch_load = torch.load

def _load_allow_full(obj, *args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(obj, *args, **kwargs)

torch.load = _load_allow_full

try:
    from lightning.fabric.plugins.io import torch_io as _torch_io
    from lightning.fabric.utilities import cloud_io as _cloud_io
    from lightning.pytorch import strategies as _strategies

    _mods = [_torch_io, _cloud_io]
    if hasattr(_strategies, "strategy"):
        _mods.append(_strategies.strategy)

    for _mod in _mods:
        if hasattr(_mod, "_load") or hasattr(_mod, "pl_load"):
            _target = getattr(_mod, "_load", None) or getattr(_mod, "pl_load", None)
            _orig_pl_load = _target

            def _pl_load_allow_full(*args, _orig=_orig_pl_load, **kwargs):
                kwargs["weights_only"] = False
                return _orig(*args, **kwargs)

            if hasattr(_mod, "_load"):
                _mod._load = _pl_load_allow_full
            if hasattr(_mod, "pl_load"):
                _mod.pl_load = _pl_load_allow_full
except Exception:
    pass

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras
from src.utils.custom_hydra_resolvers import *
from src.utils.launcher_utils import pipeline_launcher

command_line_logger = RankedLogger(__name__, rank_zero_only=True)


def inference(cfg: DictConfig) -> Dict[str, Any]:
    """Runs inference using a pre-trained model.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A dict with all instantiated objects.
    """

    with pipeline_launcher(cfg) as pipeline_modules:
        command_line_logger.info("Starting inference!")
        ckpt_path = pipeline_modules.cfg.get("ckpt_path", None)
        if not ckpt_path:
            command_line_logger.warning(
                "No ckpt_path was provided. If using a model you trained, this is mandatory. Only leave ckpt_path=None if using a pre-trained model."
            )

        pipeline_modules.trainer.predict(
            model=pipeline_modules.model,
            datamodule=pipeline_modules.datamodule,
            ckpt_path=ckpt_path,
            return_predictions=False,
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for inference.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    extras(cfg)

    # run inference
    inference(cfg)


if __name__ == "__main__":
    main()
