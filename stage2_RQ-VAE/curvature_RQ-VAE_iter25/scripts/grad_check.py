"""Layer-1/2 compatibility entrypoint for Iter25 gradient verification."""
from __future__ import annotations

import numpy as np
import torch

from mvg_check import (
    _build_model,
    _load_batch,
    _load_checkpoint_state,
    _load_training_module,
    _verify_graph_and_gradients,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("grad_check requires CUDA")
    rqtrain = _load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    batch = _load_batch(rqtrain, device)
    checkpoint_state = _load_checkpoint_state(device)
    model = _build_model(
        rqtrain, device, rqtrain._build_layer_scale_prior(), checkpoint_state
    )
    evidence = _verify_graph_and_gradients(rqtrain, model, batch)
    print("GRAD_CHECK PASS")
    print(f"batch_shape={tuple(batch.x.shape)}")
    print(f"prior={evidence['prior']}")
    print(f"component_gradient_norms={evidence['component_grad_norms']}")
    print(f"total_scale_grad_norms={evidence['total_scale_grad_norms']}")


if __name__ == "__main__":
    main()
