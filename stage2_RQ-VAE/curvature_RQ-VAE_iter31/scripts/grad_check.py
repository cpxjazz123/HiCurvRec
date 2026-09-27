"""No-argument gradient verification delegated to the Iter31 HRA MVG.

The shared MVG performs explicit loss.backward() and checks model gradients.
"""
from __future__ import annotations

import numpy as np
import torch

import mvg_check


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("grad_check requires CUDA")
    rqtrain = mvg_check._load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    fixed_c = rqtrain._load_closed_form_curvatures()
    checkpoint_state = mvg_check._load_checkpoint(device)
    batch = mvg_check._load_batch(rqtrain, device)
    model = mvg_check._build_model(rqtrain, device, fixed_c, checkpoint_state)
    mvg_check._check_layer_buffers(model, fixed_c)
    mvg_check._check_gradients(model, batch)
    print("GRAD_CHECK PASS")
    print(f"closed_form_c_l={fixed_c}")


if __name__ == "__main__":
    main()