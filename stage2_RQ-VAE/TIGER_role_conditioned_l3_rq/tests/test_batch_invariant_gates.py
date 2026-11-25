"""A global item's geometry gate must not depend on its companion ego batch."""
import numpy as np
import torch

import train_rqvae as training


def test_actual_geometry_gates_are_invariant_to_other_batch_items():
    assert torch.cuda.is_available(), 'this regression requires a launcher-owned CUDA worker'
    training.set_seed(42)
    device = torch.device('cuda', torch.cuda.current_device())
    model = training.RQVAE(training._tokenizer_config(), in_dim=768,
                           distance=training.DISTANCE).to(device).train()
    if training.XAVIER_INIT:
        training.initialize_tiger_weights(model)
    generator = np.random.default_rng(42)
    main = np.sort(generator.choice(model.store.n, size=1024, replace=False))
    auxiliary = np.sort(generator.choice(model.store.n, size=256, replace=False))
    union = np.union1d(main, auxiliary)
    _, weights = model.structure.gates_for_items(model.store, union, device)
    expected = weights.detach().clone()
    for population in (main, auxiliary):
        _, actual = model.structure.gates_for_items(model.store, population, device)
        index = torch.tensor(np.searchsorted(union, population), device=device)
        torch.testing.assert_close(actual, expected[index], atol=1e-5, rtol=1e-4,
            msg='global item geometry gates must not depend on companion batch items')
