"""Frozen red reproducer for the pre-repair shared native-mode verifier."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT/'stage2_RQ-VAE/TIGER_periodic_budget_prefix_rq'))
import torch
import validation_entry
from model.layers import VQLayer

layer = VQLayer(2, 32, sk_epsilon=.003, sk_iters=50)
cost = torch.tensor([[0., 2.], [1., 2.0009765625]])
layer.eval()
auxiliary_ids = layer._indices(cost, False).reshape(2, 1)
assert auxiliary_ids.tolist() == [[1], [0]], 'unbalanced frozen greedy hand answer'
layer.train()
balanced_ids = layer._indices(cost, False).reshape(2, 1)
print('native_auxiliary', auxiliary_ids.tolist(), 'native_balanced', balanced_ids.tolist(), flush=True)
observer = validation_entry.Observer({}, Path('.'))
observer.assignment_layers = [layer]
observer.assignment_balanced = True  # model.train(), but this captured population is auxiliary.
observer.auxiliary_ties(auxiliary_ids, auxiliary_ids, [cost], [cost])
