"""Split the GNN cost into assembly, first layer and second layer (diagnostic)."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_context import build


def main():
    context = build()
    model = context['model']
    device = context['device']
    structure = model.structure.tokens
    store = model.store
    layouts, normalized = structure._cached_store(store)
    means = structure._feature_cache[store]['devices'][str(device)]['means'][normalized.dtype]
    affine = structure._first_affine()
    ids = torch.as_tensor(np.arange(1024, dtype=np.int64), device=device)
    host = np.arange(1024, dtype=np.int64)

    report = {}
    for scale, (layout, static) in enumerate(zip(layouts, means)):
        plan = layout.plan(host)
        torch.cuda.synchronize()
        start = time.perf_counter()
        nodes, groups, sizes, incoming, outgoing = __import__(
            'tools.issue252.ego_operators', fromlist=['batch_operators']).batch_operators(
                layout, ids, normalized.dtype, plan)
        torch.cuda.synchronize()
        assembly = time.perf_counter() - start

        torch.cuda.synchronize()
        start = time.perf_counter()
        value = structure._read_sparse(normalized[nodes], groups, sizes, incoming, outgoing,
                                      static.gather(layout, ids, plan), affine)
        torch.cuda.synchronize()
        read = time.perf_counter() - start

        torch.cuda.synchronize()
        start = time.perf_counter()
        value.square().sum().backward(retain_graph=True)
        torch.cuda.synchronize()
        backward = time.perf_counter() - start

        report['scale%d' % scale] = {
            'nodes': int(len(nodes)), 'assembly': assembly, 'read_sparse': read,
            'backward': backward, 'plan_nodes': plan.nodes,
            'in_edges': plan.in_edges, 'out_edges': plan.out_edges}
    print(report)


if __name__ == '__main__':
    main()
