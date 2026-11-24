"""Split the fused GNN cost into assembly, dense layer, kernel and backward."""
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
    from tools.issue252 import ego_kernel
    from tools.issue252.sparse_adjoint import fused_directional_means

    items = np.arange(1024, dtype=np.int64)
    report = {}
    for scale, (layout, static) in enumerate(zip(layouts, means)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        geometry = ego_kernel.batch_geometry(layout, items, device)
        torch.cuda.synchronize()
        assembly = time.perf_counter() - start

        nodes = layout.nodes[geometry['layout_rows']].long()
        groups = geometry['row_ego'].long()
        sizes = torch.as_tensor(layout.node_counts[items], device=device)
        gathered = static.gather(layout, torch.as_tensor(items, device=device),
                                 layout.plan(items))
        features = normalized[nodes]
        mean_in, mean_out, nonempty_in, nonempty_out = gathered
        self_weight, self_bias, in_weight, in_bias, out_weight, out_bias = affine

        torch.cuda.synchronize()
        start = time.perf_counter()
        x = torch.tanh(features @ self_weight.T + self_bias
                       + mean_in @ in_weight.T + nonempty_in[:, None] * in_bias
                       + mean_out @ out_weight.T + nonempty_out[:, None] * out_bias)
        torch.cuda.synchronize()
        dense = time.perf_counter() - start

        x = x.detach().requires_grad_(True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        means_out = fused_directional_means(x, layout, geometry)
        torch.cuda.synchronize()
        kernel = time.perf_counter() - start

        grad = torch.randn_like(means_out[0])
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.autograd.grad(means_out[0], x, grad, retain_graph=True)
        torch.cuda.synchronize()
        backward = time.perf_counter() - start

        report['scale%d' % scale] = {'nodes': int(len(nodes)), 'geometry': assembly,
                                    'dense_layer': dense, 'fused_kernel': kernel,
                                    'kernel_backward': backward}
    print(report)


if __name__ == '__main__':
    main()
