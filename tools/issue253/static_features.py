"""Detached six-feature means over the unchanged verified full-ego layouts.

Only immutable features/topology are cached; learned projections and GNN outputs
are evaluated with current parameters at every optimizer step. No asset builder
or persistent asset write belongs here.
"""
import numpy as np
import torch

from tools.issue252.ego_operators import batch_operators


class StaticFeatureMeans:
    """One layout/device/dtype's Ain F, Aout F and nonempty masks."""

    @torch.no_grad()
    def __init__(self, layout, features, chunk=256):
        if chunk < 1:
            raise ValueError('chunk must be positive')
        count = layout.nodes.numel()
        self.incoming = features.new_empty((count, 6))
        self.outgoing = features.new_empty((count, 6))
        self.in_nonempty = torch.empty(count, dtype=torch.bool, device=features.device)
        self.out_nonempty = torch.empty_like(self.in_nonempty)
        if features.is_cuda:
            from tools.issue253.ego_kernel import batch_geometry, row_means
        for start in range(0, len(layout.node_counts), chunk):
            stop = min(start + chunk, len(layout.node_counts))
            host = np.arange(start, stop, dtype=np.int64)
            begin, end = layout.offsets_np[start], layout.offsets_np[stop]
            if features.is_cuda:
                # The fixed complete-row reduction is shared with the live GNN.
                # cuSPARSE SpMM can change these immutable means between readers.
                geometry = batch_geometry(layout, host, features.device)
                local = features[layout.nodes[geometry['layout_rows']].long()]
                row_means(layout.in_ptr, layout.in_col, local, geometry,
                          out=self.incoming[begin:end])
                row_means(layout.out_ptr, layout.out_col, local, geometry,
                          out=self.outgoing[begin:end])
                self.in_nonempty[begin:end] = layout.in_ptr[begin+1:end+1] > layout.in_ptr[begin:end]
                self.out_nonempty[begin:end] = layout.out_ptr[begin+1:end+1] > layout.out_ptr[begin:end]
            else:
                ids = torch.as_tensor(host, device=features.device)
                plan = layout.plan(host)
                nodes, _, _, incoming, outgoing = batch_operators(layout, ids, features.dtype, plan)
                local = features[nodes]
                self.incoming[begin:end] = torch.sparse.mm(incoming, local)
                self.outgoing[begin:end] = torch.sparse.mm(outgoing, local)
                self.in_nonempty[begin:end] = incoming.crow_indices().diff() > 0
                self.out_nonempty[begin:end] = outgoing.crow_indices().diff() > 0

    def gather(self, layout, ids, plan=None):
        """Gather all static rows in exactly the requested ego/node order."""
        if plan is None:
            plan = layout.plan(ids.detach().cpu().numpy() if torch.is_tensor(ids) else ids)
        starts = layout.offsets[ids]
        sizes = layout.offsets[ids + 1] - starts
        ends = sizes.cumsum(0)
        indices = (torch.repeat_interleave(starts - (ends - sizes), sizes, output_size=plan.nodes)
                   + torch.arange(plan.nodes, device=ids.device))
        return (self.incoming[indices], self.outgoing[indices],
                self.in_nonempty[indices], self.out_nonempty[indices])
