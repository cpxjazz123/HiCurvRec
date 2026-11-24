"""Detached six-feature directional means of complete, verified ego layouts.

Topology is supplied by the existing layout consumer; this module never builds a
graph or persists an asset. No learned tensor or activation belongs in this cache.
"""
import torch

from tools.issue252.ego_operators import batch_operators


class StaticFeatureMeans:
    """One immutable layout/device/dtype's Ain F, Aout F and nonempty masks."""

    @torch.no_grad()
    def __init__(self, layout, features, chunk=256):
        if chunk < 1:
            raise ValueError('chunk must be positive')
        count = layout.nodes.numel()
        self.incoming = features.new_empty((count, 6))
        self.outgoing = features.new_empty((count, 6))
        self.in_nonempty = torch.empty(count, dtype=torch.bool, device=features.device)
        self.out_nonempty = torch.empty_like(self.in_nonempty)
        # Visit all roots, not just the roots requested by the current minibatch.
        # Every block remains complete; chunk bounds the number of full egos.
        for start in range(0, layout.offsets.numel() - 1, chunk):
            stop = min(start + chunk, layout.offsets.numel() - 1)
            ids = torch.arange(start, stop, device=features.device)
            nodes, _, _, incoming, outgoing = batch_operators(layout, ids, features.dtype)
            begin, end = layout.offsets[start], layout.offsets[stop]
            local = features[nodes]
            self.incoming[begin:end] = torch.sparse.mm(incoming, local)
            self.outgoing[begin:end] = torch.sparse.mm(outgoing, local)
            self.in_nonempty[begin:end] = incoming.crow_indices().diff() > 0
            self.out_nonempty[begin:end] = outgoing.crow_indices().diff() > 0

    def gather(self, layout, ids):
        """Select complete rows in exactly batch_operators' ego/global order."""
        starts = layout.offsets[ids]
        sizes = layout.offsets[ids + 1] - starts
        ends = sizes.cumsum(0)
        indices = (torch.repeat_interleave(starts - (ends - sizes), sizes)
                   + torch.arange(ends[-1], device=ids.device))
        return (self.incoming[indices], self.outgoing[indices],
                self.in_nonempty[indices], self.out_nonempty[indices])
