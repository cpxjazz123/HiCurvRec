"""Asynchronous host-to-device copies of small integer tables.

A pageable copy blocks its caller until the pending device queue has drained,
which serialises the training step even though the copy itself is tiny. Pinned
staging buffers keep the copy asynchronous; the caching host allocator reuses
the blocks and keeps them stream-safe.
"""
import torch


def host_int64(values, device):
    """Copy host integers to the device without waiting for the device queue."""
    source = torch.as_tensor(values, dtype=torch.int64)
    staging = torch.empty(source.numel(), dtype=torch.int64, pin_memory=True)
    staging.copy_(source)
    return staging.to(device, non_blocking=True)
