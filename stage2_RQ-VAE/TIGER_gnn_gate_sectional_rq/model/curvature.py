"""Exact full-component A.1 supervision; no sampled or batch-local references."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _distance_block(x, y, squared):
    if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad):
        return checkpoint(squared, x, y, use_reentrant=False)
    return squared(x, y)


def reference_moments(q, indices, labels, squared, reference_chunk=256, node_chunk=128):
    indices = torch.as_tensor(indices,device=q.device,dtype=torch.long)
    selected = indices.detach().cpu().numpy()
    moments = q.new_zeros(len(indices))
    for component in np.unique(labels[selected]):
        positions = np.flatnonzero(labels[selected] == component)
        references = np.flatnonzero(labels == component)
        for start in range(0,len(positions),node_chunk):
            pos = torch.as_tensor(positions[start:start+node_chunk],device=q.device)
            endpoints = q[indices[pos]]
            total = q.new_zeros(len(pos))
            for offset in range(0,len(references),reference_chunk):
                ref = torch.as_tensor(references[offset:offset+reference_chunk],device=q.device)
                total = total + _distance_block(q[ref],endpoints,squared).sum(0)
            moments = moments.index_copy(0,pos,total/len(references))
    return moments


def sectional_values(q, triples, components, squared, reference_chunk=256, node_chunk=128,
                     moments=None, paired_squared=None):
    """Moment identity for identical full reference sets; all q gradients retained.

    `q` indexes the global catalogue. `squared(x,y)` is fixed and symmetric,
    returning all pairwise squared distances, never gate-dependent weights.
    """
    if reference_chunk <= 0 or node_chunk <= 0:
        raise ValueError('chunk sizes must be positive')
    triples = torch.as_tensor(triples, dtype=torch.long, device=q.device)
    labels = np.asarray(components)
    if len(labels) != len(q):
        raise ValueError('component labels must cover the entire catalogue')
    if triples.numel() == 0:
        return q.new_empty((0,))
    ids = triples.detach().cpu().numpy()
    if ids.min() < 0 or ids.max() >= len(q):
        raise ValueError('triple id outside global catalogue')
    if not np.all(labels[ids] == labels[ids[:, :1]]):
        raise ValueError('triple crosses reference components')
    if moments is None:
        unique, inverse = torch.unique(triples.reshape(-1),sorted=True,return_inverse=True)
        moments = reference_moments(q,unique,labels,squared,reference_chunk,node_chunk)
        lookup = moments[inverse].reshape(-1,3)
    else:
        lookup = moments[triples]
    pair = []
    for start in range(0, len(triples), node_chunk):
        block = triples[start:start + node_chunk]
        if paired_squared is None:
            pair.append(_distance_block(q[block[:,1]],q[block[:,2]],squared).diagonal())
        else:
            pair.append(_distance_block(q[block[:,1]],q[block[:,2]],paired_squared))
    return lookup[:, 0] - (lookup[:, 1] + lookup[:, 2]) / 2 + torch.cat(pair) / 4


def centre_huber(predicted, target, centres, valid, scale):
    if not np.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError('target scale must be positive and finite')
    valid = torch.as_tensor(valid,device=predicted.device,dtype=torch.bool)
    centres = torch.as_tensor(centres,device=predicted.device,dtype=torch.long)
    target = torch.as_tensor(target,device=predicted.device,dtype=predicted.dtype)
    if not bool(valid.any()):
        raise ValueError('no valid curvature targets')
    error = F.huber_loss(predicted[valid] / scale, target[valid] / scale,
                         reduction='none', delta=1.)
    _, index, counts = torch.unique(centres[valid],return_inverse=True,return_counts=True)
    return (error / counts[index]).sum() / len(counts)


class FrozenCurvature:
    """Read one frozen shard at a time, never cache the entire triple table."""
    def __init__(self, root):
        self.root = Path(root)
        with np.load(self.root / 'node_moments.npz') as blob:
            self.components = blob['component']
        with np.load(self.root / 'item_summary.npz') as blob:
            count, mean, variance = blob['valid'], blob['mean'], blob['variance']
            valid = count > 0
            self.valid_centres = np.flatnonzero(valid)
            self.scale = float(np.sqrt(np.sum(count[valid] * (variance[valid] + mean[valid]**2))
                                       / np.sum(count[valid])))
        self.scale = max(self.scale, np.finfo(np.float64).eps)
        with np.load(self.root / 'index.npz') as blob:
            self.shard_names = [str(name) for name in blob['shard_names']]

    def blocks(self, centres=None, triple_chunk=1024):
        wanted = None if centres is None else set(int(x) for x in centres)
        for name in self.shard_names:
            with np.load(self.root / 'shards' / name) as shard:
                centre = shard['centre']
                unique, starts, counts = np.unique(centre,return_index=True,return_counts=True)
                for item,start,count in zip(unique,starts,counts):
                    if wanted is not None and int(item) not in wanted:
                        continue
                    denominator = int(shard['valid'][start:start+count].sum())
                    if not denominator:
                        continue
                    for offset in range(int(start),int(start+count),triple_chunk):
                        end = min(offset+triple_chunk,int(start+count))
                        valid = shard['valid'][offset:end]
                        if not bool(valid.any()):
                            continue
                        triples = np.column_stack((centre[offset:end],shard['b'][offset:end],shard['c'][offset:end]))[valid]
                        yield triples, shard['curvature'][offset:end][valid], denominator

    def loss(self, q, squared, centres=None, reference_chunk=256, triple_chunk=1024,
             paired_squared=None):
        """Exact mean over requested valid centres, all their triples and full references."""
        selected = self.valid_centres if centres is None else np.intersect1d(self.valid_centres,centres)
        if len(selected) == 0:
            raise ValueError('no valid centres in requested split')
        # First pass determines endpoints only; second streams the same frozen
        # targets. Required references remain the full original components.
        needed = np.zeros(len(q),dtype=bool)
        for triples,_,_ in self.blocks(selected,triple_chunk):
            needed[triples.reshape(-1)] = True
        indices = torch.as_tensor(np.flatnonzero(needed),device=q.device)
        values = reference_moments(q,indices,self.components,squared,reference_chunk)
        moments = q.new_zeros(len(q)).index_copy(0,indices,values)
        total = q.sum()*0
        rows = 0
        for triples, target, denominator in self.blocks(selected,triple_chunk):
            prediction = sectional_values(q,triples,self.components,squared,reference_chunk,
                                           moments=moments,paired_squared=paired_squared)
            target = torch.as_tensor(target,device=q.device,dtype=q.dtype)
            total = total + F.huber_loss(prediction/self.scale,target/self.scale,
                                         reduction='sum',delta=1.) / denominator / len(selected)
            rows += len(triples)
        if not rows:
            raise ValueError('no frozen target rows read')
        return total
