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


def reference_moments(q, indices, labels, squared, reference_chunk=2048, node_chunk=512):
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


def sectional_values(q, triples, components, squared, reference_chunk=2048, node_chunk=512,
                     moments=None, paired_squared=None):
    """Moment identity for identical full reference sets; all q gradients retained.

    `q` indexes the global catalogue. `squared(x,y)` is fixed and symmetric,
    returning all pairwise squared distances, never gate-dependent weights.
    """
    if reference_chunk <= 0 or node_chunk <= 0:
        raise ValueError('chunk sizes must be positive')
    ids=triples.detach().cpu().numpy() if torch.is_tensor(triples) else np.asarray(triples)
    labels=np.asarray(components)
    if len(labels)!=len(q):
        raise ValueError('component labels must cover the entire catalogue')
    if not ids.size:
        return q.new_empty((0,))
    if ids.ndim!=2 or ids.shape[1]!=3 or ids.dtype.kind not in 'iu':
        raise ValueError('triples must contain three integer global IDs')
    if ids.min()<0 or ids.max()>=len(q):
        raise ValueError('triple id outside global catalogue')
    if not np.all(labels[ids]==labels[ids[:,:1]]):
        raise ValueError('triple crosses reference components')
    triples=torch.as_tensor(ids,dtype=torch.long,device=q.device)
    if moments is None:
        unique, inverse = torch.unique(triples.reshape(-1),sorted=True,return_inverse=True)
        moments = reference_moments(q,unique,labels,squared,reference_chunk,node_chunk)
        lookup = moments[inverse].reshape(-1,3)
    else:
        lookup = moments[triples]
    pair = []
    pair_chunk=len(triples) if paired_squared is not None else node_chunk
    for start in range(0, len(triples), pair_chunk):
        block = triples[start:start + pair_chunk]
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
            self.item_shard=blob['item_shard']

    def blocks(self, centres=None, triple_chunk=16384):
        """Stream bounded mixed-centre batches with per-row centre denominators."""
        if triple_chunk<=0:
            raise ValueError('triple chunk must be positive')
        wanted=None if centres is None else set(int(x) for x in centres)
        names=self.shard_names
        if wanted is not None:
            indices=np.asarray(sorted(wanted),dtype=np.int64)
            names=[self.shard_names[int(index)] for index in np.unique(self.item_shard[indices])
                   if int(index)>=0]
        pending_t=[];pending_y=[];pending_d=[];buffered=0
        for name in names:
            with np.load(self.root/'shards'/name) as archive:
                shard={key:archive[key] for key in ('centre','b','c','valid','curvature')}
            centre=shard['centre']
            unique,starts,counts=np.unique(centre,return_index=True,return_counts=True)
            for item,start,count in zip(unique,starts,counts):
                if wanted is not None and int(item) not in wanted:
                    continue
                denominator=int(shard['valid'][start:start+count].sum())
                if not denominator:
                    continue
                for offset in range(int(start),int(start+count),triple_chunk):
                    end=min(offset+triple_chunk,int(start+count))
                    valid=shard['valid'][offset:end]
                    if not bool(valid.any()):
                        continue
                    triples=np.column_stack((centre[offset:end],shard['b'][offset:end],
                                             shard['c'][offset:end]))[valid]
                    pending_t.append(triples)
                    pending_y.append(shard['curvature'][offset:end][valid])
                    pending_d.append(np.full(len(triples),denominator,dtype=np.int64))
                    buffered+=len(triples)
                    if buffered>=triple_chunk:
                        t=np.concatenate(pending_t);y=np.concatenate(pending_y);d=np.concatenate(pending_d)
                        yield t[:triple_chunk],y[:triple_chunk],d[:triple_chunk]
                        buffered-=triple_chunk
                        pending_t=[t[triple_chunk:]] if buffered else []
                        pending_y=[y[triple_chunk:]] if buffered else []
                        pending_d=[d[triple_chunk:]] if buffered else []
        if buffered:
            yield np.concatenate(pending_t),np.concatenate(pending_y),np.concatenate(pending_d)

    def loss(self, q, squared, centres=None, reference_chunk=2048, triple_chunk=16384,
             paired_squared=None):
        """Exact mean over requested valid centres, all their triples and full references."""
        if len(q)!=len(self.components):
            raise ValueError('curvature q must cover the full global catalogue')
        if centres is not None:
            ids=np.asarray(centres)
            if ids.ndim!=1 or ids.dtype.kind not in 'iu' or (ids<0).any() or (ids>=len(q)).any():
                raise ValueError('curvature centre outside global catalogue')
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
            denominator=torch.as_tensor(denominator,device=q.device,dtype=q.dtype)
            error=F.huber_loss(prediction/self.scale,target/self.scale,reduction='none',delta=1.)
            total=total+(error/denominator).sum()/len(selected)
            rows += len(triples)
        if not rows:
            raise ValueError('no frozen target rows read')
        self.last_rows=rows
        self.last_centres=len(selected)
        return total
