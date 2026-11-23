"""GraphMoRE Appendix A.1 statistic on an unweighted undirected projection.

This is not Ricci curvature and not a directed curvature. No training/PPR dependencies.
Missing references remain masked; disconnected estimates are component conditional.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

ZERO_TOLERANCE = 1e-12


def projection(graph):
    g = sp.csr_matrix(graph)
    if g.shape[0] != g.shape[1] or not np.isfinite(g.data).all() or (g.data < 0).any() or np.any(g.diagonal()):
        raise ValueError('finite nonnegative square loop-free graph required')
    g = g.copy(); g.eliminate_zeros()
    g.data[:] = 1
    return g.maximum(g.T).astype(np.int8)


def sample_triples(graph, seed):
    g = projection(graph)
    rng = np.random.default_rng(seed)
    degree = np.diff(g.indptr).astype(np.int64)
    population = degree * (degree - 1) // 2
    triples = []
    for m in np.flatnonzero(population):
        neighbours = g.indices[g.indptr[m]:g.indptr[m+1]]
        b, c = np.sort(rng.choice(neighbours, 2, replace=False))
        triples.append((m, b, c))
    return {'triples': np.asarray(triples, dtype=np.int64).reshape(-1,3), 'population': population}


def contributions(am, ab, ac, bc):
    """Float64 parallelogram residual; unreachable rows masked before arithmetic."""
    am, ab, ac, bc = np.broadcast_arrays(am, ab, ac, bc)
    valid = np.isfinite(am) & np.isfinite(ab) & np.isfinite(ac) & np.isfinite(bc)
    out = np.full(am.shape, np.nan, dtype=np.float64)
    out[valid] = am[valid] ** 2 + bc[valid] ** 2 / 4 - (ab[valid] ** 2 + ac[valid] ** 2) / 2
    return out


def endpoint_distance(graph, b, c):
    """g(b,c) on the unweighted undirected projection: 1 if adjacent, else 2."""
    g = projection(graph)
    return np.where(np.asarray(g[b,c]).ravel() != 0, 1., 2.)


def estimate(graph, triples, references, distances=None):
    g = projection(graph)
    t, r = np.asarray(triples), np.asarray(references)
    n = g.shape[0]
    if t.ndim != 2 or t.shape[1] != 3 or t.dtype.kind not in 'iu' or r.ndim != 1 or r.dtype.kind not in 'iu':
        raise ValueError('integer triple/reference arrays required')
    if np.any(t < 0) or np.any(t >= n) or np.any(r < 0) or np.any(r >= n):
        raise ValueError('node index out of range')
    canonical = np.column_stack((t[:,0], np.sort(t[:,1:], axis=1)))
    if len(np.unique(canonical,axis=0)) != len(t) or len(np.unique(r)) != len(r):
        raise ValueError('duplicate triples or references')
    if np.any(t[:,0] == t[:,1]) or np.any(t[:,0] == t[:,2]) or np.any(t[:,1] == t[:,2]):
        raise ValueError('coincident triple vertices')
    if len(t) and (np.any(np.asarray(g[t[:,0],t[:,1]]).ravel() != 1) or np.any(np.asarray(g[t[:,0],t[:,2]]).ravel() != 1)):
        raise ValueError('triple endpoints must neighbor center')
    if distances is None:
        distances = shortest_path(g, directed=False, unweighted=True, indices=r)
    d = np.asarray(distances)
    if d.shape != (len(r),n) or np.isnan(d).any() or (d < 0).any() or not np.all(d[np.arange(len(r)),r] == 0):
        raise ValueError('invalid reference distance matrix')
    m,b,c = t.T
    bc = endpoint_distance(graph, b, c)
    values = contributions(d[:,m].T, d[:,b].T, d[:,c].T, bc[:,None])
    count = np.isfinite(values).sum(axis=1)
    mean = np.divide(np.nansum(values,axis=1), count, out=np.full(len(t),np.nan), where=count>0)
    return {'value': mean, 'effective_references': count, 'contributions': values,
            'valid': count > 0, 'reason': np.where(count>0, '', 'no_reachable_reference'),
            'endpoint_distance': bc}


def summarize(n, triples, values):
    result = {'count': np.zeros(n,dtype=np.int64), 'valid': np.zeros(n,dtype=bool)}
    for k in ['mean','std','negative','zero','positive']:
        result[k] = np.full(n,np.nan)
    result['quantiles'] = np.full((n,5),np.nan)
    for m in np.unique(triples[:,0]):
        v = values[(triples[:,0]==m) & np.isfinite(values)]
        if not len(v): continue
        result['count'][m] = len(v); result['valid'][m] = True
        result['mean'][m] = v.mean(); result['std'][m] = v.std()
        result['negative'][m] = np.mean(v < -ZERO_TOLERANCE)
        result['zero'][m] = np.mean(np.abs(v) <= ZERO_TOLERANCE)
        result['positive'][m] = np.mean(v > ZERO_TOLERANCE)
        result['quantiles'][m] = np.quantile(v,[0,.25,.5,.75,1])
    return result
