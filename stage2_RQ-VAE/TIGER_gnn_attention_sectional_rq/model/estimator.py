"""Declared finite-population training estimator; exact full evaluation stays separate."""
import hashlib
import numpy as np
import torch

SEED = 42
TRIPLES_PER_UPDATE = 32
REFERENCES_PER_UPDATE = 32
MIN_AUX_ITEMS = 256


def sampled_values(q, triples, references, squared, paired_squared=None):
    """A.1 via endpoint distance moments, with all sampled-q gradients retained."""
    triples = torch.as_tensor(triples, dtype=torch.long, device=q.device)
    references = torch.as_tensor(references, dtype=torch.long, device=q.device)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError('triples must have three local item IDs')
    if references.ndim != 1 or references.numel() == 0:
        raise ValueError('reference sample must not be empty')
    if not triples.numel():
        return q.new_empty((0,))
    if bool((triples < 0).any()) or bool((triples >= len(q)).any()):
        raise ValueError('triple outside auxiliary item population')
    if bool((references < 0).any()) or bool((references >= len(q)).any()):
        raise ValueError('reference outside auxiliary item population')
    endpoints, inverse = torch.unique(triples.reshape(-1), sorted=True, return_inverse=True)
    moments = squared(q[references], q[endpoints]).mean(0)[inverse].reshape(-1, 3)
    b, c = q[triples[:, 1]], q[triples[:, 2]]
    pair = squared(b, c).diagonal() if paired_squared is None else paired_squared(b, c)
    return moments[:, 0] - (moments[:, 1] + moments[:, 2]) / 2 + pair / 4


class StructuralSampler:
    """Replayable centre cycle and uniform draws from immutable complete populations."""
    def __init__(self, table, offsets, components):
        self.table = table
        self.offsets = np.asarray(offsets)
        self.components = np.asarray(components)
        self.n = len(self.components)
        if (self.offsets.shape != (self.n + 1,) or self.offsets.dtype.kind not in 'iu'
                or self.offsets[0] != 0 or self.offsets[-1] != len(table)
                or np.any(np.diff(self.offsets) < 0)):
            raise ValueError('invalid full-target offsets')
        if table.dtype.names != ('b', 'c', 'target') or self.n < MIN_AUX_ITEMS:
            raise ValueError('invalid target table or infeasible auxiliary assignment population')
        self.valid_centres = np.flatnonzero(np.diff(self.offsets))
        if not len(self.valid_centres):
            raise ValueError('no valid curvature centres')
        self.order = np.random.default_rng(SEED).permutation(self.valid_centres)
        self.references = {label: np.flatnonzero(self.components == label)
                           for label in np.unique(self.components)}
        self.catalogue = np.arange(self.n, dtype=np.int64)
        self.seen_rows = np.zeros(len(table), dtype=bool)
        self.seen_centres = np.zeros(self.n, dtype=bool)
        self.seen_references = np.zeros(self.n, dtype=bool)
        self.seen_items = np.zeros(self.n, dtype=bool)
        self.unique_rows = self.unique_centres = self.unique_references = self.unique_items = 0
        self.updates = self.sampled_rows = self.sampled_references = 0
        self.digest = hashlib.sha256()
        self.begin_epoch()

    def begin_epoch(self):
        self.epoch_updates = self.epoch_rows = self.epoch_references = 0
        self.epoch_centres = []

    def draw(self, global_step, rank, world_size, *, track=True):
        if global_step < 0 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError('invalid sampling step/rank/world size')
        centre = int(self.order[(global_step * world_size + rank) % len(self.order)])
        rng = np.random.default_rng(np.random.SeedSequence([SEED, int(global_step), int(rank)]))
        start, stop = (int(x) for x in self.offsets[centre:centre + 2])
        row_ids = start + rng.choice(stop - start, min(TRIPLES_PER_UPDATE, stop - start), replace=False)
        rows = self.table[row_ids]
        triples = np.column_stack((np.full(len(rows), centre, dtype=np.int64), rows['b'], rows['c']))
        population = self.references[self.components[centre]]
        references = rng.choice(population, min(REFERENCES_PER_UPDATE, len(population)), replace=False)
        if (not np.isfinite(rows['target']).all() or (triples < 0).any()
                or (triples >= self.n).any()
                or not np.all(self.components[triples] == self.components[centre])):
            raise ValueError('invalid or cross-component frozen target')
        ids = np.unique(np.r_[triples.reshape(-1), references])
        needed = MIN_AUX_ITEMS - len(ids)
        if needed > 0:
            remaining = np.setdiff1d(self.catalogue, ids, assume_unique=True)
            ids = np.sort(np.r_[ids, rng.choice(remaining, needed, replace=False)])
        draw = {'centre': centre, 'ids': ids, 'triples_global': triples,
                'triples': np.searchsorted(ids, triples), 'references_global': references,
                'references': np.searchsorted(ids, references), 'target': rows['target'].copy(),
                'row_ids': row_ids, 'reference_population': len(population),
                'centre_population': len(self.order), 'triple_population': stop - start,
                'global_step': int(global_step), 'rank': int(rank), 'world_size': int(world_size)}
        if track:
            self._record(draw)
        return draw

    def _record(self, draw):
        centre, rows, references, ids = (draw[key] for key in
                                        ('centre', 'row_ids', 'references_global', 'ids'))
        self.unique_centres += int(not self.seen_centres[centre])
        self.unique_rows += int((~self.seen_rows[rows]).sum())
        self.unique_references += int((~self.seen_references[references]).sum())
        self.unique_items += int((~self.seen_items[ids]).sum())
        self.seen_centres[centre] = True
        self.seen_rows[rows] = self.seen_references[references] = self.seen_items[ids] = True
        self.updates += 1
        self.sampled_rows += len(rows)
        self.sampled_references += len(references)
        self.epoch_updates += 1
        self.epoch_rows += len(rows)
        self.epoch_references += len(references)
        self.epoch_centres.append(centre)
        for values in (np.array([draw['global_step'], draw['rank'], centre], dtype='<i8'),
                       rows, references, ids):
            self.digest.update(np.asarray(values, dtype='<i8').tobytes())

    def receipt(self):
        return {'updates': self.updates, 'sampled_target_occurrences': self.sampled_rows,
                'sampled_references': self.sampled_references,
                'unique_centres': self.unique_centres, 'centre_population': len(self.order),
                'unique_target_rows': self.unique_rows, 'target_population': len(self.table),
                'unique_references': self.unique_references, 'unique_aux_items': self.unique_items,
                'epoch_updates': self.epoch_updates, 'epoch_target_occurrences': self.epoch_rows,
                'epoch_reference_draws': self.epoch_references, 'epoch_centres': list(self.epoch_centres),
                'sampling_sha256': self.digest.hexdigest(), 'seed': SEED,
                'objective': 'biased stochastic training estimator; not full-catalogue loss'}
