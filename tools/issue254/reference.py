"""Independent recomputation of the paired Stage3 metrics.

This module never imports the Stage3 trainer.  It re-derives every target rank
from the raw prediction files (the model's own candidate list) and rebuilds the
recall/NDCG sums from scratch, so that the trainer's own per-user dump and
shard sums are an independent check rather than the source of truth.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple



def load_prediction_rows(run_dir: Path) -> List[dict]:
    """Read every shard's prediction dump of one Stage3 generation.

    Each line is ``{"global_row", "target_item", "target_sid", "target_rank",
    "topk": [{"item", "sid_generated", "sid_canonical"}]}``.
    """
    rows: List[dict] = []
    for shard in sorted((run_dir/'predictions').glob('rank*.jsonl')):
        with shard.open('r', encoding='utf-8') as handle:
            for local_row_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                rows.append({
                    'global_row': int(payload['global_row']),
                    'target_item': int(payload['target_item']),
                    'target_sid': list(payload['target_sid']),
                    'dumped_rank': payload['target_rank'],
                    'candidates': [int(entry['item']) for entry in payload['topk']],
                    'candidate_sids': [list(entry['sid_canonical']) for entry in payload['topk']],
                    'generated_sids': [entry.get('sid_generated') for entry in payload['topk']],
                    'local_row_index': local_row_index,
                    'shard': shard.stem,
                })
    return rows


def recompute_rank(candidates: Sequence[int], target_item: int) -> Optional[int]:
    """The zero-based position of the target in the candidate list, else None."""
    for position, item in enumerate(candidates):
        if item == target_item:
            return position
    return None


def recompute_ranks(rows: Iterable[dict]) -> List[Optional[int]]:
    """Re-derive the target rank of every row from its candidate list."""
    return [recompute_rank(row['candidates'], row['target_item']) for row in rows]


def metric_sums(ranks: Sequence[Optional[int]], cutoffs: Sequence[int] = (5, 10)
                ) -> Tuple[Dict[int, float], Dict[int, float], int]:
    """Independent histogram reduction, not the production per-sample predicate."""
    from collections import Counter
    histogram = Counter(rank for rank in ranks if rank is not None)
    if any(not isinstance(rank, int) or rank < 0 for rank in histogram):
        raise ValueError('invalid rank')
    recalls = {k: float(sum(histogram.get(r, 0) for r in range(k))) for k in cutoffs}
    ndcgs = {k: math.fsum(histogram.get(r, 0) / math.log2(r + 2) for r in range(k))
             for k in cutoffs}
    return recalls, ndcgs, len(ranks)


def cross_check_dumped_ranks(rows: Sequence[dict], ranks: Sequence[Optional[int]]) -> List[dict]:
    """Every row whose recomputed rank differs from the trainer's dumped rank."""
    mismatches = []
    for row, rank in zip(rows, ranks):
        dumped = row['dumped_rank']
        if (dumped is None) != (rank is None) or (dumped is not None and int(dumped) != rank):
            mismatches.append({'global_row': row['global_row'], 'dumped': dumped,
                                'recomputed': rank})
    return mismatches
