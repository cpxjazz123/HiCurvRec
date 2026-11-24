"""Evaluation-only layered SID record; no training or Stage3-score reads.

Definitions are frozen in results/research/issue252/LAYER_METRICS_CONTRACT.md.
The unavailable legacy G3 is marked missing, never guessed from a different
statistic. Exact uniform-negative expectations create no new sampled assets.
"""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from tools.issue252.archive_audit import sha256, validate_binding

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
ARM = 'TIGER_gnn_attention_sectional_rq'


def _rows(rows):
    rows = np.asarray(rows)
    if (rows.ndim != 2 or rows.shape[1] not in (3, 4) or not len(rows)
            or rows.dtype.kind not in 'iu' or np.any(rows[:, :3] < 0)):
        raise ValueError('SID requires nonempty integer global catalogue and three codes')
    return rows[:, :3]


def _na(reason):
    return {'status': 'NA', 'reason': reason}


def _entropy(counts, total):
    return -math.fsum((count / total) * math.log2(count / total) for count in counts)


def catalogue_metrics(rows):
    rows = _rows(rows)
    prefixes = [Counter(map(tuple, rows[:, :level])) for level in (1, 2, 3)]
    entropies = [_entropy(counts.values(), len(rows)) for counts in prefixes]
    return {'items': len(rows), 'distinct_l2_prefixes': len(prefixes[1]),
            'distinct_l3_triples': len(prefixes[2]),
            'l3_collision': 1. - len(prefixes[2]) / len(rows),
            'l3_refinement_gain_bits': entropies[2] - entropies[1],
            'prefix_entropy_bits': entropies,
            'codebook_occupancy': [len(np.unique(rows[:, level])) for level in range(3)]}


def validation_metrics(sids, examples):
    sids = _rows(sids)
    size = len(sids)
    prefixes = [[tuple(row[:level]) for row in sids] for level in (1, 2, 3)]
    counts = [Counter(values) for values in prefixes]
    positive, negative = [0, 0, 0], [0., 0., 0.]
    effective = empty = exhausted = total = 0
    for example in examples:
        total += 1
        history, seen = list(example['history']), list(example.get('seen_history', []))
        target = int(example['target'])
        if any(not 0 <= int(item) < size for item in (*history, *seen, target)):
            raise ValueError('validation item outside global catalogue')
        if not history:
            empty += 1
            continue
        anchor = int(history[-1])
        excluded = set(map(int, history)) | set(map(int, seen)) | {anchor, target}
        population = size - len(excluded)
        if not population:
            exhausted += 1
            continue
        effective += 1
        for level in range(3):
            prefix = prefixes[level][anchor]
            positive[level] += prefix == prefixes[level][target]
            remaining = counts[level][prefix] - sum(prefixes[level][item] == prefix for item in excluded)
            negative[level] += remaining / population
    positive_rates = [value / effective for value in positive] if effective else None
    negative_rates = [value / effective for value in negative] if effective else None
    er = (positive_rates[0] / negative_rates[0] if effective and negative_rates[0] > 0
          else _na('no usable rows or zero negative prefix probability'))
    if effective and positive[1] and negative[1]:
        positive_survival, negative_survival = positive[2] / positive[1], negative[2] / negative[1]
        difference = positive_survival - negative_survival
        log_enrichment = (math.log2(positive_survival / negative_survival)
                          if positive_survival > 0 and negative_survival > 0
                          else _na('zero conditional probability; finite logarithm undefined'))
    else:
        difference = log_enrichment = _na('no positive or negative L2-prefix survivor')
    return {'scope': 'recommendation valid last-history-to-target; exact legal uniform catalogue negatives',
            'rows': total, 'effective_rows': effective, 'excluded_empty_history': empty,
            'excluded_empty_negative_population': exhausted,
            'positive_hits_by_level': positive, 'expected_negative_mass_by_level': negative,
            'positive_hit_rates_by_level': positive_rates, 'negative_hit_probabilities_by_level': negative_rates,
            'l1_positive_hit_rate': positive_rates[0] if effective else _na('no usable rows'),
            'l1_negative_hit_probability': negative_rates[0] if effective else _na('no usable rows'),
            'l1_hard_er': er, 'negative_sampling': False,
            'conditional_g3': {'status': 'MISSING',
                'reason': 'legacy defining implementation removed; no surviving formula, not invented'},
            'conditional_log_enrichment_bits': log_enrichment,
            'conditional_prefix_survival_difference': difference}


def resolve_arm(repo, sid, arm=None):
    """The arm that owns this SID: an explicit value, else the immutable run's
    own arm, else the historical constant for a pre-workspace result path."""
    from tools.issue252.run_workspace import run_from_sid
    if arm is not None:
        if not isinstance(arm, str) or not arm.strip():
            raise ValueError('explicit arm must be a nonempty string')
        return arm
    run = run_from_sid(Path(repo).resolve(), Path(sid).resolve())
    return run['arm'] if run is not None else ARM


def record_layers(sid_path, *, repo=ROOT, arm=None):
    import pyarrow.parquet as pq
    from tools.issue252.final_evaluation import load_sid

    repo, sid = Path(repo).resolve(), Path(sid_path).resolve()
    if not sid.is_relative_to(repo / 'results/stage2_RQ-VAE'):
        raise ValueError('SID is outside formal results')
    binding = sid.with_name('archive_binding.json')
    receipt = json.loads(binding.read_text())
    validate_binding(binding, sid, sid.with_name('rqvae_best.pth'), receipt['source_tree'])
    arm = resolve_arm(repo, sid, arm)
    rows = load_sid(sid, 24587, (256, 256, 256))
    records = {}
    for name in ('train', 'valid'):
        path = repo / 'results/stage0_build_parquet' / (name + '.parquet')
        records[name] = {'path': str(path), 'sha256': sha256(path)}
    examples = pq.read_table(records['valid']['path'], columns=['history', 'seen_history', 'target']).to_pylist()
    digest = sha256(sid)
    output = repo / 'results/stage2_RQ-VAE' / arm / 'stage2_metrics' / (digest + '.json')
    report = {'schema': 'genrec.issue252.layer_metrics/1', 'arm': arm, 'run_id': receipt['run_id'],
              'sid': {'path': str(sid), 'sha256': digest}, 'inputs': records,
              'catalogue': catalogue_metrics(rows), 'validation': validation_metrics(rows, examples),
              'output': str(output), 'read_stage3_scores': False,
              'evaluator': {'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)},
              'contract': {'path': 'results/research/issue252/LAYER_METRICS_CONTRACT.md',
                           'sha256': sha256(repo / 'results/research/issue252/LAYER_METRICS_CONTRACT.md')}}
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text() != text:
            raise ValueError('refusing to overwrite historical layered metrics')
    else:
        with output.open('x') as handle:
            handle.write(text)
    return {'path': str(output), 'sha256': sha256(output), 'report': report}
