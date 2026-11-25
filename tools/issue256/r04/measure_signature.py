"""Measure the preregistered R04 signature on the exported SID.

The R04 contract predicts the level-3 refinement, not the level-1 partition:
the exported three-code table must have H(L3 | L1 L2) <= 1.2 bits and at most
12000 distinct triples, while the level-1 statistics stay put (absolute L1
prefix carry inside 0.0496..0.0696).  Failure criterion: H >= 1.5 bits.

Everything here is exact counting over the persisted hard SID; the fourth
column (the export collision counter) is not a semantic code and is excluded.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path
import sys

from tools.issue256.r04.identity import ARM, ROOT, EVIDENCE, RUN_ID, RESULT


def entropy(counts, total):
    return -sum(k / total * math.log2(k / total) for k in counts.values())


def main() -> int:
    sid_path = RESULT / 'item_sids.json'
    payload = json.loads(sid_path.read_text())
    rows = [tuple(int(v) for v in row[:3]) for row in payload.values()]
    total = len(rows)
    h1 = entropy(collections.Counter(row[0] for row in rows), total)
    h12 = entropy(collections.Counter(row[:2] for row in rows), total)
    h123 = entropy(collections.Counter(rows), total)
    distinct = len(set(rows))
    metrics_path = ROOT / 'results/stage2_RQ-VAE' / ARM / 'stage2_metrics' / (
        hashlib.sha256(sid_path.read_bytes()).hexdigest() + '.json')
    layered = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    catalogue = layered.get('catalogue', {})
    validation = layered.get('validation', {})
    carry = validation.get('l1_positive_hit_rate')
    measured = {
        'items': total,
        'H_L1': round(h1, 6),
        'H_L2_given_L1': round(h12 - h1, 6),
        'H_L3_given_L1L2': round(h123 - h12, 6),
        'H_L1L2L3': round(h123, 6),
        'distinct_L1': len({row[0] for row in rows}),
        'distinct_triples': distinct,
        'collision': round(1.0 - distinct / total, 6),
        'absolute_l1_prefix_carry': carry,
        'distinct_two_code_prefixes': catalogue.get('distinct_l2_prefixes'),
        'l3_collision_layered': catalogue.get('l3_collision'),
        'l3_refinement_gain_bits': catalogue.get('l3_refinement_gain_bits'),
    }
    prediction = {
        'H_L3_given_L1L2_max_bits': 1.2,
        'distinct_triples_max': 12000,
        'absolute_l1_prefix_carry_window': [0.0496, 0.0696],
        'failure_threshold_bits': 1.5,
    }
    signature_met = (measured['H_L3_given_L1L2'] <= 1.2
                     and measured['distinct_triples'] <= 12000)
    carry_in_window = (carry is not None
                       and 0.0496 <= float(carry) <= 0.0696)
    verdict = {
        'schema': 'genrec.issue256.r04.signature/1',
        'round_id': 'r04',
        'run_id': RUN_ID,
        'arm': ARM,
        'sid': {'path': str(sid_path), 'sha256': hashlib.sha256(sid_path.read_bytes()).hexdigest()},
        'layered_metrics': ({'path': str(metrics_path),
                             'sha256': hashlib.sha256(metrics_path.read_bytes()).hexdigest()}
                            if metrics_path.is_file() else None),
        'measured': measured,
        'prediction': prediction,
        'signature_met': signature_met,
        'level1_unchanged': carry_in_window,
        'falsified': (measured['H_L3_given_L1L2'] >= 1.5) or (not signature_met),
    }
    (EVIDENCE / 'SIGNATURE.json').write_text(json.dumps(verdict, indent=2) + '\n')
    print(json.dumps({'measured': measured, 'signature_met': signature_met,
                      'level1_unchanged': carry_in_window, 'falsified': verdict['falsified']},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
