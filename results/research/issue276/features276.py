"""Issue #276 — compute frozen SID-only layer features for SIDs not covered by #275.

Reuses exactly the #275 evaluator path (`tools.issue252.layer_metrics` + `final_evaluation.load_sid`)
and the same parquet inputs; writes one JSON per SID content hash under
results/research/issue276/features/layer/<sha256>.json (never overwrites an existing file).

Candidate SIDs: every `item_sids.json` under results/research/issue139/_r2_download/ and every
recovered copy under results/research/issue276/recovered_sids/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path('.').resolve()
sys.path.insert(0, str(REPO))
from tools.issue252 import layer_metrics as lm          # noqa: E402  (frozen evaluator)
from tools.issue252.final_evaluation import load_sid    # noqa: E402  (frozen loader)

OUT = REPO / 'results/research/issue276/features/layer'
OUT.mkdir(parents=True, exist_ok=True)
VALID = REPO / 'results/stage0_build_parquet/valid.parquet'
TRAIN = REPO / 'results/stage0_build_parquet/train.parquet'


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon_mapping(rows) -> dict:
    """item -> ordered first three codes (the discrete SID identity)."""
    return {int(k): [int(c) for c in v[:3]] for k, v in rows.items()}


def main() -> None:
    candidates = []
    for root in (REPO / 'results/research/issue139/_r2_download',
                 REPO / 'results/research/issue276/recovered_sids',
                 REPO / 'results/research/stage3_queue/_stage'):
        candidates.extend(sorted(root.glob('*/item_sids.json')))
    candidates.extend(sorted(REPO.glob('results/stage2_RQ-VAE/**/item_sids.json')))
    seen, todo = set(), []
    for f in candidates:
        h = sha_file(f)
        if h in seen:
            continue
        seen.add(h)
        if (OUT / f'{h}.json').exists():
            continue
        todo.append((h, f))
    print('candidates', len(candidates), 'unique_new', len(todo), flush=True)

    examples = pq.read_table(VALID, columns=['history', 'seen_history', 'target']).to_pylist()
    rows_out = []
    for i, (h, f) in enumerate(todo):
        try:
            raw = json.loads(f.read_text())
            normalized = False
            try:
                rows = load_sid(f, 24587, (256, 256, 256))
            except ValueError:
                # legacy export shape (5 integer columns). The discrete SID is the ordered first
                # three codes; the trailing counters are not part of it. Keep the first three and
                # zero the counter, and record the normalisation explicitly.
                rows = [[int(c) for c in raw[k][:3]] + [0] for k in sorted(raw, key=int)]
                normalized = True
            rec = {'schema': 'genrec.issue276.layer_features/1',
                   'sid': {'path': str(f), 'sha256': h},
                   'mapping_sha256': hashlib.sha256(
                       json.dumps(canon_mapping(raw), sort_keys=True).encode()).hexdigest(),
                   'format_normalized_from': {'path': str(f), 'columns': len(next(iter(raw.values())))}
                       if normalized else None,
                   'catalogue': lm.catalogue_metrics(rows),
                   'validation': lm.validation_metrics(rows, examples),
                   'evaluator': {'path': str(Path(lm.__file__)), 'sha256': sha_file(Path(lm.__file__))},
                   'inputs': {'valid': {'path': str(VALID), 'sha256': sha_file(VALID)},
                              'train': {'path': str(TRAIN), 'sha256': sha_file(TRAIN)}}}
            (OUT / f'{h}.json').write_text(json.dumps(rec, indent=1, allow_nan=False) + '\n')
            rows_out.append({'sid_sha256': h, 'status': 'computed', 'path': str(f)})
        except Exception as exc:  # recorded, never guessed
            rows_out.append({'sid_sha256': h, 'status': f'error: {exc!r}'[:200], 'path': str(f)})
        print(i + 1, 'done', rows_out[-1]['status'], flush=True)
    json.dump({'schema': 'genrec.issue276.layer_features_run/1', 'rows': rows_out},
              open(REPO / 'results/research/issue276/features/layer_run.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
