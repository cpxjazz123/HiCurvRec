"""Issue #276 — read-only recovery of historical SID identity (channels: disk, R2 cache, ledger,
inventory, cohort). Writes RECOVERY.json / RECOVERY.md under results/research/issue276/.

Identity rule: a SID is identified by the sha256 of its item_sids.json content. A record binds to a
SID only when the record's sid_sha256 equals that content hash; names and paths are never enough.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

P = Path('results/research/issue276')
REPO = Path('.')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def index_sids() -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for p in REPO.glob('results/**/item_sids.json'):
        # skip the analysis trees themselves (no SID exports there)
        if 'issue276' in str(p):
            continue
        idx.setdefault(sha256(p), []).append(str(p))
    return idx


def load_json(p: str | Path):
    try:
        return json.load(open(p))
    except Exception:
        return None


def inventory_records():
    d = load_json('results/research/issue275/RUN_INVENTORY.json') or {}
    return d.get('records', []), d.get('units', [])


def cohort_rows():
    d = load_json('results/research/issue270/COHORT.json') or {}
    return d.get('rows', [])


def ledger_entries():
    d = load_json('results/research/issue139/r2_ledger.json') or {}
    out = []
    for number, e in d.items():
        if not isinstance(e, dict):
            continue
        files = e.get('files') or {}
        sid_sha = files.get('item_sids.json') if isinstance(files, dict) else None
        out.append({
            'number': number,
            'arm': e.get('arm') or e.get('key') or '',
            'key': e.get('key'),
            'sid_sha256': sid_sha,
            'recall_source': e.get('recall_source'),
            'recall_paired': e.get('recall_paired'),
            'retired': e.get('retired'),
            'withdrawn': e.get('withdrawn'),
            'note': e.get('note'),
        })
    return out


def main() -> None:
    P.mkdir(parents=True, exist_ok=True)
    idx = index_sids()
    inv_records, inv_units = inventory_records()
    cohort = cohort_rows()
    ledger = ledger_entries()
    ledger_by_sha = {e['sid_sha256']: e for e in ledger if e.get('sid_sha256')}

    records = []
    for r in inv_records:
        records.append({
            'channel': 'inventory',
            'arm': r.get('arm'),
            'recall@10': r.get('recall@10'),
            'sid_sha256': r.get('sid_sha256'),
            'sid': r.get('sid'),
            'checkpoint': r.get('checkpoint'),
            'checkpoint_sha256': r.get('checkpoint_sha256'),
            'test_final': r.get('test_final'),
            'world_size': r.get('world_size'),
            'epochs': r.get('epochs'),
            'record_verified': r.get('record_verified'),
            'archive_binding': r.get('archive_binding'),
        })
    for r in cohort:
        records.append({
            'channel': 'cohort',
            'arm': r.get('arm'),
            'recall@10': r.get('recall@10'),
            'sid_sha256': r.get('sid_sha256'),
            'sid': r.get('sid'),
            'checkpoint': r.get('checkpoint'),
            'checkpoint_sha256': r.get('checkpoint_sha256'),
            'test_final': r.get('stage3_final'),
            'world_size': None,
            'epochs': None,
            'record_verified': r.get('record_verified'),
            'binding': r.get('binding'),
        })

    # attach SID existence and ledger evidence
    for r in records:
        s = r.get('sid_sha256')
        r['sid_on_disk'] = idx.get(s, []) if s else []
        le = ledger_by_sha.get(s) if s else None
        r['r2_number'] = le['number'] if le else None
        r['r2_arm'] = le['arm'] if le else None

    # ledger entries with no matching on-disk SID still carry a content hash: report them
    ledger_orphans = [e for e in ledger if e.get('sid_sha256') and e['sid_sha256'] not in idx
                      and not any((Path('results/research/issue139/_r2_download') / e['number'] / 'item_sids.json').exists()
                                  for _ in [0])]
    r2_cache = {}
    for d in (REPO / 'results/research/issue139/_r2_download').glob('*'):
        f = d / 'item_sids.json'
        if f.is_file():
            r2_cache[d.name] = sha256(f)

    # ---- the 12 arms named by the issue ----
    named = ['r48k2', 'r48k1', 'r48k2u', 'BEHAVIOR_R48',
             'behavior_residual_geometry_transport', 'crowd_couser_w3', 'crowd_posattract_w3',
             'resample', 'resample_fn_mask', 'resample_l1only', 'resample_s43', 'resample_s44']
    corrected = {r['arm']: r for r in (load_json('results/research/sid_utilization/metrics_corrected.json') or [])}
    named_report = []
    for w in named:
        recs = [r for r in records if w in (r.get('arm') or '')]
        led = [e for e in ledger if w in (e.get('arm') or '')]
        corr = {k: v for k, v in corrected.items() if w in k}
        named_report.append({
            'query': w,
            'records': recs,
            'ledger': led,
            'metrics_corrected': corr,
            'bound_records': sum(1 for r in recs if r['sid_on_disk']),
            'recovered_sids': sorted({s for r in recs for s in r['sid_on_disk']}),
        })

    high = [r for r in records if isinstance(r.get('recall@10'), (int, float)) and r['recall@10'] > 0.06]
    out = {
        'schema': 'genrec.issue276.recovery/1',
        'sid_files_indexed': len(idx),
        'records': len(records),
        'inventory_units': len(inv_units),
        'cohort_rows': len(cohort),
        'ledger_entries': len(ledger),
        'r2_cache_sids': r2_cache,
        'named_arms': named_report,
        'high_recall_records': high,
        'high_recall_bound': sum(1 for r in high if r['sid_on_disk']),
        'ledger_orphans': len(ledger_orphans),
    }
    (P / 'recovery/RECOVERY.json').parent.mkdir(parents=True, exist_ok=True)
    (P / 'recovery/RECOVERY.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print('records', len(records), 'high', len(high), 'high_bound', out['high_recall_bound'],
          'sid_files', len(idx), 'r2_cache', len(r2_cache), 'ledger_orphans', len(ledger_orphans))
    for n in named_report:
        print(f"{n['query']:42s} records={len(n['records']):3d} bound={n['bound_records']:3d} "
              f"ledger={len(n['ledger']):2d} corrected={len(n['metrics_corrected'])} sids={len(n['recovered_sids'])}")


if __name__ == '__main__':
    main()
