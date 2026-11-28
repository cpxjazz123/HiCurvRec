"""Issue #276 — recover deleted historical SIDs from git history (read-only) and bind them to their
Stage3 records by era + content evidence.

Identity evidence levels (recorded per candidate):
  L1 ledger   : the R2 ledger entry for the arm carries the sha256 of item_sids.json -> exact content match
  L2 collision: the SID's computed L3 collision equals metrics_corrected[arm].collision -> content match
  L3 era+name : only the directory name and the live window match -> weak, marked as such

Outputs: results/research/issue276/recovery/git_sids.json and recovered_sids/<arm>/item_sids.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

P = Path('results/research/issue276')
OUT = P / 'recovered_sids'
MON = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
       'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collision_of(content: bytes) -> tuple[float, int]:
    d = json.loads(content.decode('utf-8'))
    triples = {tuple(v[:3]) for v in d.values() if isinstance(v, list) and len(v) >= 3}
    n = len(d)
    return (1.0 - len(triples) / n if n else float('nan')), n


def rec_date(rec: dict) -> str | None:
    s = rec.get('test_final') or rec.get('stage3_final') or ''
    m = re.search(r'([A-Z][a-z]{2})-(\d{2})-(\d{4})', s)
    if not m:
        return None
    return f"{m.group(3)}-{MON[m.group(1)]:02d}-{m.group(2)}"


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main() -> None:
    inv = load_json('results/research/issue275/RUN_INVENTORY.json') or {}
    coh = load_json('results/research/issue270/COHORT.json') or {}
    corrected = {r['arm']: r for r in (load_json('results/research/sid_utilization/metrics_corrected.json') or [])}
    ledger = load_json('results/research/issue139/r2_ledger.json') or {}
    ledger_arm = {}
    for number, e in ledger.items():
        if not isinstance(e, dict):
            continue
        arm = e.get('arm') or e.get('key') or ''
        files = e.get('files') or {}
        ledger_arm[str(arm)] = {'number': number,
                                'sid_sha256': files.get('item_sids.json') if isinstance(files, dict) else None}

    records = []
    for r in inv.get('records', []):
        records.append({'channel': 'inventory', 'arm': r.get('arm'), 'recall': r.get('recall@10'),
                        'date': rec_date(r), 'sid_sha256': r.get('sid_sha256'),
                        'test_final': r.get('test_final')})
    for r in coh.get('rows', []):
        records.append({'channel': 'cohort', 'arm': r.get('arm'), 'recall': r.get('recall@10'),
                        'date': rec_date(r), 'sid_sha256': r.get('sid_sha256'),
                        'test_final': r.get('stage3_final')})

    named = ['r48k2', 'r48k1', 'r48k2u', 'BEHAVIOR_R48', 'behavior_residual_geometry_transport',
             'crowd_couser_w3', 'crowd_posattract_w3', 'resample', 'resample_fn_mask',
             'resample_l1only', 'resample_s43', 'resample_s44']
    arms = sorted({r['arm'] for r in records if r.get('arm')})
    extra = [a for a in corrected if a not in arms]
    arms = sorted(set(arms) | set(extra))

    out = []
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        if arm is None:
            continue
        # deletion events that removed */<arm>/item_sids.json
        log = sh('git', 'log', '--all', '--diff-filter=D', '--format=@%H %ad', '--date=short',
                 '--name-only', '--', f'*{arm}/item_sids.json')
        events, cur = [], None
        for line in log.splitlines():
            if line.startswith('@'):
                h, d = line[1:].split()
                cur = {'commit': h, 'deleted': d, 'paths': []}
                events.append(cur)
            elif line.strip() and cur is not None:
                cur['paths'].append(line.strip())
        rel_records = [r for r in records if r.get('arm') == arm and isinstance(r.get('recall'), (int, float))]
        versions = []
        for ev in events:
            for path in ev['paths']:
                try:
                    content = subprocess.run(['git', 'show', f"{ev['commit']}^:{path}"],
                                             capture_output=True, check=True).stdout
                except subprocess.CalledProcessError:
                    versions.append({'commit': ev['commit'], 'deleted': ev['deleted'], 'path': path,
                                     'error': 'blob missing in parent'})
                    continue
                h = sha(content)
                coll, n = collision_of(content)
                versions.append({'commit': ev['commit'], 'deleted': ev['deleted'], 'path': path,
                                 'sha256': h, 'collision': coll, 'n_items': n, 'bytes': len(content)})
        # bind each record to the live SID version at its run date
        binds = []
        for r in rel_records:
            want = [(v, i) for i, v in enumerate(versions) if v.get('sha256')]
            if not want:
                binds.append({'record': r, 'sid': None, 'why': 'no_sid_version_in_git'})
                continue
            if r['date']:
                after = [v for v, _ in want if v['deleted'] >= r['date']]
                chosen = after[0] if after else want[-1][0]
            else:
                chosen = want[-1][0]
            binds.append({'record': r, 'sid': chosen['sha256'], 'why': 'era'})
        # identity level of each chosen version
        for b in binds:
            v = next((x for x in versions if x.get('sha256') == b['sid']), None) if b['sid'] else None
            if v is None:
                b['identity'] = None
                continue
            le = ledger_arm.get(arm)
            corr = corrected.get(arm)
            if le and le.get('sid_sha256') == v['sha256']:
                b['identity'] = 'L1_ledger'
            elif corr and isinstance(corr.get('collision'), (int, float)) and \
                    abs(corr['collision'] - v['collision']) < 1e-12:
                b['identity'] = 'L2_collision'
            else:
                b['identity'] = 'L3_era_name'
        # materialise the chosen version
        for b in binds:
            if b['sid'] and not (OUT / arm / 'item_sids.json').exists():
                v = next(x for x in versions if x.get('sha256') == b['sid'])
                content = subprocess.run(['git', 'show', f"{v['commit']}^:{v['path']}"],
                                         capture_output=True, check=True).stdout
                (OUT / arm).mkdir(parents=True, exist_ok=True)
                (OUT / arm / 'item_sids.json').write_bytes(content)
        out.append({'arm': arm, 'records': rel_records, 'versions': versions, 'binds': binds,
                    'named_by_issue': any(z in arm for z in named) if arm else False})

    (P / 'recovery').mkdir(exist_ok=True)
    (P / 'recovery/git_sids.json').write_text(json.dumps({'schema': 'genrec.issue276.git_sids/1',
                                                          'arms': out}, indent=1) + '\n')
    print('arms', len(out), 'arms_with_versions', sum(1 for a in out if a['versions']),
          'binds', sum(len(a['binds']) for a in out),
          'identity_levels', {lvl: sum(1 for a in out for b in a['binds'] if b.get('identity') == lvl)
                              for lvl in ('L1_ledger', 'L2_collision', 'L3_era_name', None)})


if __name__ == '__main__':
    main()
