"""Item-vector discriminativeness vs Stage3 recall, over every reachable arm.

Inputs : stage1 sentence embeddings, per-arm rqvae_best.pth, item_sids.json.
Outputs: results/research/sid_utilization/vector_stats_all.json
"""
import collections
import json

import numpy as np
import torch
from pathlib import Path

ROOT = Path('.')
OUT = ROOT / 'results/research/sid_utilization/vector_stats_all.json'


def norm(name):
    name = name.lower()
    out = {name, name.replace('tiger_', ''), name.replace('tiger_sample_', '')}
    if '__' in name:
        out |= {name.split('__')[-1], name.split('__')[0]}
    return out


def recall_index():
    index = {}
    for f in (ROOT / 'results/stage3_T5Train').glob('**/test_final.json'):
        arm = f.relative_to(ROOT / 'results/stage3_T5Train').parts[0]
        try:
            value = json.loads(f.read_text()).get('test_recall@10')
        except Exception:
            continue
        if value is not None:
            index[arm] = max(index.get(arm, -1.0), float(value))
    return index


def ledger_names():
    path = ROOT / 'results/research/issue139/r2_ledger.json'
    if not path.is_file():
        return {}
    ledger = json.loads(path.read_text())
    out = {}
    for key, entry in ledger.items():
        pair = entry.get('pair') or {}
        out[key] = pair.get('variant') or pair.get('stage2_rel') or entry.get('arm')
    return out


def encoder(sd, embeddings):
    weights = sorted(
        (int(k.split('.')[2]), v) for k, v in sd.items()
        if k.startswith('encoder.mlp') and k.endswith('weight') and v.ndim == 2)
    if not weights:
        return None
    if weights[0][1].shape[1] != embeddings.shape[1]:
        return None
    x = embeddings
    for i, (_, w) in enumerate(weights):
        x = x @ w.t()
        if i < len(weights) - 1:
            x = torch.relu(x)
    return x


def codebooks(sd):
    euclid = [v for k, v in sorted(sd.items()) if 'vq_layers' in k and k.endswith('embed.weight')]
    mixed = [v for k, v in sorted(sd.items()) if '.levels.' in k and k.endswith('embed.weight')]
    return euclid or mixed


def vector_stats(z, tokens, tables):
    rng = np.random.default_rng(0)
    zc = z - z.mean(0, keepdims=True)
    cov = zc.T @ zc / max(1, z.shape[0] - 1)
    lam = np.linalg.eigvalsh(cov).clip(1e-12, None)
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    sample = rng.choice(len(z), min(2000, len(z)), replace=False)
    a, b = z[sample[:1000]], z[sample[1000:]]
    between = float(np.linalg.norm(a - b, axis=1).mean())
    groups = collections.defaultdict(list)
    for position, token in enumerate(tokens):
        groups[tuple(token)].append(position)
    within = []
    for group in [g for g in groups.values() if len(g) >= 2][:400]:
        i, j = rng.choice(group, 2, replace=False)
        within.append(np.linalg.norm(z[i] - z[j]))
    within_mean = float(np.mean(within)) if within else float('nan')
    quant_err = float('nan')
    if tables:
        try:
            q = None
            for level, table in enumerate(tables[:tokens.shape[1]]):
                picked = table[torch.tensor(tokens[:, level])]
                q = picked if q is None else q + picked
            if q.shape[1] == z.shape[1]:
                quant_err = float(torch.linalg.vector_norm(z - q, dim=1).mean())
        except Exception:
            pass
    return {'pr': pr, 'between': between, 'within': within_mean,
            'sep': between / within_mean if within_mean and within_mean > 0 else float('nan'),
            'quant_err': quant_err,
            'distinct_ratio': len(groups) / len(z),
            'n_codes': [int(t.max()) + 1 for t in tables] if tables else []}


def main():
    embeddings = torch.tensor(np.load('stage1_GeneEmbedding/output/sentence_t5.npy').astype('float32'))
    print('embeddings', tuple(embeddings.shape), flush=True)
    recalls = recall_index()
    r2_names = ledger_names()
    seen = {}
    rows = []

    for sid_path in sorted(ROOT.glob('results/**/item_sids.json')):
        if '_admission_' in str(sid_path):
            continue
        folder = sid_path.parent
        ckpt = folder / 'rqvae_best.pth'
        if not ckpt.is_file():
            continue
        parts = str(folder).split('/')
        if '_r2_download' in parts:
            key = parts[parts.index('_r2_download') + 1]
            name = r2_names.get(key)
            if name is None:
                continue
            recall = None
            summary = folder / 'recall.json'
            if summary.is_file():
                try:
                    recall = float(json.loads(summary.read_text())['test_recall@10'])
                except Exception:
                    recall = None
            if recall is None:
                recall = next((v for k, v in recalls.items() if k.lower() == name.lower()), None)
        else:
            name = folder.name
            recall = next((v for k, v in recalls.items() if name in norm(k)), None)
        if recall is None or name in seen:
            continue
        seen[name] = True
        sid = json.loads(sid_path.read_text())
        keys = list(sid)
        width = min(3, min(len(v) for v in sid.values()))
        tokens = np.array([sid[k][:width] for k in keys], dtype=np.int64)
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = state.get('state_dict', state)
        z = encoder(sd, embeddings)
        if z is None or z.shape[0] != len(tokens):
            print('skip', name, 'encoder mismatch', flush=True)
            continue
        stats = vector_stats(z, tokens, codebooks(sd))
        rows.append({'arm': name, 'recall': recall, 'sid_path': str(sid_path), 'ckpt': str(ckpt), **stats})
        print(f"{recall:.4f}  {name[:44]:44s} PR {stats['pr']:5.2f} sep {stats['sep']:.2f} "
              f"qerr {stats['quant_err']:.3f} distinct {stats['distinct_ratio']:.3f}", flush=True)

    OUT.write_text(json.dumps(rows, indent=1))
    print(f'total arms: {len(rows)}  -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
