"""Issue #276 — corrected-forward vector features (Linear bias included, Nov-19 forward).

For every stored vector_stats_all row whose SID and checkpoint are both on disk:
  PR(z)        participation ratio of the covariance of the encoder output z (bias included)
  PR(hard q)   participation ratio of the covariance of the hard-quantised vector
  quant_err    mean ||z - sum_l codebook_l[token_l]||
  recon_mse    mean ||decoder(q) - x||^2 with the corrected decoder forward (bias included)
  recon_cos    mean cosine(decoder(q), x)
  collision    computed from the ordered first three codes
Writes results/research/issue276/vector_forward.json. Read-only on inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

P = Path('results/research/issue276')
EMB = Path('stage1_GeneEmbedding/output/sentence_t5.npy')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mlp(sd, prefix, x, relu_last=False):
    layers = sorted((int(k.split('.')[2]), v, sd.get(k.replace('weight', 'bias')))
                    for k, v in sd.items()
                    if k.startswith(prefix) and k.endswith('weight') and v.ndim == 2)
    if not layers:
        return None
    if x.shape[1] != layers[0][1].shape[1]:
        return None
    for i, (_, w, b) in enumerate(layers):
        x = x @ w.t()
        if b is not None:
            x = x + b
        if i < len(layers) - 1 or relu_last:
            x = torch.relu(x)
    return x


def codebooks(sd):
    euclid = [v for k, v in sorted(sd.items()) if 'vq_layers' in k and k.endswith('embed.weight')]
    mixed = [v for k, v in sorted(sd.items()) if '.levels.' in k and k.endswith('embed.weight')]
    return euclid or mixed


def pr(z: torch.Tensor) -> float:
    zc = z - z.mean(0, keepdim=True)
    cov = zc.t() @ zc / max(1, z.shape[0] - 1)
    lam = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def main() -> None:
    rows_in = json.load(open('results/research/sid_utilization/vector_stats_all.json'))
    emb = torch.tensor(np.load(EMB).astype('float32'), device=DEV)
    print('embeddings', tuple(emb.shape), 'device', DEV, flush=True)
    out = []
    for r in rows_in:
        sid_path, ckpt = Path(r['sid_path']), Path(r['ckpt'])
        rec = {'arm': r['arm'], 'recall': r['recall'], 'sid_path': str(sid_path), 'ckpt': str(ckpt)}
        if not (sid_path.is_file() and ckpt.is_file()):
            rec['status'] = 'missing_sid_or_ckpt'
            out.append(rec)
            continue
        try:
            sid = json.loads(sid_path.read_text())
            keys = list(sid)
            width = min(3, min(len(v) for v in sid.values()))
            tokens = torch.tensor([[int(c) for c in sid[k][:width]] for k in keys], dtype=torch.long, device=DEV)
            state = torch.load(ckpt, map_location='cpu', weights_only=False)
            sd = state.get('state_dict', state)
            if isinstance(sd, dict) and any(isinstance(v, dict) for v in sd.values()):
                for key in ('model', 'state_dict', 'net', 'module'):
                    if isinstance(sd.get(key), dict):
                        sd = sd[key]
                        break
            sd = {k: v.to(DEV) for k, v in sd.items() if isinstance(v, torch.Tensor)}
            z = mlp(sd, 'encoder.mlp', emb)
            if z is None or z.shape[0] != tokens.shape[0]:
                rec['status'] = 'encoder_mismatch'
                out.append(rec)
                continue
            tables = [t.to(DEV) for t in codebooks(sd)]
            q = None
            if tables and all(t.shape[1] == z.shape[1] for t in tables[:tokens.shape[1]]):
                for level, table in enumerate(tables[:tokens.shape[1]]):
                    picked = table[tokens[:, level]]
                    q = picked if q is None else q + picked
            rec.update({'status': 'ok', 'n_items': int(tokens.shape[0]),
                        'sid_sha256': sha(sid_path), 'ckpt_sha256': sha(ckpt),
                        'pr_z': pr(z), 'tables': len(tables)})
            if q is not None:
                rec['pr_q'] = pr(q)
                rec['quant_err'] = float(torch.linalg.vector_norm(z - q, dim=1).mean())
                recon = mlp(sd, 'decoder.mlp', q)
                if recon is not None and recon.shape == emb.shape:
                    rec['recon_mse'] = float(((recon - emb) ** 2).mean())
                    rec['recon_cos'] = float(torch.nn.functional.cosine_similarity(recon, emb, dim=1).mean())
            triples = {tuple(int(c) for c in v[:3]) for v in sid.values()}
            rec['collision'] = 1.0 - len(triples) / max(1, len(sid))
            # agreement with the stored (bias-less) series
            rec['stored_pr'] = r.get('pr')
        except Exception as exc:
            rec['status'] = f'error: {exc!r}'[:200]
        out.append(rec)
        print(rec.get('status'), rec['arm'][:40],
              rec.get('pr_z'), rec.get('pr_q'), rec.get('quant_err'), rec.get('recon_mse'), flush=True)
        del sd
        if DEV == 'cuda':
            torch.cuda.empty_cache()
    (P / 'vector_forward.json').write_text(json.dumps({'schema': 'genrec.issue276.vector_forward/1',
                                                       'device': DEV, 'rows': out}, indent=1) + '\n')
    print('rows', len(out), 'ok', sum(1 for r in out if r.get('status') == 'ok'))


if __name__ == '__main__':
    main()
