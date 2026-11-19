"""Vector-space features of the item representation, per arm.

Computes, from the arm's checkpoint and the stage 1 embeddings: the encoder output
norm (``norm_z``), the codeword-sum norm (``norm_q``), the relative quantisation
error, the k-NN hub ratio, and keeps the participation ratio / separation that
``vector_stats_all.py`` already produced.  Writes ``vector_features.json``.
"""
import json

import numpy as np
import torch
from pathlib import Path

OUT = Path('results/research/sid_utilization/vector_features.json')
SOURCE = Path('results/research/sid_utilization/vector_stats_all.json')
EMBEDDINGS = Path('stage1_GeneEmbedding/output/sentence_t5.npy')
CHUNK = 8192


def encoder(sd, embeddings):
    weights = sorted((int(k.split('.')[2]), v) for k, v in sd.items()
                     if k.startswith('encoder.mlp') and k.endswith('weight'))
    x = torch.as_tensor(embeddings)
    with torch.no_grad():
        for index, (_, weight) in enumerate(weights):
            x = x @ weight.t()
            if index < len(weights) - 1:
                x = torch.relu(x)
    return x


def tables(sd):
    return ([v for k, v in sorted(sd.items()) if 'vq_layers' in k and k.endswith('embed.weight')]
            or [v for k, v in sorted(sd.items()) if '.levels.' in k and k.endswith('embed.weight')])


def main():
    rows = [r for r in json.loads(SOURCE.read_text())
            if np.isfinite(r['sep']) and np.isfinite(r['quant_err']) and r['quant_err'] > 0]
    embeddings = np.load(EMBEDDINGS).astype('float32')
    rng = np.random.default_rng(0)
    for row in rows:
        payload = json.loads(Path(row['sid_path']).read_text())
        keys = list(payload)
        depth = min(3, min(len(value) for value in payload.values()))
        tokens = np.array([payload[key][:depth] for key in keys], dtype=np.int64)
        state = torch.load(row['ckpt'], map_location='cpu', weights_only=False)
        state = state.get('state_dict', state)
        with torch.no_grad():
            z = encoder(state, embeddings)
            quantised = None
            for level, table in enumerate(tables(state)[:depth]):
                picked = table[torch.as_tensor(tokens[:, level])]
                quantised = picked if quantised is None else quantised + picked
        zn = float(torch.linalg.vector_norm(z, dim=1).mean())
        qn = float(torch.linalg.vector_norm(quantised, dim=1).mean())
        error = float(torch.linalg.vector_norm(z - quantised, dim=1).mean())
        wide = z.numpy()
        sample = rng.choice(len(wide), min(3000, len(wide)), replace=False)
        distance = ((wide[sample][:, None, :] - wide[sample][None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(distance, np.inf)
        neighbours = np.argsort(distance, axis=1)[:, :10]
        indegree = np.bincount(neighbours.ravel(), minlength=len(sample))
        row.update({'norm_z': zn, 'norm_q': qn,
                    'rel_qerr': error / zn if zn else float('nan'),
                    'qerr_over_qnorm': error / qn if qn else float('nan'),
                    'hub': float(indegree.max() / max(indegree.mean(), 1e-9))})
        print(f"{row['recall']:.4f} {row['arm'][:38]:38s} norm_z {zn:.3f} rel_qerr "
              f"{row['rel_qerr']:.2f} hub {row['hub']:.1f}", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print('wrote', OUT, 'arms:', len(rows))


if __name__ == '__main__':
    main()
