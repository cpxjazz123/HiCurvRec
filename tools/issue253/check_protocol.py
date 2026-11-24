"""Re-derive the frozen C1 protocol from the CURRENT candidate source.

The check is a CPU introspection of the actual model and trainer: shapes, the
expert population, the assignment floor, the levels that use Sinkhorn, and the
frozen training recipe. It writes C1_PROTOCOL.json with the source hashes it
read, so the credential can never be backdated to an earlier source.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
ARM = 'TIGER_three_expert_attention_rq'
TREE = ROOT/'stage2_RQ-VAE'/ARM
OUTPUT = ROOT/'results/research/issue253/C1_PROTOCOL.json'
FROZEN = {'input_dim': 768, 'latent_dim': 32,
          'books': [[256, 32], [256, 32], [256, 32]],
          'queries': [3, 32], 'curvatures': [-1, 0, 1], 'floor': 1,
          'sinkhorn_layers': [3], 'epochs': 3000, 'batch_per_rank': 1024, 'seed': 42}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sys.path[:0] = [str(TREE), str(ROOT)]
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    import torch
    import train_rqvae as training
    from model import geometry
    from model import layers as layer_module

    config = training._tokenizer_config()
    model = training.RQVAE(config, in_dim=FROZEN['input_dim'], distance=training.DISTANCE)
    books = [[int(layer.n_embed), int(layer.embed.weight.shape[1])]
             for layer in model.rq.vq_layers]
    queries = list(model.structure.attention.query.shape)
    # The production rule: the last level keeps Sinkhorn, earlier levels do not.
    last = len(model.rq.vq_layers) - 1
    sinkhorn_levels = [index + 1 for index, layer in enumerate(model.rq.vq_layers)
                       if bool(layer.use_sk) and index == last]
    facts = {
        'arm': training.ARM_NAME,
        'input_dim': int(model.encoder_sizes[0]),
        'latent_dim': int(model.encoder_sizes[-1]),
        'books': books,
        'queries': queries,
        'curvatures': list(geometry.CURVATURES),
        'floor': int(layer_module.ASSIGNMENT_FLOOR),
        'sinkhorn_layers': sinkhorn_levels,
        'layer_use_sk_flags': [bool(layer.use_sk) for layer in model.rq.vq_layers],
        'epochs': int(training.EPOCHS),
        'batch_per_rank': int(training.BATCH_SIZE_PER_RANK),
        'seed': int(training.SEED),
    }
    problems = []
    for key, value in FROZEN.items():
        if facts.get(key) != value:
            problems.append('%s is %r, the frozen value is %r' % (key, facts.get(key), value))
    if facts['arm'] != ARM:
        problems.append('arm identity is %r' % facts['arm'])
    payload = {
        'status': 'FAIL' if problems else 'PASS',
        'detail': '; '.join(problems) or 'every frozen protocol value matches the current source',
        'command': [sys.executable, '-m', 'tools.issue253.check_protocol'],
        'pid': os.getpid(),
        'training': False,
        'cuda_used': bool(torch.cuda.is_available() and torch.cuda.is_initialized()),
        **facts,
        'config_sha256': training.CONFIG.sha256,
        'sources': {str((TREE/name).relative_to(ROOT)): sha256(TREE/name)
                    for name in ('train_rqvae.py', 'train_config.py', 'config/train_config.json',
                                 'model/mechanism.py', 'model/geometry.py', 'model/layers.py',
                                 'model/structure.py', 'model/attention.py',
                                 'assignment_floor.py')},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'status': payload['status'],
                      'detail': payload['detail']}, indent=2))
    return 0 if payload['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
