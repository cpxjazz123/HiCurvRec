"""Read-only complete recommendation validation using the exact formal Stage3 model.

The framework launches four workers only after the formal Stage3 test has ended.
No checkpoint selection, training, test replay, or changed decoding semantics.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys

from tools.issue256.r03.identity import ROOT, ARM, RUN_ID, SOURCE, EVIDENCE
from tools.issue252.archive_audit import sha256
from tools.research.stage2 import launcher

PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10'
POINTER = EVIDENCE / 'FINAL_STAGE3.json'
OUTPUT = EVIDENCE / 'recommendation_validation'
VALID = ROOT / 'results/stage0_build_parquet/valid.parquet'
VALID_SHA256 = '48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9'
TRAINER = ROOT / 'stage3_T5Train/train_HG-Rec.py'
TRAINER_SHA256 = '54ca6bfbcd55a42e718660dd0004370a41a5e1c2c5038a86bbfc84f32b963690'


def record(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path)}


def _require_complete_population(population):
    import pyarrow.parquet as pq

    expected = pq.ParquetFile(VALID).metadata.num_rows
    if population <= 0 or population != expected:
        raise ValueError('validation population differs from the complete frozen split')


def inputs():
    pointer = json.loads(POINTER.read_text())
    test_path = Path(pointer['test_final']['path'])
    if record(test_path) != pointer['test_final']:
        raise ValueError('formal Stage3 result changed')
    final = json.loads(test_path.read_text())
    if (final['stage2_run_id'] != RUN_ID or final['stage3_completed_epochs'] != 150
            or final['stage3_world_size'] != 4 or final['n_eval'] != 57439
            or final['stage3_trainer_sha256'] != TRAINER_SHA256
            or sha256(TRAINER) != TRAINER_SHA256 or sha256(VALID) != VALID_SHA256):
        raise ValueError('validation is not paired with the frozen complete formal generation')
    config = final['stage3_input_config']
    if (config['beam_size'] != 20 or config['infer_size'] != 1024 or config['seed'] != 42
            or config['topk_list'] != [5, 10] or not config['no_eval'] or config['skip_test']):
        raise ValueError('frozen inference/configuration drift')
    sid = Path(final['sid_path'])
    if sha256(sid) != final['sid_sha256'] or Path(config['code_path']).resolve() != sid.resolve():
        raise ValueError('formal SID/configuration mismatch')
    checkpoint = Path(final['best_checkpoint'])
    checkpoint.relative_to((ROOT / 'results/stage3_T5Train' / ARM).resolve())
    sources = [TRAINER, Path(__file__)]
    for folder in ('model', 'data'):
        sources.extend(sorted((TRAINER.parent / folder).rglob('*.py')))
    identity = {'run_id': RUN_ID, 'stage3_run_id': final['stage3_run_id'],
        'test_final': record(test_path), 'checkpoint': record(checkpoint), 'sid': record(sid),
        'valid': record(VALID), 'sources': {str(p): sha256(p) for p in sources}, 'config': config}
    return final, identity


def training_module():
    if any(name == 'model' or name.startswith('model.') for name in sys.modules):
        raise RuntimeError('validation must use an isolated Stage3 model import namespace')
    sys.path.insert(0, str(TRAINER.parent))
    spec = importlib.util.spec_from_file_location('issue256_frozen_hgrec', TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_rank():
    import torch

    final, identity = inputs()
    config = dict(identity['config'])
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise ValueError('complete validation requires four disjoint rank shards')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    trainer = training_module()
    if config['fast']:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    trainer.set_seed(config['seed'])
    dataset = trainer._build_dataset(config, str(VALID), 'evaluation', Path(final['sid_path']))
    _require_complete_population(len(dataset))
    if (dataset.vocab_size != config['vocab_size'] or dataset.eos_token != config['eos_token_id']
            or dataset.n_digit != config['sid_length']
            or dataset.max_token_seq_len != config['max_token_seq_len']):
        raise ValueError('validation SID namespace differs from the formal checkpoint')
    indices = list(range(rank, len(dataset), world))
    loader = trainer.GenRecDataLoader(trainer.Subset(dataset, indices),
        batch_size=config['infer_size'], shuffle=False, sampler=None,
        sample_collator=trainer.make_sample_collator(dataset, include_eval_metadata=True),
        max_token_seq_len=dataset.max_token_seq_len, include_seen=True,
        **trainer._loader_options(config))
    model = trainer.HG_Rec(config).to(device)
    model.load_state_dict(torch.load(final['best_checkpoint'], map_location=device))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = OUTPUT / ('rank%d.jsonl' % rank)
    recalls, ndcgs, count = trainer.evaluate(model, loader, dataset, config['topk_list'],
        config['beam_size'], device, exclude_history=config['exclude_history'],
        use_bf16=config['bf16'], sid_dump_path=str(predictions), sid_dump_rank=rank, sid_dump_world=world)
    if count != len(indices) or inputs()[1] != identity:
        raise ValueError('validation population or paired inputs changed')
    report = {'rank': rank, 'world_size': world, 'n_eval': count, 'population': len(dataset),
        'recall_sums': recalls, 'ndcg_sums': ndcgs, 'predictions': record(predictions),
        'identity': identity, 'worker_pid': os.getpid(), 'device': str(device),
        'formal_training': False, 'test_replay': False}
    (OUTPUT / ('rank%d.json' % rank)).write_text(json.dumps(report, indent=2) + '\n')
    return 0


def reduce_rows():
    _, identity = inputs()
    shards = [json.loads((OUTPUT / ('rank%d.json' % r)).read_text()) for r in range(4)]
    population = shards[0]['population']
    _require_complete_population(population)
    ranks = [None] * population
    seen = set()
    for rank, shard in enumerate(shards):
        if shard['identity'] != identity or shard['rank'] != rank or shard['population'] != population:
            raise ValueError('mixed validation shards')
        path = Path(shard['predictions']['path'])
        if record(path) != shard['predictions']:
            raise ValueError('validation prediction rows changed')
        rows = [json.loads(line) for line in path.open()]
        if [row['global_row'] for row in rows] != list(range(rank, population, 4)):
            raise ValueError('missing, duplicate or misordered validation identity')
        if len(rows) != shard['n_eval']:
            raise ValueError('validation denominator mismatch')
        for row in rows:
            number, target_rank = row['global_row'], row['target_rank']
            if number in seen or target_rank is not None and (type(target_rank) is not int or not 0 <= target_rank < 10):
                raise ValueError('invalid validation row/rank')
            seen.add(number)
            ranks[number] = target_rank
    if seen != set(range(population)):
        raise ValueError('incomplete validation population')
    result = {'n_eval': population}
    for k in (5, 10):
        hits = [r for r in ranks if r is not None and r < k]
        result['validation_recall@%d' % k] = len(hits) / population
        result['validation_ndcg@%d' % k] = math.fsum(1 / math.log2(r + 2) for r in hits) / population
        for metric, expected in (('recall', len(hits)), ('ndcg', math.fsum(1 / math.log2(r + 2) for r in hits))):
            actual = math.fsum(s[metric + '_sums'][str(k)] for s in shards)
            if not math.isclose(actual, expected, rel_tol=1.e-10, abs_tol=1.e-10):
                raise ValueError('independent validation reduction disagrees with trainer')
    receipt = {'schema': 'genrec.issue256.recommendation_validation/1', 'status': 'COMPLETE',
        'identity': identity, 'metrics': result,
        'shards': [record(OUTPUT / ('rank%d.json' % r)) for r in range(4)],
        'used_for_checkpoint_selection': False, 'training': False, 'test_access': False}
    (OUTPUT / 'VALIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return 0


def main():
    if launcher.is_child_rank():
        return infer_rank()
    inputs()
    if any(OUTPUT.glob('rank*.json')) or any(OUTPUT.glob('rank*.jsonl')):
        raise ValueError('validation evidence already exists; do not silently replay or overwrite')
    code = launcher.launch(SOURCE / 'validation_inference_entry.py', OUTPUT / 'launcher',
                           python=PYTHON, log_name='validation_inference.log')
    return reduce_rows() if code == 0 else code


if __name__ == '__main__':
    raise SystemExit(main())
