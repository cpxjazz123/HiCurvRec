"""Read the exact final hard SID once on CUDA; never train or rewrite an export."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tools.issue256.identity import ROOT, ARM, RUN_ID, SOURCE, EVIDENCE, PYTHON
from tools.issue252.archive_audit import sha256
from tools.issue252.final_evaluation import _training_module, load_bound_checkpoint, load_sid
from tools.issue252.run_workspace import load_run
from tools.issue253.final_evaluation import _books
from tools.issue255.evaluate_arm import validate_catalogue
from tools.issue255 import mechanism_metrics as metrics
from tools.research.stage2 import launcher

OUTPUT = EVIDENCE / 'evaluation'


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def context():
    run = load_run(ROOT, RUN_ID, ARM)
    result = Path(run['result'])
    paths = {'checkpoint': result / 'rqvae_best.pth', 'sid': result / 'item_sids.json',
             'binding': result / 'archive_binding.json', 'export_manifest': result / 'export_manifest.json'}
    payload, binding = load_bound_checkpoint(tree=run['source'], **paths)
    sources = [Path(__file__), SOURCE / 'catalogue_audit_entry.py',
               ROOT / 'tools/issue255/evaluate_arm.py', ROOT / 'tools/issue255/mechanism_metrics.py',
               ROOT / 'tools/issue252/final_evaluation.py', ROOT / 'tools/issue253/final_evaluation.py',
               EVIDENCE / 'CONTRACT.md']
    identity = {'run_id': RUN_ID, 'arm': ARM, 'source_tree': run['source'],
                'workspace': record(run['workspace']), 'artifacts': {k: record(v) for k, v in paths.items()},
                'source_files': {str(p): sha256(p) for p in sources},
                'checkpoint_provenance': payload['provenance']}
    return run, paths, payload, identity


def audit():
    import torch

    run, paths, payload, identity = context()
    generation = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = OUTPUT / 'extractions' / generation
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'EXTRACTION.json'
    array_path = output / 'catalogue.npz'
    rows = load_sid(paths['sid'], 24587, (256, 256, 256))
    books = _books(payload)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['context'] != identity or receipt['artifact'] != record(array_path):
            raise ValueError('existing extraction identity or data changed')
        with np.load(array_path, allow_pickle=False) as blob:
            arrays = {key: blob[key] for key in blob.files}
        validate_catalogue(arrays, rows, books, embedding_shape=(24587, 768), token_columns=5)
    else:
        if array_path.exists():
            raise ValueError('partial generation exists without a complete extraction receipt')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
        torch.set_num_threads(1)
        trainer = _training_module(Path(run['source']))
        trainer.set_seed(trainer.SEED)
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=768,
                              distance=trainer.DISTANCE).to(device)
        model.load_state_dict(payload['state_dict'], strict=True)
        model.eval()
        for layer in model.rq.vq_layers:
            layer._skip_ddp_reduce = True
        embeddings = torch.from_numpy(np.load(trainer.EMBEDDING_FILE)).to(device)
        with torch.no_grad():
            actual = model.get_indices(embeddings, infer_use_sk=False).cpu().numpy()
            if not np.array_equal(actual, rows[:, :3]):
                raise ValueError('actual CUDA IDs differ from the canonical final SID')
            if not np.array_equal(trainer._extend_collisions(actual, [256] * 3), rows):
                raise ValueError('actual CUDA collision suffix differs from the canonical final SID')
            arrays = {'sid': rows, 'prefixes': model.last_prefixes.cpu().numpy(),
                      'weights': model.last_weights.cpu().numpy(),
                      'attention': model.catalogue_attention.cpu().numpy(),
                      'token_mask': model.structure.last_token_mask.cpu().numpy(),
                      'reconstruction': model.decoder(model.last_q).cpu().numpy()}
        validate_catalogue(arrays, rows, books, embedding_shape=(24587, 768), token_columns=5)
        if context()[-1] != identity:
            raise ValueError('bound source/artifacts changed during extraction')
        partial = array_path.with_suffix('.partial')
        with partial.open('wb') as stream:
            np.savez(stream, **arrays)
        partial.replace(array_path)
        receipt = {'schema': 'genrec.issue256.catalogue_extraction/1', 'status': 'COMPLETE',
                   'context': identity, 'artifact': record(array_path),
                   'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                              for key, value in arrays.items()},
                   'execution': {'worker_pid': os.getpid(), 'rank': 0,
                       'device': str(device), 'device_name': torch.cuda.get_device_name(device),
                       'matmul_precision': torch.get_float32_matmul_precision(),
                       'seed': trainer.SEED, 'no_grad': True, 'inference_mode': False,
                       'actual_sid_reexport_equal': True, 'all_levels_bitwise_equal': True,
                       'formal_training': False}}
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pointer = {'receipt': record(receipt_path), 'catalogue': record(array_path),
               'context': identity, 'generation': generation}
    (OUTPUT / 'EXTRACTION.json').write_text(json.dumps(pointer, indent=2) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'catalogue': pointer['catalogue'],
                      'receipt': pointer['receipt']}, indent=2), flush=True)
    return 0


def main():
    if launcher.is_child_rank():
        return audit() if int(os.environ['RANK']) == 0 else 0
    return launcher.launch(SOURCE / 'catalogue_audit_entry.py', OUTPUT / 'cuda_extraction' / RUN_ID,
                           python=PYTHON, log_name='catalogue_extraction.log')


if __name__ == '__main__':
    raise SystemExit(main())
