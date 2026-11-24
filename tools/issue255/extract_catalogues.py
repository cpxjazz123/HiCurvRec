"""No-argument CUDA catalogue audit, exclusively through the Stage2 launcher.

Finish this command before starting the separate CPU six-cell evaluation. No
training, checkpoint/SID replacement, asset construction or scale fitting occurs.
Only rank zero constructs models; both exact generations run sequentially there.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys

import numpy as np

from tools.issue255 import evaluate_arm as evaluation
from tools.issue255.identity import ROOT, ARM, RUN_ID, SOURCE
from tools.research.stage2 import contract, launcher

OUTPUT = evaluation.OUTPUT
RECEIPT = OUTPUT / 'EXTRACTION.json'
LAUNCH_OUTPUT = OUTPUT / 'cuda_extraction' / RUN_ID
ENTRY = SOURCE / 'catalogue_audit_entry.py'
TOKEN_COLUMNS = {'issue253': 3, 'issue255': 5}


def _forget_source(tree):
    """Release only this exact imported generation before loading the next one."""
    tree = Path(tree).resolve()
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).resolve().is_relative_to(tree):
            del sys.modules[name]
    while str(tree) in sys.path:
        sys.path.remove(str(tree))


def extract_generation(run, paths, payload, name, device):
    """Observe one real no_grad get_indices export; never replace assignment."""
    import torch
    from tools.issue252.final_evaluation import _training_module, load_sid
    from tools.issue253.final_evaluation import _books

    tree = Path(run['source'])
    trainer = _training_module(tree)
    # Exactly the source trainer's seed, deterministic algorithms and high GEMM
    # precision must be restored separately for EACH generation, before CUDA work.
    trainer.set_seed(trainer.SEED)
    evaluation.require(torch.get_float32_matmul_precision() == 'high'
                       and torch.are_deterministic_algorithms_enabled()
                       and torch.backends.cudnn.deterministic
                       and not torch.backends.cudnn.benchmark,
                       'source trainer did not restore frozen numerical policy')
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=768,
                         distance=trainer.DISTANCE).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    rows = load_sid(paths['sid'], evaluation.metrics.ITEMS, [256] * 3)
    books = _books(payload)
    embeddings = np.load(evaluation.EMBEDDINGS, allow_pickle=False)
    module = sys.modules[type(model).quantize.__module__]
    original = module.select
    selected = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        evaluation.require(not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
                           and result['q'].device == device
                           and torch.equal(result['q'], result['hard']),
                           'selection is not the actual CUDA no_grad hard forward')
        selected.append(result['q'].detach())
        return result

    module.select = observe
    layers = model.rq.vq_layers
    for layer in layers:
        layer._skip_ddp_reduce = True  # Identical to the trainer's rank-zero export.
    try:
        with torch.no_grad():
            actual_ids = model.get_indices(torch.from_numpy(embeddings).to(device),
                                           infer_use_sk=False).cpu().numpy().astype(np.int64)
            evaluation.require(len(selected) == 3, 'export must make exactly one three-level catalogue pass')
            evaluation.require(np.array_equal(actual_ids, rows[:, :3]),
                               'CUDA hard IDs differ from actual exported SID')
            extended = trainer._extend_collisions(actual_ids, [256] * 3)
            evaluation.require(np.array_equal(extended, rows),
                               'CUDA fourth collision suffix differs from actual exported SID')
            running = np.zeros((len(rows), 32), dtype=np.float32)
            prefixes = []
            for value in selected:
                running = running + value.cpu().numpy()
                prefixes.append(running.copy())
            prefixes = np.stack(prefixes)
            evaluation.metrics.assert_prefix_identity(prefixes, rows, books)
            q = model.last_q
            evaluation.require(q.device == device and np.array_equal(q.cpu().numpy(), prefixes[2]),
                               'CUDA final quantized output differs from hard SID prefix')
            if hasattr(model, 'last_prefixes'):
                evaluation.require(np.array_equal(model.last_prefixes.cpu().numpy(), prefixes),
                                   'CUDA production cumulative prefixes differ from actual selections')
            weights = model.last_weights.cpu().numpy()
            attention = model.catalogue_attention.cpu().numpy()
            mask = model.structure.last_token_mask.cpu().numpy()
            if name == 'issue253':
                evaluation.require(weights.shape == (len(rows), 3)
                                   and attention.shape == (len(rows), 3, 3),
                                   'archived shared gate/attention schema changed')
                # Persist the actual shared gate, not fictional per-level learning.
                weights = np.repeat(weights[:, None, :], 3, axis=1)
                attention = np.repeat(attention[:, None, :, :], 3, axis=1)
            arrays = {'prefixes': prefixes, 'sid': rows, 'weights': weights,
                      'attention': attention, 'token_mask': mask,
                      'reconstruction': model.decoder(q).cpu().numpy()}
    finally:
        module.select = original
        for layer in layers:
            layer._skip_ddp_reduce = False
    evaluation.validate_catalogue(arrays, rows, books,
        embedding_shape=(evaluation.metrics.ITEMS, 768), token_columns=TOKEN_COLUMNS[name])
    output = OUTPUT / (name + '_catalogue.npz')
    partial = output.with_name(output.name + '.partial')
    with partial.open('wb') as handle:
        np.savez(handle, **arrays)
    partial.replace(output)
    execution = {'device_type': 'cuda', 'device': str(device),
        'device_name': torch.cuda.get_device_name(device), 'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda, 'pid': os.getpid(), 'rank': 0,
        'seed': int(trainer.SEED), 'matmul_precision': torch.get_float32_matmul_precision(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'catalogue_passes': len(selected) // 3, 'no_grad': True, 'inference_mode': False,
        'actual_sid_reexport_equal': True, 'all_levels_bitwise_equal': True,
        'shared_gate_across_levels': name == 'issue253'}
    return {'artifact': evaluation.record(output), 'execution': execution,
            'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                       for key, value in arrays.items()}}


def audit():
    """Rank zero extracts all GPU data before publishing one complete receipt."""
    import torch
    evaluation.require(int(os.environ['RANK']) == 0
                       and int(os.environ['WORLD_SIZE']) == contract.WORLD_SIZE
                       and int(os.environ['MASTER_PORT']) == contract.STAGE2_PORT,
                       'catalogue audit requires the frozen launcher rank/world/port')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(1)
    runs, paths, payloads, _, _, context = evaluation.bound_generations()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if RECEIPT.exists():
        RECEIPT.unlink()
    catalogues = {}
    for name, run in runs.items():
        try:
            catalogues[name] = extract_generation(run, paths[name], payloads[name], name, device)
        finally:
            _forget_source(run['source'])
            gc.collect()
            torch.cuda.empty_cache()
    # Sources, both bound exports and all input bytes must still be identical.
    current = evaluation.bound_generations()[-1]
    evaluation.require(current == context, 'extraction provenance changed during CUDA catalogue passes')
    receipt = {'schema': 'genrec.issue255.catalogue_extraction/1', 'status': 'COMPLETE',
        'context': context, 'metric': evaluation.common_metric(), 'catalogues': catalogues,
        'framework': {'entry': evaluation.record(ENTRY), 'world_size': contract.WORLD_SIZE,
                      'port': contract.STAGE2_PORT, 'launcher': evaluation.record(Path(launcher.__file__)),
                      'serial_guard': str(launcher.SERIAL_LOCK),
                      'launch_directory': str(LAUNCH_OUTPUT), 'model_ranks': [0]},
        'sid_protocol': 'exact first three hard IDs plus untouched fourth collision suffix',
        'population': {'items': evaluation.metrics.ITEMS, 'levels': 3,
                       'curvature_triples_per_cell': evaluation.metrics.EXPECTED_TRIPLES}}
    evaluation.write_json(RECEIPT, receipt)
    print(json.dumps({'status': 'COMPLETE', 'receipt': evaluation.record(RECEIPT),
                      'catalogues': {name: item['artifact'] for name, item in catalogues.items()}}), flush=True)
    return 0


def main():
    evaluation.require(len(sys.argv) == 1, 'catalogue extraction accepts no arguments; source constants only')
    sys.dont_write_bytecode = True
    if launcher.is_child_rank():
        return audit() if int(os.environ['RANK']) == 0 else 0
    status = launcher.launch(ENTRY, LAUNCH_OUTPUT, python=contract.STAGE2_PYTHON,
                             log_name='catalogue_extraction.log')
    evaluation.require(status == 0, 'CUDA catalogue launcher failed; see catalogue_extraction.log')
    evaluation.require(RECEIPT.is_file(), 'CUDA catalogue launcher returned without EXTRACTION.json')
    print(json.dumps({'status': 'COMPLETE', 'receipt': evaluation.record(RECEIPT)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
