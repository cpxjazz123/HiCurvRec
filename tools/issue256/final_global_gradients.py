"""One train-only final-weight global VJP measurement, never optimizer continuation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from tools.issue256.identity import ROOT, SOURCE, RUN_ID, ARM, EVIDENCE, PYTHON
from tools.issue252.archive_audit import sha256
from tools.research.stage2 import launcher

OUTPUT = EVIDENCE / 'final_global_gradient'


def group_statistics(groups, original, curvature, weight):
    rows = {}
    for name, members in groups.items():
        pairs = []
        for member in members:
            parameter, index = member if isinstance(member, tuple) else (member, None)
            a, b = original[id(parameter)], curvature[id(parameter)]
            pairs.append((a.double() if index is None else a[index].double(),
                          b.double() if index is None else b[index].double()))
        aa = sum(float(a.square().sum()) for a, _ in pairs)
        bb = sum(float(b.square().sum()) for _, b in pairs)
        ab = sum(float((a * b).sum()) for a, b in pairs)
        dot = sum(float((a * (a + weight * b)).sum()) for a, b in pairs)
        rows[name] = {'original_norm': aa ** .5, 'curvature_norm': bb ** .5,
            'weighted_ratio': weight * (bb / aa) ** .5 if aa else None,
            'original_curvature_cosine': ab / (aa * bb) ** .5 if aa and bb else None,
            'original_dot_combined': dot,
            'normalized_original_dot_combined': dot / aa if aa else None,
            'raw_combined_negative_is_original_ascent': dot < 0}
    return rows


def child():
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from tools.issue252.final_evaluation import _training_module, load_bound_checkpoint
    from tools.issue252.run_workspace import load_run

    rank, world, local_rank = (int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 4:
        raise ValueError('diagnostic requires actual world4/global4096')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group('nccl')
    started = time.monotonic()
    try:
        run = load_run(ROOT, RUN_ID, ARM)
        result, tree = Path(run['result']), Path(run['source'])
        paths = {'checkpoint': result / 'rqvae_best.pth', 'sid': result / 'item_sids.json',
                 'binding': result / 'archive_binding.json', 'export_manifest': result / 'export_manifest.json'}
        payload, _ = load_bound_checkpoint(tree=tree, **paths)
        trainer = _training_module(tree)
        trainer.set_seed(trainer.SEED + rank)
        vectors = trainer.maybe_apply_pca(trainer.load_embeddings(trainer.EMBEDDING_FILE),
                                         trainer.PCA_DIM, trainer.SEED)
        catalogue_cpu = torch.from_numpy(vectors)
        catalogue = catalogue_cpu.to(device)
        frame = trainer.pd.read_parquet(trainer.TRAIN_FILE)
        ids = np.unique(frame['target'].to_numpy(dtype=np.int64))
        dataset = trainer._BatchedTensorDataset(catalogue_cpu[torch.from_numpy(ids)], torch.from_numpy(ids))
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                     seed=trainer.SEED, drop_last=False)
        sampler.set_epoch(2999)
        loader = DataLoader(dataset, batch_size=trainer.BATCH_SIZE_PER_RANK, sampler=sampler,
            num_workers=trainer.NUM_WORKERS, pin_memory=True, collate_fn=trainer._batch_passthrough,
            persistent_workers=False)
        batch, item_ids = trainer._calibration_batch(loader, device)
        if len(item_ids) != 1024:
            raise ValueError('diagnostic lost the full actual per-rank population')
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=768, distance=trainer.DISTANCE).to(device)
        model.load_state_dict(payload['state_dict'], strict=True)
        model.train()
        from model.calibration import coupled_groups, mean_gradients
        groups = coupled_groups(model)
        coupled_ids = {id(member[0] if isinstance(member, tuple) else member)
                       for members in groups.values() for member in members}
        parameters = tuple(parameter for parameter in model.parameters() if id(parameter) in coupled_ids)
        step = 2999 * len(loader)
        sample = model.structural_sampler().draw(step, rank, world, track=False)
        output = model(batch, item_ids, catalogue_embeddings=catalogue, structural_sample=sample)
        original, _ = trainer.CRITERION(output[0], batch, output[1])
        curvature = output[4]
        combined = model.combined_loss(original, curvature)
        if not all(bool(torch.isfinite(loss)) for loss in (original, curvature, combined)):
            raise ValueError('nonfinite actual final-weight objective')
        og = torch.autograd.grad(original, parameters, retain_graph=True, allow_unused=True)
        cg = torch.autograd.grad(curvature, parameters, allow_unused=True)
        if any(gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in (*og, *cg)):
            raise ValueError('a coupled actual-loss VJP is absent or nonfinite')
        weight = float(model.curvature_weight)
        local = group_statistics(groups, dict(zip(map(id, parameters), og)),
                                 dict(zip(map(id, parameters), cg)), weight)
        sample_sha = hashlib.sha256(json.dumps(sample, sort_keys=True,
            default=lambda value: value.tolist(), allow_nan=False).encode()).hexdigest()
        local_receipt = {'rank': rank, 'pid': os.getpid(), 'device': str(device),
            'items': len(item_ids), 'item_ids_sha256': hashlib.sha256(item_ids.cpu().numpy().tobytes()).hexdigest(),
            'structural_sample_sha256': sample_sha, 'structural_step': step,
            'diagnostic_seed': trainer.SEED + rank,
            'losses': {'original': float(original.detach()), 'curvature': float(curvature.detach()),
                       'combined': float(combined.detach())}, 'local_module_statistics': local}
        # These are actual VJP vectors. Averaging local scalar norms is forbidden.
        mean_gradients(og)
        mean_gradients(cg)
        global_rows = group_statistics(groups, dict(zip(map(id, parameters), og)),
                                       dict(zip(map(id, parameters), cg)), weight)
        ranks = [None] * world
        dist.all_gather_object(ranks, local_receipt)
        torch.cuda.synchronize(device)
        if rank == 0:
            report = {'schema': 'genrec.issue256.final_global_gradients/1', 'run_id': RUN_ID,
                'world_size': world, 'global_batch': sum(r['items'] for r in ranks),
                'checkpoint': {'path': str(paths['checkpoint']), 'sha256': sha256(paths['checkpoint'])},
                'source_tree': str(tree), 'source_workspace_sha256': sha256(run['workspace']),
                'lambda': weight, 'groups': global_rows, 'ranks': ranks,
                'averaging': 'SUM/world on actual original and curvature VJP vectors before norms/dots',
                'groups_beyond_initial_budget': [name for name, row in global_rows.items()
                    if row['weighted_ratio'] is not None and row['weighted_ratio'] > 1 + 2.e-6],
                'raw_original_ascent_groups': [name for name, row in global_rows.items()
                                             if row['raw_combined_negative_is_original_ascent']],
                'elapsed_seconds': time.monotonic() - started,
                'evaluator': {'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)},
                'question': {'path': str(EVIDENCE / 'POST_ROUND_DIAGNOSIS.md'),
                             'sha256': sha256(EVIDENCE / 'POST_ROUND_DIAGNOSIS.md')},
                'optimizer_step': False, 'coefficient_recalibration': False, 'test_access': False,
                'limitations': 'One fresh-RNG final-weight global4096 train batch; not the formal RNG state, all epochs, an AdamW step or Recall causality.'}
            OUTPUT.mkdir(parents=True, exist_ok=True)
            (OUTPUT / 'GLOBAL_FINAL_GRADIENTS.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        return 0
    finally:
        dist.destroy_process_group()


def main():
    if launcher.is_child_rank():
        return child()
    return launcher.launch(SOURCE / 'final_global_gradient_entry.py', OUTPUT,
                           python=PYTHON, log_name='final_global_gradients.log')


if __name__ == '__main__':
    raise SystemExit(main())
