"""Post-training actual-loss replay on a disposable final-weight model, never a run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.issue256.r03.identity import ROOT, EVIDENCE, RUN_ID
from tools.issue252.archive_audit import sha256


def probe(tree, payload):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from tools.issue252.final_evaluation import _training_module

    trainer = _training_module(Path(tree))
    trainer.set_seed(trainer.SEED)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    catalogue_cpu = torch.from_numpy(trainer.load_embeddings(trainer.EMBEDDING_FILE))
    catalogue = catalogue_cpu.to(device)
    train_frame = trainer.pd.read_parquet(trainer.TRAIN_FILE)
    ids = np.unique(train_frame['target'].to_numpy(dtype=np.int64))
    dataset = trainer._BatchedTensorDataset(catalogue_cpu[torch.from_numpy(ids)], torch.from_numpy(ids))
    sampler = DistributedSampler(dataset, num_replicas=4, rank=0, shuffle=True,
                                 seed=trainer.SEED, drop_last=False)
    sampler.set_epoch(2999)
    loader = DataLoader(dataset, batch_size=trainer.BATCH_SIZE_PER_RANK, sampler=sampler,
                        num_workers=trainer.NUM_WORKERS, pin_memory=True,
                        collate_fn=trainer._batch_passthrough, persistent_workers=False)
    batch, item_ids = trainer._calibration_batch(loader, device)
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=768, distance=trainer.DISTANCE).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.train()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    from model.calibration import coupled_groups
    groups = coupled_groups(model)
    groups['decoder'] = tuple(model.decoder.parameters())
    parameters = tuple(model.parameters())
    sample = model.structural_sampler().draw(2999 * len(loader), 0, 4, track=False)
    output = model(batch, item_ids, catalogue_embeddings=catalogue, structural_sample=sample)
    original, components = trainer.CRITERION(output[0], batch, output[1])
    curvature = output[4]
    total = model.combined_loss(original, curvature)
    losses = {'original': original, 'curvature': curvature,
              'reconstruction': components['reconstruction'], 'combined': total}
    gradients = {name: dict(zip(map(id, parameters), torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True))) for name, loss in losses.items()}

    def selected(members, values):
        result = []
        for member in members:
            parameter, index = member if isinstance(member, tuple) else (member, None)
            value = values.get(id(parameter))
            result.append(None if value is None else value if index is None else value[index])
        return result

    def stats(values):
        present = [g for g in values if g is not None]
        if not all(bool(torch.isfinite(g).all()) for g in present):
            raise RuntimeError('nonfinite post-training module gradient')
        return {'none': len(values) - len(present), 'tensors': len(values),
                'zero': sum(not bool(g.any()) for g in present),
                'norm': sum(float(g.double().square().sum()) for g in present) ** .5}

    decoded = torch.zeros_like(model.last_structural_q)
    prefix_exact = []
    for level, book in enumerate(model.rq.vq_layers):
        decoded = decoded + book.embed(model.last_structural_tokens[:, level])
        prefix_exact.append(torch.equal(decoded, model.last_prefixes[level]))
    if not all(prefix_exact) or not torch.equal(decoded, model.last_structural_q):
        raise RuntimeError('final-weight training hard prefixes differ from selected codewords')
    rows = {}
    weight = float(model.curvature_weight)
    for name, members in groups.items():
        module = {key: stats(selected(members, value)) for key, value in gradients.items()}
        og, cg = selected(members, gradients['original']), selected(members, gradients['curvature'])
        pairs = [(a.double(), b.double()) for a, b in zip(og, cg) if a is not None and b is not None]
        on, cn = module['original']['norm'], module['curvature']['norm']
        module['weighted_curvature_over_original'] = weight * cn / on if on else None
        module['original_curvature_cosine'] = (
            sum(float((a * b).sum()) for a, b in pairs) / (on * cn) if on and cn else None)
        rows[name] = module
    before = {id(p): p.detach().clone() for p in parameters}
    # A probe of update reachability, not optimizer continuation: checkpoints do
    # not contain Adam moments and the formal final learning rate is not reused.
    optimizer_class = torch.optim.Adagrad if trainer.OPTIMIZER.lower() == 'adagrad' else torch.optim.AdamW
    optimizer = optimizer_class(parameters, lr=trainer.LR, weight_decay=trainer.WEIGHT_DECAY)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    delta = {id(p): p.detach() - before[id(p)] for p in parameters}
    for name, members in groups.items():
        rows[name]['fresh_optimizer_update'] = stats(selected(members, delta))
    path = EVIDENCE / 'POST_TRAINING_MODULE_CHECKS.json'
    report = {'schema': 'genrec.issue256.final_wiring/1', 'run_id': RUN_ID,
        'checkpoint_epoch': payload['epoch'], 'source_tree': str(Path(tree).resolve()),
        'evaluator': {'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)},
        'population': 'actual1024-item rank0/world4 first batch for epoch3000; diagnostic replay',
        'batch_item_ids_sha256': hashlib.sha256(item_ids.cpu().numpy().tobytes()).hexdigest(),
        'batch_items': len(item_ids), 'structural_global_step': 2999 * len(loader),
        'lambda': weight, 'losses': {key: float(value.detach()) for key, value in losses.items()},
        'modules': rows, 'actual_hard_prefixes_exact': prefix_exact,
        'formal_checkpoint_changed': False, 'optimizer_continuation': False,
        'update_probe': {'optimizer': trainer.OPTIMIZER, 'lr': trainer.LR,
                         'weight_decay': trainer.WEIGHT_DECAY, 'fresh_state': True},
        'limitations': 'Single-rank final-weight diagnostics, not the initial global VJP budget; no claim of later norm safety or an extra formal training run.'}
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return {'path': str(path), 'sha256': sha256(path)}
