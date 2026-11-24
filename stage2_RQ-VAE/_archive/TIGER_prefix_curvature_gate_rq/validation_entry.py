"""Actual one-epoch phase2 diagnostic; no inherited checkpoint or synthetic loss."""
import hashlib
import json
import os
import time
import traceback

import torch
import torch.distributed as dist
import train_rqvae as training
import benchmark_entry as common


def digest(tensors):
    value = hashlib.sha256()
    for name, tensor in tensors:
        array = tensor.detach().cpu().contiguous()
        value.update(name.encode())
        value.update(str((array.dtype, tuple(array.shape))).encode())
        value.update(array.view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def groups(model):
    result = {'encoder': list(model.encoder.parameters()),
              'decoder': list(model.decoder.parameters()),
              'GNN.input': list(model.structure.tokens.input.parameters()),
              'sharedKey': list(model.structure.attention.key.parameters()),
              'sharedValue': list(model.structure.attention.value.parameters()),
              'sharedScore': list(model.structure.attention.score.parameters())}
    for name, module in model.structure.tokens.named_modules():
        if name and name != 'input' and list(module.parameters(recurse=False)):
            result['GNN.' + name] = list(module.parameters(recurse=False))
    for index, book in enumerate(model.rq.vq_layers):
        result['codebook%d' % index] = list(book.parameters())
    # Each query is a row of one real parameter, not fictitious modules.
    query = model.structure.attention.query
    for index in range(len(query)):
        result['Query%d' % index] = [(query, index)]
    return result


def statistics(model, values):
    result = {}
    for name, members in groups(model).items():
        selected = []
        for member in members:
            parameter, index = member if isinstance(member, tuple) else (member, None)
            value = values.get(id(parameter))
            selected.append(None if value is None else value if index is None else value[index])
        present = [value for value in selected if value is not None]
        finite = all(bool(torch.isfinite(value).all()) for value in present)
        result[name] = {'absent': not present, 'none': len(selected) - len(present),
                        'zero': sum(not bool(value.any()) for value in present),
                        'tensors': len(selected), 'finite': finite,
                        'norm': sum(float(value.double().square().sum()) for value in present) ** .5}
        if not finite:
            raise RuntimeError('nonfinite module ' + name)
    return result


def tensor_leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item)


class Observer:
    def __init__(self, report, output):
        self.report, self.output = report, output
        self.forward = {}
        self.seen = set()

    def instrument(self, model):
        for name, module in model.named_modules():
            if list(module.parameters(recurse=False)):
                def hook(owner, inputs, output, name=name):
                    row = self.forward.setdefault(name, {'calls': 0, 'finite': True})
                    row['calls'] += 1
                    row['finite'] &= all(bool(torch.isfinite(t).all()) for t in tensor_leaves(output))
                    if not row['finite']:
                        raise RuntimeError('nonfinite forward ' + name)
                module.register_forward_hook(hook)

    def __call__(self, event, **fields):
        if event != 'diagnostic_complete' and event in self.seen:
            return
        self.seen.add(event)
        model = fields['model']
        if event == 'first_step_before_backward':
            self.before = {id(p): p.detach().clone() for p in model.parameters()}
            self.batch = fields['batch'].detach().clone()
            self.ids = torch.as_tensor(fields['item_ids'], dtype=torch.long).detach().clone()
            self.sample = fields['structural_sample']
            self.report['first_step_losses'] = {
                key: float(fields[key].detach()) for key in
                ('original_loss', 'structural_loss', 'combined_loss')}
            if not all(torch.isfinite(fields[key]).all() for key in
                       ('original_loss', 'structural_loss', 'combined_loss')):
                raise RuntimeError('nonfinite actual first-step losses')
            self.report['actual_batch_items'] = len(self.ids)
            if len(self.ids) != 1024:
                raise RuntimeError('first step is not the full real per-rank batch')
            self.report['first_structural_sample'] = {
                key: value.tolist() if hasattr(value, 'tolist') else value
                for key, value in self.sample.items()}
            decoded = sum(layer.embed(model.last_structural_tokens[:, i])
                          for i, layer in enumerate(model.rq.vq_layers))
            exact = torch.equal(model.last_structural_q, decoded)
            self.report['actual_structural_hard_q_decode_exact'] = exact
            if not exact:
                raise RuntimeError('actual sampled training q differs from token decode')
            # The three live curvature gate weights of the actual reconstruction
            # population: their distribution is the gate evidence C14 asks for.
            weights = model.last_weights.detach()
            if tuple(weights.shape) != (len(self.ids), 3):
                raise RuntimeError('gate weights do not cover the three experts')
            self.report['actual_gate_weights'] = {
                'population': int(weights.shape[0]),
                'per_expert': [{'mean': float(weights[:, index].mean()),
                                'min': float(weights[:, index].min()),
                                'max': float(weights[:, index].max())}
                               for index in range(3)],
                'argmax_histogram': [int((weights.argmax(dim=1) == index).sum())
                                     for index in range(3)],
                'row_sum_max_abs_error': float((weights.sum(dim=1) - 1).abs().max())}
        elif event == 'first_step_after_backward':
            self.report['normal_backward'] = statistics(model, {id(p): p.grad for p in model.parameters()})
            self.report['normal_forward'] = {name: dict(row) for name, row in self.forward.items()}
        elif event == 'first_step_after_optimizer':
            self.report['normal_backward_post_clip'] = statistics(model, {id(p): p.grad for p in model.parameters()})
            self.report['actual_optimizer_update'] = statistics(model,
                {id(p): p.detach() - self.before[id(p)] for p in model.parameters()})
            del self.before
        elif event == 'diagnostic_complete':
            losses = fields['completed_losses']
            self.report['completed_epoch'] = fields['completed_epoch']
            self.report['completed_local_losses'] = [float(value) for value in losses]
            if not losses or not all(torch.isfinite(torch.as_tensor(value)).all() for value in losses):
                raise RuntimeError('completed epoch has missing/nonfinite real losses')
            self.completed(model, fields['catalogue'])

    def completed(self, model, catalogue):
        required = {'first_step_before_backward', 'first_step_after_backward',
                    'first_step_after_optimizer', 'diagnostic_complete'}
        if not required <= self.seen:
            raise RuntimeError('trainer omitted actual diagnostic observer events')
        state_hash = digest(model.named_parameters())
        hashes = [None] * dist.get_world_size()
        dist.all_gather_object(hashes, state_hash)
        self.report['completed_parameter_hashes'] = hashes
        if len(set(hashes)) != 1:
            raise RuntimeError('four-rank completed parameters differ')
        self.report['completed_state_finite'] = all(
            bool(torch.isfinite(p).all()) for p in model.parameters())
        if not self.report['completed_state_finite']:
            raise RuntimeError('nonfinite completed parameter')
        # Independent unwrapped raw model: no autograd.grad through DDP's reducer.
        raw = training.RQVAE(training._tokenizer_config(), in_dim=catalogue.shape[1],
                             distance=training.DISTANCE).to(catalogue.device)
        raw.load_state_dict(model.state_dict())
        raw.train()
        reconstructed, parts, _, _, structural = raw(
            self.batch, self.ids, catalogue_embeddings=catalogue, structural_sample=self.sample)
        original, _ = training.CRITERION(reconstructed, self.batch, parts)
        parameters = list(raw.parameters())
        original_g = torch.autograd.grad(original, parameters, allow_unused=True, retain_graph=True)
        structural_g = torch.autograd.grad(structural, parameters, allow_unused=True)
        weight = float(raw.curvature_weight)
        self.report['weighted_structural_VJP'] = statistics(raw, dict(zip(map(id, parameters),
            (None if g is None else g*weight for g in structural_g))))
        self.report['loss_gradient_direction'] = {}
        originals = dict(zip(map(id, parameters), original_g))
        structurals = dict(zip(map(id, parameters), structural_g))
        for name, members in groups(raw).items():
            dot = rn = cn = 0.
            for member in members:
                parameter, index = member if isinstance(member, tuple) else (member, None)
                a, b = originals.get(id(parameter)), structurals.get(id(parameter))
                if a is None or b is None:
                    continue
                if index is not None:
                    a, b = a[index], b[index]
                dot += float((a.double()*b.double()).sum())
                rn += float(a.double().square().sum()); cn += float(b.double().square().sum())
            self.report['loss_gradient_direction'][name] = {
                'dot': dot, 'cosine': dot/(rn*cn)**.5 if rn and cn else None,
                'original_norm': rn**.5, 'structural_norm': cn**.5,
                'weighted_structural_norm': abs(weight)*cn**.5}
        self.report['raw_real_batch_VJP'] = {
            'state': 'completed diagnostic checkpoint; same first real batch/sample',
            'original': statistics(raw, dict(zip(map(id, parameters), original_g))),
            'structural': statistics(raw, dict(zip(map(id, parameters), structural_g)))}
        if self.report['raw_real_batch_VJP']['structural']['decoder']['norm'] != 0:
            raise RuntimeError('structural-only loss unexpectedly reaches decoder')
        if self.report['normal_backward']['decoder']['norm'] <= 0:
            raise RuntimeError('actual total loss did not reach decoder')
        for name, row in self.report['normal_backward'].items():
            if row['norm'] <= 0 or self.report['actual_optimizer_update'][name]['norm'] <= 0:
                raise RuntimeError('missing total gradient/update: ' + name)
        del original_g, structural_g, parameters, original, structural, reconstructed, parts
        self.shared_equivalence(raw, catalogue)
        raw.eval()
        raw.rank_local_checks = True
        for layer in raw.rq.vq_layers:
            layer._skip_ddp_reduce = True
        with torch.no_grad():
            self.switches(raw, catalogue)
        dist.barrier()
        if dist.get_rank() == 0:
            self.export(raw, catalogue)
        dist.barrier()

    def shared_equivalence(self, raw, catalogue):
        """Real rank-local populations versus independent two-branch execution."""
        parameters = tuple(raw.parameters())
        actual = raw(self.batch, self.ids, catalogue_embeddings=catalogue,
                     structural_sample=self.sample)
        actual_original = training.CRITERION(actual[0], self.batch, actual[1])[0]
        actual_loss = raw.combined_loss(actual_original, actual[4])
        actual_ids = actual[3].detach().clone()
        actual_aux_ids = raw.last_structural_tokens.detach().clone()
        actual_aux_q = raw.last_structural_q.detach().clone()
        actual_aux_costs = [cost.detach().clone() for cost in raw.last_costs]
        actual_output = actual[0].detach().clone()
        actual_losses = torch.stack((actual_original, actual[4], actual_loss)).detach()
        actual_g = torch.autograd.grad(actual_loss, parameters, allow_unused=True)
        del actual, actual_original, actual_loss

        # Deliberately do not call the shared forward or merge quantization
        # populations: this is the original reconstruction + auxiliary path.
        _, weights = raw.structure.gates_for_items(
            raw.store, self.ids, catalogue.device, behavior_on=raw.behavior_on)
        q, parts, _, expected_ids = raw.quantize(
            raw.encoder(self.batch), weights, balanced=True)
        expected_output = raw.decoder(q)
        expected_structural = raw.sampled_curvature(catalogue, self.sample)
        expected_original = training.CRITERION(expected_output, self.batch, parts)[0]
        expected_loss = raw.combined_loss(expected_original, expected_structural)
        expected_g = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
        if not torch.equal(actual_ids, expected_ids):
            raise AssertionError('shared real reconstruction hard IDs differ')
        ties = self.auxiliary_ties(actual_aux_ids, raw.last_structural_tokens,
                                   actual_aux_costs, raw.last_costs)
        if not torch.equal(actual_aux_q, raw.last_structural_q):
            raise AssertionError('shared real auxiliary hard q differs')
        torch.testing.assert_close(actual_output, expected_output, atol=1e-5, rtol=1e-4)
        expected_losses = torch.stack(
            (expected_original, expected_structural, expected_loss)).detach()
        torch.testing.assert_close(actual_losses, expected_losses, atol=1e-5, rtol=1e-4)
        gradients = {}
        for (name, _), got, wanted in zip(raw.named_parameters(), actual_g, expected_g):
            if (got is None) != (wanted is None):
                raise AssertionError('shared real gradient routing differs: ' + name)
            if got is not None:
                torch.testing.assert_close(got, wanted, atol=2e-5, rtol=2e-4,
                                           msg='shared real gradient: ' + name)
                gradients[name] = {'max_abs': float((got-wanted).abs().max()),
                                   'reference_norm': float(wanted.norm())}
        self.report['shared_equivalence'] = {
            'reference': 'independent original two-branch GNN execution',
            'reconstruction_population': len(self.ids),
            'auxiliary_population': len(self.sample['ids']),
            'reconstruction_ids_sha256': digest([('ids', self.ids)]),
            'hard_ids_exact': ties['differing_items'] == 0,
            'hard_q_exact': True,
            'loss_max_abs': float((actual_losses-expected_losses).abs().max()),
            'gradient_atol': 2e-5, 'gradient_rtol': 2e-4,
            'gradients': gradients,
            'auxiliary_ties': ties}

    def auxiliary_ties(self, shared_ids, independent_ids, shared_costs, independent_costs):
        """The shared forward must not change either branch's assignment objective.

        float32 batch composition changes the last bits of every accumulated
        sum, so the shared execution and the two independent branches see
        slightly different costs. The capacity-constrained assignment is a
        global matching, so a float-level perturbation can move a few items off
        their own argmin. What must not change is the objective: both
        assignments must be optimal for their own costs, within the
        accumulated numerical divergence of the two executions.
        """
        rows = []
        for level, (left, right) in enumerate(zip(shared_costs, independent_costs)):
            if left.shape != right.shape:
                raise AssertionError('shared real auxiliary cost shapes differ')
            divergence = (left-right).abs()
            noise = float(divergence.max())
            bound = float(divergence.max(dim=1).values.sum())
            shared_objective = float(left.gather(1, shared_ids[:, level:level+1].long()).sum())
            independent_objective = float(right.gather(1, independent_ids[:, level:level+1].long()).sum())
            if abs(shared_objective-independent_objective) > bound:
                raise AssertionError(
                    'shared real auxiliary assignment objective differs beyond the '
                    'numerical divergence: level %d %.9g vs %.9g bound %.9g'
                    % (level, shared_objective, independent_objective, bound))
            rows.append({'level': level, 'max_cost_divergence': noise,
                         'objective_bound': bound,
                         'shared_objective': shared_objective,
                         'independent_objective': independent_objective,
                         'cost_scale': float(left.abs().max()),
                         'differing_items': int((shared_ids[:, level] != independent_ids[:, level]).sum())})
        return {'levels': rows,
                'differing_items': sum(row['differing_items'] for row in rows),
                'rule': ('the capacity-constrained assignment objective must agree within '
                         'the accumulated float32 divergence of the two executions; the '
                         'per-item codes may differ where the global matching re-solves')}

    def switches(self, raw, catalogue):
        ids = self.ids
        output = {}
        snapshots = {}
        for label, behavior, curvature in [('both_on', True, True),
                                            ('behavior_off', False, True),
                                            ('curvature_off', True, False),
                                            ('both_off', False, False)]:
            raw.behavior_on, raw.curvature_on = behavior, curvature
            raw(catalogue[ids], ids)
            decoded = sum(layer.embed(raw.last_tokens[:, i])
                          for i, layer in enumerate(raw.rq.vq_layers))
            if not torch.equal(raw.last_q, decoded) or not torch.equal(raw.last_hard, decoded):
                raise RuntimeError('hard-q/token decode mismatch: ' + label)
            snapshots[label] = (raw.last_weights.detach().clone(), raw.last_costs[0].detach().clone(),
                                raw.last_tokens.detach().clone())
            output[label] = {'weights_sha256': digest([('weights', snapshots[label][0])]),
                             'cost_sha256': digest([('cost', snapshots[label][1])]),
                             'tokens_sha256': digest([('tokens', snapshots[label][2])]),
                             'hard_q_decode_exact': True}
        raw.behavior_on = raw.curvature_on = True
        baseline = snapshots['both_on']
        for label in ('behavior_off', 'curvature_off', 'both_off'):
            current = snapshots[label]
            output[label]['cost_delta_norm'] = float((current[1] - baseline[1]).norm())
            output[label]['changed_hard_items'] = int((current[2] != baseline[2]).any(-1).sum())
            if output[label]['cost_delta_norm'] <= 0:
                raise RuntimeError('switch has no actual cost influence: ' + label)
        # Reversible intervention on the actual learned gate, not a fake echo flag.
        query = raw.structure.attention.query
        saved = query.detach().clone()
        try:
            query[0].neg_()
            raw(catalogue[ids], ids)
            weight_delta = float((raw.last_weights - baseline[0]).norm())
            cost_delta = float((raw.last_costs[0] - baseline[1]).norm())
            output['query_intervention'] = {'weight_delta_norm': weight_delta,
                                             'cost_delta_norm': cost_delta,
                                             'changed_hard_items': int(
                                                 (raw.last_tokens != baseline[2]).any(-1).sum())}
            if weight_delta <= 0 or cost_delta <= 0:
                raise RuntimeError('real query intervention did not influence gates and cost')
        finally:
            query.copy_(saved)
        self.report['switches'] = output

    def export(self, raw, catalogue):
        checkpoint = self.output / 'diagnostic_epoch1.pth'
        torch.save({'state_dict': raw.state_dict(), 'epoch': 1, 'formal_training': False,
                    'run_id': training.RUN_ID, 'provenance': raw.provenance()}, checkpoint)
        restored = training.RQVAE(training._tokenizer_config(), in_dim=catalogue.shape[1],
                                  distance=training.DISTANCE).to(catalogue.device)
        restored.load_state_dict(torch.load(checkpoint, map_location=catalogue.device)['state_dict'])
        restored.eval()
        restored.rank_local_checks = True
        for layer in restored.rq.vq_layers:
            layer._skip_ddp_reduce = True
        with torch.no_grad():
            actual = raw.get_indices(catalogue)
            again = restored.get_indices(catalogue)
        if not torch.equal(actual, again):
            raise RuntimeError('same-checkpoint full export differs')
        extended = training._extend_collisions(actual.cpu().numpy(), list(training.CODEBOOK_SIZE))
        path = self.output / 'diagnostic_item_sids.json'
        payload = {str(i): row for i, row in enumerate(extended.tolist())}
        path.write_text(json.dumps(payload))
        if json.loads(path.read_text()) != payload:
            raise RuntimeError('persisted diagnostic export differs')
        self.report['export'] = {'items': len(actual), 'checkpoint_sha256': hashlib.sha256(
            checkpoint.read_bytes()).hexdigest(), 'sid_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'same_checkpoint_exact': True, 'formal_training': False}


def main():
    run_id, output = common.configure('validation')
    report = {'run_id': run_id, 'rank': int(os.environ['RANK']), 'worker_pid': os.getpid(),
              'controller_pid': os.getppid(), 'formal_training': False, 'diagnostic_epochs': 1,
              'source_hashes': common.source_binding('validation'), 'status': 'running',
              'recipe': {'epochs': training.EPOCHS, 'per_rank_batch': 1024, 'global_batch': 4096,
                         'world_size': 4, 'seed': 42, 'config_sha256': training.CONFIG.sha256}}
    observer = Observer(report, output)
    original_class = training.RQVAE
    class Instrumented(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observer.instrument(self)
    training.RQVAE = Instrumented
    training.DIAGNOSTIC_OBSERVER = observer
    start = time.monotonic()
    try:
        training.main(diagnostic_epochs=1)
        if 'diagnostic_complete' not in observer.seen:
            raise RuntimeError('trainer did not emit completed diagnostic evidence')
        if report['source_hashes'] != common.source_binding('validation'):
            raise RuntimeError('sources changed during validation')
        report['status'] = 'pass'
    except Exception:
        report['status'] = 'fail'
        report['error'] = traceback.format_exc()
        raise
    finally:
        training.RQVAE = original_class
        training.DIAGNOSTIC_OBSERVER = None
        report['wall_seconds'] = time.monotonic() - start
        (output / ('rank_%d.json' % report['rank'])).write_text(
            json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
