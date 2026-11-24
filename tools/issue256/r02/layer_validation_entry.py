"""Pre-dispatch one-epoch diagnostic of the preregistered candidate source.

No formal checkpoint/SID is written or replaced.
The existing actual-trainer observer supplies DDP/update/export checks; this
observer adds the layer axis and individual real structural-loss VJPs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
from tools.issue256.r02.identity import SOURCE as TREE
sys.path.insert(0, str(TREE))
sys.path.insert(1, str(ROOT))

import torch
import validation_entry as original
import model.mechanism as mechanism

_CAPTURED = ()
_aggregate = mechanism.aggregate_layer_losses


def capture_layer_losses(values):
    global _CAPTURED
    _CAPTURED = tuple(values)
    return _aggregate(values)


class LayerObserver(original.Observer):
    def __init__(self, report, output):
        super().__init__(report, output)
        report['diagnostic_source'] = {
            'path': str(Path(__file__).resolve()),
            'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        report['timing_scope'] = 'pre-dispatch verification of the preregistered global-budget policy'
        report['executed_source_tree'] = str(TREE)

    def __call__(self, event, **fields):
        if event != 'first_step_before_backward':
            return super().__call__(event, **fields)
        if event in self.seen:
            return
        self.seen.add(event)
        model = fields['model']
        self.before = {id(p): p.detach().clone() for p in model.parameters()}
        self.batch = fields['batch'].detach().clone()
        self.ids = torch.as_tensor(fields['item_ids'], dtype=torch.long).detach().clone()
        self.sample = fields['structural_sample']
        self.report['first_step_losses'] = {
            key: float(fields[key].detach())
            for key in ('original_loss', 'structural_loss', 'combined_loss')}
        if not all(torch.isfinite(fields[key]).all() for key in self.report['first_step_losses']):
            raise RuntimeError('nonfinite actual first-step loss')
        if len(self.ids) != 1024:
            raise RuntimeError('diagnostic must use the actual full per-rank training batch')
        self.report['actual_batch_items'] = len(self.ids)
        self.report['first_structural_sample'] = {
            key: value.tolist() if hasattr(value, 'tolist') else value
            for key, value in self.sample.items()}
        decoded = torch.zeros_like(model.last_structural_q)
        prefix_exact = []
        for level, book in enumerate(model.rq.vq_layers):
            decoded = decoded + book.embed(model.last_structural_tokens[:, level])
            prefix_exact.append(torch.equal(decoded, model.last_prefixes[level]))
        if not all(prefix_exact) or not torch.equal(decoded, model.last_structural_q):
            raise RuntimeError('actual training hard prefix differs from selected SID codewords')
        self.report['actual_structural_hard_q_decode_exact'] = True
        self.report['actual_structural_prefix_decode_exact'] = prefix_exact
        weights = model.last_weights.detach()
        if tuple(weights.shape) != (len(self.ids), 3, 3):
            raise RuntimeError('gate must have item, level and expert axes')
        if not torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-6, rtol=0):
            raise RuntimeError('expert gates do not normalize independently per level')
        self.report['actual_gate_weights'] = {
            'population': len(weights), 'shape': list(weights.shape),
            'per_level': [
                {'mean': weights[:, level].mean(0).tolist(),
                 'min': weights[:, level].amin(0).tolist(),
                 'max': weights[:, level].amax(0).tolist(),
                 'argmax_histogram': torch.bincount(weights[:, level].argmax(-1), minlength=3).tolist()}
                for level in range(3)],
            'row_sum_max_abs_error': float((weights.sum(-1) - 1).abs().max())}
        attention = model.structure.last_attention.detach()
        if attention.shape[-1] != 5:
            raise RuntimeError('actual attention must retain four real tokens and the explicit null token')
        if not torch.allclose(attention.sum(-1), torch.ones_like(attention[..., 0]), atol=1e-6, rtol=0):
            raise RuntimeError('token attention is not normalized')
        self.report['actual_token_attention_shape'] = list(attention.shape)
        self.verify_initial_budget(model, fields)

    def verify_initial_budget(self, model, fields):
        """Independent raw replica and norm reducer; no VJP through DDP hooks."""
        import torch.distributed as dist
        from model.calibration import mean_gradients
        device = fields['catalogue'].device
        # Constructing a diagnostic replica must not consume training RNG.
        with torch.random.fork_rng(devices=[device.index]):
            raw = original.training.RQVAE(original.training._tokenizer_config(),
                in_dim=fields['catalogue'].shape[1], distance=original.training.DISTANCE).to(device)
        raw.load_state_dict(model.state_dict())
        raw.train()
        raw.rank_local_checks = True
        result = raw(fields['batch'], fields['item_ids'],
                     catalogue_embeddings=fields['catalogue'],
                     structural_sample=fields['structural_sample'])
        loss, _ = original.training.CRITERION(result[0], fields['batch'], result[1])
        parameters = tuple(p for name, p in raw.named_parameters() if not name.startswith('decoder.'))
        og = torch.autograd.grad(loss, parameters, retain_graph=True)
        cg = torch.autograd.grad(result[4], parameters)
        # Independent reduction, not the production norm/budget helper.
        for gradient in (*og, *cg):
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(dist.get_world_size())
        a = original.statistics(raw, dict(zip(map(id, parameters), og)))
        b = original.statistics(raw, dict(zip(map(id, parameters), cg)))
        rows = {}
        weight = float(model.curvature_weight)
        for name in a:
            if name == 'decoder':
                continue
            on, cn = a[name]['norm'], b[name]['norm']
            if on <= 0 or cn <= 0:
                raise AssertionError('missing actual global coupled-module VJP: ' + name)
            rows[name] = {'original_norm': on, 'curvature_norm': cn,
                          'weighted_ratio': weight * cn / on}
        if len(rows) != 17 or max(row['weighted_ratio'] for row in rows.values()) > 1 + 2e-6:
            raise AssertionError('actual initial global module budget exceeded')
        if abs(max(row['weighted_ratio'] for row in rows.values()) - 1) > 2e-6:
            raise AssertionError('actual initial global coefficient is not maximal')
        weights = [None] * 4
        dist.all_gather_object(weights, weight)
        if len(set(weights)) != 1:
            raise AssertionError('four-rank calibrated coefficients differ')
        # Actual production collective, with cancellation before the norm.
        probe = torch.tensor([float(dist.get_rank()) - 1.5, 2.], device=device)
        mean_gradients((probe,))
        if not torch.equal(probe, torch.tensor([0., 2.], device=device)):
            raise AssertionError('global VJPs must be averaged as vectors before norms')
        self.report['initial_global_module_budget'] = {
            'lambda': weight, 'rank_lambdas': weights, 'modules': rows,
            'cancellation_probe': probe.tolist(), 'actual_batch_items': len(fields['batch']),
            'independent_raw_replica': True, 'passed': True}

    def completed(self, model, catalogue):
        # A separate unwrapped model avoids invoking autograd.grad through the
        # DDP reducer. The losses and sampling are exactly the trainer's.
        raw = original.training.RQVAE(original.training._tokenizer_config(),
            in_dim=catalogue.shape[1], distance=original.training.DISTANCE).to(catalogue.device)
        raw.load_state_dict(model.state_dict())
        raw.train()
        result = raw(self.batch, self.ids, catalogue_embeddings=catalogue,
                     structural_sample=self.sample)
        losses = _CAPTURED
        if len(losses) != 3:
            raise RuntimeError('the production structural objective did not expose three levels')
        torch.testing.assert_close(result[4], torch.stack(losses).mean(), atol=1e-7, rtol=1e-6)
        _, components = original.training.CRITERION(result[0], self.batch, result[1])
        reconstruction_loss = components['reconstruction']
        parameters = tuple(raw.parameters())
        reconstruction_g = torch.autograd.grad(reconstruction_loss, parameters, allow_unused=True, retain_graph=True)
        weight = float(raw.curvature_weight) / 3
        rows = []
        for level, loss in enumerate(losses):
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
            values = dict(zip(map(id, parameters), gradients))
            stats = original.statistics(raw, values)
            if stats['encoder']['norm'] <= 0 or stats['codebook%d' % level]['norm'] <= 0:
                raise RuntimeError('individual prefix loss has no encoder/current-book gradient')
            for future in range(level + 1, 3):
                if stats['codebook%d' % future]['norm'] != 0:
                    raise RuntimeError('a future codebook directly influences an earlier prefix')
            for group in ('GNN.input', 'sharedKey', 'sharedValue', 'sharedScore', 'Query%d' % level):
                if stats[group]['norm'] <= 0:
                    raise RuntimeError('individual prefix loss has missing gradient: ' + group)
            rg = [g for (name, _), g in zip(raw.named_parameters(), reconstruction_g) if name.startswith('encoder.')]
            cg = [g for (name, _), g in zip(raw.named_parameters(), gradients) if name.startswith('encoder.')]
            pairs = [(a.double(), b.double()) for a, b in zip(rg, cg) if a is not None and b is not None]
            rn = sum(float(a.square().sum()) for a, _ in pairs) ** 0.5
            cn = sum(float(b.square().sum()) for _, b in pairs) ** 0.5
            cosine = sum(float((a * b).sum()) for a, b in pairs) / (rn * cn)
            rows.append({'level': level + 1, 'loss_unweighted': float(loss.detach()),
                         'lambda_alpha': weight, 'modules_unweighted': stats,
                         'modules_weighted': original.statistics(raw, {
                             key: None if g is None else g * weight for key, g in values.items()}),
                         'encoder_reconstruction_cosine': cosine,
                         'future_codebooks_zero': True})
        self.report['individual_layer_losses'] = rows
        del result, reconstruction_loss, reconstruction_g, gradients, parameters, losses, raw
        super().completed(model, catalogue)


def main():
    mechanism.aggregate_layer_losses = capture_layer_losses
    original.Observer = LayerObserver
    original.main()


if __name__ == '__main__':
    main()
