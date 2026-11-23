"""No-argument, four-rank controlled microstep proof; never a training epoch."""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import tarfile
from datetime import timedelta
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import train_rqvae as training
from model.mechanism import RQVAE

ROOT = training.REPO_ROOT
OUTPUT = ROOT / 'results/research/issue251/runtime'
PRETRAINED = ROOT / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth'
CONTROLLED_MICROSTEP = True
BATCH = 256


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_binding():
    tree = Path(__file__).resolve().parent
    paths = sorted(tree.rglob('*.py')) + sorted((tree / 'config').glob('*.json'))
    paths += [ROOT / name for name in ('assets/manifest.json',
        'tools/issue251/launch_validation.py','tools/research/stage2/launcher.py',
        'tools/graph_infra/ego_gnn.py','tools/graph_infra/assets.py',
        'results/research/issue251/CONTRACT.md')]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def groups(model):
    return {'encoder': list(model.encoder.parameters()),
            'GNN': list(model.structure.tokens.parameters()),
            'Q': [model.structure.attention.query],
            'K': list(model.structure.attention.key.parameters()),
            'V': list(model.structure.attention.value.parameters()),
            'score': list(model.structure.attention.score.parameters()),
            'codebook': list(model.rq.parameters()), 'decoder': list(model.decoder.parameters())}


def gradients(loss, model, retain_graph=False):
    parameters = list(model.parameters())
    values = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    lookup = {id(p): g for p, g in zip(parameters, values)}
    result = {}
    for name, members in groups(model).items():
        present = [lookup[id(p)] for p in members if lookup[id(p)] is not None]
        for value in present:
            if not bool(torch.isfinite(value).all()):
                raise AssertionError('nonfinite ' + name + ' gradient')
        norm = sum(float(g.double().square().sum()) for g in present) ** .5
        result[name] = {'norm': norm, 'none': len(members) - len(present),
                        'zero': sum(int(not bool(g.any())) for g in present), 'parameters': len(members)}
    return result, values


def synchronized(tensors):
    maximum = 0.
    for tensor in tensors:
        if tensor is None:
            continue
        if not bool(torch.isfinite(tensor).all()):
            raise AssertionError('nonfinite synchronized tensor')
        reference = tensor.detach().clone()
        dist.broadcast(reference, src=0)
        difference = float((tensor.detach() - reference).abs().max())
        maximum = max(maximum, difference)
    if maximum > 1e-6:
        raise AssertionError('rank mismatch: ' + str(maximum))
    return maximum


def decode(model, tokens):
    return sum(layer.embed(tokens[:, index]) for index, layer in enumerate(model.rq.vq_layers))


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 4:
        raise RuntimeError('validation requires all four ranks')
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl',timeout=timedelta(hours=1))
    device = torch.device('cuda', local_rank)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {'rank': rank, 'world_size': 4, 'seed': 42, 'controlled_microstep': CONTROLLED_MICROSTEP,
              'formal_training': False, 'timings_seconds': {}, 'status': 'running'}
    report.update({'run_id':'issue251_validation_'+str(os.getppid()),
                   'worker_pid':os.getpid(),'controller_pid':os.getppid(),
                   'master_port':int(os.environ['MASTER_PORT'])})
    started = time.perf_counter()
    def timed(name, action):
        report['phase']=name
        (OUTPUT/('rank_%d.json'%rank)).write_text(json.dumps(report,indent=2,allow_nan=False))
        print('[issue251 rank%d] %s start'%(rank,name),flush=True)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = action()
        torch.cuda.synchronize()
        report['timings_seconds'][name] = time.perf_counter() - begin
        print('[issue251 rank%d] %s finished %.3fs'%(rank,name,report['timings_seconds'][name]),flush=True)
        return result
    try:
        binding = source_binding()
        embeddings = torch.from_numpy(np.load(training.EMBEDDING_FILE)).to(device=device, dtype=torch.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != 768:
            raise AssertionError('expected original semantic embedding catalogue')
        model = RQVAE(training._tokenizer_config(), in_dim=embeddings.shape[1], distance=training.DISTANCE).to(device)
        initial = torch.load(PRETRAINED, map_location='cpu')
        incompatible = model.load_state_dict(initial['state_dict'], strict=False)
        allowed_buffers = {'tangent_scale', 'curvature_weight', 'target_scale'}
        if incompatible.unexpected_keys or any(not (key.startswith('structure.') or key in allowed_buffers)
                                              for key in incompatible.missing_keys):
            raise AssertionError(str(incompatible))
        if len(embeddings) != model.store.n:
            raise AssertionError('catalogue item identities do not match assets')
        report['pretrained'] = {'path': str(PRETRAINED.relative_to(ROOT)), 'sha256': sha256(PRETRAINED),
                                'missing_new_state': incompatible.missing_keys}
        report['source_hashes'] = binding
        report['embedding_sha256'] = sha256(training.EMBEDDING_FILE)
        report['catalogue_size'] = len(embeddings)
        calibration_centres = [int(model.targets.valid_centres[0])]
        centres = [int(model.targets.valid_centres[rank])]
        calibration_ids = torch.arange(BATCH,device=device)
        ids = torch.arange(rank*BATCH,(rank+1)*BATCH,device=device)
        batch = embeddings[ids]
        report['curvature_centres'] = centres
        report['reference_count'] = int(np.sum(model.targets.components == model.targets.components[centres[0]]))
        report['valid_triples'] = sum(len(triples) for triples, _, _ in model.targets.blocks(centres))
        if not report['valid_triples']:
            raise AssertionError('fixed centre has no valid triples')
        model.freeze_tangent_scale(embeddings)
        report['calibration'] = timed('initial_gradient_calibration', lambda:
            model.calibrate_curvature_weight(embeddings[calibration_ids], calibration_ids,
                                             embeddings, calibration_centres))
        dist.broadcast(model.curvature_weight,0)
        frozen_lambda = model.curvature_weight.detach().clone()
        report['tangent_scale'] = float(model.tangent_scale)
        report['target_scale'] = float(model.target_scale)
        model.train()
        report['last_layer_sinkhorn'] = bool(model.rq.vq_layers[-1].use_sk)
        if not report['last_layer_sinkhorn']:
            raise AssertionError('original last-layer Sinkhorn disabled')
        reconstructed, parts, _, tokens = model(batch, ids)
        original, reconstruction = model.compute_loss(batch, reconstructed, parts)
        if not torch.equal(model.last_hard, decode(model, tokens)):
            raise AssertionError('hard quantization differs from decoded SID')
        if not torch.equal(model.last_q, model.last_hard):
            raise AssertionError('surrogate forward is not hard quantization')
        report['initial_original_loss'] = float(original)
        report['initial_reconstruction_loss'] = float(reconstruction)
        report['original_gradients'], original_gradients = gradients(original, model)
        del reconstructed, parts, original, reconstruction
        def curvature_proof():
            q = model.catalogue_q(embeddings)
            if len(q) != len(embeddings):
                raise AssertionError('curvature catalogue truncated')
            if not torch.equal(model.last_hard, decode(model, model.last_tokens)):
                raise AssertionError('global hard quantization differs from decoded SID')
            loss = model.curvature_loss(q, centres)
            norms, values = gradients(loss, model)
            return float(loss), norms, values
        curvature, norms, curvature_gradients = timed('full_catalogue_A1_gradients', curvature_proof)
        report['initial_curvature_loss'] = curvature
        report['curvature_gradients'] = norms
        for name in ('encoder','GNN','Q','K','V','score','codebook'):
            if norms[name]['norm'] <= 0:
                raise AssertionError('missing actual A1 gradient: ' + name)
        if norms['decoder']['norm'] != 0:
            raise AssertionError('curvature unexpectedly reaches decoder')
        dot = sum(float((a.double() * b.double()).sum()) for a,b in zip(original_gradients,curvature_gradients)
                  if a is not None and b is not None)
        a_norm = sum(float(a.double().square().sum()) for a in original_gradients if a is not None) ** .5
        b_norm = sum(float(b.double().square().sum()) for b in curvature_gradients if b is not None) ** .5
        report['loss_gradient_cosine'] = dot / (a_norm * b_norm)
        report['unweighted_gradient_ratio'] = b_norm / a_norm
        report['weighted_gradient_ratio'] = float(frozen_lambda) * b_norm / a_norm
        positions={id(parameter):i for i,parameter in enumerate(model.parameters())}
        report['module_gradient_comparison']={}
        for group,members in groups(model).items():
            on=report['original_gradients'][group]['norm'];cn=norms[group]['norm']
            group_dot=0.
            for parameter in members:
                a=original_gradients[positions[id(parameter)]]
                b=curvature_gradients[positions[id(parameter)]]
                if a is not None and b is not None:
                    group_dot+=float((a.double()*b.double()).sum())
            report['module_gradient_comparison'][group]={
                'raw_curvature_norm':cn,'weighted_curvature_norm':float(frozen_lambda)*cn,
                'original_norm':on,'weighted_ratio':float(frozen_lambda)*cn/on if on else None,
                'cosine':group_dot/(on*cn) if on and cn else None}
        report['query_curvature_gradient_per_expert']=curvature_gradients[
            positions[id(model.structure.attention.query)]].double().norm(dim=-1).cpu().tolist()
        reference_gradients=[]
        for parameter,original_gradient,curvature_gradient in zip(
                model.parameters(),original_gradients,curvature_gradients):
            expected=torch.zeros_like(parameter)
            if original_gradient is not None:
                expected.add_(original_gradient)
            if curvature_gradient is not None:
                expected.add_(curvature_gradient,alpha=float(frozen_lambda))
            dist.all_reduce(expected)
            reference_gradients.append(expected/4)
        del original_gradients,curvature_gradients
        model.cut_selection_cost = True
        _, cut_norms, _ = timed('detached_cost_A1_negative', curvature_proof)
        model.cut_selection_cost = False
        report['detached_cost_curvature_gradients'] = cut_norms
        if any(cut_norms[name]['norm'] != 0 for name in ('GNN','Q','K','V','score')):
            raise AssertionError('detached assignment cost did not cut actual A1 structure route')
        sid_before = model.last_tokens.detach().clone()
        cost_before = model.last_costs[0][:8].detach().clone()
        with torch.no_grad():
            gate_before = model.structure.gates_for_items(model.store,ids[:8],device)[1].detach().clone()
        wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=training.LR, weight_decay=training.WEIGHT_DECAY)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        def microstep():
            optimizer.zero_grad(set_to_none=True)
            reconstructed, parts, _, _, curv = wrapped(batch, ids, catalogue_embeddings=embeddings,
                                                       curvature_centres=centres)
            original, _ = model.compute_loss(batch, reconstructed, parts)
            loss = model.combined_loss(original, curv)
            if not bool(torch.isfinite(loss)):
                raise AssertionError('nonfinite combined loss')
            loss.backward()
            relative_error=max(float((p.grad-e).abs().max() /
                (1+p.grad.abs().max()+e.abs().max()))
                for p,e in zip(model.parameters(),reference_gradients) if p.grad is not None)
            report['DDP_vs_independent_rank_mean_relative_error']=relative_error
            if relative_error>1e-4:
                raise AssertionError('DDP gradient differs from mean of independent real losses')
            report['gradient_rank_max_difference'] = synchronized([p.grad for p in model.parameters()])
            report['preclip_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            report['postclip_norm']=sum(float(p.grad.double().square().sum())
                for p in model.parameters() if p.grad is not None)**.5
            optimizer.step()
            return {'original': float(original), 'curvature': float(curv), 'combined': float(loss)}
        report['microstep_losses'] = timed('DDP_controlled_microstep', microstep)
        report['learning_rate'] = training.LR
        report['clip_norm'] = 1.0
        report['parameter_rank_max_difference'] = synchronized(list(model.parameters()))
        report['parameter_update_norms'] = {name: float((parameter.detach()-before[name]).double().norm())
                                            for name, parameter in model.named_parameters()}
        names={id(parameter):name for name,parameter in model.named_parameters()}
        report['module_update_norms']={group:sum(float(
            (parameter.detach()-before[names[id(parameter)]]).double().square().sum())
            for parameter in members)**.5 for group,members in groups(model).items()}
        if any(report['module_update_norms'][group]<=0 for group in ('GNN','Q','K','V','score')):
            raise AssertionError('controlled update did not change all trainable structure groups')
        del before
        if not torch.equal(frozen_lambda, model.curvature_weight):
            raise AssertionError('calibration changed during microstep')
        dist.barrier()
        # Other ranks wait in the barrier: any inference collective deadlocks and fails the run.
        if rank == 0:
            def export_reload():
                model.eval()
                held = torch.tensor([0, centres[0], len(embeddings)-1, 0], device=device)
                first = model.get_indices(embeddings[held], item_ids=held)
                with torch.no_grad():
                    gate_after=model.structure.gates_for_items(model.store,ids[:8],device)[1]
                report['attention_token_axes']=model.structure.last_attention.detach().cpu().tolist()
                report['attention_valid_token_mask']=model.structure.last_token_mask.cpu().tolist()
                report['expert_geometry_weights']=gate_after.cpu().tolist()
                report['gate_weight_change_norm']=float((gate_after-gate_before).norm())
                report['selection_cost_change_norm']=float((model.last_costs[0][:8]-cost_before).norm())
                report['hard_sid_changed_items']=int((model.last_tokens!=sid_before).any(-1).sum())
                if report['gate_weight_change_norm']<=0 or report['selection_cost_change_norm']<=0:
                    raise AssertionError('controlled update did not change gates and selection costs')
                if report['hard_sid_changed_items']==0:
                    raise AssertionError('controlled microstep did not change any actual hard SID')
                second = model.get_indices(embeddings[held], item_ids=held)
                if not torch.equal(first, second):
                    raise AssertionError('repeated real-ID inference differs')
                state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                checkpoint_path = OUTPUT / 'controlled_microstep.pth'
                torch.save({'state_dict': state, 'source_hashes': binding, 'asset_manifest_sha256': sha256(ROOT/'assets/manifest.json'),
                            'pretrained_sha256': report['pretrained']['sha256'], 'controlled_microsteps': 1,
                            'config': vars(training._tokenizer_config()),
                            'provenance':model.provenance()}, checkpoint_path)
                with tarfile.open(OUTPUT / 'source_snapshot.tar.gz', 'w:gz') as archive:
                    for source in binding:
                        archive.add(ROOT / source, arcname=source)
                reloaded = RQVAE(training._tokenizer_config(), in_dim=embeddings.shape[1], distance=training.DISTANCE).to(device)
                reloaded.load_checkpoint(checkpoint_path)
                reloaded.eval()
                all_ids = torch.arange(len(embeddings), device=device)
                actual = model.get_indices(embeddings, item_ids=all_ids)
                restored = reloaded.get_indices(embeddings, item_ids=all_ids)
                if not torch.equal(actual, restored):
                    raise AssertionError('checkpoint export reload changed SID')
                if not torch.equal(first, restored[held]):
                    raise AssertionError('held real-ID inference differs from catalogue export')
                path = OUTPUT / 'item_sids.json'
                path.write_text(json.dumps({str(i): row for i,row in enumerate(actual.cpu().tolist())}))
                disk = json.loads(path.read_text())
                if [disk[str(i)] for i in range(len(embeddings))] != restored.cpu().tolist():
                    raise AssertionError('persisted export differs')
                return {'checkpoint_sha256': sha256(checkpoint_path), 'export_sha256': sha256(path),
                        'items': len(actual), 'held_item_ids': held.cpu().tolist(), 'exact_reload': True,
                        'rank0_inference_without_collectives': True},reloaded
            report['export'],benchmark_model = timed('rank0_export_reload',export_reload)
            def full_target_proof():
                q=benchmark_model.catalogue_q(embeddings)
                loss=benchmark_model.curvature_loss(q)
                module_norms,values=gradients(loss,benchmark_model)
                if benchmark_model.targets.last_rows!=28391948:
                    raise AssertionError('full target traversal count mismatch')
                if not bool((benchmark_model.last_token_counts==2).all()):
                    raise AssertionError('catalogue token coverage incomplete')
                attention_file=OUTPUT/'catalogue_attention.npz'
                np.savez_compressed(attention_file,
                    token_attention=benchmark_model.catalogue_attention.cpu().numpy(),
                    expert_weights=benchmark_model.last_weights.detach().cpu().numpy(),
                    token_counts=benchmark_model.last_token_counts.cpu().numpy())
                return {'loss':float(loss),'valid_triples':benchmark_model.targets.last_rows,
                        'centres':benchmark_model.targets.last_centres,
                        'module_gradients':module_norms,'full_catalogue_items':len(q),
                        'tokens_per_item':2,'attention_sha256':sha256(attention_file),
                        'optimizer_steps':0}
            report['full_target_end_to_end']=timed('all_28391948_targets_end_to_end',full_target_proof)
        dist.barrier()
        if binding != source_binding():
            raise AssertionError('source changed during validation')
        report['status'] = 'pass'
    except Exception:
        report['status'] = 'fail'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        report['cuda_peak_bytes'] = torch.cuda.max_memory_allocated(device)
        (OUTPUT / ('rank_%d.json' % rank)).write_text(json.dumps(report, indent=2, allow_nan=False))
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
