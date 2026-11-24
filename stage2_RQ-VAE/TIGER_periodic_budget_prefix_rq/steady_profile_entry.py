"""Profile complete warmed epoch 2 of three actual diagnostic training epochs.

The frozen recipe/scheduler remain 3000 epochs. Nested instrumentation is not a
throughput benchmark and does not decide the inclusive 60-epoch speed gate.
"""
import functools
import numpy as np
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

BOOT_TIME = time.perf_counter()
REPO = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'tools/research/stage2/contract.py').is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
from tools.issue252.profiling import Tracker
from tools.issue252.profile_units import operator_record


class SteadyTracker(Tracker):
    def __init__(self, output, boot_time, training):
        super().__init__(output, boot_time, training)
        self.enabled = False
        self.boundaries = []
        self.completed_epochs = 0
        self.diagnostic_end = None
        self.window_seconds = None
        self.reserved_peak = 0
        self.total_steps = 0
        self.scheduler_steps = 0
        self.window_closed = False
        self.current_hop = 0
        self.hop_by_layout = {}
        self.workloads = []

    def sample_peak(self):
        peak = super().sample_peak()
        if self.enabled:
            self.reserved_peak = max(self.reserved_peak, torch.cuda.max_memory_reserved())
        return peak

    def wrap(self, parent, name, label, observe=None):
        original = getattr(parent, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            actual = label(*args, **kwargs) if callable(label) else label
            if observe is not None:
                observe(args, kwargs, None, self.stage, actual)
            with self.section(actual):
                result = original(*args, **kwargs)
            if observe is not None and result is not None:
                observe(args, kwargs, result, self.stage, actual)
            return result

        setattr(parent, name, wrapped)

    def start_window(self):
        torch.cuda.synchronize()
        self.initialization_seconds = time.perf_counter() - self.boot_time
        self.init_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self.stage = 'training_step'
        self.profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False, with_stack=False)
        self.profiler.__enter__()
        self.window_start = time.perf_counter()
        self.enabled = True

    def stop_window(self):
        torch.cuda.synchronize()
        self.window_seconds = time.perf_counter() - self.window_start
        self.sample_peak()
        self.enabled = False
        self.profiler.__exit__(None, None, None)
        self.window_closed = True

    def install_record_hook(self):
        original = self.training._record

        @functools.wraps(original)
        def record(rank, event, **fields):
            # The trainer calls this on EVERY rank before _record's rank-0 guard.
            # Keep the original event/log semantics, including all barriers.
            result = original(rank, event, **fields)
            if event == 'epoch_boundary':
                self.boundaries.append(dict(fields))
                self.completed_epochs = fields['epoch']
                if fields['epoch'] == 1:
                    self.start_window()
                elif fields['epoch'] == 2:
                    self.stop_window()
            elif event == 'diagnostic_end':
                self.diagnostic_end = dict(fields)
            return result

        self.training._record = record

    def install(self):
        import model.geometry as geometry
        import model.curvature as curvature
        import model.selection as selection
        import model.layers as layers
        import model.structure as structure
        import model.estimator as estimator
        import model.mechanism as mechanism
        from model.layer_gate import LayerExpertAttention
        from losses import TotalLoss

        self.install_record_hook()
        self.wrap(structure, 'batch_operators', 'ego_static_operator_assembly', self.observe_ego)
        self.wrap(structure.SubgraphTokens, '_read_sparse',
                  lambda *a, **kw: 'GNN_%dhop' % self.current_hop, self.observe_shapes)
        token_function = structure.SubgraphTokens.tokens_for_items
        def tokens(owner, store, ids, device, chunk=1024):
            if not self.enabled:
                return token_function(owner, store, ids, device, chunk=chunk)
            host = np.asarray(ids.detach().cpu().numpy() if torch.is_tensor(ids) else ids,
                              dtype=np.int64).reshape(-1)
            unique = np.unique(host)
            layouts, _ = owner._cached_store(store)
            self.hop_by_layout.update({id(layout): hop for hop, layout in enumerate(layouts, 1)})
            scope = [label for stage, label in self.active]
            for hop, layout in enumerate(layouts, 1):
                self.workloads.append({'hop': hop, 'requested_items': len(host),
                    'unique_items': len(unique), 'ordered_ids': host.tolist(),
                    'nodes': int(layout.node_counts[unique].sum()),
                    'directed_edges': int(layout.in_edge_counts[unique].sum()),
                    'caller_scopes': scope})
            with self.section('full_ego_tokens'):
                return token_function(owner, store, ids, device, chunk=chunk)
        structure.SubgraphTokens.tokens_for_items = tokens
        geometry_function = structure.ego_kernel.batch_geometry
        def layout_geometry(layout, *args, **kwargs):
            if not self.enabled:
                return geometry_function(layout, *args, **kwargs)
            self.current_hop = self.hop_by_layout[id(layout)]
            with self.section('cpu_layout_index_transfer_%dhop' % self.current_hop):
                return geometry_function(layout, *args, **kwargs)
        structure.ego_kernel.batch_geometry = layout_geometry
        self.wrap(structure, '_gather_features',
                  lambda *a, **kw: 'static_feature_gather_%dhop' % self.current_hop, self.observe_shapes)
        self.wrap(structure.StaticFeatureMeans, 'gather',
                  lambda *a, **kw: 'static_neighbour_mean_gather_%dhop' % self.current_hop, self.observe_shapes)
        self.wrap(structure, 'directional_means',
                  lambda *a, **kw: 'second_directional_aggregation_%dhop' % self.current_hop, self.observe_shapes)
        self.wrap(structure, '_second_affine',
                  lambda *a, **kw: 'second_affine_activation_%dhop' % self.current_hop, self.observe_shapes)
        self.wrap(structure, '_pool_readout',
                  lambda *a, **kw: 'pooling_%dhop' % self.current_hop, self.observe_shapes)
        factory = structure._compiled_affine_layer
        @functools.lru_cache(maxsize=1)
        def measured_affine_factory():
            compiled = factory()
            def measured(*args, **kwargs):
                if not self.enabled:
                    return compiled(*args, **kwargs)
                with self.section('first_affine_%dhop' % self.current_hop):
                    return compiled(*args, **kwargs)
            return measured
        structure._compiled_affine_layer = measured_affine_factory
        self.wrap(mechanism.RQVAE, 'quantize', 'separate_rank_local_quantization', self.observe_shapes)
        self.wrap(mechanism.RQVAE, '_identities', 'cpu_identity_index', self.observe_shapes)
        self.wrap(geometry, 'fixed_squared', 'three_geometry_fixed_distance', self.observe_shapes)
        mechanism.fixed_squared = geometry.fixed_squared
        self.wrap(LayerExpertAttention, 'forward', 'Attention', self.observe_shapes)
        self.wrap(layers.MLP, 'forward',
                  lambda owner, x: 'Encoder' if x.shape[-1] == 768 else 'Decoder', self.observe_shapes)
        for name, label in (('forward', 'production_forward'),
                            ('catalogue_q', 'whole_catalogue_forward'),
                            ('sampled_curvature', 'sampled_structural_forward'),
                            ('assignment_cost', 'three_geometry_assignment_cost'),
                            ('combined_loss', 'combined_loss')):
            self.wrap(mechanism.RQVAE, name, label, self.observe_shapes)
        self.wrap(estimator.StructuralSampler, 'draw', 'structural_sample', self.observe_ego)
        self.wrap(estimator, 'sampled_values', 'sampled_triple_statistic', self.observe_shapes)
        mechanism.sampled_values = estimator.sampled_values
        self.wrap(geometry, 'expert_squared', 'three_geometry_distances', self.observe_shapes)
        self.wrap(layers.VQLayer, 'sinkhorn', 'hard_Sinkhorn', self.observe_shapes)
        self.wrap(selection, 'probabilities', 'soft_allocation', self.observe_shapes)
        self.wrap(layers, 'constrained_assignment', 'assignment_floor', self.observe_shapes)
        for name in ('verify_assignment', 'verify_codebook', 'verify_margins', 'verify_separation'):
            self.wrap(layers, name, 'assignment_guards', self.observe_shapes)
        self.wrap(curvature, 'reference_moments', 'full_reference_distances', self.observe_shapes)
        self.wrap(curvature, 'sectional_values', 'triple_statistic', self.observe_shapes)
        self.wrap(curvature.FrozenCurvature, 'loss', 'full_curvature_loss', self.observe_curvature)
        self.wrap(TotalLoss, '__call__', 'criterion', self.observe_shapes)
        self.wrap(torch.autograd, 'backward', 'backward', self.observe_shapes)
        self.wrap(torch.nn.utils, 'clip_grad_norm_', 'gradient_clip', self.observe_shapes)
        for name in ('all_reduce', 'all_gather_object', 'barrier', 'broadcast'):
            self.wrap(dist, name, 'communication::' + name)

        tracker = self
        loader = self.training.DataLoader

        class MeasuredLoader:
            def __init__(self, inner):
                self.inner = inner

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def __len__(self):
                return len(self.inner)

            def __iter__(self):
                iterator = iter(self.inner)
                while True:
                    try:
                        if tracker.enabled:
                            with tracker.section('data_read'):
                                batch = next(iterator)
                        else:
                            batch = next(iterator)
                    except StopIteration:
                        return
                    yield batch

        def make_loader(*args, **kwargs):
            inner = loader(*args, **kwargs)
            tracker.loader = {'items': len(inner.dataset), 'steps_per_epoch': len(inner),
                              'batch_per_rank': inner.batch_size}
            return MeasuredLoader(inner)

        self.training.DataLoader = make_loader
        for cls in (torch.optim.AdamW, torch.optim.Adagrad):
            original = cls.step

            @functools.wraps(original)
            def optimizer_step(optimizer, *args, _original=original, **kwargs):
                if tracker.enabled:
                    with tracker.section('optimizer'):
                        result = _original(optimizer, *args, **kwargs)
                    tracker.steps += 1
                else:
                    result = _original(optimizer, *args, **kwargs)
                tracker.total_steps += 1
                return result

            cls.step = optimizer_step
        scheduler_step = torch.optim.lr_scheduler.LambdaLR.step

        @functools.wraps(scheduler_step)
        def measured_scheduler(scheduler, *args, **kwargs):
            if tracker.enabled:
                with tracker.section('scheduler'):
                    result = scheduler_step(scheduler, *args, **kwargs)
                tracker.scheduler_steps += 1
                return result
            return scheduler_step(scheduler, *args, **kwargs)

        torch.optim.lr_scheduler.LambdaLR.step = measured_scheduler

    def results(self):
        totals = {}
        for stage, label, wall, start, end in self.rows:
            row = totals.setdefault(label, {'calls': 0, 'cpu_dispatch_wall_seconds': 0.,
                                          'cuda_stream_span_seconds': 0.,
                                          'phase_peak_allocated_bytes': self.phase_peaks[(stage, label)],
                                          'footprint': dict(self.footprints[(stage, label)])})
            row['calls'] += 1
            row['cpu_dispatch_wall_seconds'] += wall
            row['cuda_stream_span_seconds'] += start.elapsed_time(end) / 1000
        operators = []
        if self.window_closed:
            for event in self.profiler.key_averages():
                row = operator_record(event)
                if row['self_device_seconds'] > 0 or any(
                        word in event.key.lower() for word in ('allreduce', 'nccl', 'c10d')):
                    operators.append(row)
        operators.sort(key=lambda row: row['self_device_seconds'], reverse=True)
        return {'profiled_epoch': 2, 'profiled_optimizer_steps': self.steps,
                'profiled_scheduler_steps': self.scheduler_steps,
                'completed_optimizer_steps': self.total_steps,
                'completed_epochs': self.completed_epochs, 'epoch_boundaries': self.boundaries,
                'diagnostic_end': self.diagnostic_end, 'loader': self.loader,
                'warmup_through_epoch1_seconds': self.initialization_seconds,
                'warmup_end_allocated_peak_bytes': self.init_peak,
                'profile_window_wall_seconds': self.window_seconds,
                'profile_window_peak_allocated_bytes': self.step_peak,
                'profile_window_peak_reserved_bytes': self.reserved_peak,
                'phases': [{'phase': name, **row} for name, row in sorted(totals.items())],
                'cuda_operators': operators, 'ego_workloads': self.workloads,
                'phase_memory_scope': 'Actual absolute allocator allocated high-water marks while each span is active; child peaks propagate to parents. Reserved high-water mark covers the whole warmed window.',
                'interpretation': 'Instrumented full epoch2 after actual epoch1 initialization/cold start. Three production diagnostic epochs, no formal checkpoint/SID. Nested/inclusive phase spans overlap and MUST NOT be summed. CUDA operator times are exclusive self device times; concurrent kernels/ranks are not an additive epoch estimate. This profile makes no speed acceptance claim; the inclusive 60-epoch benchmark is still required.'}


def source_binding(training):
    import benchmark_entry
    files = benchmark_entry.source_binding('steady_profile')['files']
    for name in ('tools/issue252/profiling.py', 'tools/issue252/profile_units.py'):
        path = training.REPO_ROOT / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def main():
    import train_rqvae as training
    if int(os.environ.get('WORLD_SIZE', '0')) != 4 or int(os.environ.get('MASTER_PORT', '0')) != 50202:
        raise RuntimeError('steady profile requires the sanctioned four-rank/50202 launcher')
    if training._single_card_mode() or (training.EPOCHS, training.BATCH_SIZE_PER_RANK, training.SEED) != (3000, 1024, 42):
        raise RuntimeError('frozen four-rank training recipe changed')
    controller = os.getppid()
    start_tick = (Path('/proc') / str(controller) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    run_id = 'issue256_r02_steady_profile_' + str(controller) + '_' + start_tick
    output = training.REPO_ROOT / 'results/research/issue256/rounds/r02/steady_profile' / run_id
    output.mkdir(parents=True, exist_ok=True)
    training.RUN_ID = run_id
    training.RESULT_DIR = output
    training.LOG_DIR = output / 'logs'
    training.METRICS_PATH = training.LOG_DIR / 'training_metrics.jsonl'
    training.CHECKPOINT = output / 'forbidden_formal_checkpoint.pth'
    training.OUTPUT_SID = output / 'forbidden_formal_sids.npy'
    training.OUTPUT_JSON = output / 'forbidden_formal_item_sids.json'
    rank = int(os.environ['RANK'])
    before = source_binding(training)
    tracker = SteadyTracker(output, BOOT_TIME, training)
    report = {'status': 'FAIL', 'run_id': run_id, 'rank': rank, 'worker_pid': os.getpid(),
              'controller_pid': controller, 'formal_training': False, 'diagnostic_epochs': 3,
              'world_size': 4, 'global_batch': 4096, 'master_port': 50202,
              'source_hashes_before': before,
              'recipe': {'epochs': training.EPOCHS, 'batch_per_rank': training.BATCH_SIZE_PER_RANK,
                         'seed': training.SEED, 'config_sha256': training.CONFIG.sha256,
                         'config': json.loads(Path(training.CONFIG.path).read_text())}}
    error = None
    try:
        tracker.install()
        training.main(diagnostic_epochs=3)
        if [row['epoch'] for row in tracker.boundaries] != [1, 2, 3] or not tracker.window_closed:
            raise RuntimeError('missing complete all-rank epoch2 profile or three diagnostic epochs')
        if tracker.loader is None or tracker.steps != tracker.loader['steps_per_epoch'] or tracker.steps != 6:
            raise RuntimeError('profile did not cover all six actual epoch2 optimizer steps')
        if tracker.scheduler_steps != tracker.steps or tracker.total_steps != 18:
            raise RuntimeError('actual diagnostic optimizer/scheduler steps are incomplete')
        if tracker.diagnostic_end is None or tracker.diagnostic_end != {
                'completed_epochs': 3, 'epoch_budget': 3000, 'checkpoint_written': False, 'sid_exported': False}:
            raise RuntimeError('missing bounded diagnostic completion without formal artifacts')
        if any(p.exists() for p in (training.CHECKPOINT, training.OUTPUT_SID, training.OUTPUT_JSON)):
            raise RuntimeError('diagnostic incorrectly emitted formal checkpoint/SID')
    except BaseException:
        error = traceback.format_exc()
    finally:
        if tracker.enabled:
            tracker.stop_window()
        after = source_binding(training)
        report['source_hashes_after'] = after
        report['source_unchanged'] = before == after
        if before != after:
            error = (error or '') + '\nSource/input binding changed during steady profile'
        report.update(tracker.results())
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        report['hardware'] = {'hostname': os.uname().nodename,
                              'local_rank': int(os.environ['LOCAL_RANK']),
                              'cuda_device': torch.cuda.current_device(), 'gpu': properties.name,
                              'total_memory_bytes': properties.total_memory,
                              'compute_capability': [properties.major, properties.minor],
                              'visible_device_count': torch.cuda.device_count(),
                              'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                              'cudnn': torch.backends.cudnn.version(), 'nccl': torch.cuda.nccl.version()}
        report['error'] = error
        report['status'] = 'PROFILED' if error is None else 'FAIL'
        (output / ('rank_%d.json' % rank)).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('[issue256 steady profile]', rank, report['status'], run_id, flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
    if error is not None:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
