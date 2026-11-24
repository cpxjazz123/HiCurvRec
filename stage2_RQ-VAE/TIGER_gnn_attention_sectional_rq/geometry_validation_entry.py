"""Four-rank geometry equivalence and native CUDA-launch diagnostic, not training.

Only the sanctioned source-fixed launcher starts this entry. Ambient references
are the candidate's existing independent references, evaluated in float64.
The resource gate counts observed CUDA compute events, never compiler flags.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'stage2_RQ-VAE').is_dir()
            and (p / 'tools/research/stage2/contract.py').is_file())
TREE = Path(__file__).resolve().parent
# Both imports must work with cwd=result_dir and no PYTHONPATH.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TREE))
OUTPUT = ROOT / 'results/research/issue252/geometry_validation'
WORLD_SIZE = 4
PORT = 50202
PROFILE_SHAPE = (1024, 256, 32)
WARMUPS = 4
REPEATS = 3
KERNEL_LIMIT = 100
TOLERANCES = {'float32': {'atol': 1e-5, 'rtol': 1e-4},
              'float64': {'atol': 1e-9, 'rtol': 1e-8}}
FLOOR_SNAPSHOT = ROOT / 'results/research/issue252/profiling_baseline/source_snapshot.tar.gz'
FLOOR_MEMBER = str(TREE.relative_to(ROOT) / 'assignment_floor.py')


def source_inventory():
    from tools.issue252.training_admission import source_binding

    result = source_binding(ROOT, TREE, 'validation')
    for path in (ROOT / 'tools/issue252/launch_geometry_validation.py', FLOOR_SNAPSHOT):
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def new_report(rank, run_id):
    return {'schema_version': 1, 'run_id': run_id, 'rank': rank,
            'worker_pid': os.getpid(), 'world_size': WORLD_SIZE, 'port': PORT,
            'command': [sys.executable] + sys.argv,
            'launcher_command': [sys.executable, str(ROOT / 'tools/issue252/launch_geometry_validation.py')],
            'cwd': str(Path.cwd()), 'device': None, 'profile_shape': list(PROFILE_SHAPE),
            'formal_training': False, 'optimizer_created': False,
            'tolerances': TOLERANCES, 'source_before': source_inventory(),
            'source_after': None, 'numeric_checks': [], 'selection_checks': [],
            'floor_checks': [], 'profiles': [], 'status': 'RUNNING', 'error': None,
            'interpretation': 'Geometry only; instrumented CUDA spans are not epoch speed.'}


def is_compute_kernel(name):
    lowered = name.lower()
    return 'memcpy' not in lowered and 'memset' not in lowered


def assert_resource(row):
    count = row['cuda_kernel_count']
    assert 0 < count < KERNEL_LIMIT, (
        'Actual warmed geometry forward/backward CUDA compute launches must be '
        'positive and < %d; observed %d' % (KERNEL_LIMIT, count))
    assert row['cuda_kernel_duration_us'] > 0, 'CUDA profiling produced no positive compute duration'


def runtime_directories(output, rank):
    # Set these before importing torch/Inductor/Triton. No user cache/temp override.
    cache = output / 'cache' / ('rank_%d' % rank)
    locations = {'TMPDIR': cache / 'temporary', 'TEMP': cache / 'temporary',
                 'TMP': cache / 'temporary', 'TORCHINDUCTOR_CACHE_DIR': cache / 'inductor',
                 'TRITON_CACHE_DIR': cache / 'triton', 'XDG_CACHE_HOME': cache / 'xdg',
                 'TORCH_EXTENSIONS_DIR': cache / 'extensions'}
    for key, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    return {key: str(path) for key, path in locations.items()}


def load_reference():
    spec = importlib.util.spec_from_file_location('issue252_geometry_reference',
                                                TREE / 'tests/reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_experts(torch, reference, x, y, scale=1.0, curvature_on=True):
    # The existing ambient API takes matrices only. Reshaping singleton-broadcast
    # views adapts its input layout without substituting production arithmetic.
    x = x.reshape(-1, 32).double() / scale
    y = y.reshape(-1, 32).double() / scale
    euclidean = (x[:, None] - y[None]).square().sum(-1)
    if not curvature_on:
        return euclidean.unsqueeze(-1).expand(*euclidean.shape, 5)
    return torch.stack([reference.lorentz_squared(x, y, c) if c < 0 else
                        reference.sphere_squared(x, y, c) if c > 0 else euclidean
                        for c in (-3, -1, 0, 1, 3)], dim=-1)


def checked(torch, actual, expected, dtype, label):
    tolerance = TOLERANCES[str(dtype).split('.')[-1]]
    assert bool(torch.isfinite(actual).all()), label + ': nonfinite production result'
    assert bool(torch.isfinite(expected).all()), label + ': nonfinite independent reference'
    torch.testing.assert_close(actual.double(), expected.double(), **tolerance,
                               msg=label)
    error = (actual.double() - expected.double()).abs()
    return {'label': label, 'dtype': str(dtype), 'shape': list(actual.shape),
            'max_abs_error': float(error.max()), 'atol': tolerance['atol'],
            'rtol': tolerance['rtol'], 'passed': True}


def numerical_checks(torch, geometry, reference, report, device):
    generator = torch.Generator(device=device).manual_seed(42)
    normal_x = torch.randn((6, 32), generator=generator, device=device,
                           dtype=torch.float64) * .12
    normal_y = torch.randn((5, 32), generator=generator, device=device,
                           dtype=torch.float64) * .14
    edge_x = torch.zeros((6, 32), device=device, dtype=torch.float64)
    edge_x[:, 0] = torch.tensor([0., .2, .2, -.2, 1e-5, math.pi / math.sqrt(3)],
                                device=device, dtype=torch.float64)
    edge_y = torch.zeros((5, 32), device=device, dtype=torch.float64)
    edge_y[:, 0] = torch.tensor([0., .2, -.2, 2e-5, math.pi / math.sqrt(3)],
                                device=device, dtype=torch.float64)
    for dtype in (torch.float32, torch.float64):
        for case, original_x, original_y in (('normal', normal_x, normal_y),
                                              ('edge_degenerate', edge_x, edge_y)):
            for broadcast in (False, True):
                a, b = original_x.to(dtype), original_y.to(dtype)
                x, y = (a[:, None], b[None]) if broadcast else (a, b)
                weights = torch.softmax(torch.arange(30, device=device, dtype=dtype)
                                        .reshape(6, 5) / 7, -1)
                for curvature_on in (True, False):
                    expected = reference_experts(torch, reference, x, y, 1.3, curvature_on)
                    label = '%s/%s/curvature_%s' % (
                        case, 'prebroadcast' if broadcast else 'matrix', curvature_on)
                    outputs = [('expert_squared', geometry.expert_squared(
                        x, y, scale=1.3, curvature_on=curvature_on), expected),
                        ('mixed_cost', geometry.mixed_cost(
                            x, y, weights, scale=1.3, curvature_on=curvature_on),
                         (expected * weights.double()[:, None]).sum(-1))]
                    if curvature_on:
                        outputs.append(('fixed_squared', geometry.fixed_squared(x, y, scale=1.3),
                                        expected.mean(-1)))
                    for name, actual, trusted in outputs:
                        report['numeric_checks'].append(checked(
                            torch, actual, trusted, dtype, label + '/' + name))
                if case == 'edge_degenerate':
                    # Ambient acosh/acos has undefined backward at coincidence;
                    # do not pretend those NaNs are a trustworthy gradient oracle.
                    a = a.detach().requires_grad_()
                    diagonal = geometry.expert_squared(a, a).diagonal(dim1=0, dim2=1)
                    gradient, = torch.autograd.grad(diagonal.sum(), (a,))
                    report['numeric_checks'].append(checked(
                        torch, diagonal, torch.zeros_like(diagonal), dtype,
                        'degenerate/exact_self_distance'))
                    report['numeric_checks'].append(checked(
                        torch, gradient, torch.zeros_like(gradient), dtype,
                        'degenerate/exact_self_gradient'))
        # Smooth input gradients are checked against autograd of the independent
        # ambient formulas, including both matrix and prebroadcast layouts.
        for broadcast in (False, True):
            for name in ('expert_squared', 'mixed_cost', 'fixed_squared'):
                x = normal_x.to(dtype).detach()
                y = normal_y.to(dtype).detach()
                if broadcast:
                    x, y = x[:, None], y[None]
                x, y = x.requires_grad_(), y.requires_grad_()
                w = torch.softmax(torch.arange(30, device=device, dtype=dtype)
                                  .reshape(6, 5) / 7, -1).detach().requires_grad_()
                rx = x.detach().double().requires_grad_()
                ry = y.detach().double().requires_grad_()
                rw = w.detach().double().requires_grad_()
                expected = reference_experts(torch, reference, rx, ry, 1.3)
                if name == 'mixed_cost':
                    actual = geometry.mixed_cost(x, y, w, scale=1.3)
                    expected = (expected * rw[:, None]).sum(-1)
                    leaves, ref_leaves = (x, y, w), (rx, ry, rw)
                else:
                    actual = getattr(geometry, name)(x, y, scale=1.3)
                    if name == 'fixed_squared':
                        expected = expected.mean(-1)
                    leaves, ref_leaves = (x, y), (rx, ry)
                gradients = torch.autograd.grad(actual.mean(), leaves)
                trusted = torch.autograd.grad(expected.mean(), ref_leaves)
                for index, (value, target) in enumerate(zip(gradients, trusted)):
                    report['numeric_checks'].append(checked(
                        torch, value, target, dtype, '%s/%s/gradient_%d' % (
                            name, 'prebroadcast' if broadcast else 'matrix', index)))


def paired_training_checks(torch, geometry, reference, report, device):
    """Real estimator layout:singleton,28 and32 differentiable rowwise pairs."""
    for count in (1, 28, 32):
        base = torch.linspace(-.3, .4, count * 32, device=device).reshape(count, 32)
        other = torch.cos(torch.arange(count * 32, device=device).float()).reshape(count, 32) * .1
        for name in ('fixed_squared', 'expert_squared', 'mixed_cost'):
            a, b = base.detach().requires_grad_(), other.detach().requires_grad_()
            w = torch.softmax(torch.arange(count * 5, device=device).float().reshape(count, 5) / 7, -1).requires_grad_()
            ra, rb, rw = (value.detach().double().requires_grad_() for value in (a, b, w))
            trusted = reference_experts(torch, reference, ra, rb, 1.3)
            trusted = trusted.diagonal(dim1=0, dim2=1).T[:, None, :]
            if name == 'mixed_cost':
                actual = geometry.mixed_cost(a[:, None], b[:, None], w, scale=1.3)
                trusted = (trusted * rw[:, None]).sum(-1)
                leaves, reference_leaves = (a, b, w), (ra, rb, rw)
            else:
                actual = getattr(geometry, name)(a[:, None], b[:, None], scale=1.3)
                if name == 'fixed_squared':
                    trusted = trusted.mean(-1)
                leaves, reference_leaves = (a, b), (ra, rb)
            label = 'paired_training/%d/%s' % (count, name)
            report['numeric_checks'].append(checked(torch, actual, trusted, torch.float32, label))
            gradients = torch.autograd.grad(actual.mean(), leaves)
            expected = torch.autograd.grad(trusted.mean(), reference_leaves)
            for index, (got, want) in enumerate(zip(gradients, expected)):
                report['numeric_checks'].append(checked(
                    torch, got, want, torch.float32, label + '/gradient_%d' % index))


def retained_graph_checks(torch, geometry, reference, report, device):
    """Calibration and real-loss VJPs retain a graph before the next VJP."""
    for name in ('fixed_squared', 'expert_squared', 'mixed_cost'):
        a = torch.linspace(-.3, .4, 28 * 32, device=device).reshape(28, 32).requires_grad_()
        b = (torch.cos(torch.arange(32 * 32, device=device).float()).reshape(32, 32) * .1).requires_grad_()
        w = torch.softmax(torch.arange(28 * 5, device=device).float().reshape(28, 5) / 7, -1).requires_grad_()
        ra, rb, rw = (value.detach().double().requires_grad_() for value in (a, b, w))
        trusted = reference_experts(torch, reference, ra, rb, 1.3)
        if name == 'mixed_cost':
            actual = geometry.mixed_cost(a, b, w, scale=1.3)
            trusted = (trusted * rw[:, None]).sum(-1)
            leaves, reference_leaves = (a, b, w), (ra, rb, rw)
        else:
            actual = getattr(geometry, name)(a, b, scale=1.3)
            if name == 'fixed_squared':
                trusted = trusted.mean(-1)
            leaves, reference_leaves = (a, b), (ra, rb)
        for retained in (True, False):
            objective = actual.mean() if retained else actual.square().mean()
            expected = trusted.mean() if retained else trusted.square().mean()
            gradients = torch.autograd.grad(objective, leaves, retain_graph=retained)
            reference_gradients = torch.autograd.grad(expected, reference_leaves, retain_graph=retained)
            for index, (got, want) in enumerate(zip(gradients, reference_gradients)):
                report['numeric_checks'].append(checked(
                    torch, got, want, torch.float32,
                    'retained_graph/%s/%s/gradient_%d' % (name, retained, index)))


def load_original_floor(report):
    import tarfile
    import types

    with tarfile.open(FLOOR_SNAPSHOT, 'r:gz') as archive:
        member = archive.extractfile(FLOOR_MEMBER)
        if member is None:
            raise RuntimeError('Preserved original floor source is missing from snapshot')
        source = member.read()
    report['floor_reference'] = {'archive': str(FLOOR_SNAPSHOT), 'member': FLOOR_MEMBER,
        'archive_sha256': hashlib.sha256(FLOOR_SNAPSHOT.read_bytes()).hexdigest(),
        'member_sha256': hashlib.sha256(source).hexdigest()}
    module = types.ModuleType('issue252_original_floor')
    # Execute the preserved source directly; never extract files or reconstruct
    # an allegedly equivalent version of the original floor.
    exec(compile(source, str(FLOOR_SNAPSHOT) + ':' + FLOOR_MEMBER, 'exec'), module.__dict__)
    return module


def floor_checks(torch, original, report, device):
    import assignment_floor as candidate_floor
    from tools.research.stage2.anticollapse import floor as framework_floor
    from model import layers

    fixtures = [
        ('healthy', [[0, 5, 8], [9, 0, 7], [6, 2, 0], [0, 3, 4]], 1, [0, 1, 2, 0]),
        ('unique_contested', [[10 * row + col for col in range(4)] for row in range(7)],
         1, [1, 2, 3, 0, 0, 0, 0]),
        ('unique_multi_round', [[0, 1, 5], [0, 4, 2], [0, 8, 9], [0, 10, 11]],
         1, [1, 2, 0, 0]),
        ('argmin_ties', [[0, 0], [0, 0], [0, 0], [0, 0]], 1, [1, 0, 0, 0]),
        ('equal_deficit_priorities', [[0, 0, 0, 0] for _ in range(7)], 1, None),
        ('floor_disabled_ties', [[0, 0], [1, 1]], 0, [0, 0])]
    routes = [('framework', framework_floor.constrained_assignment),
              ('candidate', candidate_floor.constrained_assignment),
              ('actual_layer_alias', layers.constrained_assignment)]
    for name, values, required, hand_known in fixtures:
        cost = torch.tensor(values, device=device, dtype=torch.float32)
        expected_ids, expected_report = original.constrained_assignment(cost, required)
        if hand_known is not None:
            assert torch.equal(expected_ids, torch.tensor(hand_known, device=device)), (
                'preserved original floor violates hand-known fixture ' + name)
        for route, function in routes:
            actual_ids, actual_report = function(cost, required)
            assert actual_ids.is_cuda and actual_ids.dtype == expected_ids.dtype
            assert torch.equal(actual_ids, expected_ids), route + '/floor IDs differ: ' + name
            assert actual_report == expected_report, route + '/floor report differs: ' + name
            report['floor_checks'].append({'fixture': name, 'route': route,
                'cost': values, 'dtype': 'torch.float32', 'device': str(device),
                'floor': required, 'ids': actual_ids.tolist(), 'original_ids': expected_ids.tolist(),
                'hand_known_ids': hand_known, 'assignment_report': actual_report, 'passed': True})


def selection_checks(torch, geometry, selection, reference, original_floor, config, report, device):
    from model.layers import VQLayer, ASSIGNMENT_FLOOR

    for dtype in (torch.float32, torch.float64):
        codes = torch.zeros((256, 32), device=device, dtype=dtype)
        codes[:, 0] = -2 - torch.arange(256, device=device, dtype=dtype) / 256
        codes[0, :2] = torch.tensor([1., .6], device=device, dtype=dtype)
        codes[1, :2] = torch.tensor([1.8, 0.], device=device, dtype=dtype)
        # Keep a whole batch and coverage of every real codeword. The probe row
        # is free to change its winner without a floor-induced forced reassignment.
        x = codes[torch.arange(1024, device=device) % 256].clone()
        x[0].zero_()
        x[0, 0] = 1
        layer = VQLayer(256, 32, beta=config['BETA'], sk_epsilon=-1,
                        sk_iters=config['SK_ITERS']).to(device=device, dtype=dtype)
        layer.eval()
        ids = []
        for expert in (2, 0):
            w = torch.zeros((1024, 5), device=device, dtype=dtype)
            w[:, expert] = 1
            actual = geometry.mixed_cost(x, codes, w)
            expected = (reference_experts(torch, reference, x, codes) * w.double()[:, None]).sum(-1)
            report['numeric_checks'].append(checked(
                torch, actual, expected, dtype, 'controlled_weight/expert_%d' % expert))
            actual_ids = layer._indices(actual, False)
            expected_ids, _ = original_floor.constrained_assignment(expected, ASSIGNMENT_FLOOR)
            result = selection.select(actual, codes, balanced=False, epsilon=-1,
                iterations=config['SK_ITERS'], training=False, hard_ids=actual_ids)
            assert torch.equal(result['ids'], expected_ids), 'controlled actual-layer IDs differ from independent cost/original floor'
            assert torch.equal(result['hard'], codes[expected_ids]), 'selected codeword differs'
            ordered = expected[0].sort(-1).values
            margin = float(ordered[1] - ordered[0])
            assert margin > 2 * float((actual[0].double() - expected[0]).abs().max()), 'intervention cost margin is unresolved'
            ids.append(int(result['ids'][0].item()))
            report['selection_checks'].append({'kind': 'weight_intervention', 'dtype': str(dtype),
                'expert': expert, 'probe_id': ids[-1], 'independent_cost_margin': margin,
                'batch_size': 1024, 'codebook_shape': [256, 32], 'actual_layer_indices': True,
                'assignment_floor': ASSIGNMENT_FLOOR, 'passed': True})
        assert ids == [0, 1], 'controlled expert intervention did not change hard IDs'
        generator = torch.Generator(device=device).manual_seed(42)
        for level in range(config['CODEBOOK_NUM']):
            codes = torch.randn((256, 32), device=device, dtype=dtype, generator=generator) * .14
            x = codes[torch.arange(1024, device=device) % 256] + torch.randn(
                (1024, 32), device=device, dtype=dtype, generator=generator) * .01
            w = torch.softmax(torch.randn((1024, 5), device=device, dtype=dtype,
                                          generator=generator), -1)
            actual = geometry.mixed_cost(x, codes, w, scale=1.3)
            expected = (reference_experts(torch, reference, x, codes, 1.3)
                        * w.double()[:, None]).sum(-1)
            report['numeric_checks'].append(checked(
                torch, actual, expected, dtype, 'frozen_selection/level_%d' % level))
            balanced = level == config['CODEBOOK_NUM'] - 1
            epsilon = config['SK_EPSILON'] if balanced else -1
            layer = VQLayer(256, 32, beta=config['BETA'], sk_epsilon=epsilon,
                            sk_iters=config['SK_ITERS']).to(device=device, dtype=dtype)
            for training in (False, True):
                layer.train(training)
                actual_ids = layer._indices(actual, False)
                if balanced and training:
                    # Preserve the dtype of centering in the actual frozen path.
                    score = reference.frozen_sinkhorn(expected.to(dtype), epsilon, config['SK_ITERS'])
                    expected_ids, _ = original_floor.constrained_assignment(-score, ASSIGNMENT_FLOOR)
                    ordered = score.sort(-1, descending=True).values
                    margin = float((ordered[:, 0] - ordered[:, 1]).min())
                    margin_kind = 'independent_frozen_probability_before_floor'
                else:
                    expected_ids, _ = original_floor.constrained_assignment(expected, ASSIGNMENT_FLOOR)
                    ordered = expected.sort(-1).values
                    margin = float((ordered[:, 1] - ordered[:, 0]).min())
                    margin_kind = 'independent_cost_before_floor'
                result = selection.select(actual, codes, balanced=balanced, epsilon=epsilon,
                    iterations=config['SK_ITERS'], training=training, hard_ids=actual_ids)
                assert margin > 0, 'independent frozen assignment has a tied winner'
                assert torch.equal(result['ids'], expected_ids), 'actual frozen layer selection differs from independent cost/original floor'
                assert torch.equal(result['hard'], codes[expected_ids]), 'actual frozen selected codewords differ'
                report['selection_checks'].append({'kind': 'actual_frozen_selection',
                    'dtype': str(dtype), 'level': level, 'training': training,
                    'balanced': balanced, 'rows': len(x), 'codebook_shape': [256, 32],
                    'winner_margin': margin, 'margin_kind': margin_kind,
                    'actual_layer_indices': True, 'assignment_floor': ASSIGNMENT_FLOOR,
                    'max_cost_abs_error': float((actual.double() - expected).abs().max()),
                    'ids': result['ids'].tolist(), 'passed': True})


def profile_geometry(torch, geometry, report, device):
    generator = torch.Generator(device=device).manual_seed(42)
    x = (torch.randn((1024, 32), device=device, generator=generator) * .12).requires_grad_()
    y = (torch.randn((256, 32), device=device, generator=generator) * .14).requires_grad_()
    weights = torch.softmax(torch.randn((1024, 5), device=device, generator=generator), -1)
    weights = weights.detach().requires_grad_()
    for name in ('expert_squared', 'mixed_cost', 'fixed_squared'):
        def step():
            if name == 'mixed_cost':
                output = geometry.mixed_cost(x, y, weights, scale=1.3)
                leaves = (x, y, weights)
            else:
                output = getattr(geometry, name)(x, y, scale=1.3)
                leaves = (x, y)
            # No optimizer, no parameter update, no model or training loop.
            return torch.autograd.grad(output.mean(), leaves)
        for _ in range(WARMUPS):
            step()
        torch.cuda.synchronize(device)
        for repeat in range(REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            began = time.perf_counter()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                start.record()
                gradients = step()
                end.record()
                torch.cuda.synchronize(device)
            rows = {}
            cpu = {}
            kernel_count = 0
            kernel_us = 0.0
            for event in profiler.events():
                event_type = str(event.device_type).rsplit('.', 1)[-1]
                if event_type == 'CUDA' and is_compute_kernel(event.name):
                    duration = getattr(event, 'device_time_total', None)
                    if duration is None:
                        duration = event.cuda_time_total
                    row = rows.setdefault(event.name, {'name': event.name, 'calls': 0, 'duration_us': 0.0})
                    row['calls'] += 1
                    row['duration_us'] += float(duration)
                    kernel_count += 1
                    kernel_us += float(duration)
                elif event_type == 'CPU' and event.name.startswith('aten::'):
                    cpu[event.name] = cpu.get(event.name, 0) + 1
            report['profiles'].append({'function': name, 'repeat': repeat, 'dtype': 'torch.float32',
                'shape': list(PROFILE_SHAPE), 'warmups': WARMUPS,
                'cuda_kernel_count': kernel_count, 'cuda_kernel_duration_us': kernel_us,
                'instrumented_cuda_span_ms': start.elapsed_time(end),
                'instrumented_wall_seconds': time.perf_counter() - began,
                'kernel_limit_exclusive': KERNEL_LIMIT,
                'event_definition': 'DeviceType.CUDA compute events; exclude memcpy/memset; no CPU/runtime events',
                'kernels': sorted(rows.values(), key=lambda row: (-row['calls'], row['name'])),
                'native_cpu_aten_calls': cpu,
                'gradients_finite': all(bool(torch.isfinite(value).all()) for value in gradients)})
            assert report['profiles'][-1]['gradients_finite'], 'profiled gradients are nonfinite'
    # Keep all nine raw measurements even when the existing eager core is RED.
    for row in report['profiles']:
        assert_resource(row)


def main():
    rank = int(os.environ.get('RANK', '-1'))
    output = Path.cwd().resolve()
    if output.parent != OUTPUT or rank not in range(WORLD_SIZE):
        raise RuntimeError('Use tools/issue252/launch_geometry_validation.py via the sanctioned launcher')
    report = new_report(rank, output.name)
    report['runtime_directories'] = runtime_directories(output, rank)
    report_path = output / ('rank_%d.json' % rank)
    torch = None
    distributed = None
    try:
        import torch
        import torch.distributed as distributed
        assert int(os.environ['WORLD_SIZE']) == WORLD_SIZE, 'four ranks are required'
        assert int(os.environ['MASTER_PORT']) == PORT, 'the frozen Stage2 port is required'
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(42)
        distributed.init_process_group('nccl')
        properties = torch.cuda.get_device_properties(device)
        report['device'] = {'name': properties.name, 'index': local_rank,
                            'total_memory_bytes': properties.total_memory,
                            'capability': list(torch.cuda.get_device_capability(device)),
                            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        from model import geometry, selection
        reference = load_reference()
        config = json.loads((TREE / 'config/train_config.json').read_text())
        assert (config['CODEBOOK_DIM'], config['CODEBOOK_NUM'], config['CODEBOOK_SIZE'],
                config['BATCH_SIZE_PER_RANK'], config['SEED']) == (32, 3, [256, 256, 256], 1024, 42)
        report['frozen_configuration'] = config
        original_floor = load_original_floor(report)
        errors = []
        phases = [
            ('floor', lambda: floor_checks(torch, original_floor, report, device)),
            ('numeric', lambda: numerical_checks(torch, geometry, reference, report, device)),
            ('paired_training', lambda: paired_training_checks(torch, geometry, reference, report, device)),
            ('retained_graph', lambda: retained_graph_checks(torch, geometry, reference, report, device)),
            ('selection', lambda: selection_checks(torch, geometry, selection, reference,
                                                  original_floor, config, report, device)),
            ('resource', lambda: profile_geometry(torch, geometry, report, device))]
        for phase, check in phases:
            try:
                check()
            except Exception:
                errors.append(phase + ':\n' + traceback.format_exc())
        report['error'] = '\n'.join(errors) if errors else None
        report['status'] = 'FAIL' if errors else 'PASS'
    except Exception:
        report['error'] = traceback.format_exc()
        report['status'] = 'FAIL'
    finally:
        report['runtime_directories'] = {
            name: os.environ.get(name) for name in
            ('TEMP', 'TMP', 'TMPDIR', 'TORCHINDUCTOR_CACHE_DIR',
             'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR')}
        report['source_after'] = source_inventory()
        if report['source_after'] != report['source_before']:
            report['status'] = 'FAIL'
            report['error'] = (report['error'] or '') + '\nSource inventory changed during diagnostic'
        # Write local evidence before any collective can fail or another worker exits.
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    if distributed is not None and distributed.is_initialized():
        try:
            failed = torch.tensor([int(report['status'] != 'PASS')], device=device)
            distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
            report['all_ranks_passed'] = not bool(failed.item())
            if not report['all_ranks_passed']:
                report['status'] = 'FAIL'
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        finally:
            distributed.destroy_process_group()
    print(json.dumps({'report': str(report_path), 'rank': rank, 'status': report['status'],
                      'run_id': output.name}), flush=True)
    if report['status'] != 'PASS':
        raise RuntimeError('Geometry validation failed; original numeric/resource error preserved in ' + str(report_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
