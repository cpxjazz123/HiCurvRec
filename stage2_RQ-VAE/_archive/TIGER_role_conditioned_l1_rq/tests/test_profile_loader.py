"""The actual profiler must preserve loader behavior and executed-source identity."""
from pathlib import Path
import subprocess
import sys

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))


def test_actual_measured_loader_preserves_rng_and_first_calibration_batch(tmp_path):
    # SteadyTracker.install modifies torch methods: a child process keeps this
    # real integration test isolated from every other test in the full suite.
    code = r'''
import torch
from pathlib import Path
import sys
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
import train_rqvae as training
from steady_profile_entry import SteadyTracker
tracker = SteadyTracker(Path(sys.argv[2]), 0., training)
tracker.install()
population = torch.arange(8192)
dataset = training._BatchedTensorDataset(population.float()[:, None], population)
generator = torch.Generator().manual_seed(42)
loader = training.DataLoader(dataset, batch_size=1024, shuffle=True,
    generator=generator, collate_fn=training._batch_passthrough)
state = torch.random.get_rng_state()
gstate = generator.get_state()
with torch.random.fork_rng(devices=[]):
    expected_batch, expected_ids = next(iter(loader))
generator.set_state(gstate)
batch, ids = training._calibration_batch(loader, 'cpu')
assert torch.equal(ids, expected_ids), 'measured-loader calibration population changed'
assert torch.equal(batch, expected_batch), 'measured-loader calibration features changed'
assert torch.equal(torch.random.get_rng_state(), state), 'measured-loader global RNG changed'
assert torch.equal(generator.get_state(), gstate), 'measured-loader explicit RNG changed'
assert tracker.completed_epochs == 0 and not tracker.enabled, 'calibration peek must not be a measured epoch'
'''
    done = subprocess.run([sys.executable, '-c', code, str(TREE), str(tmp_path)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_changed_actual_tracker_source_invalidates_profile_identity(tmp_path, monkeypatch):
    import benchmark_entry
    import steady_profile_entry
    tree = tmp_path / 'stage2_RQ-VAE/TIGER_role_conditioned_l1_rq'
    tree.mkdir(parents=True)
    (tree / 'train_rqvae.py').write_text('VALUE=1\n')
    (tree / 'benchmark_entry.py').write_text('VALUE=1\n')
    files = []
    for name in ('tools/issue252/profiling.py', 'tools/issue252/profile_units.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('VALUE=1\n')
        files.append(path)
    monkeypatch.setattr(benchmark_entry, 'ROOT', tmp_path)
    monkeypatch.setattr(benchmark_entry, '__file__', str(tree / 'benchmark_entry.py'))
    for path in files:
        before = steady_profile_entry.source_binding()
        path.write_text('VALUE=2\n')
        after = steady_profile_entry.source_binding()
        assert before != after, 'changed executed profiler dependency must invalidate its identity'
