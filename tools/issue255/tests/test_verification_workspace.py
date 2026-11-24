"""A result tree named stage2_RQ-VAE must not redirect verification."""
from pathlib import Path
import runpy
import shutil

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())


def test_verification_runs_against_source_not_result_shadow(tmp_path):
    repo = tmp_path / 'repository'
    candidate = repo / 'stage2_RQ-VAE/TIGER_prefix_curvature_gate_rq'
    candidate.mkdir(parents=True)
    (repo / 'results/stage2_RQ-VAE').mkdir(parents=True)
    marker = repo / 'tools/research/stage2/contract.py'
    marker.parent.mkdir(parents=True)
    marker.write_text('')
    script = repo / 'results/research/issue255/run_tests.py'
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / 'results/research/issue255/run_tests.py', script)
    namespace = runpy.run_path(str(script))
    assert namespace['TREE'] == candidate
    assert namespace['TREE'].is_dir()
    assert namespace['OUTPUT'] == script.parent
