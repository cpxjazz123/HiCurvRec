"""The probes and the admission must judge a tree the way it will be run.

Three engineering faults of the first candidate are reproduced here, because a
fix without a test that fails on the old behaviour is a fix that can be undone:

* the entry point derives the repository root from the stage2_RQ-VAE marker, so
  a tree under results/ resolves a DIFFERENT root than the promoted tree; the
  admission therefore runs the promoted copy;
* the gradient and the four-rank probes demanded a model interface the frozen
  baseline does not have (a forward taking the item ids and returning five
  values), so the baseline itself could not be probed;
* the behavioural differential has to call the surface's own callable and to
  compare the implementations where the surface is defined - inside the ball.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.research.stage2 import checks, contract  # noqa: E402


def test_the_admission_runs_the_tree_where_it_will_train(tmp_path):
    """The promoted copy resolves the repository root, the round tree does not."""
    tree = tmp_path / "candidate"
    (tree / "model").mkdir(parents=True)
    (tree / "train_rqvae.py").write_text(
        "from pathlib import Path\n"
        "def _repository_root() -> Path:\n"
        "    here = Path(__file__).resolve()\n"
        "    for parent in here.parents:\n"
        "        if (parent / 'stage2_RQ-VAE').is_dir():\n"
        "            return parent\n"
        "    raise RuntimeError('no root')\n"
        "REPO_ROOT = _repository_root()\n", encoding="utf-8")
    promoted = checks._admission_tree(tree)
    try:
        assert promoted.parent.name == "stage2_RQ-VAE"
        assert promoted.is_dir()
        namespace = {"__file__": str(promoted / "train_rqvae.py"), "__name__": "tree"}
        exec(compile((promoted / "train_rqvae.py").read_text(encoding="utf-8"),
                     str(promoted / "train_rqvae.py"), "exec"), namespace)
        assert Path(namespace["REPO_ROOT"]).resolve() == contract.resolve_root()
    finally:
        checks._discard_admission_tree(tree)
    assert not promoted.exists()


def test_the_probes_adapt_to_the_model_interface():
    """One interface is the baseline's, the other a candidate's: both work."""
    from tools.research.stage2 import gradient_probe

    class Baseline:
        def forward(self, embeddings):
            return embeddings * 2.0, torch_zero(), 0, embeddings

    class Candidate:
        def forward(self, embeddings, ids):
            return embeddings * 2.0, torch_zero(), 0, embeddings, embeddings

    def torch_zero():
        import torch
        return torch.zeros(1)

    import torch
    embeddings = torch.randn(4, 3, requires_grad=True)
    ids = torch.arange(4)
    values, reference = gradient_probe._call_model(Baseline(), embeddings, ids)
    assert values[0].shape == embeddings.shape
    assert reference.shape == embeddings.shape
    values, reference = gradient_probe._call_model(Candidate(), embeddings, ids)
    assert values[0].shape == embeddings.shape
    assert reference.shape == embeddings.shape


def test_a_ddp_wrapper_does_not_look_like_a_two_argument_model():
    """A DDP wrapper's forward takes *inputs, which is not a second argument."""
    import torch
    from tools.research.stage2 import ddp_probe

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, *inputs, **kwargs):
            return self.linear(inputs[0])

    wrapper = Wrapper()
    embeddings = torch.randn(4, 3)
    ids = torch.arange(4)
    values, reference = ddp_probe._call_model(wrapper, embeddings, ids)
    assert values[0].shape == embeddings.shape


def test_the_differential_compares_the_surface_s_own_callable():
    """The surface names the callable; the differential has to use that name."""
    from tools.research.rounds import module_differential

    code = (
        "import math\n"
        "import torch\n"
        "def distance(x, y, curvature):\n"
        "    return torch.sqrt((x - y) ** 2).sum(-1) * curvature\n"
    )
    implementations = [{"name": "coding_1", "code": code},
                        {"name": "coding_2", "code": code}]
    spec = {"continuous_surface": {"callable": "distance(x, y, curvature)"}}
    verdict = module_differential.compare_modules(implementations, spec)
    assert verdict["status"] == "AGREE", verdict.get("reason")
    assert verdict["entries"][0]["entry"] == "distance"


def test_the_differential_samples_inside_the_ball():
    """The surface is defined inside the ball; outside it the artanh argument
    saturates and correct implementations differ on the spelling of the guard."""
    from tools.research.rounds import module_differential

    arms = module_differential._arms(
        {"continuous_surface": {"callable": "distance(x, y, curvature)"}}, seed=17)
    curvature = 0.25
    for name, values in arms.items():
        if name.startswith("__"):
            continue
        left, right = values[0], values[1]
        bound = 1.0 / (curvature ** 0.5)
        assert float(left.norm(dim=-1).max()) < bound
        assert float(right.norm(dim=-1).max()) < bound
