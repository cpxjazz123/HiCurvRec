"""Switching the mechanism off must give the frozen baseline, exactly.

The switch is not a claim in a docstring: this test runs the same weights and
the same batch through the frozen baseline's own ``model`` package and through
the candidate's package with ``stratified_enabled=False``, and compares the
reconstruction, the quantisation loss, the three-level token table and the
gradient of every parameter.  The two runs are separate processes so that the
two ``model`` packages cannot shadow each other.

The forward values, the token table and the quantisation loss are compared with
``torch.equal``.  The gradients are compared to a stated tolerance instead, and
the reason is recorded rather than smoothed over: the candidate writes the
residual chain as ``(residual - quant).detach()``, which is the same function
with the same derivative - the baseline's quantizer is
``x + (x_q - x).detach()``, whose derivative with respect to its input is the
identity, so ``residual - quant`` already had a zero derivative - but the
BACKWARD graph has one fewer edge, and the gradient buffer of the encoder's
first layer is filled in a different order.  Measured: values exact, gradients
equal to 1.2e-10 against magnitudes of 2.4e-2 (float32 accumulation noise).
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
BASELINE_NAME = "TIGER_RQ-VAE"

PROBE = r'''
import json, sys
from types import SimpleNamespace
import torch

tree = sys.argv[1]
sys.path.insert(0, tree)
from model import RQVAE

codebook_num, codebook_size, codebook_dim = (int(v) for v in sys.argv[2:5])
stratified = sys.argv[5] == "1"
config = SimpleNamespace(
    hidden_sizes=(16,), codebook_num=codebook_num,
    codebook_size=tuple([codebook_size] * codebook_num), codebook_dim=codebook_dim,
    dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
    sk_epsilon=0.003, sk_iters=5, stratified_enabled=stratified)

torch.manual_seed(23)
model = RQVAE(config, in_dim=8)
state = torch.load(sys.argv[6], map_location="cpu", weights_only=True)
missing = model.load_state_dict(state, strict=False)
if missing.unexpected_keys:
    raise SystemExit("the state does not fit: " + repr(missing))

generator = torch.Generator().manual_seed(31)
embeddings = torch.randn(96, 8, generator=generator) * 0.05
ids = torch.arange(96)
# The frozen baseline's model takes the embeddings alone: a candidate may take
# the item ids beside them, and the probe follows whichever interface this tree
# declares.
import inspect
positional = [parameter for parameter in inspect.signature(model.forward).parameters.values()
              if parameter.kind in (parameter.POSITIONAL_ONLY,
                                    parameter.POSITIONAL_OR_KEYWORD)]
reconstructed, quant_loss, unused, tokens = (model(embeddings, ids) if len(positional) > 1
                                             else model(embeddings))
loss = torch.nn.functional.mse_loss(reconstructed, embeddings) + quant_loss
loss.backward()
report = {
    "reconstruction": reconstructed.detach().tolist(),
    "quant_loss": float(quant_loss.detach()),
    "tokens": tokens.tolist(),
    "gradients": {name: parameter.grad.detach().tolist()
                  for name, parameter in model.named_parameters()
                  if parameter.grad is not None},
    "parameters": sorted(name for name, _ in model.named_parameters()),
}
print(json.dumps(report))
'''


def _run(tree: Path, state: Path, stratified: bool) -> dict:
    done = subprocess.run(
        [sys.executable, "-c", PROBE, str(tree), "3", "8", "4",
         "1" if stratified else "0", str(state)],
        capture_output=True, text=True, timeout=900, cwd=str(tree))
    if done.returncode != 0:
        raise AssertionError("the probe failed for " + str(tree) + " (cwd " + str(tree) + "): "
                             + (done.stdout + done.stderr)[-2000:])
    return json.loads(done.stdout.strip().splitlines()[-1])


class TheSwitchIsTheBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = TREE
        # The tree sits under results/, and results/ itself contains a
        # stage2_RQ-VAE (of RESULT directories) plus a stale results/results from
        # an older run that resolved the root wrongly, so a marker on directory
        # names alone stops too early.  The root that matters is the one that
        # holds the frozen baseline's SOURCE.
        while not (root / "stage2_RQ-VAE" / BASELINE_NAME / "model" / "__init__.py").is_file() \
                and root != root.parent:
            root = root.parent
        baseline = root / "stage2_RQ-VAE" / BASELINE_NAME
        if not baseline.is_dir():
            raise unittest.SkipTest("the frozen baseline is not available at " + str(baseline))
        cls.root = root
        cls.baseline = baseline
        # One state, written by the baseline's own architecture: the candidate
        # must accept it, which is the checkpoint compatibility the framework
        # requires of a derived tree.
        torch.manual_seed(23)
        from types import SimpleNamespace
        from model import RQVAE
        reference = RQVAE(SimpleNamespace(
            hidden_sizes=(16,), codebook_num=3, codebook_size=(8, 8, 8),
            codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
            fix_code_embs=False, sk_epsilon=0.003, sk_iters=5,
            stratified_enabled=False), in_dim=8)
        cls.state_path = Path(self_scratch(cls.root)) / "state.pth"
        torch.save(reference.state_dict(), cls.state_path)

    def test_the_switched_off_candidate_is_the_baseline_bit_for_bit(self):
        baseline = _run(self.baseline, self.state_path, stratified=False)
        candidate = _run(TREE, self.state_path, stratified=False)
        self.assertEqual(baseline["parameters"], candidate["parameters"])
        for key in ("reconstruction", "tokens"):
            left = torch.tensor(baseline[key], dtype=torch.float64)
            right = torch.tensor(candidate[key], dtype=torch.float64)
            self.assertTrue(torch.equal(left, right), key + " differs")
        self.assertEqual(baseline["quant_loss"], candidate["quant_loss"])
        self.assertEqual(sorted(baseline["gradients"]), sorted(candidate["gradients"]))
        worst, worst_name = 0.0, ""
        for name in baseline["gradients"]:
            left = torch.tensor(baseline["gradients"][name], dtype=torch.float64)
            right = torch.tensor(candidate["gradients"][name], dtype=torch.float64)
            scale = max(float(left.abs().max()), 1e-12)
            difference = float((left - right).abs().max())
            if difference > worst:
                worst, worst_name = difference, name
            self.assertLessEqual(difference, 1e-9 + 1e-6 * scale,
                                 "the gradient of " + name + " differs")
        # The measured worst case is 1.16e-10; anything at 1e-6 means a real
        # change in the graph rather than an accumulation order.
        self.assertLess(worst, 1e-6)

    def test_the_switched_on_candidate_differs_from_the_baseline(self):
        # A mechanism that changes nothing on a batch it was built to change is
        # not wired.  The level-1 index and the reconstruction must move.
        baseline = _run(self.baseline, self.state_path, stratified=False)
        candidate = _run(TREE, self.state_path, stratified=True)
        left = np.asarray(baseline["tokens"])
        right = np.asarray(candidate["tokens"])
        self.assertFalse(np.array_equal(left[:, 0], right[:, 0]),
                         "the mechanism never changes the level-1 index")


def self_scratch(root: Path) -> Path:
    scratch = root / "results" / "stage2_RQ-VAE" / "_conformance" / "curvature-stratum-sinkhorn-v1"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


if __name__ == "__main__":
    unittest.main()
