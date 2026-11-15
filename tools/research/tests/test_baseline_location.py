"""The frozen baseline must be where the framework says it is.

The owner moved the sample trees under ``stage2_RQ-VAE/baseline/`` on 2026-11-15. Three separate places
name that tree -- the resolver constant, the evidence contract and the implementer prompt -- and a stale
one fails closed only at the moment a round tries to start, which is the worst time to discover it. These
tests pin all three to one directory, so a future move has to update them together or fail loudly here.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]
ROUNDS = HERE / "tools" / "research" / "rounds"
sys.path.insert(0, str(HERE / "tools"))

from research.rounds import baseline as baseline_module  # noqa: E402
from research.rounds import evidence as evidence_module  # noqa: E402


class BaselineLocationTest(unittest.TestCase):
    def setUp(self):
        self.baseline = baseline_module
        self.evidence = evidence_module

    def test_the_canonical_baseline_directory_exists(self):
        self.assertTrue(self.baseline.CANONICAL_BASELINE.is_dir(),
                        f"frozen baseline missing at {self.baseline.CANONICAL_BASELINE}")

    def test_the_resolver_returns_the_canonical_tree(self):
        resolved = self.baseline.baseline_path()
        self.assertEqual(resolved.resolve(), self.baseline.CANONICAL_BASELINE.resolve())

    def test_the_evidence_contract_points_at_the_same_tree(self):
        contract = self.evidence.DATA_CONTRACT["baseline_tree"]
        self.assertTrue((HERE / contract).is_dir(), f"contract path does not exist: {contract}")
        self.assertEqual((HERE / contract).resolve(), self.baseline.CANONICAL_BASELINE.resolve())

    def test_the_implementer_prompt_names_the_same_tree(self):
        text = (ROUNDS / "agents.py").read_text()
        self.assertIn(str(self.baseline.CANONICAL_BASELINE), text)

    def test_the_frozen_tree_still_holds_the_trainer(self):
        self.assertTrue((self.baseline.CANONICAL_BASELINE / "train_rqvae.py").is_file())


if __name__ == "__main__":
    unittest.main()
