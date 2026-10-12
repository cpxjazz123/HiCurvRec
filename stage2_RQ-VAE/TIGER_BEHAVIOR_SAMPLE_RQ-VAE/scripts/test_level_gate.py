from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_rqvae as train


class LevelGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1701)
        self.residuals = torch.randn(3, 8, 6, dtype=torch.float32)
        self.source_ids = torch.tensor([0, 1, 2, 3])
        self.target_ids = torch.tensor([4, 5, 6, 7])

    def original_loss(self, residuals, source_ids, target_ids):
        batch_size = int(source_ids.shape[0])
        losses = []
        for level in range(residuals.shape[0]):
            source = residuals[level, :batch_size]
            candidates = residuals[level, batch_size:]
            logits = -torch.cdist(source, candidates, p=2) / train.BEHAVIOR_TEMPERATURE
            loss = train._contrastive_ce(logits, source_ids, target_ids, None)
            if loss is not None:
                losses.append(loss)
        if not losses:
            return residuals.sum() * 0.0
        return torch.stack(losses).mean()

    def test_all_levels_match_original_bit_for_bit(self):
        previous = train.ACTIVE_LEVELS
        try:
            train.ACTIVE_LEVELS = (0, 1, 2)
            actual = train.behavior_contrastive_loss(
                self.residuals, self.source_ids, self.target_ids
            )
            expected = self.original_loss(self.residuals, self.source_ids, self.target_ids)
            self.assertTrue(torch.equal(actual, expected))
        finally:
            train.ACTIVE_LEVELS = previous

    def _assert_only_level_changes_loss(self, active_level):
        previous = train.ACTIVE_LEVELS
        try:
            train.ACTIVE_LEVELS = (active_level,)
            original = train.behavior_contrastive_loss(
                self.residuals, self.source_ids, self.target_ids
            )
            for level in range(3):
                changed = self.residuals.clone()
                changed[level] += torch.randn_like(changed[level]) * 3.0
                changed_loss = train.behavior_contrastive_loss(
                    changed, self.source_ids, self.target_ids
                )
                if level == active_level:
                    self.assertFalse(torch.equal(original, changed_loss))
                else:
                    self.assertTrue(torch.equal(original, changed_loss))
        finally:
            train.ACTIVE_LEVELS = previous

    def test_level_zero_only(self):
        self._assert_only_level_changes_loss(0)

    def test_level_one_only(self):
        self._assert_only_level_changes_loss(1)

    def test_level_two_only(self):
        self._assert_only_level_changes_loss(2)

    def test_level_two_gradients_reach_encoder_and_earlier_quantizers(self):
        encoder = nn.Linear(6, 6, bias=False)
        quantizers = nn.ModuleList(nn.Linear(6, 6, bias=False) for _ in range(3))
        embeddings = torch.randn(8, 6)
        residual = encoder(embeddings)
        traces = []
        for quantizer in quantizers:
            traces.append(residual)
            residual = residual - quantizer(residual)
        residuals = torch.stack(traces)
        previous = train.ACTIVE_LEVELS
        try:
            train.ACTIVE_LEVELS = (2,)
            loss = train.behavior_contrastive_loss(
                residuals, self.source_ids, self.target_ids
            )
            loss.backward()
        finally:
            train.ACTIVE_LEVELS = previous
        parameters = [encoder.weight, quantizers[0].weight, quantizers[1].weight]
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        for parameter in [encoder.weight, *(q.weight for q in quantizers)]:
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_gate_keeps_same_contributing_rows(self):
        previous = train.ACTIVE_LEVELS
        rows_by_setting = []
        rows_reference = None
        def recording_ce(logits, source_ids, target_ids, successor_indicator=None):
            duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
            duplicate_targets.fill_diagonal_(False)
            valid = source_ids.ne(target_ids)
            rows = tuple(torch.nonzero(valid, as_tuple=False).flatten().tolist())
            labels = torch.arange(len(source_ids), device=source_ids.device)
            rows_by_setting.append((rows, tuple(labels[valid].tolist())))
            if not bool(valid.any()):
                return None
            labels = torch.arange(len(source_ids), device=source_ids.device)
            return F.cross_entropy(logits[valid], labels[valid])

        try:
            with patch.object(train, "_contrastive_ce", side_effect=recording_ce):
                for active in ((0, 1, 2), (0,), (1,), (2,)):
                    rows_by_setting.clear()
                    train.ACTIVE_LEVELS = active
                    train.behavior_contrastive_loss(
                        self.residuals, self.source_ids, self.target_ids
                    )
                    self.assertTrue(rows_by_setting)
                    self.assertEqual(len(set(rows_by_setting)), 1)
                    current_rows = rows_by_setting[0]
                    if rows_reference is None:
                        rows_reference = current_rows
                    else:
                        self.assertEqual(current_rows, rows_reference)
        finally:
            train.ACTIVE_LEVELS = previous

if __name__ == "__main__":
    unittest.main(verbosity=2)
