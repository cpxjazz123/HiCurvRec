"""Issue #125: the forensic layer must not invent a beam trace or a mapping.

These tests are the cheap part of the guarantee; the expensive part is the
reproduction of each arm's reported Recall@10, which the arm runner asserts.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.research.diagnostics import sid_generation_forensics as forensics


class CanonicalErTests(unittest.TestCase):
    def setUp(self):
        self.sid = np.array([[1, 1, 1, 0]] * 4 + [[2, 2, 2, 0]] * 2, dtype=np.int64)

    def test_a_constant_first_level_makes_the_ratio_one(self):
        payload = {"anchors": np.array([0, 1, 2]),
                   "positives": np.array([3, 4, 5]),
                   "negatives": np.array([[0, 1]] * 3)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            np.savez(path, **payload)
            original = forensics.PAIRS
            forensics.PAIRS = path
            try:
                result = forensics.canonical_er(np.zeros((6, 4), dtype=np.int64))
            finally:
                forensics.PAIRS = original
        self.assertAlmostEqual(result["L1"]["positive_hit_rate"], 1.0)
        self.assertAlmostEqual(result["L1"]["negative_hit_rate"], 1.0)
        self.assertAlmostEqual(result["L1"]["expected_rank_ratio"], 1.0)

    def test_a_separating_first_level_gives_a_ratio_above_one(self):
        sid = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0],
                        [2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]], dtype=np.int64)
        payload = {"anchors": np.array([0, 2]), "positives": np.array([1, 3]),
                   "negatives": np.array([[0, 2], [0, 2]])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            np.savez(path, **payload)
            original = forensics.PAIRS
            forensics.PAIRS = path
            try:
                result = forensics.canonical_er(sid)
            finally:
                forensics.PAIRS = original
        self.assertAlmostEqual(result["L1"]["positive_hit_rate"], 1.0)
        self.assertAlmostEqual(result["L1"]["negative_hit_rate"], 0.5)
        self.assertAlmostEqual(result["L1"]["expected_rank_ratio"], 2.0)

    def test_an_undefined_ratio_is_reported_not_faked(self):
        sid = np.array([[1, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0]] +
                       [[1, 0, 0, 0]] * 3, dtype=np.int64)
        payload = {"anchors": np.array([0]), "positives": np.array([2]),
                   "negatives": np.array([[1]])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            np.savez(path, **payload)
            original = forensics.PAIRS
            forensics.PAIRS = path
            try:
                result = forensics.canonical_er(sid)
            finally:
                forensics.PAIRS = original
        self.assertEqual(result["L1"]["negative_hit_rate"], 0.0)
        self.assertIsNone(result["L1"]["expected_rank_ratio"])
        self.assertEqual(result["L1"]["ratio_undefined_draws"], forensics.N_BOOTSTRAP)


class StructureTests(unittest.TestCase):
    def test_a_collapsed_level_is_visible_as_one_bucket(self):
        sid = np.array([[7, 3, 1, 0], [7, 3, 2, 0], [7, 4, 1, 0]], dtype=np.int64)
        structure = forensics.sid_structure(sid)
        self.assertEqual(structure["levels"]["L1"]["distinct"], 1)
        self.assertAlmostEqual(structure["levels"]["L1"]["largest_bucket_share"], 1.0)
        self.assertEqual(structure["levels"]["L2"]["distinct"], 2)
        self.assertEqual(structure["levels"]["L3"]["distinct"], 3)
        self.assertEqual(structure["first_level_usage"]["codes_used"], 1)
        self.assertAlmostEqual(structure["first_level_usage"]["entropy_bits"], 0.0)

    def test_distinct_triples_and_shared_items_are_counted(self):
        sid = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 2, 0]], dtype=np.int64)
        structure = forensics.sid_structure(sid)
        self.assertEqual(structure["collision_extension"]["distinct_triples"], 2)
        self.assertEqual(structure["collision_extension"]["items_sharing_a_triple"], 1)


class BeamStateTests(unittest.TestCase):
    """The survivor filter must follow the scorer's own scores, not the ancestry."""

    def test_a_filled_slot_is_not_alive(self):
        torch, _, _ = _torch_and_t5()
        steps = [{"indices": torch.tensor([0, 0]),
                  "scores": torch.tensor([-0.5, -1e9])}]
        live = _alive_chain(steps, 2)
        self.assertEqual(live[1], [True, False])

    def test_a_child_of_a_dead_parent_is_not_alive(self):
        torch, _, _ = _torch_and_t5()
        steps = [{"indices": torch.tensor([0, 1]),
                  "scores": torch.tensor([-0.5, -1e9])},
                 {"indices": torch.tensor([0, 1]),
                  "scores": torch.tensor([-1.0, -0.9])}]
        live = _alive_chain(steps, 2)
        self.assertEqual(live[1], [True, False])
        self.assertEqual(live[2], [True, False])


class _Forced:
    """A deterministic distribution where the second first-token beam dies at depth 2.

    Without the live-beam trace a reader would see two final sequences and assume
    both first tokens survived; the trace shows the prune.
    """

    def __init__(self, inner, config):
        self.inner = inner
        self.config = config
        self.step = 0

    def __call__(self, *args, **kwargs):
        torch, _, _ = _torch_and_t5()
        out = self.inner(*args, **kwargs)
        # With ``use_cache`` the decoder ids of every step but the first hold only
        # the token just consumed, which is exactly the previous generated token.
        decoder = kwargs.get("decoder_input_ids")
        logits = torch.full_like(out.logits, -1e4)
        if self.step == 0:
            logits[:, -1, 5] = 0.0
            logits[:, -1, 9] = -1.0
        else:
            last = decoder[:, -1]
            path5 = last == 5
            logits[path5, -1, 6] = 0.0
            logits[path5, -1, 8] = -0.1
            logits[~path5, -1, 8] = -5.0
            logits[~path5, -1, 6] = -6.0
        self.step += 1
        out.logits = logits
        return out


class BeamSurvivalTests(unittest.TestCase):
    def test_the_pruned_first_token_is_absent_from_the_live_beams(self):
        torch, T5Config, T5ForConditionalGeneration = _torch_and_t5()
        forensics._install_beam_probe()
        config = T5Config(vocab_size=32, d_model=16, d_ff=32, num_layers=1,
                          num_decoder_layers=1, num_heads=2, d_kv=8, dropout_rate=0.0,
                          pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
        torch.manual_seed(0)
        model = T5ForConditionalGeneration(config).eval()
        original_forward = model.forward
        model.forward = _Forced(original_forward, model.config)
        encoder_input = torch.zeros((1, 4), dtype=torch.long)
        recorder = forensics._BeamRecorder()
        previous, forensics._RECORDER = forensics._RECORDER, recorder
        try:
            generated = model.generate(input_ids=encoder_input, num_beams=2,
                                       num_return_sequences=2, max_new_tokens=3)
            steps = list(recorder.steps)
        finally:
            forensics._RECORDER = previous
        self.assertEqual(len(steps), 3)
        live = _alive_chain(steps, width=2)
        prompt = steps[0]["input_ids"].shape[1]
        after_first = steps[1]["input_ids"][:, prompt:]
        alive_after_first = {tuple(row) for row, keep in zip(after_first.tolist(), live[1]) if keep}
        self.assertEqual(alive_after_first, {(5,), (9,)})
        after_second = steps[2]["input_ids"][:, prompt:]
        alive_after_second = {tuple(row) for row, keep in zip(after_second.tolist(), live[2]) if keep}
        self.assertEqual(alive_after_second, {(5, 6), (5, 8)})
        self.assertFalse(any(prefix[:1] == (9,) for prefix in alive_after_second))
        self.assertEqual(tuple(generated.shape), (2, 4))


class _EosForced:
    """Force token 5, then EOS, so a hypothesis is finalized while others stay live."""

    def __init__(self, inner, config):
        self.inner = inner
        self.config = config
        self.step = 0

    def __call__(self, *args, **kwargs):
        torch, _, _ = _torch_and_t5()
        out = self.inner(*args, **kwargs)
        logits = torch.full_like(out.logits, -1e4)
        if self.step == 0:
            logits[:, -1, 5] = 0.0
            logits[:, -1, 6] = -0.5
        else:
            logits[:, -1, 1] = 0.0      # EOS
            logits[:, -1, 7] = -0.2
        self.step += 1
        out.logits = logits
        return out


class FinalizedBeamTests(unittest.TestCase):
    def test_a_finalized_hypothesis_is_recorded_separately_from_the_live_beams(self):
        torch, T5Config, T5ForConditionalGeneration = _torch_and_t5()
        forensics._install_beam_probe()
        config = T5Config(vocab_size=32, d_model=16, d_ff=32, num_layers=1,
                          num_decoder_layers=1, num_heads=2, d_kv=8, dropout_rate=0.0,
                          pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
        torch.manual_seed(0)
        model = T5ForConditionalGeneration(config).eval()
        model.forward = _EosForced(model.forward, model.config)
        recorder = forensics._BeamRecorder()
        previous, forensics._RECORDER = forensics._RECORDER, recorder
        try:
            model.generate(input_ids=torch.zeros((1, 4), dtype=torch.long), num_beams=2,
                           num_return_sequences=2, max_new_tokens=3)
            steps = list(recorder.steps)
            finalized = list(recorder.finalized)
        finally:
            forensics._RECORDER = previous
        self.assertTrue(finalized, "an EOS-ended hypothesis must be recorded")
        finalized_ids = [entry["ids"].tolist() for entry in finalized]
        self.assertTrue(any(ids[:2] == [0, 5] for ids in finalized_ids),
                        "the finalized hypothesis keeps its own token ids: " + str(finalized_ids))
        live = _alive_chain(steps, width=2)
        self.assertEqual(len(live), len(steps) + 1)
        # A finalized hypothesis is not an active beam: it must not be counted as one.
        last_live = live[-1]
        self.assertLessEqual(sum(last_live), 2)


def _torch_and_t5():
    import torch
    from transformers import T5Config, T5ForConditionalGeneration
    return torch, T5Config, T5ForConditionalGeneration


def _alive_chain(steps, width):
    live = [[row % width == 0 for row in range(len(steps[0]["indices"]))]]
    for step in steps:
        parent_live = live[-1]
        scores = step["scores"]
        alive = []
        for row in range(len(step["indices"])):
            parent = int(step["indices"][row])
            alive.append(bool(parent_live[parent]) and float(scores[row]) > -1e8)
        live.append(alive)
    return live


class PostprocessTests(unittest.TestCase):
    """The decomposition must follow the production evaluator's own rules."""

    @staticmethod
    def _select(rows, code_to_item, excluded, fallback, width, max_k=10):
        selected, chosen = [], set()
        for beam in range(width):
            item = code_to_item.get(tuple(rows[beam]))
            if item is None or item in chosen or item in excluded:
                continue
            selected.append(item)
            chosen.add(item)
            if len(selected) == max_k:
                break
        from_fallback = []
        if len(selected) < max_k:
            for item in fallback:
                if item in chosen or item in excluded:
                    continue
                selected.append(item)
                chosen.add(item)
                from_fallback.append(item)
                if len(selected) == max_k:
                    break
        return selected, from_fallback

    def test_invalid_duplicate_and_seen_rows_are_dropped_in_beam_order(self):
        rows = [[1, 1, 1, 0], [1, 1, 1, 0], [2, 2, 2, 0], [3, 3, 3, 0]]
        mapping = {(1, 1, 1, 0): 10, (2, 2, 2, 0): 20, (3, 3, 3, 0): 30}
        selected, fallback = self._select(rows, mapping, {20}, [99, 10], 4)
        self.assertEqual(selected, [10, 30, 99, 10][:2] + [99])
        self.assertEqual(fallback, [99])
        self.assertNotIn(20, selected)

    def test_the_fallback_only_fills_short_lists(self):
        rows = [[5, 5, 5, 0]] * 10
        mapping = {(5, 5, 5, 0): 50}
        selected, fallback = self._select(rows, mapping, set(), [1, 2, 3], 10)
        self.assertEqual(selected, [50, 1, 2, 3])
        self.assertEqual(fallback, [1, 2, 3])

    def test_an_excluded_target_is_never_selected(self):
        rows = [[4, 4, 4, 0]]
        mapping = {(4, 4, 4, 0): 40}
        selected, _ = self._select(rows, mapping, {40}, [7], 1, max_k=2)
        self.assertEqual(selected, [7])
        self.assertNotIn(40, selected)


class ImmutabilityTests(unittest.TestCase):
    def test_the_module_writes_only_inside_its_own_report_directory(self):
        allowed = forensics.OUT.resolve()
        self.assertTrue(str(allowed).endswith("results/research/sid_generation_forensics"))
        for path in (forensics.OUT, forensics.OUT / "cache"):
            self.assertEqual(Path(path).resolve().relative_to(forensics.REPO.resolve()).parts[:2],
                             ("results", "research"))

    def test_the_sid_loader_rejects_a_non_rectangular_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "item_sids.json"
            path.write_text(json.dumps({"0": [1, 2], "1": [1]}))
            with self.assertRaises(ValueError):
                forensics.load_sid(path)

    def test_the_sid_loader_keys_by_item_id_not_by_dictionary_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "item_sids.json"
            path.write_text(json.dumps({"1": [3, 3], "0": [2, 2]}))
            table = forensics.load_sid(path)
            self.assertEqual(table[0].tolist(), [2, 2])
            self.assertEqual(table[1].tolist(), [3, 3])

    def test_a_gap_in_the_item_ids_is_a_data_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "item_sids.json"
            path.write_text(json.dumps({"0": [1, 1], "2": [2, 2]}))
            with self.assertRaises(ValueError):
                forensics.load_sid(path)


if __name__ == "__main__":
    unittest.main()
