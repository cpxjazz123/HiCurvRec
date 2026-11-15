"""Run the real ``evaluate`` on a stub model and check the SID dump it writes.

The pure-format test does not prove the evaluator actually writes the file, and a full Stage 3 run costs
hours.  This drives ``evaluate`` with two rows, a stub generator that returns known SID tuples, and a
stub dataset whose code table maps those tuples to items, then reads the JSONL back.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE.parent / "train_HG-Rec.py"

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - the analysis interpreter has no torch
    torch = None


def stub_t5_if_unavailable():
    """The analysis interpreter ships transformers without T5 support; the evaluator never instantiates it."""
    try:
        import transformers
    except ModuleNotFoundError:
        return
    try:
        transformers.T5ForConditionalGeneration
    except Exception:
        class _StubT5:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("stub T5 must not be instantiated in this test")

        transformers.T5Config = _StubT5
        transformers.T5ForConditionalGeneration = _StubT5


def load_trainer():
    stub_t5_if_unavailable()
    sys.path.insert(0, str(TRAINER.parent))
    spec = importlib.util.spec_from_file_location("hg_rec_trainer", TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubModel:
    def __init__(self, rows):
        self.rows = rows

    def eval(self):
        return self

    def generate(self, **kwargs):
        return self.rows


class StubDataset:
    n_digit = 3
    code_to_item = {(1, 2, 3): 7, (4, 5, 6): 9, (7, 8, 9): 11}
    item_to_code = {item: code for code, item in code_to_item.items()}
    fallback_item_ids = tuple(range(20, 40))


@unittest.skipIf(torch is None, "torch is not installed in this interpreter")
class EvaluateDumpSmokeTest(unittest.TestCase):
    def test_evaluate_writes_one_line_per_row_with_sids(self):
        trainer = load_trainer()
        # two rows, four beams each; the last tuple duplicates the first and must be dropped
        beams = [
            [0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9], [0, 1, 2, 3], [0, 4, 5, 6],   # row 0
            [0, 4, 5, 6], [0, 7, 8, 9], [0, 1, 2, 3], [0, 7, 8, 9], [0, 4, 5, 6],   # row 1
        ]
        model = StubModel(torch.tensor(beams, dtype=torch.long))
        batch = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "target_item": torch.tensor([9, 7]),
            "labels": torch.zeros(2, 4, dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory) / "predictions" / "rank1.jsonl"
            recalls, ndcgs, n_eval = trainer.evaluate(
                model, [batch], StubDataset(), topk_list=[5], beam_size=5,
                device=torch.device("cpu"), sid_dump_path=str(dump),
                sid_dump_rank=1, sid_dump_world=4,
            )
            lines = [json.loads(line) for line in dump.read_text().splitlines()]
        self.assertEqual(n_eval, 2)
        self.assertEqual(len(lines), 2)
        self.assertEqual([row["global_row"] for row in lines], [1, 5])
        # row 0's beams map to items 7, 9, 11 so target 9 sits at rank 1; row 1's map to 9, 11, 7
        # so target 7 sits at rank 2
        self.assertEqual([row["target_rank"] for row in lines], [1, 2])
        for row in lines:
            self.assertEqual(len(row["topk"]), 5)
            # the item at target_rank is the target, so its generated SID is the target SID
            hit = row["topk"][row["target_rank"]]
            self.assertEqual(hit["item"], row["target_item"])
            self.assertEqual(hit["sid_generated"], row["target_sid"])
        self.assertEqual([entry["item"] for entry in lines[0]["topk"][:3]], [7, 9, 11])
        self.assertEqual(lines[0]["topk"][2]["sid_canonical"], [7, 8, 9])
        filler = [entry for entry in lines[0]["topk"] if entry["sid_generated"] is None]
        self.assertTrue(filler, "the deterministic fallback fill must be visible and unlabelled")
        self.assertGreater(recalls[5], 0)


if __name__ == "__main__":
    unittest.main()
