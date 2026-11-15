"""The SID prediction dump must record the predicted items *and* their SID tokens.

The evaluator already knew the target's rank but threw the prediction away, so a wrong row said only
"not in the top 10" with no way to see which SID it had chosen instead.  This pins the line format: a
generated SID for items that came from the beam, `None` for the deterministic fallback fill (which is
not a model prediction), the canonical SID of every predicted item, and an unchanged `None` rank when
the target missed the list.
"""

import ast
import json
import unittest
from pathlib import Path

TRAINER_PATH = Path(__file__).resolve().parents[1] / "train_HG-Rec.py"
_tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
_function = next(node for node in _tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "sid_prediction_line")
_module = ast.Module(body=[_function], type_ignores=[])
_namespace = {}
exec(compile(_module, str(TRAINER_PATH), "exec"), _namespace)
sid_prediction_line = _namespace["sid_prediction_line"]

CANONICAL = {7: (1, 2, 3), 9: (4, 5, 6), 11: (7, 8, 9)}


class SidPredictionDumpTest(unittest.TestCase):
    def test_full_row_records_generated_and_canonical_sids(self):
        line = sid_prediction_line(
            global_row=4021, target_item=9, target_sid=(4, 5, 6), target_rank=2,
            selected=[7, 9, 11], generated_sids=[(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            canonical_sids=CANONICAL)
        self.assertEqual(line["global_row"], 4021)
        self.assertEqual(line["target_rank"], 2)
        self.assertEqual(line["target_sid"], [4, 5, 6])
        self.assertEqual([entry["item"] for entry in line["topk"]], [7, 9, 11])
        self.assertEqual([entry["sid_generated"] for entry in line["topk"]],
                         [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual([entry["sid_canonical"] for entry in line["topk"]],
                         [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        json.dumps(line)          # must be serialisable as written

    def test_miss_keeps_none_rank_and_fallback_has_no_generated_sid(self):
        line = sid_prediction_line(
            global_row=5, target_item=11, target_sid=(7, 8, 9), target_rank=None,
            selected=[7, 9], generated_sids=[(1, 2, 3), None], canonical_sids=CANONICAL)
        self.assertIsNone(line["target_rank"])
        self.assertEqual(line["topk"][1]["sid_generated"], None)
        self.assertEqual(line["topk"][1]["sid_canonical"], [4, 5, 6])

    def test_unknown_item_has_no_canonical_sid(self):
        line = sid_prediction_line(
            global_row=0, target_item=7, target_sid=(1, 2, 3), target_rank=0,
            selected=[999], generated_sids=[(1, 1, 1)], canonical_sids=CANONICAL)
        self.assertIsNone(line["topk"][0]["sid_canonical"])


if __name__ == "__main__":
    unittest.main()
