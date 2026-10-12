import ast
import unittest
from pathlib import Path

TRAINER_PATH = Path("/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/train_HG-Rec.py")
_tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
_function = next(
    node
    for node in _tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "global_row_index"
)
_module = ast.Module(body=[_function], type_ignores=[])
_namespace = {}
exec(compile(_module, str(TRAINER_PATH), "exec"), _namespace)
global_row_index = _namespace["global_row_index"]


class GlobalRowIndexTest(unittest.TestCase):
    def test_rank_strided_subset_mapping(self):
        self.assertEqual(
            [global_row_index(2, local, 4) for local in range(4)],
            [2, 6, 10, 14],
        )
        self.assertEqual(global_row_index(0, 0, 4), 0)


if __name__ == "__main__":
    unittest.main()
