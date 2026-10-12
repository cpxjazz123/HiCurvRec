import ast
import os
import tempfile
import threading
import unittest
from pathlib import Path

TRAINER_PATH = Path(__file__).resolve().parents[1] / "train_HG-Rec.py"
_TREE = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
_FUNCTIONS = {
    node.name: node
    for node in _TREE.body
    if isinstance(node, ast.FunctionDef)
}
_SYNC_NODES = [
    node
    for node in _TREE.body
    if isinstance(node, ast.ClassDef) and node.name == "RunDirectoryNotPublishedError"
]
_SYNC_NODES.extend(_FUNCTIONS[name] for name in ("_publish_run_dir", "_read_run_dir"))
_NAMESPACE = {"os": os, "tempfile": tempfile}
exec(compile(ast.Module(body=_SYNC_NODES, type_ignores=[]), str(TRAINER_PATH), "exec"), _NAMESPACE)
RunDirectoryNotPublishedError = _NAMESPACE["RunDirectoryNotPublishedError"]
_publish_run_dir = _NAMESPACE["_publish_run_dir"]
_read_run_dir = _NAMESPACE["_read_run_dir"]

_PROTOCOL_VALUES = {
    "NUM_EPOCHS": 150,
    "NO_EVAL": True,
    "SKIP_TEST": False,
    "SEED": 42,
    "BEAM_SIZE": 20,
    "TOPK_LIST": [5, 10],
    "SCREEN_BASELINE_LOG": "",
}


class RunDirectorySyncTest(unittest.TestCase):
    def test_empty_file_is_rejected_with_named_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_dir.txt"
            path.touch()
            with self.assertRaises(RunDirectoryNotPublishedError):
                _read_run_dir(str(path))

    def test_completed_publish_and_reused_value_read_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "run_dir.txt")
            for value in ("run-2026-10-13", "run-2026-10-13"):
                _publish_run_dir(directory, value)
                self.assertEqual(_read_run_dir(path), value)
                self.assertEqual(Path(path).read_text(encoding="utf-8"), value + "\n")

    def test_racing_reader_sees_only_complete_old_or_new_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "run_dir.txt")
            _publish_run_dir(directory, "previous-run")
            staged = threading.Event()
            permit_replace = threading.Event()
            original_replace = os.replace

            def paused_replace(source, destination):
                staged.set()
                if not permit_replace.wait(timeout=5):
                    raise AssertionError("test did not release the publisher")
                original_replace(source, destination)

            os.replace = paused_replace
            try:
                publisher = threading.Thread(
                    target=_publish_run_dir, args=(directory, "current-run")
                )
                publisher.start()
                self.assertTrue(staged.wait(timeout=5), "publisher did not stage its file")
                observed = [_read_run_dir(path)]
                permit_replace.set()
                publisher.join(timeout=5)
                self.assertFalse(publisher.is_alive(), "publisher did not finish")
                observed.append(_read_run_dir(path))
            finally:
                permit_replace.set()
                os.replace = original_replace
            self.assertEqual(observed, ["previous-run", "current-run"])
            self.assertTrue(all(value in {"previous-run", "current-run"} for value in observed))

    def test_frozen_protocol_values_are_unchanged(self):
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in _TREE.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in _PROTOCOL_VALUES
        }
        self.assertEqual(
            {name: assignments[name] for name in _PROTOCOL_VALUES}, _PROTOCOL_VALUES
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
