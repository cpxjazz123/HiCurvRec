"""The grouped executor must keep per-suite evidence and fail closed."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "run_suites.py"


def _write_case(directory: Path, name: str, body: str) -> None:
    (directory / ("test_" + name + ".py")).write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n" + body, encoding="utf-8")


class GroupedRunnerTests(unittest.TestCase):
    def _run(self, *args):
        done = subprocess.run([sys.executable, str(RUNNER), *map(str, args)],
                              capture_output=True, text=True, timeout=300)
        return done, json.loads(done.stdout.strip().splitlines()[-1])

    def test_two_suites_share_one_process_and_report_separately(self):
        with tempfile.TemporaryDirectory() as work:
            case = Path(work)
            _write_case(case, "math", "    def test_math(self):\n        self.assertEqual(1 + 1, 2)\n")
            _write_case(case, "gradient", "    def test_gradient(self):\n        self.assertTrue(True)\n")
            done, payload = self._run(case, "math", "gradient")
            self.assertEqual(done.returncode, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(set(payload["suites"]), {"math", "gradient"})
            self.assertEqual(payload["suites"]["math"]["tests_run"], 1)
            self.assertEqual(payload["suites"]["gradient"]["tests_run"], 1)
            self.assertIn("test_math", payload["suites"]["math"]["output"])

    def test_zero_tests_is_a_failure_not_a_pass(self):
        with tempfile.TemporaryDirectory() as work:
            case = Path(work)
            # A module with no test method: unittest collects zero tests.
            (case / "test_ddp.py").write_text("import unittest\n\n\nclass T(unittest.TestCase):\n    pass\n",
                                              encoding="utf-8")
            done, payload = self._run(case, "ddp")
            self.assertNotEqual(done.returncode, 0)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["suites"]["ddp"]["tests_run"], 0)
            self.assertFalse(payload["suites"]["ddp"]["ok"])

    def test_skipped_and_expected_failure_are_failures(self):
        with tempfile.TemporaryDirectory() as work:
            case = Path(work)
            _write_case(case, "entrypoint",
                        "    @unittest.skip('not really run')\n"
                        "    def test_skipped(self):\n        pass\n")
            done, payload = self._run(case, "entrypoint")
            self.assertNotEqual(done.returncode, 0)
            self.assertEqual(payload["suites"]["entrypoint"]["skipped"], 1)
            self.assertFalse(payload["suites"]["entrypoint"]["ok"])

    def test_import_error_is_a_failed_suite(self):
        with tempfile.TemporaryDirectory() as work:
            case = Path(work)
            (case / "test_math.py").write_text("import does_not_exist_anywhere\n", encoding="utf-8")
            done, payload = self._run(case, "math")
            self.assertNotEqual(done.returncode, 0)
            self.assertFalse(payload["suites"]["math"]["ok"])
            self.assertGreaterEqual(payload["suites"]["math"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
